"""Fit both models, score them on the same held-out customers, write the report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cohorts import cohort_retention, retention_matrix
from .config import GROSS_MARGIN, SPLIT
from .data import transactions
from .evaluate import decile_table, point_metrics
from .models import (
    LGBM_FEATURES,
    add_engineered_features,
    fit_bgnbd,
    fit_gamma_gamma,
    fit_lightgbm,
    predict_expected_value,
    split_customers,
)
from .rfm import summary

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def run(draws: int = 1000, chains: int = 2) -> dict:
    orders = transactions()
    features = add_engineered_features(summary(orders))
    horizon = int(features["holdout_days"].iloc[0])

    train, test = split_customers(features)

    # BG/NBD and Gamma-Gamma are unsupervised with respect to the holdout, but
    # they are still fitted on the training customers only, so that both
    # approaches are scored on customers neither has seen.
    bgnbd = fit_bgnbd(train, draws=draws, chains=chains)
    gamma_gamma = fit_gamma_gamma(train, draws=draws, chains=chains)
    test = test.reset_index(drop=True)
    test["pred_bgnbd"] = predict_expected_value(bgnbd, gamma_gamma, test, horizon)

    lgbm = fit_lightgbm(train)
    test["pred_lgbm"] = lgbm.predict(test[LGBM_FEATURES]).clip(min=0)

    results = {
        "horizon_days": horizon,
        "calibration_end": SPLIT.calibration_end,
        "holdout_end": SPLIT.holdout_end,
        "customers_total": int(len(features)),
        "customers_train": int(len(train)),
        "customers_test": int(len(test)),
        "repeat_rate": float((features["frequency"] > 0).mean()),
        "holdout_active_rate": float((test["holdout_revenue"] > 0).mean()),
        "bgnbd": point_metrics(test["holdout_revenue"], test["pred_bgnbd"]),
        "lightgbm": point_metrics(test["holdout_revenue"], test["pred_lgbm"]),
    }

    REPORTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    deciles = {
        "bgnbd": decile_table(test["holdout_revenue"], test["pred_bgnbd"]),
        "lightgbm": decile_table(test["holdout_revenue"], test["pred_lgbm"]),
    }
    for name, table in deciles.items():
        table.to_csv(REPORTS / f"deciles_{name}.csv", index=False)

    retention = cohort_retention(orders)
    retention_matrix(retention).to_csv(REPORTS / "cohort_retention.csv")

    # Margin, not revenue: a retention budget is spent against contribution.
    test["expected_margin_bgnbd"] = test["pred_bgnbd"] * GROSS_MARGIN
    test.to_parquet(PROCESSED / "scored_customers.parquet", index=False)

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    return {"results": results, "deciles": deciles, "test": test, "retention": retention}


if __name__ == "__main__":
    output = run()
    print(json.dumps(output["results"], indent=2))
    for name, table in output["deciles"].items():
        print(f"\n=== deciles: {name}")
        print(table.to_string(index=False))
