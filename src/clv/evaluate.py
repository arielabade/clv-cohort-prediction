"""Holdout evaluation.

Point accuracy is reported, but the decile table is the one that matters for a
budget decision: a retention programme does not spend on the average customer,
it spends on the top of a ranking. A model can have mediocre MAE and still sort
customers well enough to be worth using.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    spearman = stats.spearmanr(actual, predicted)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "spearman": float(spearman.statistic),
    }


def decile_table(actual: pd.Series, predicted: pd.Series, deciles: int = 10) -> pd.DataFrame:
    """Rank customers by prediction and report what each decile was actually worth."""
    frame = pd.DataFrame({"actual": actual.to_numpy(), "predicted": predicted.to_numpy()})
    # Rank first so ties do not collapse the bins: with many identical
    # predictions, qcut on the raw values would drop whole deciles.
    frame["rank"] = frame["predicted"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(frame["rank"], deciles, labels=range(1, deciles + 1))

    table = (
        frame.groupby("decile", observed=True)
        .agg(
            customers=("actual", "size"),
            predicted_mean=("predicted", "mean"),
            actual_mean=("actual", "mean"),
            actual_total=("actual", "sum"),
        )
        .reset_index()
    )
    overall = frame["actual"].mean()
    table["lift_vs_average"] = table["actual_mean"] / overall
    table["share_of_actual_value"] = table["actual_total"] / frame["actual"].sum()
    return table
