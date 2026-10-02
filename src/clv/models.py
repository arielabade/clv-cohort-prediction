"""The two approaches being compared.

BG/NBD + Gamma-Gamma is a generative model of buying behaviour: it assumes a
purchase process and a dropout process and infers their parameters. It needs no
labels, so it can be fitted on the calibration window alone.

LightGBM is supervised: it learns the mapping from calibration features to
realised holdout spend. It can exploit patterns the probabilistic model has no
way to express, but it needs labelled history to learn from, and that history
has to come from somewhere.

The comparison is only fair if both are scored on customers neither of them was
fitted on, which is why `split_customers` exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel

RANDOM_SEED = 20261002

LGBM_FEATURES = [
    "frequency",
    "recency",
    "T",
    "monetary_value",
    "calibration_revenue",
    "avg_order_value",
    "purchase_rate",
]


def split_customers(features: pd.DataFrame, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a random set of CUSTOMERS, not of transactions.

    The calibration/holdout split is already temporal; this second split exists
    so the supervised model has customers it has never seen. Splitting on
    transactions instead would leak a customer's own future spend into their
    training row.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = features.sample(frac=1.0, random_state=rng.integers(2**31)).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - test_size))
    return shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy()


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    working = features.copy()
    orders = working["frequency"] + 1
    working["avg_order_value"] = working["calibration_revenue"] / orders
    # Purchases per day of observed tenure. Guarded because a customer whose
    # first and only order is on the last calibration day has T = 0.
    working["purchase_rate"] = np.where(working["T"] > 0, orders / working["T"], 0.0)
    return working


def fit_bgnbd(features: pd.DataFrame, draws: int = 1000, chains: int = 2) -> BetaGeoModel:
    data = features[["customer_id", "frequency", "recency", "T"]].copy()
    model = BetaGeoModel(data=data)
    model.build_model()
    model.fit(draws=draws, chains=chains, random_seed=RANDOM_SEED, progressbar=False)
    return model


def fit_gamma_gamma(features: pd.DataFrame, draws: int = 1000, chains: int = 2) -> GammaGammaModel:
    """Fit on repeat buyers only.

    Gamma-Gamma models the value of repeat transactions, so a customer with
    frequency 0 has no repeat value to learn from and must be excluded. Feeding
    them in with monetary_value 0 drags the spend estimate toward zero.
    """
    repeat_buyers = features[(features["frequency"] > 0) & (features["monetary_value"] > 0)]
    data = repeat_buyers[["customer_id", "monetary_value", "frequency"]].copy()
    model = GammaGammaModel(data=data)
    model.build_model()
    model.fit(draws=draws, chains=chains, random_seed=RANDOM_SEED, progressbar=False)
    return model


def predict_expected_value(
    bgnbd: BetaGeoModel,
    gamma_gamma: GammaGammaModel,
    features: pd.DataFrame,
    horizon_days: int,
) -> pd.Series:
    """Expected spend over `horizon_days`: expected purchases x expected spend each.

    Computed explicitly rather than through expected_customer_lifetime_value so
    that the two components can be inspected separately, and so the horizon
    matches the holdout window exactly.
    """
    purchase_data = features[["customer_id", "frequency", "recency", "T"]].copy()
    expected_purchases = (
        bgnbd.expected_purchases(data=purchase_data, future_t=horizon_days)
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )

    spend_data = features[["customer_id", "monetary_value", "frequency"]].copy()
    expected_spend = (
        gamma_gamma.expected_customer_spend(data=spend_data).mean(dim=("chain", "draw")).to_numpy()
    )

    return pd.Series(expected_purchases * expected_spend, index=features.index, name="predicted")


def fit_lightgbm(train: pd.DataFrame) -> LGBMRegressor:
    """Deliberately small model.

    A few thousand customers and seven features do not support a large
    ensemble; the shallow, regularised configuration is there to stop it
    memorising the training customers.
    """
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(train[LGBM_FEATURES], train["holdout_revenue"])
    return model
