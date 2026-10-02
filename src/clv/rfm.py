"""Calibration/holdout RFM summary.

Follows the BG/NBD convention, which trips people up:

    frequency  number of REPEAT purchases, so a customer who bought once has
               frequency 0, not 1
    recency    age at the last purchase, i.e. days between first and last
               purchase. Not "days since last purchase", which is the meaning
               the word carries in an RFM segmentation deck
    T          days between the first purchase and the end of the observation
               window

Getting recency backwards silently inverts the model's notion of who is alive.
"""

from __future__ import annotations

import pandas as pd

from .config import SPLIT


def summary(transactions: pd.DataFrame, calibration_end: str | None = None,
            holdout_end: str | None = None) -> pd.DataFrame:
    """Build per-customer calibration features and holdout outcomes.

    Only customers who made their first purchase during the calibration window
    are included. A customer first seen in the holdout window has no
    calibration history to predict from.
    """
    calibration_end = pd.Timestamp(calibration_end or SPLIT.calibration_end)
    holdout_end = pd.Timestamp(holdout_end or SPLIT.holdout_end)

    calibration = transactions[transactions["invoice_date"] <= calibration_end]
    holdout = transactions[
        (transactions["invoice_date"] > calibration_end)
        & (transactions["invoice_date"] <= holdout_end)
    ]

    first = calibration.groupby("customer_id")["invoice_date"].min()
    last = calibration.groupby("customer_id")["invoice_date"].max()
    counts = calibration.groupby("customer_id").size()  # purchase occasions, one per day

    features = pd.DataFrame(
        {
            "frequency": counts - 1,  # repeat purchases
            "recency": (last - first).dt.days,
            "T": (calibration_end - first).dt.days,
        }
    )

    # Gamma-Gamma is fitted on the value of REPEAT purchases, so the first
    # order is excluded from the monetary average. Including it biases the
    # spend estimate toward the acquisition basket, which behaves differently.
    repeats = calibration.sort_values("invoice_date").groupby("customer_id").apply(
        lambda group: group.iloc[1:]["revenue"].mean(), include_groups=False
    )
    features["monetary_value"] = repeats.fillna(0.0)

    features["calibration_revenue"] = calibration.groupby("customer_id")["revenue"].sum()
    features["holdout_purchases"] = holdout.groupby("customer_id").size()
    features["holdout_revenue"] = holdout.groupby("customer_id")["revenue"].sum()
    features[["holdout_purchases", "holdout_revenue"]] = features[
        ["holdout_purchases", "holdout_revenue"]
    ].fillna(0.0)

    features["holdout_days"] = (holdout_end - calibration_end).days
    return features.reset_index().rename(columns={"index": "customer_id"})
