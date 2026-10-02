"""Monthly acquisition cohorts and their retention."""

from __future__ import annotations

import pandas as pd


def cohort_retention(transactions: pd.DataFrame) -> pd.DataFrame:
    """Share of each acquisition cohort that purchased in month N after acquisition.

    Month 0 is the acquisition month and is 100% by construction, so it is kept
    as the denominator rather than presented as a result.
    """
    working = transactions.copy()
    working["order_month"] = working["invoice_date"].dt.to_period("M")
    working["cohort"] = working.groupby("customer_id")["order_month"].transform("min")
    working["month_index"] = (
        (working["order_month"] - working["cohort"]).apply(lambda offset: offset.n)
    )

    active = (
        working.groupby(["cohort", "month_index"])["customer_id"]
        .nunique()
        .reset_index(name="active_customers")
    )
    size = active[active["month_index"] == 0][["cohort", "active_customers"]].rename(
        columns={"active_customers": "cohort_size"}
    )
    merged = active.merge(size, on="cohort")
    merged["retention"] = merged["active_customers"] / merged["cohort_size"]
    return merged


def retention_matrix(retention: pd.DataFrame) -> pd.DataFrame:
    matrix = retention.pivot(index="cohort", columns="month_index", values="retention")
    matrix.index = matrix.index.astype(str)
    return matrix
