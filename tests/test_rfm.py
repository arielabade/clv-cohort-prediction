"""The RFM conventions are easy to get backwards, so they are pinned here."""

import pandas as pd

from clv.rfm import summary


def _orders(rows):
    frame = pd.DataFrame(rows, columns=["customer_id", "invoice_date", "revenue"])
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    return frame


def test_frequency_counts_repeat_purchases_not_total():
    # Three purchase occasions means frequency 2 under the BG/NBD convention.
    orders = _orders([(1, "2011-01-01", 100.0), (1, "2011-02-01", 100.0), (1, "2011-03-01", 100.0)])
    row = summary(orders, "2011-06-09", "2011-12-09").iloc[0]
    assert row["frequency"] == 2


def test_single_purchase_customer_has_frequency_zero():
    orders = _orders([(1, "2011-01-01", 100.0)])
    row = summary(orders, "2011-06-09", "2011-12-09").iloc[0]
    assert row["frequency"] == 0


def test_recency_is_age_at_last_purchase_not_days_since():
    # First 2011-01-01, last 2011-03-01: recency is 59 days (the gap between
    # them), NOT the distance from the last purchase to the period end.
    orders = _orders([(1, "2011-01-01", 100.0), (1, "2011-03-01", 100.0)])
    row = summary(orders, "2011-06-09", "2011-12-09").iloc[0]
    assert row["recency"] == 59
    assert row["T"] == 159


def test_monetary_value_excludes_the_first_order():
    # Gamma-Gamma models repeat-purchase value, so the acquisition basket of
    # 1000 must not drag the average up.
    orders = _orders(
        [(1, "2011-01-01", 1000.0), (1, "2011-02-01", 100.0), (1, "2011-03-01", 200.0)]
    )
    row = summary(orders, "2011-06-09", "2011-12-09").iloc[0]
    assert row["monetary_value"] == 150.0


def test_customers_first_seen_in_holdout_are_excluded():
    orders = _orders([(1, "2011-01-01", 100.0), (2, "2011-08-01", 500.0)])
    result = summary(orders, "2011-06-09", "2011-12-09")
    assert set(result["customer_id"]) == {1}


def test_holdout_outcomes_are_measured_after_the_cutoff_only():
    orders = _orders([(1, "2011-01-01", 100.0), (1, "2011-08-01", 400.0), (1, "2012-05-01", 999.0)])
    row = summary(orders, "2011-06-09", "2011-12-09").iloc[0]
    assert row["holdout_purchases"] == 1
    assert row["holdout_revenue"] == 400.0  # the 2012 order is past holdout_end
