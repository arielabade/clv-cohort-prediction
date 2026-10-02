import pandas as pd

from clv.cohorts import cohort_retention, retention_matrix


def _orders(rows):
    frame = pd.DataFrame(rows, columns=["customer_id", "invoice_date", "revenue"])
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    return frame


def test_acquisition_month_is_fully_retained_by_construction():
    orders = _orders([(1, "2011-01-05", 10.0), (2, "2011-01-20", 10.0)])
    result = cohort_retention(orders)
    assert result[result["month_index"] == 0]["retention"].eq(1.0).all()


def test_retention_is_share_of_the_original_cohort():
    orders = _orders(
        [(1, "2011-01-05", 10.0), (2, "2011-01-20", 10.0), (1, "2011-02-10", 10.0)]
    )
    result = cohort_retention(orders)
    month_one = result[(result["month_index"] == 1)].iloc[0]
    assert month_one["retention"] == 0.5  # one of two customers came back


def test_matrix_is_cohorts_by_month_index():
    orders = _orders([(1, "2011-01-05", 10.0), (1, "2011-02-10", 10.0)])
    matrix = retention_matrix(cohort_retention(orders))
    assert list(matrix.columns) == [0, 1]
