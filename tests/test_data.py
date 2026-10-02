import pandas as pd

from clv.data import clean


def _raw(rows):
    frame = pd.DataFrame(
        rows, columns=["Invoice", "Customer ID", "InvoiceDate", "Quantity", "Price"]
    )
    frame["Invoice"] = frame["Invoice"].astype("string")
    frame["StockCode"] = "X"
    frame["Description"] = "item"
    frame["Country"] = "UK"
    return frame


def test_cancellations_are_dropped():
    """A return must not read as a second purchase."""
    raw = _raw([("489434", 1, "2009-12-01", 5, 2.0), ("C489434", 1, "2009-12-02", -5, 2.0)])
    result = clean(raw)
    assert len(result) == 1
    assert not result["invoice"].str.startswith("C").any()


def test_rows_without_a_customer_are_dropped():
    raw = _raw([("489434", 1, "2009-12-01", 5, 2.0), ("489435", None, "2009-12-01", 5, 2.0)])
    assert len(clean(raw)) == 1


def test_revenue_is_quantity_times_price():
    raw = _raw([("489434", 1, "2009-12-01", 5, 2.5)])
    assert clean(raw)["revenue"].iloc[0] == 12.5
