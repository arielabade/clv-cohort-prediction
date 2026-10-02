"""Dataset, cleaning rules and the calibration/holdout split.

The split is the most consequential choice in this project, so it lives here
rather than inside a function.
"""

from __future__ import annotations

from dataclasses import dataclass

# Online Retail II, UCI Machine Learning Repository.
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
EXCEL_NAME = "online_retail_II.xlsx"

# Published shape of the dataset, asserted on load so a changed or truncated
# download fails the run instead of quietly shifting every result.
EXPECTED_RAW_ROWS = 1_067_371


@dataclass(frozen=True)
class Cleaning:
    """What counts as a purchase.

    cancellations
        Invoices prefixed with "C" are returns. They carry negative quantities
        and must not be counted as transactions, or a customer who bought once
        and returned it looks like a two-purchase customer.
    require_customer_id
        About a fifth of rows have no customer. They are real revenue but they
        cannot be attributed to a person, and CLV is a per-person quantity.
    """

    drop_cancellations: bool = True
    require_customer_id: bool = True
    min_quantity: int = 1
    min_price: float = 0.01


@dataclass(frozen=True)
class Split:
    """Time-based calibration/holdout split.

    Split by DATE, never at random. A random split would let a customer's
    later purchases train a model that is then scored on their earlier ones,
    which leaks the future into the past and inflates every metric.

    The dataset runs 2009-12-01 to 2011-12-09. Calibration takes the first
    ~21 months, holdout the final ~6, which is long enough for a quarterly
    purchase cycle to show up at least once.
    """

    calibration_end: str = "2011-06-09"
    holdout_end: str = "2011-12-09"


CLEANING = Cleaning()
SPLIT = Split()

# Monetary assumptions. Online Retail II is a wholesale/gift business and
# publishes revenue, not margin, so margin is a declared assumption.
GROSS_MARGIN = 0.30
