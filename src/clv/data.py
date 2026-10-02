"""Load Online Retail II, clean it, and cache it as Parquet.

The source is a 45MB Excel workbook that takes about 90 seconds to parse, so
the parsed form is cached. Reading Excel on every run would make the project
unpleasant enough that nobody reproduces it.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import CLEANING, DATA_URL, EXCEL_NAME, EXPECTED_RAW_ROWS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE = RAW_DIR / "online_retail_ii.parquet"


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "online_retail_ii.zip"
    workbook = RAW_DIR / EXCEL_NAME
    if not workbook.exists():
        if not archive.exists():
            print(f"downloading {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(EXCEL_NAME, RAW_DIR)
    return workbook


def load_raw(force: bool = False) -> pd.DataFrame:
    """Parse both sheets into one frame, cached as Parquet."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    workbook = download()
    print("parsing workbook (about 90 seconds)")
    sheets = pd.ExcelFile(workbook)
    frame = pd.concat([sheets.parse(name) for name in sheets.sheet_names], ignore_index=True)

    # Invoice, StockCode and Description mix integers and strings in the source
    # file; coerce them so the cached schema is stable.
    for column in ("Invoice", "StockCode", "Description", "Country"):
        frame[column] = frame[column].astype("string")

    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"expected {EXPECTED_RAW_ROWS:,} rows, found {len(frame):,}")

    frame.to_parquet(CACHE, index=False)
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce to attributable purchases.

    Returns one row per line item with a `revenue` column.
    """
    working = frame.copy()
    if CLEANING.drop_cancellations:
        is_cancellation = working["Invoice"].str.startswith("C").fillna(False)
        working = working[~is_cancellation]
    if CLEANING.require_customer_id:
        working = working[working["Customer ID"].notna()]
    working = working[
        (working["Quantity"] >= CLEANING.min_quantity) & (working["Price"] >= CLEANING.min_price)
    ]

    working = working.rename(
        columns={"Customer ID": "customer_id", "InvoiceDate": "invoice_date", "Invoice": "invoice"}
    )
    working["customer_id"] = working["customer_id"].astype(int)
    working["revenue"] = working["Quantity"] * working["Price"]
    return working[["customer_id", "invoice", "invoice_date", "Quantity", "Price", "revenue", "Country"]]


def transactions(force: bool = False) -> pd.DataFrame:
    """One row per customer per PURCHASE DAY.

    Not per invoice. BG/NBD assumes at most one transaction per customer per
    time unit, and the time unit here is a day. In this dataset 9.1% of
    customer-days carry more than one invoice — a single shopping session split
    across several documents — which would inflate observed frequency by 11.7%
    and, with it, every predicted purchase count.

    `invoices` is kept so the split can be inspected rather than taken on
    trust.
    """
    lines = clean(load_raw(force=force))
    lines = lines.copy()
    lines["invoice_date"] = pd.to_datetime(lines["invoice_date"]).dt.normalize()
    occasions = (
        lines.groupby(["customer_id", "invoice_date"])
        .agg(revenue=("revenue", "sum"), invoices=("invoice", "nunique"))
        .reset_index()
    )
    return occasions.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)
