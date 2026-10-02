# Data

Populated by the pipeline; not committed.

```
data/raw/        Online Retail II workbook + Parquet cache (~90s to parse once)
data/processed/  scored_customers.parquet, written by the pipeline
```

## Source

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning
Repository. A UK online giftware retailer, largely wholesale buyers, 2009-12-01 to 2011-12-09.

**All data is real.** Nothing in this project is simulated. The raw row count is asserted against the
published figure (1,067,371) on load, so a changed download fails the run rather than shifting the
results.

The only declared assumption is a 30% gross margin, used to convert predicted revenue into
contribution. It lives in `src/clv/config.py` and affects no model.
