import numpy as np
import pandas as pd

from clv.models import add_engineered_features, split_customers


def _features(n=100):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "frequency": rng.integers(0, 10, n),
            "recency": rng.integers(0, 300, n),
            "T": rng.integers(1, 400, n),
            "monetary_value": rng.gamma(2, 100, n),
            "calibration_revenue": rng.gamma(2, 500, n),
            "holdout_revenue": rng.gamma(2, 300, n),
        }
    )


def test_split_is_deterministic():
    first, _ = split_customers(_features())
    second, _ = split_customers(_features())
    pd.testing.assert_frame_equal(first, second)


def test_split_partitions_customers_without_overlap():
    train, test = split_customers(_features())
    assert set(train["customer_id"]).isdisjoint(set(test["customer_id"]))
    assert len(train) + len(test) == 100


def test_purchase_rate_guards_against_zero_tenure():
    """A customer whose only order lands on the last calibration day has T = 0."""
    features = pd.DataFrame(
        {"frequency": [0], "T": [0], "calibration_revenue": [50.0], "monetary_value": [0.0]}
    )
    result = add_engineered_features(features)
    assert np.isfinite(result["purchase_rate"]).all()
    assert result["purchase_rate"].iloc[0] == 0.0


def test_avg_order_value_divides_by_occasions_not_repeats():
    features = pd.DataFrame(
        {"frequency": [1], "T": [100], "calibration_revenue": [200.0], "monetary_value": [100.0]}
    )
    # frequency 1 means two purchase occasions, so AOV is 100, not 200.
    assert add_engineered_features(features)["avg_order_value"].iloc[0] == 100.0
