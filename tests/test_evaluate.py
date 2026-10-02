import numpy as np
import pandas as pd

from clv.evaluate import decile_table, point_metrics


def test_point_metrics_report_direction_of_error():
    actual = pd.Series([10.0, 20.0, 30.0])
    over = pd.Series([15.0, 25.0, 35.0])
    assert point_metrics(actual, over)["bias"] == 5.0
    assert point_metrics(actual, actual)["mae"] == 0.0


def test_perfect_ranking_scores_spearman_one():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert point_metrics(actual, pd.Series([9.0, 10.0, 11.0, 12.0]))["spearman"] == 1.0


def test_top_decile_holds_the_highest_actual_value():
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.gamma(2.0, 100.0, size=1000))
    table = decile_table(actual, actual)  # a perfect model
    assert table.iloc[0]["lift_vs_average"] > table.iloc[-1]["lift_vs_average"]
    assert table["share_of_actual_value"].sum() == np.float64(1.0).round(6)


def test_tied_predictions_still_produce_ten_deciles():
    """Guards the rank-then-bin step.

    A model that predicts the same value for many customers would collapse
    whole deciles if qcut were applied to the raw predictions.
    """
    actual = pd.Series(np.arange(1000, dtype=float))
    flat = pd.Series(np.zeros(1000))
    assert len(decile_table(actual, flat)) == 10
