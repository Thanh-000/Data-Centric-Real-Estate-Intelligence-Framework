import pandas as pd

from dc_reif.uncertainty import build_prediction_intervals, conformal_quantile


def test_uncertainty_exports_conformal_interval_helpers():
    residuals = pd.Series([1.0, 2.0, 3.0, 4.0])
    q_hat = conformal_quantile(residuals, alpha=0.1)
    intervals = build_prediction_intervals(pd.Series([10.0, 12.0]), q_hat=q_hat)
    assert q_hat > 0
    assert "interval_width" in intervals.columns

