import pandas as pd

from dc_reif.uncertainty import build_prediction_intervals, calibrate_local_conformal, conformal_quantile


def test_uncertainty_exports_conformal_interval_helpers():
    residuals = pd.Series([1.0, 2.0, 3.0, 4.0])
    q_hat = conformal_quantile(residuals, alpha=0.1)
    intervals = build_prediction_intervals(pd.Series([10.0, 12.0]), q_hat=q_hat)
    assert q_hat > 0
    assert "interval_width" in intervals.columns


def test_local_conformal_applies_upper_tail_safety_correction():
    calibration = pd.DataFrame(
        {
            "observed_price": [100, 110, 120, 130, 500, 550, 600, 650, 900, 1000],
            "fair_value_hat": [100, 110, 120, 130, 500, 550, 600, 650, 900, 1000],
            "segment_label": ["segment_a"] * 10,
        }
    )
    prediction = pd.DataFrame(
        {
            "fair_value_hat": [100, 950],
            "segment_label": ["segment_a", "segment_a"],
        }
    )

    rows, artifacts = calibrate_local_conformal(
        calibration,
        prediction,
        alpha=0.1,
        min_price_band_samples=1,
        min_segment_samples=1,
        upper_tail_band_count=1,
        upper_tail_multiplier=1.2,
    )

    upper_tail = rows.loc[rows["fair_value_hat"].eq(950)].iloc[0]
    assert upper_tail["upper_tail_adjusted"]
    assert upper_tail["q_hat"] >= upper_tail["price_band_q_hat"] * 1.2
    assert artifacts.calibration_summary["upper_tail_multiplier"] == 1.2
