import pandas as pd

from dc_reif.product_analytics import abstention_summary, slice_metrics


def test_product_analytics_reports_abstention_and_slice_metrics():
    frame = pd.DataFrame(
        [
            {
                "zipcode": "98001",
                "segment_label": "segment_0",
                "observed_price": 500000,
                "fair_value_hat": 480000,
                "lower_bound": 420000,
                "upper_bound": 540000,
                "interval_width": 120000,
                "anomaly_flag": "within_expected_range",
                "grade": 7,
                "house_age": 20,
            },
            {
                "zipcode": "98001",
                "segment_label": "segment_0",
                "observed_price": 900000,
                "fair_value_hat": None,
                "lower_bound": None,
                "upper_bound": None,
                "interval_width": None,
                "anomaly_flag": "insufficient_history",
                "grade": 10,
                "house_age": 5,
            },
            {
                "zipcode": "98002",
                "segment_label": "segment_1",
                "observed_price": 300000,
                "fair_value_hat": 410000,
                "lower_bound": 360000,
                "upper_bound": 460000,
                "interval_width": 100000,
                "anomaly_flag": "potentially_under_valued",
                "grade": 6,
                "house_age": 65,
            },
        ]
    )

    abstention = abstention_summary(frame, ["zipcode"])
    assert abstention["zipcode"].loc[abstention["zipcode"]["zipcode"] == "98001", "abstention_count"].iloc[0] == 1

    metrics = slice_metrics(frame, ["zipcode"])
    assert {"mae", "rmse", "interval_coverage", "abstention_rate"}.issubset(metrics["zipcode"].columns)
