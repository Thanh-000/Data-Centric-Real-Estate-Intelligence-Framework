from __future__ import annotations

import json

import pandas as pd

from dc_reif.report_results import build_final_results_summary, build_report_results_pack


def test_report_results_pack_assembles_canonical_summary(temp_config):
    temp_config.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    temp_config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    temp_config.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "model_name": "random_forest",
                "validation_rmse": 129084.55,
                "validation_mae": 70975.29,
                "validation_mape": 13.5,
                "validation_r2": 0.8594,
                "test_rmse": 147002.45,
                "test_mae": 79150.17,
                "test_mape": 13.24,
                "test_r2": 0.8476,
            },
            {
                "model_name": "linear_regression",
                "validation_rmse": 147987.98,
                "validation_mae": 95296.02,
                "validation_mape": 20.4,
                "validation_r2": 0.8152,
                "test_rmse": 177181.17,
                "test_mae": 100810.06,
                "test_mape": 19.21,
                "test_r2": 0.7787,
            },
        ]
    ).to_csv(temp_config.paths.tables_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"feature": "grade", "importance": 0.33},
            {"feature": "sqft_living", "importance": 0.25},
            {"feature": "lat", "importance": 0.15},
            {"feature": "long", "importance": 0.06},
            {"feature": "sqft_living15", "importance": 0.03},
        ]
    ).to_csv(temp_config.paths.tables_dir / "feature_importance.csv", index=False)
    pd.DataFrame(
        [
            {"segment_label": "segment_0", "sqft_living": 1500, "grade": 7, "condition": 3, "house_age": 35, "lat": 47.5, "long": -122.2, "count": 120},
            {"segment_label": "segment_1", "sqft_living": 2300, "grade": 9, "condition": 4, "house_age": 20, "lat": 47.6, "long": -122.1, "count": 140},
        ]
    ).to_csv(temp_config.paths.tables_dir / "cluster_profiles.csv", index=False)
    pd.DataFrame(
        [
            {
                "property_id": "1",
                "observed_price": 700000,
                "fair_value_hat": 500000,
                "lower_bound": 450000,
                "upper_bound": 550000,
                "interval_width": 100000,
                "segment_label": "segment_0",
                "data_quality_flag": "ok",
                "valuation_gap": 200000,
                "anomaly_score": 2.0,
                "anomaly_flag": "potentially_over_valued",
                "top_drivers": "grade, sqft_living, lat",
            },
            {
                "property_id": "2",
                "observed_price": 300000,
                "fair_value_hat": 450000,
                "lower_bound": 400000,
                "upper_bound": 500000,
                "interval_width": 100000,
                "segment_label": "segment_0",
                "data_quality_flag": "ok",
                "valuation_gap": -150000,
                "anomaly_score": -1.5,
                "anomaly_flag": "potentially_under_valued",
                "top_drivers": "grade, sqft_living, lat",
            },
            {
                "property_id": "3",
                "observed_price": 480000,
                "fair_value_hat": 475000,
                "lower_bound": 425000,
                "upper_bound": 525000,
                "interval_width": 100000,
                "segment_label": "segment_1",
                "data_quality_flag": "ok",
                "valuation_gap": 5000,
                "anomaly_score": 0.05,
                "anomaly_flag": "within_expected_range",
                "top_drivers": "grade, sqft_living, lat",
            },
            {
                "property_id": "4",
                "observed_price": 510000,
                "fair_value_hat": None,
                "lower_bound": None,
                "upper_bound": None,
                "interval_width": None,
                "segment_label": "segment_1",
                "data_quality_flag": "ok",
                "valuation_gap": None,
                "anomaly_score": None,
                "anomaly_flag": "insufficient_history",
                "top_drivers": "grade, sqft_living, lat",
            },
        ]
    ).to_csv(temp_config.paths.tables_dir / "property_intelligence_table.csv", index=False)

    (temp_config.paths.figures_dir / "price_distribution.png").write_bytes(b"png")
    (temp_config.paths.figures_dir / "log_price_distribution.png").write_bytes(b"png")
    (temp_config.paths.figures_dir / "spatial_price_map.png").write_bytes(b"png")
    (temp_config.paths.figures_dir / "temporal_trend.png").write_bytes(b"png")
    (temp_config.paths.figures_dir / "feature_importance.png").write_bytes(b"png")
    (temp_config.paths.figures_dir / "shap_summary.png").write_bytes(b"png")

    (temp_config.paths.reports_dir / "cluster_summary.json").write_text(
        json.dumps({"n_clusters": 6, "silhouette_score": 0.1784, "davies_bouldin_index": 1.4803}),
        encoding="utf-8",
    )
    (temp_config.paths.reports_dir / "uncertainty_metrics.json").write_text(
        json.dumps({"empirical_coverage": 0.8858, "average_interval_width": 309509.04, "q_hat": 154754.52}),
        encoding="utf-8",
    )
    (temp_config.paths.reports_dir / "pipeline_summary.md").write_text("summary", encoding="utf-8")

    summary = build_final_results_summary(temp_config)
    artifacts = build_report_results_pack(temp_config)

    assert summary["core_valuation_metrics"]["selected_model"] == "random_forest"
    assert summary["pricing_anomaly_results"]["potentially_over_valued"] == 1
    assert summary["uncertainty_results"]["interval_coverage"] == 0.8858
    assert artifacts["summary_json"].exists()
    assert artifacts["pack_case_examples"].exists()
