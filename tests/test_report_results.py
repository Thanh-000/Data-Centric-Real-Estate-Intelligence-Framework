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
                "model_name": "xgboost",
                "validation_rmse": 110852.91,
                "validation_mae": 65525.94,
                "validation_mape": 12.93,
                "validation_r2": 0.8963,
                "test_rmse": 121629.66,
                "test_mae": 70271.43,
                "test_mape": 12.43,
                "test_r2": 0.8957,
            },
        ]
    ).to_csv(temp_config.paths.tables_dir / "valuation_metrics.csv", index=False)
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
                "id": "1",
                "date": "2014-05-01",
                "price": 700000,
                "zipcode": "98001",
                "lat": 47.5,
                "long": -122.2,
                "sqft_living": 1800,
                "sqft_lot": 5000,
                "sqft_above": 1400,
                "sqft_basement": 400,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "floors": 2.0,
                "yr_built": 1980,
                "yr_renovated": 0,
                "grade": 7,
                "sqft_living15": 1750,
                "sqft_lot15": 5200,
            },
            {
                "id": "2",
                "date": "2014-05-02",
                "price": 300000,
                "zipcode": "98001",
                "lat": 47.5,
                "long": -122.2,
                "sqft_living": 1200,
                "sqft_lot": 4500,
                "sqft_above": 1000,
                "sqft_basement": 200,
                "bedrooms": 2,
                "bathrooms": 1.5,
                "floors": 1.0,
                "yr_built": 1975,
                "yr_renovated": 0,
                "grade": 6,
                "sqft_living15": 1250,
                "sqft_lot15": 4700,
            },
            {
                "id": "3",
                "date": "2014-05-03",
                "price": 480000,
                "zipcode": "98002",
                "lat": 47.6,
                "long": -122.1,
                "sqft_living": 1600,
                "sqft_lot": 4000,
                "sqft_above": 1600,
                "sqft_basement": 0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "floors": 1.0,
                "yr_built": 1995,
                "yr_renovated": 2005,
                "grade": 8,
                "sqft_living15": 1650,
                "sqft_lot15": 4100,
            },
            {
                "id": "4",
                "date": "2014-05-04",
                "price": 510000,
                "zipcode": "98002",
                "lat": 47.6,
                "long": -122.1,
                "sqft_living": 1700,
                "sqft_lot": 4300,
                "sqft_above": 1500,
                "sqft_basement": 200,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "floors": 2.0,
                "yr_built": 1990,
                "yr_renovated": 0,
                "grade": 8,
                "sqft_living15": 1750,
                "sqft_lot15": 4400,
            },
        ]
    ).to_csv(temp_config.feature_dataset_path, index=False)
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
                "sale_date": "2014-05-01",
                "zipcode": "98001",
                "sqft_living": 1800,
                "grade": 7,
                "house_age": 34,
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
                "sale_date": "2014-05-02",
                "zipcode": "98001",
                "sqft_living": 1200,
                "grade": 6,
                "house_age": 39,
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
                "sale_date": "2014-05-03",
                "zipcode": "98002",
                "sqft_living": 1600,
                "grade": 8,
                "house_age": 19,
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
                "sale_date": "2014-05-04",
                "zipcode": "98002",
                "sqft_living": 1700,
                "grade": 8,
                "house_age": 24,
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
        json.dumps({"n_clusters": 6, "silhouette_score": 0.1839, "davies_bouldin_index": 1.6291}),
        encoding="utf-8",
    )
    (temp_config.paths.reports_dir / "uncertainty_metrics.json").write_text(
        json.dumps({"empirical_coverage": 0.8831, "average_interval_width": 280938.25, "q_hat": 140469.12}),
        encoding="utf-8",
    )
    (temp_config.paths.reports_dir / "pipeline_summary.md").write_text("summary", encoding="utf-8")

    summary = build_final_results_summary(temp_config)
    artifacts = build_report_results_pack(temp_config)

    assert summary["core_valuation_metrics"]["selected_model"] == "xgboost"
    assert summary["pricing_anomaly_results"]["potentially_over_valued"] == 1
    assert summary["uncertainty_results"]["interval_coverage"] == 0.8831
    assert artifacts["summary_json"].exists()
    assert artifacts["pack_case_examples"].exists()
    assert artifacts["pack_error_by_segment"].exists()
    assert artifacts["pack_anomaly_by_price_band"].exists()
    assert artifacts["pack_improved_segment_profiles"].exists()
