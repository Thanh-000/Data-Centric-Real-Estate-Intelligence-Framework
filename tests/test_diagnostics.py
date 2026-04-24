import pandas as pd

from dc_reif.pipeline import run_full_pipeline
from dc_reif.report_results import build_report_results_pack


def test_diagnostics_pack_writes_report_ready_slice_outputs(sample_dataframe, temp_config):
    raw_path = temp_config.data_dir / temp_config.data_filename
    sample_dataframe.to_csv(raw_path, index=False)

    run_full_pipeline(temp_config)
    artifacts = build_report_results_pack(temp_config)

    required_artifacts = [
        "pack_error_by_segment",
        "pack_error_by_price_band",
        "pack_coverage_by_segment",
        "pack_coverage_by_price_band",
        "pack_anomaly_by_segment",
        "pack_anomaly_by_price_band",
        "pack_improved_segment_profiles",
        "pack_anomaly_casebook",
        "pack_interpretation_notes",
        "pack_geospatial_notes",
        "pack_xgboost_parameters",
        "pack_segmentation_selection",
        "pack_local_conformal",
    ]
    for artifact_key in required_artifacts:
        assert artifacts[artifact_key].exists()

    error_by_segment = pd.read_csv(artifacts["pack_error_by_segment"])
    coverage_by_price_band = pd.read_csv(artifacts["pack_coverage_by_price_band"])
    anomaly_by_price_band = pd.read_csv(artifacts["pack_anomaly_by_price_band"])
    anomaly_casebook = pd.read_csv(artifacts["pack_anomaly_casebook"])

    assert {"segment_label", "count", "mae", "rmse"}.issubset(error_by_segment.columns)
    assert {"price_band", "empirical_coverage", "average_interval_width"}.issubset(coverage_by_price_band.columns)
    assert {"price_band", "potentially_over_valued", "over_rate"}.issubset(anomaly_by_price_band.columns)
    assert {"property_id", "observed_price", "segment_label", "anomaly_flag", "support_score"}.issubset(anomaly_casebook.columns)
