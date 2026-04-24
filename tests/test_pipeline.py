import pandas as pd

from dc_reif.pipeline import run_full_pipeline


def test_pipeline_produces_property_intelligence_outputs(sample_dataframe, temp_config):
    raw_path = temp_config.data_dir / temp_config.data_filename
    sample_dataframe.to_csv(raw_path, index=False)

    outputs = run_full_pipeline(temp_config)

    property_table = pd.read_csv(outputs["property_intelligence"])
    required_columns = {
        "property_id",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "q_hat",
        "predicted_price_band",
        "segment_label",
        "anomaly_flag",
        "anomaly_score",
        "support_score",
        "slice_risk_level",
        "confidence_note",
        "why_flagged",
        "evidence_strength",
        "top_drivers",
        "data_quality_flag",
    }
    assert required_columns.issubset(property_table.columns)
    assert len(property_table) > 0
