from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dc_reif.anomaly import compute_pricing_anomalies
from dc_reif.clustering import assign_submarket_segments, fit_submarket_clustering
from dc_reif.config import ProjectConfig
from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.data_download import download_dataset
from dc_reif.data_ingestion import load_raw_data
from dc_reif.data_validation import validate_schema, validation_report_frame
from dc_reif.explainability import (
    build_top_driver_map,
    global_feature_importance,
    plot_feature_importance,
    shap_explanations,
)
from dc_reif.feature_engineering import assert_no_target_leakage, build_feature_matrix
from dc_reif.reporting import create_eda_figures, save_dataframe, save_json, write_summary_report
from dc_reif.splitting import chronological_split
from dc_reif.uncertainty import build_prediction_intervals, conformal_quantile, evaluate_interval_quality
from dc_reif.utils import get_logger, write_json
from dc_reif.valuation import train_and_select_model

LOGGER = get_logger(__name__)


def run_full_pipeline(config: ProjectConfig) -> dict[str, str]:
    config.paths.ensure()
    dataset_path = config.data_dir / config.data_filename
    if not dataset_path.exists() or config.force_download:
        dataset_path = download_dataset(config)

    raw_df, manifest = load_raw_data(dataset_path, config.manifest_path)
    validation_report = validate_schema(raw_df, config.required_columns)
    if validation_report.missing_columns:
        raise ValueError(f"Missing required columns: {validation_report.missing_columns}")

    save_json(validation_report.to_dict(), config.validation_report_path)
    validation_report_csv = save_dataframe(validation_report_frame(validation_report), config.paths.tables_dir / "data_quality_report.csv")

    cleaning_result = clean_king_county_data(raw_df)
    cleaned_df = cleaning_result.dataframe
    write_json(config.paths.reports_dir / "cleaning_summary.json", cleaning_result.summary)
    save_dataframe(cleaned_df, config.cleaned_dataset_path)

    feature_set = build_feature_matrix(cleaned_df)
    modeling_df = feature_set.dataframe.sort_values([config.date_column, config.id_column]).reset_index(drop=True)
    cluster_artifacts = fit_submarket_clustering(modeling_df.iloc[: int(len(modeling_df) * config.train_fraction)].copy(), random_state=config.random_state)
    modeling_df["segment_label"] = assign_submarket_segments(modeling_df, cluster_artifacts)
    predictive_features = feature_set.predictive_features + ["segment_label"]
    assert_no_target_leakage(predictive_features)
    save_dataframe(modeling_df, config.feature_dataset_path)
    save_dataframe(cluster_artifacts.cluster_profiles, config.paths.tables_dir / "cluster_profiles.csv")
    save_json(
        {
            "n_clusters": cluster_artifacts.n_clusters,
            "silhouette_score": cluster_artifacts.silhouette,
            "davies_bouldin_index": cluster_artifacts.davies_bouldin,
            "min_keep_cluster": cluster_artifacts.min_keep_cluster,
            "min_local_cluster": cluster_artifacts.min_local_cluster,
        },
        config.paths.reports_dir / "cluster_summary.json",
    )

    split_bundle = chronological_split(
        modeling_df,
        date_column=config.date_column,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    valuation = train_and_select_model(
        train_df=split_bundle.train_df,
        validation_df=split_bundle.validation_df,
        train_validation_df=split_bundle.train_validation_df,
        test_df=split_bundle.test_df,
        feature_columns=predictive_features,
        target_column=config.target_column,
        n_splits=config.n_splits,
        random_state=config.random_state,
    )
    save_dataframe(valuation.model_comparison, config.paths.tables_dir / "model_comparison.csv")

    fair_value_hat = pd.Series(np.nan, index=modeling_df.index, name="fair_value_hat")
    fair_value_hat.loc[valuation.fair_value_hat_oof.index] = valuation.fair_value_hat_oof
    fair_value_hat.loc[valuation.fair_value_hat_test.index] = valuation.fair_value_hat_test

    oof_mask = valuation.fair_value_hat_oof.notna()
    residuals = split_bundle.train_validation_df.loc[oof_mask, config.target_column] - valuation.fair_value_hat_oof.loc[oof_mask]
    q_hat = conformal_quantile(residuals, alpha=config.alpha)
    intervals = build_prediction_intervals(fair_value_hat, q_hat=q_hat)
    interval_metrics = evaluate_interval_quality(
        split_bundle.test_df[config.target_column],
        intervals.loc[split_bundle.test_df.index, "lower_bound"],
        intervals.loc[split_bundle.test_df.index, "upper_bound"],
    )
    save_json({"q_hat": q_hat, **interval_metrics}, config.paths.reports_dir / "uncertainty_metrics.json")

    property_frame = pd.DataFrame(
        {
            "property_id": modeling_df[config.id_column].astype(str),
            "observed_price": modeling_df[config.target_column],
            "fair_value_hat": intervals["fair_value_hat"],
            "lower_bound": intervals["lower_bound"],
            "upper_bound": intervals["upper_bound"],
            "interval_width": intervals["interval_width"],
            "segment_label": modeling_df["segment_label"],
            "data_quality_flag": modeling_df["data_quality_flag"],
        }
    )
    property_frame = compute_pricing_anomalies(property_frame)

    importance_df = global_feature_importance(valuation.model_pipeline, valuation.model_name)
    importance_plot = plot_feature_importance(importance_df, config.paths.figures_dir / "feature_importance.png")
    save_dataframe(importance_df, config.paths.tables_dir / "feature_importance.csv")

    explain_sample = (
        property_frame.loc[property_frame["anomaly_flag"] != "insufficient_history"]
        .assign(abs_score=lambda frame: frame["anomaly_score"].abs())
        .sort_values("abs_score", ascending=False)
        .head(3)
    )
    shap_path, local_driver_map = shap_explanations(
        valuation.model_pipeline,
        dataset=modeling_df,
        feature_columns=predictive_features,
        output_path=config.paths.figures_dir / "shap_summary.png",
        local_sample_ids=explain_sample["property_id"].astype(str).tolist(),
        id_column=config.id_column,
    )
    property_frame["top_drivers"] = build_top_driver_map(
        modeling_df,
        id_column=config.id_column,
        importance_df=importance_df,
        local_driver_map=local_driver_map,
    )
    save_dataframe(property_frame, config.paths.tables_dir / "property_intelligence_table.csv")

    eda_figures = create_eda_figures(modeling_df, config.paths.figures_dir)
    save_json(manifest, config.paths.reports_dir / "raw_manifest_copy.json")
    joblib.dump(valuation.model_pipeline, config.paths.artifacts_dir / f"{valuation.model_name}_pipeline.joblib")
    joblib.dump(cluster_artifacts, config.paths.artifacts_dir / "submarket_clustering.joblib")

    summary_lines = [
        "# DC-REIF Pipeline Summary",
        "",
        f"- Selected valuation model: {valuation.model_name}",
        f"- Validation report: {config.validation_report_path}",
        f"- Cleaned rows retained: {cleaning_result.summary['rows_out']}",
        f"- OOF residual conformal q-hat: {q_hat:.2f}",
        f"- Test empirical coverage: {interval_metrics['empirical_coverage']:.3f}",
        f"- Test average interval width: {interval_metrics['average_interval_width']:.2f}",
        f"- Potentially under-valued sales: {(property_frame['anomaly_flag'] == 'potentially_under_valued').sum()}",
        f"- Potentially over-valued sales: {(property_frame['anomaly_flag'] == 'potentially_over_valued').sum()}",
        "",
        "This system performs pricing anomaly detection on sale prices, not strict asking-price mispricing detection.",
    ]
    summary_report = write_summary_report(summary_lines, config.paths.reports_dir / "pipeline_summary.md")

    outputs = {
        "dataset_path": str(dataset_path),
        "manifest": str(config.manifest_path),
        "validation_report_json": str(config.validation_report_path),
        "validation_report_csv": str(validation_report_csv),
        "clean_dataset": str(config.cleaned_dataset_path),
        "feature_dataset": str(config.feature_dataset_path),
        "model_comparison": str(config.paths.tables_dir / "model_comparison.csv"),
        "property_intelligence": str(config.paths.tables_dir / "property_intelligence_table.csv"),
        "feature_importance_plot": str(importance_plot),
        "summary_report": str(summary_report),
    }
    if shap_path:
        outputs["shap_summary"] = str(shap_path)
    outputs.update({name: str(path) for name, path in eda_figures.items()})
    LOGGER.info("Pipeline complete.")
    return outputs
