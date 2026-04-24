from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dc_reif.anomaly import compute_pricing_anomalies
from dc_reif.anomaly.pricing import enrich_pricing_anomalies
from dc_reif.config import ProjectConfig
from dc_reif.data_download import download_dataset
from dc_reif.explainability import (
    build_top_driver_map,
    global_feature_importance,
    plot_feature_importance,
    shap_explanations,
)
from dc_reif.feature_store import assert_no_target_leakage, build_feature_matrix
from dc_reif.governance import clean_king_county_data, load_raw_data, validate_schema, validation_report_frame
from dc_reif.market_representation import assign_submarket_segments, fit_submarket_clustering
from dc_reif.property_ledger import build_property_ledger
from dc_reif.reporting import create_eda_figures, save_dataframe, save_json, write_summary_report
from dc_reif.uncertainty import build_prediction_intervals, calibrate_local_conformal, evaluate_interval_quality
from dc_reif.utils import get_logger, write_json
from dc_reif.valuation_core import chronological_split, train_and_select_model

LOGGER = get_logger(__name__)


def _coverage_by_group(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scored = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    for group_value, frame in scored.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: group_value,
                "count": int(len(frame)),
                "empirical_coverage": float(
                    ((frame["observed_price"] >= frame["lower_bound"]) & (frame["observed_price"] <= frame["upper_bound"])).mean()
                ),
                "average_interval_width": float(frame["interval_width"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _error_by_group(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scored = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    for group_value, frame in scored.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: group_value,
                "count": int(len(frame)),
                "mae": float((frame["observed_price"] - frame["fair_value_hat"]).abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(frame["observed_price"] - frame["fair_value_hat"])))),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _price_band(series: pd.Series, n_bands: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="string")
    n_quantiles = min(n_bands, max(valid.nunique(), 1))
    labels = [f"Q{index}" for index in range(1, n_quantiles + 1)]
    ranked = valid.rank(method="first")
    bands = pd.qcut(ranked, q=n_quantiles, labels=labels)
    output = pd.Series(pd.NA, index=series.index, dtype="string")
    output.loc[valid.index] = bands.astype("string")
    return output


def run_full_pipeline(config: ProjectConfig, include_enhanced_features: bool = True) -> dict[str, str]:
    config.paths.ensure()
    dataset_path = config.data_dir / config.data_filename
    if not dataset_path.exists() or config.force_download:
        dataset_path = download_dataset(config)

    raw_df, manifest = load_raw_data(dataset_path, config.manifest_path)
    validation_report = validate_schema(raw_df, config.required_columns)
    if validation_report.missing_columns:
        raise ValueError(f"Missing required columns: {validation_report.missing_columns}")

    save_json(validation_report.to_dict(), config.validation_report_path)
    validation_report_csv = save_dataframe(
        validation_report_frame(validation_report),
        config.paths.tables_dir / "data_quality_report.csv",
    )

    cleaning_result = clean_king_county_data(raw_df)
    cleaned_df = cleaning_result.dataframe
    write_json(config.paths.reports_dir / "cleaning_summary.json", cleaning_result.summary)
    save_dataframe(cleaned_df, config.cleaned_dataset_path)

    feature_set = build_feature_matrix(cleaned_df, include_enhanced_features=include_enhanced_features)
    modeling_df = feature_set.dataframe.sort_values([config.date_column, config.id_column]).reset_index(drop=True)
    cluster_artifacts = fit_submarket_clustering(
        modeling_df.iloc[: int(len(modeling_df) * config.train_fraction)].copy(),
        random_state=config.random_state,
        include_enhanced_features=include_enhanced_features,
    )
    modeling_df["segment_label"] = assign_submarket_segments(modeling_df, cluster_artifacts)
    predictive_features = feature_set.predictive_features + ["segment_label"]
    assert_no_target_leakage(predictive_features)
    save_dataframe(modeling_df, config.feature_dataset_path)
    save_dataframe(cluster_artifacts.cluster_profiles, config.paths.tables_dir / "cluster_profiles.csv")
    save_dataframe(cluster_artifacts.selection_summary, config.paths.tables_dir / "segmentation_selection_grid.csv")
    save_json(
        {
            "n_clusters": cluster_artifacts.n_clusters,
            "silhouette_score": cluster_artifacts.silhouette,
            "davies_bouldin_index": cluster_artifacts.davies_bouldin,
            "min_keep_cluster": cluster_artifacts.min_keep_cluster,
            "min_local_cluster": cluster_artifacts.min_local_cluster,
            "feature_columns": cluster_artifacts.feature_columns,
            "selection_details": cluster_artifacts.selection_details,
        },
        config.paths.reports_dir / "cluster_summary.json",
    )
    save_json(
        {
            "selected_k": cluster_artifacts.n_clusters,
            "selection_details": cluster_artifacts.selection_details,
            "feature_columns": cluster_artifacts.feature_columns,
            "selection_grid_file": str(config.paths.tables_dir / "segmentation_selection_grid.csv"),
        },
        config.paths.reports_dir / "segmentation_selection_summary.json",
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
    save_dataframe(valuation.valuation_metrics, config.paths.tables_dir / "valuation_metrics.csv")
    save_dataframe(valuation.selection_summary, config.paths.tables_dir / "xgboost_selection_grid.csv")
    save_json(
        {
            "selected_model": valuation.model_name,
            "selected_parameters": valuation.selected_parameters,
            "target_strategy": valuation.target_strategy,
            "high_price_weight": valuation.high_price_weight,
            "selection_grid_file": str(config.paths.tables_dir / "xgboost_selection_grid.csv"),
            "selected_validation_rmse": float(valuation.valuation_metrics.loc[0, "validation_rmse"]),
            "selected_test_rmse": float(valuation.valuation_metrics.loc[0, "test_rmse"]),
        },
        config.paths.reports_dir / "xgboost_selection_summary.json",
    )

    fair_value_hat = pd.Series(np.nan, index=modeling_df.index, name="fair_value_hat")
    fair_value_hat.loc[valuation.fair_value_hat_oof.index] = valuation.fair_value_hat_oof
    fair_value_hat.loc[valuation.fair_value_hat_test.index] = valuation.fair_value_hat_test

    calibration_mask = valuation.fair_value_hat_oof.notna()
    calibration_frame = pd.DataFrame(
        {
            "observed_price": split_bundle.train_validation_df.loc[calibration_mask, config.target_column],
            "fair_value_hat": valuation.fair_value_hat_oof.loc[calibration_mask],
            "segment_label": split_bundle.train_validation_df.loc[calibration_mask, "segment_label"],
        }
    )
    prediction_frame = pd.DataFrame(
        {
            "fair_value_hat": fair_value_hat,
            "segment_label": modeling_df["segment_label"],
        },
        index=modeling_df.index,
    )
    local_prediction_frame, calibration_artifacts = calibrate_local_conformal(
        calibration_frame=calibration_frame,
        prediction_frame=prediction_frame,
        alpha=config.alpha,
    )
    intervals = build_prediction_intervals(fair_value_hat, q_hat=local_prediction_frame["q_hat"])
    interval_metrics = evaluate_interval_quality(
        split_bundle.test_df[config.target_column],
        intervals.loc[split_bundle.test_df.index, "lower_bound"],
        intervals.loc[split_bundle.test_df.index, "upper_bound"],
    )

    property_frame = pd.DataFrame(
        {
            "property_id": modeling_df[config.id_column].astype(str),
            "sale_date": modeling_df[config.date_column].dt.strftime("%Y-%m-%d"),
            "zipcode": modeling_df["zipcode"].astype(str),
            "observed_price": modeling_df[config.target_column],
            "fair_value_hat": intervals["fair_value_hat"],
            "lower_bound": intervals["lower_bound"],
            "upper_bound": intervals["upper_bound"],
            "interval_width": intervals["interval_width"],
            "q_hat": intervals["q_hat"],
            "predicted_price_band": local_prediction_frame["predicted_price_band"],
            "price_band_support_n": local_prediction_frame["price_band_support_n"],
            "segment_support_n": local_prediction_frame["segment_support_n"],
            "segment_label": modeling_df["segment_label"],
            "sqft_living": modeling_df["sqft_living"],
            "grade": modeling_df["grade"],
            "house_age": modeling_df["house_age"],
            "data_quality_flag": modeling_df["data_quality_flag"],
        }
    )
    property_frame = compute_pricing_anomalies(property_frame)
    property_frame = enrich_pricing_anomalies(
        property_frame,
        global_q_hat=calibration_artifacts.global_q_hat,
        min_segment_support=200,
        min_price_band_support=300,
    )

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
    property_ledger = build_property_ledger(property_frame)
    save_dataframe(property_ledger, config.paths.tables_dir / "property_intelligence_table.csv")

    test_frame = property_frame.loc[split_bundle.test_df.index].copy()
    test_frame["evaluation_price_band"] = _price_band(test_frame["observed_price"])
    test_coverage_by_price_band = _coverage_by_group(test_frame.rename(columns={"evaluation_price_band": "price_band"}), "price_band")
    q5_coverage = float(
        test_coverage_by_price_band.loc[test_coverage_by_price_band["price_band"] == "Q5", "empirical_coverage"].iloc[0]
        if (test_coverage_by_price_band["price_band"] == "Q5").any()
        else interval_metrics["empirical_coverage"]
    )
    save_json(
        {
            "q_hat": float(calibration_artifacts.global_q_hat),
            **interval_metrics,
        },
        config.paths.reports_dir / "uncertainty_metrics.json",
    )
    save_dataframe(calibration_artifacts.price_band_summary, config.paths.tables_dir / "local_conformal_by_price_band.csv")
    save_dataframe(calibration_artifacts.segment_summary, config.paths.tables_dir / "local_conformal_by_segment.csv")
    save_json(
        {
            **calibration_artifacts.calibration_summary,
            "global_empirical_coverage": float(interval_metrics["empirical_coverage"]),
            "global_average_interval_width": float(interval_metrics["average_interval_width"]),
            "q5_empirical_coverage": q5_coverage,
            "price_band_summary_file": str(config.paths.tables_dir / "local_conformal_by_price_band.csv"),
            "segment_summary_file": str(config.paths.tables_dir / "local_conformal_by_segment.csv"),
        },
        config.paths.reports_dir / "local_conformal_calibration_summary.json",
    )

    eda_figures = create_eda_figures(modeling_df, config.paths.figures_dir)
    save_json(manifest, config.paths.reports_dir / "raw_manifest_copy.json")
    joblib.dump(
        {
            "pipeline": valuation.model_pipeline,
            "selected_parameters": valuation.selected_parameters,
            "target_strategy": valuation.target_strategy,
            "high_price_weight": valuation.high_price_weight,
        },
        config.paths.artifacts_dir / f"{valuation.model_name}_pipeline.joblib",
    )
    joblib.dump(cluster_artifacts, config.paths.artifacts_dir / "submarket_clustering.joblib")

    summary_lines = [
        "# DC-REIF Pipeline Summary",
        "",
        f"- Selected valuation model: {valuation.model_name}",
        f"- Target strategy: {valuation.target_strategy}",
        f"- High-price sample weight: {valuation.high_price_weight:.2f}",
        f"- Selected KMeans segments: {cluster_artifacts.n_clusters}",
        f"- Validation report: {config.validation_report_path}",
        f"- Cleaned rows retained: {cleaning_result.summary['rows_out']}",
        f"- Local conformal global q-hat: {calibration_artifacts.global_q_hat:.2f}",
        f"- Test empirical coverage: {interval_metrics['empirical_coverage']:.3f}",
        f"- Test Q5 empirical coverage: {q5_coverage:.3f}",
        f"- Test average interval width: {interval_metrics['average_interval_width']:.2f}",
        f"- Potentially under-valued sales: {(property_frame['anomaly_flag'] == 'potentially_under_valued').sum()}",
        f"- Potentially over-valued sales: {(property_frame['anomaly_flag'] == 'potentially_over_valued').sum()}",
        "",
        "This system performs Pricing Anomaly Detection on realized sale prices and should not be interpreted as a listing-price decision rule.",
    ]
    summary_report = write_summary_report(summary_lines, config.paths.reports_dir / "pipeline_summary.md")

    outputs = {
        "dataset_path": str(dataset_path),
        "manifest": str(config.manifest_path),
        "validation_report_json": str(config.validation_report_path),
        "validation_report_csv": str(validation_report_csv),
        "clean_dataset": str(config.cleaned_dataset_path),
        "feature_dataset": str(config.feature_dataset_path),
        "valuation_metrics": str(config.paths.tables_dir / "valuation_metrics.csv"),
        "xgboost_selection_grid": str(config.paths.tables_dir / "xgboost_selection_grid.csv"),
        "xgboost_selection_summary": str(config.paths.reports_dir / "xgboost_selection_summary.json"),
        "segmentation_selection_grid": str(config.paths.tables_dir / "segmentation_selection_grid.csv"),
        "segmentation_selection_summary": str(config.paths.reports_dir / "segmentation_selection_summary.json"),
        "local_conformal_summary": str(config.paths.reports_dir / "local_conformal_calibration_summary.json"),
        "property_intelligence": str(config.paths.tables_dir / "property_intelligence_table.csv"),
        "feature_importance_plot": str(importance_plot),
        "summary_report": str(summary_report),
    }
    if shap_path:
        outputs["shap_summary"] = str(shap_path)
    outputs.update({name: str(path) for name, path in eda_figures.items()})
    LOGGER.info("Pipeline complete.")
    return outputs
