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
from dc_reif.product_analytics import interval_width_summary, threshold_sensitivity, write_analysis_tables
from dc_reif.property_ledger import build_property_ledger
from dc_reif.reporting import (
    create_eda_figures,
    create_residual_diagnostics,
    save_dataframe,
    save_json,
    write_summary_report,
    write_trust_summary,
)
from dc_reif.uncertainty import build_prediction_intervals, calibrate_local_conformal, evaluate_interval_quality
from dc_reif.utils import get_logger, write_json
from dc_reif.valuation_core import (
    chronological_split,
    evaluate_model_suite,
    train_and_select_model,
    train_quantile_interval_artifacts,
)

LOGGER = get_logger(__name__)


def _model_comparison_markdown(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty:
        return []
    rows = [
        "",
        "## Model Baseline Comparison",
        "",
        "| Model | Test RMSE | Test MAE | Test MAPE | Test R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in dataframe.sort_values("test_rmse").itertuples(index=False):
        rows.append(
            f"| {str(row.model_name)} | ${float(row.test_rmse):,.0f} | ${float(row.test_mae):,.0f} | {float(row.test_mape):.2f}% | {float(row.test_r2):.3f} |"
        )
    return rows


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
                "mape": float(
                    (((frame["observed_price"] - frame["fair_value_hat"]).abs() / frame["observed_price"].replace(0, np.nan)).mean())
                    * 100
                ),
                "wmape": float(
                    (frame["observed_price"] - frame["fair_value_hat"]).abs().sum()
                    / max(frame["observed_price"].abs().sum(), 1e-9)
                    * 100
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _interval_quality_by_band(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual, "lower_bound": lower, "upper_bound": upper}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["price_band", "count", "empirical_coverage", "average_interval_width"])
    frame["price_band"] = _price_band(frame["actual"])
    rows: list[dict[str, object]] = []
    for price_band, group in frame.groupby("price_band", dropna=False):
        rows.append(
            {
                "price_band": price_band,
                "count": int(len(group)),
                "empirical_coverage": float(
                    ((group["actual"] >= group["lower_bound"]) & (group["actual"] <= group["upper_bound"])).mean()
                ),
                "average_interval_width": float((group["upper_bound"] - group["lower_bound"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("price_band").reset_index(drop=True)


def _synthetic_recall_for_bounds(
    frame: pd.DataFrame,
    *,
    lower_col: str,
    upper_col: str,
    method: str,
    n_per_direction: int = 100,
    shocks: tuple[float, ...] = (0.30, 0.40, 0.50),
    random_state: int = 42,
) -> pd.DataFrame:
    candidates = frame.loc[
        frame["observed_price"].notna()
        & frame["fair_value_hat"].notna()
        & frame[lower_col].notna()
        & frame[upper_col].notna()
        & frame["observed_price"].between(frame[lower_col], frame[upper_col])
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["method", "scenario", "shock", "sample_size", "detected", "recall"])

    sample_size = min(n_per_direction, len(candidates))
    rows: list[dict[str, object]] = []
    for shock in shocks:
        for scenario, multiplier, expected_flag in [
            ("synthetic_over_value", 1.0 + shock, "potentially_over_valued"),
            ("synthetic_under_value", 1.0 - shock, "potentially_under_valued"),
        ]:
            sample = candidates.sample(n=sample_size, random_state=random_state + len(rows)).copy()
            synthetic_observed = sample["observed_price"] * multiplier
            synthetic_flag = np.select(
                [
                    synthetic_observed < sample[lower_col],
                    synthetic_observed > sample[upper_col],
                ],
                ["potentially_under_valued", "potentially_over_valued"],
                default="within_expected_range",
            )
            detected = int((synthetic_flag == expected_flag).sum())
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "shock": float(shock),
                    "sample_size": int(sample_size),
                    "detected": detected,
                    "recall": detected / sample_size if sample_size else np.nan,
                }
            )

    results = pd.DataFrame(rows)
    for shock, group in results.groupby("shock", sort=True):
        total_sample = int(group["sample_size"].sum())
        total_detected = int(group["detected"].sum())
        results.loc[len(results)] = {
            "method": method,
            "scenario": "overall",
            "shock": float(shock),
            "sample_size": total_sample,
            "detected": total_detected,
            "recall": total_detected / total_sample if total_sample else np.nan,
        }
    return results


def _interval_comparison(
    *,
    test_actual: pd.Series,
    conformal_lower: pd.Series,
    conformal_upper: pd.Series,
    quantile_lower: pd.Series,
    quantile_upper: pd.Series,
    synthetic_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    band_rows: list[pd.DataFrame] = []
    for method, lower, upper in [
        ("conformal", conformal_lower, conformal_upper),
        ("quantile_xgb", quantile_lower, quantile_upper),
    ]:
        metrics = evaluate_interval_quality(test_actual, lower, upper)
        by_band = _interval_quality_by_band(test_actual, lower, upper)
        by_band.insert(0, "method", method)
        band_rows.append(by_band)
        coverage_map = by_band.set_index("price_band")["empirical_coverage"].to_dict()
        width_map = by_band.set_index("price_band")["average_interval_width"].to_dict()
        recall_overall = synthetic_results.loc[
            synthetic_results["method"].eq(method) & synthetic_results["scenario"].eq("overall")
        ].set_index("shock")["recall"].to_dict()
        rows.append(
            {
                "method": method,
                "avg_width": float(metrics["average_interval_width"]),
                "coverage": float(metrics["empirical_coverage"]),
                "q1_coverage": float(coverage_map.get("Q1", np.nan)),
                "q5_coverage": float(coverage_map.get("Q5", np.nan)),
                "q1_avg_width": float(width_map.get("Q1", np.nan)),
                "q5_avg_width": float(width_map.get("Q5", np.nan)),
                "recall_30pct": float(recall_overall.get(0.30, np.nan)),
                "recall_40pct": float(recall_overall.get(0.40, np.nan)),
                "recall_50pct": float(recall_overall.get(0.50, np.nan)),
            }
        )
    return pd.DataFrame(rows), pd.concat(band_rows, ignore_index=True) if band_rows else pd.DataFrame()


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


def _model_confidence(evidence_strength: object, slice_risk_level: object) -> str:
    evidence = str(evidence_strength)
    risk = str(slice_risk_level)
    if evidence == "strong" and risk in {"low", "medium"}:
        return "Higher"
    if evidence in {"moderate", "strong"} and risk != "high":
        return "Medium"
    return "Lower"


def _review_note(row: pd.Series) -> str:
    label_names = {
        "potentially_over_valued": "Over-valued",
        "potentially_under_valued": "Under-valued",
        "insufficient_history": "Low support",
        "within_expected_range": "Within range",
    }
    signal = label_names.get(str(row.get("anomaly_flag")), str(row.get("anomaly_flag", "Unknown")))
    confidence = _model_confidence(row.get("evidence_strength"), row.get("slice_risk_level"))
    if signal == "Low support":
        return f"Local evidence is limited. Review comparable sales before using this estimate. Confidence: {confidence}."
    drivers = str(row.get("top_drivers", "")).strip()
    driver_text = f" Main drivers: {drivers}." if drivers else ""
    return f"{signal} candidate for human review. {row.get('why_flagged', '')}{driver_text} Confidence: {confidence}."


def _model_flagged_cases(property_ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "property_id",
        "sale_date",
        "zipcode",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "model_signal",
        "model_confidence",
        "human_review_note",
        "top_drivers",
    ]
    flagged = property_ledger.loc[
        property_ledger["anomaly_flag"].isin(["potentially_over_valued", "potentially_under_valued", "insufficient_history"])
    ].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    flagged["model_signal"] = flagged["anomaly_flag"].map(
        {
            "potentially_over_valued": "Over-valued",
            "potentially_under_valued": "Under-valued",
            "insufficient_history": "Low support",
        }
    )
    flagged["model_confidence"] = flagged.apply(
        lambda row: _model_confidence(row.get("evidence_strength"), row.get("slice_risk_level")),
        axis=1,
    )
    flagged["human_review_note"] = flagged.apply(_review_note, axis=1)
    return flagged[[column for column in columns if column in flagged.columns]].sort_values(
        "observed_price",
        ascending=False,
    )


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
            "segmentation_method": cluster_artifacts.segmentation_method,
        },
        config.paths.reports_dir / "cluster_summary.json",
    )
    save_json(
        {
            "selected_k": cluster_artifacts.n_clusters,
            "segmentation_method": cluster_artifacts.segmentation_method,
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
        train_end_date=config.train_end_date,
        validation_end_date=config.validation_end_date,
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
    baseline_suite = evaluate_model_suite(
        train_df=split_bundle.train_df,
        validation_df=split_bundle.validation_df,
        train_validation_df=split_bundle.train_validation_df,
        test_df=split_bundle.test_df,
        feature_columns=predictive_features,
        target_column=config.target_column,
        model_names=["median_baseline", "linear_regression", "random_forest"],
        random_state=config.random_state,
    )
    model_comparison = pd.concat(
        [baseline_suite.valuation_metrics, valuation.valuation_metrics.assign(model_name="xgboost_selected")],
        ignore_index=True,
    ).sort_values("validation_rmse")
    save_dataframe(model_comparison, config.paths.tables_dir / "model_baseline_comparison.csv")
    LOGGER.info(
        "Model comparison complete: %s",
        ", ".join(model_comparison["model_name"].astype(str).tolist()),
    )
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
            "baseline_comparison_file": str(config.paths.tables_dir / "model_baseline_comparison.csv"),
        },
        config.paths.reports_dir / "xgboost_selection_summary.json",
    )

    fair_value_hat = pd.Series(np.nan, index=modeling_df.index, name="fair_value_hat")
    fair_value_hat.loc[valuation.fair_value_hat_oof.index] = valuation.fair_value_hat_oof
    fair_value_hat.loc[valuation.fair_value_hat_test.index] = valuation.fair_value_hat_test
    fair_value_hat = fair_value_hat.fillna(valuation.fair_value_hat_all.reindex(modeling_df.index))

    validation_prediction = valuation.validation_predictions.get(valuation.model_name)
    if validation_prediction is None:
        validation_prediction = next(iter(valuation.validation_predictions.values()))
    calibration_frame = pd.DataFrame(
        {
            "observed_price": split_bundle.validation_df[config.target_column],
            "fair_value_hat": validation_prediction.reindex(split_bundle.validation_df.index),
            "segment_label": split_bundle.validation_df["segment_label"],
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
    conformal_intervals = intervals.copy()

    quantile_intervals = train_quantile_interval_artifacts(
        train_df=split_bundle.train_df,
        full_df=modeling_df,
        test_df=split_bundle.test_df,
        feature_columns=predictive_features,
        target_column=config.target_column,
        random_state=config.random_state,
    )
    synthetic_frame = pd.DataFrame(
        {
            "observed_price": modeling_df[config.target_column],
            "fair_value_hat": conformal_intervals["fair_value_hat"],
            "lower_bound_conformal": conformal_intervals["lower_bound"],
            "upper_bound_conformal": conformal_intervals["upper_bound"],
            "lower_bound_qr": quantile_intervals.lower_bound_all.reindex(modeling_df.index),
            "upper_bound_qr": quantile_intervals.upper_bound_all.reindex(modeling_df.index),
        },
        index=modeling_df.index,
    )
    interval_synthetic_recall = pd.concat(
        [
            _synthetic_recall_for_bounds(
                synthetic_frame,
                lower_col="lower_bound_conformal",
                upper_col="upper_bound_conformal",
                method="conformal",
            ),
            _synthetic_recall_for_bounds(
                synthetic_frame,
                lower_col="lower_bound_qr",
                upper_col="upper_bound_qr",
                method="quantile_xgb",
            ),
        ],
        ignore_index=True,
    )
    interval_comparison, interval_comparison_by_band = _interval_comparison(
        test_actual=split_bundle.test_df[config.target_column],
        conformal_lower=conformal_intervals.loc[split_bundle.test_df.index, "lower_bound"],
        conformal_upper=conformal_intervals.loc[split_bundle.test_df.index, "upper_bound"],
        quantile_lower=quantile_intervals.lower_bound_test,
        quantile_upper=quantile_intervals.upper_bound_test,
        synthetic_results=interval_synthetic_recall,
    )
    quantile_row = interval_comparison.loc[interval_comparison["method"].eq("quantile_xgb")].iloc[0]
    conformal_row = interval_comparison.loc[interval_comparison["method"].eq("conformal")].iloc[0]
    interval_method = "quantile_xgb"
    quantile_rejection_reasons: list[str] = []
    if float(quantile_row["coverage"]) < 0.90:
        quantile_rejection_reasons.append("overall coverage below 90%")
    if float(quantile_row["q5_coverage"]) < 0.88:
        quantile_rejection_reasons.append("Q5 coverage below 88%")
    if float(quantile_row["avg_width"]) >= float(conformal_row["avg_width"]):
        quantile_rejection_reasons.append("average interval width is not lower than conformal")
    if float(quantile_row["recall_30pct"]) < 0.50:
        quantile_rejection_reasons.append("30% synthetic recall below 50%")
    if quantile_rejection_reasons:
        interval_method = "conformal"
    interval_comparison = interval_comparison.assign(
        selected_for_decision_layer=lambda frame: frame["method"].eq(interval_method),
        selection_rationale=(
            "selected: quantile_xgb met coverage, width, and synthetic recall gates"
            if interval_method == "quantile_xgb"
            else "kept conformal: " + "; ".join(quantile_rejection_reasons)
        ),
    )
    save_dataframe(interval_comparison, config.paths.tables_dir / "interval_comparison.csv")
    save_dataframe(interval_comparison_by_band, config.paths.tables_dir / "interval_comparison_by_price_band.csv")
    save_dataframe(interval_synthetic_recall, config.paths.tables_dir / "synthetic_anomaly_recall_comparison.csv")

    if interval_method == "quantile_xgb":
        intervals = pd.DataFrame(index=modeling_df.index)
        intervals["fair_value_hat"] = fair_value_hat
        intervals["lower_bound"] = quantile_intervals.lower_bound_all.reindex(modeling_df.index)
        intervals["upper_bound"] = quantile_intervals.upper_bound_all.reindex(modeling_df.index)
        intervals["interval_width"] = intervals["upper_bound"] - intervals["lower_bound"]
        intervals["q_hat"] = intervals["interval_width"] / 2.0
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
            "lat": modeling_df["lat"],
            "long": modeling_df["long"],
            "observed_price": modeling_df[config.target_column],
            "fair_value_hat": intervals["fair_value_hat"],
            "lower_bound": intervals["lower_bound"],
            "upper_bound": intervals["upper_bound"],
            "interval_width": intervals["interval_width"],
            "q_hat": intervals["q_hat"],
            "interval_method": interval_method,
            "lower_bound_conformal": conformal_intervals["lower_bound"],
            "upper_bound_conformal": conformal_intervals["upper_bound"],
            "interval_width_conformal": conformal_intervals["interval_width"],
            "lower_bound_qr": quantile_intervals.lower_bound_all.reindex(modeling_df.index),
            "upper_bound_qr": quantile_intervals.upper_bound_all.reindex(modeling_df.index),
            "interval_width_qr": (
                quantile_intervals.upper_bound_all.reindex(modeling_df.index)
                - quantile_intervals.lower_bound_all.reindex(modeling_df.index)
            ),
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
    save_dataframe(
        threshold_sensitivity(property_ledger),
        config.paths.tables_dir / "anomaly_threshold_sensitivity.csv",
    )
    write_analysis_tables(
        interval_width_summary(property_ledger, ["predicted_price_band", "segment_label"]),
        config.paths.tables_dir,
        "interval_width",
    )

    test_frame = property_frame.loc[split_bundle.test_df.index].copy()
    test_frame["evaluation_price_band"] = _price_band(test_frame["observed_price"])
    test_coverage_by_price_band = _coverage_by_group(test_frame.rename(columns={"evaluation_price_band": "price_band"}), "price_band")
    test_error_by_price_band = _error_by_group(test_frame.rename(columns={"evaluation_price_band": "price_band"}), "price_band")
    save_dataframe(test_coverage_by_price_band, config.paths.tables_dir / "test_interval_coverage_by_price_band.csv")
    save_dataframe(test_error_by_price_band, config.paths.tables_dir / "test_error_by_price_band.csv")
    q5_coverage = float(
        test_coverage_by_price_band.loc[test_coverage_by_price_band["price_band"] == "Q5", "empirical_coverage"].iloc[0]
        if (test_coverage_by_price_band["price_band"] == "Q5").any()
        else interval_metrics["empirical_coverage"]
    )
    q5_interval_width = float(
        test_coverage_by_price_band.loc[test_coverage_by_price_band["price_band"] == "Q5", "average_interval_width"].iloc[0]
        if (test_coverage_by_price_band["price_band"] == "Q5").any()
        else interval_metrics["average_interval_width"]
    )
    save_json(
        {
            "q_hat": float(calibration_artifacts.global_q_hat),
            **interval_metrics,
            "selected_interval_method": interval_method,
            "quantile_rejection_reasons": quantile_rejection_reasons,
            "interval_comparison_file": str(config.paths.tables_dir / "interval_comparison.csv"),
        },
        config.paths.reports_dir / "uncertainty_metrics.json",
    )
    save_dataframe(calibration_artifacts.price_band_summary, config.paths.tables_dir / "local_conformal_by_price_band.csv")
    save_dataframe(calibration_artifacts.segment_summary, config.paths.tables_dir / "local_conformal_by_segment.csv")
    save_json(
        {
            **calibration_artifacts.calibration_summary,
            "calibration_source": "chronological_validation_holdout",
            "calibration_rows": int(len(calibration_frame.dropna(subset=["observed_price", "fair_value_hat"]))),
            "global_empirical_coverage": float(interval_metrics["empirical_coverage"]),
            "global_average_interval_width": float(interval_metrics["average_interval_width"]),
            "q5_empirical_coverage": q5_coverage,
            "selected_interval_method": interval_method,
            "quantile_rejection_reasons": quantile_rejection_reasons,
            "price_band_summary_file": str(config.paths.tables_dir / "local_conformal_by_price_band.csv"),
            "segment_summary_file": str(config.paths.tables_dir / "local_conformal_by_segment.csv"),
        },
        config.paths.reports_dir / "local_conformal_calibration_summary.json",
    )

    eda_figures = create_eda_figures(modeling_df, config.paths.figures_dir)
    residual_figures = create_residual_diagnostics(property_ledger, config.paths.figures_dir)
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
        f"- Market segmentation: {cluster_artifacts.segmentation_method} ({cluster_artifacts.n_clusters} segments)",
        f"- Validation report: {config.validation_report_path}",
        f"- Cleaned rows retained: {cleaning_result.summary['rows_out']}",
        f"- Local conformal global q-hat: {calibration_artifacts.global_q_hat:.2f}",
        f"- Selected interval method: {interval_method}",
        f"- Quantile comparison rationale: {interval_comparison.loc[0, 'selection_rationale']}",
        f"- Test empirical coverage: {interval_metrics['empirical_coverage']:.3f}",
        f"- Test Q5 empirical coverage: {q5_coverage:.3f}",
        f"- Test average interval width: {interval_metrics['average_interval_width']:.2f}",
        f"- Potentially under-valued sales: {(property_frame['anomaly_flag'] == 'potentially_under_valued').sum()}",
        f"- Potentially over-valued sales: {(property_frame['anomaly_flag'] == 'potentially_over_valued').sum()}",
        "",
        "This system performs Pricing Anomaly Detection on realized sale prices and should not be interpreted as a listing-price decision rule.",
    ]
    summary_lines.extend(_model_comparison_markdown(model_comparison))
    summary_report = write_summary_report(summary_lines, config.paths.reports_dir / "pipeline_summary.md")
    trust_summary = write_trust_summary(
        valuation_metrics=valuation.valuation_metrics,
        interval_metrics=interval_metrics,
        q5_coverage=q5_coverage,
        q5_interval_width=q5_interval_width,
        property_ledger=property_ledger,
        output_path=config.paths.reports_dir / "trust_summary.md",
        interval_comparison=interval_comparison,
        selected_interval_method=interval_method,
        quantile_rejection_reasons=quantile_rejection_reasons,
    )
    model_flagged_cases = save_dataframe(
        _model_flagged_cases(property_ledger),
        config.paths.tables_dir / "model_flagged_cases.csv",
    )

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
        "baseline_models": "median_baseline, linear_regression, random_forest",
        "model_baseline_comparison": str(config.paths.tables_dir / "model_baseline_comparison.csv"),
        "anomaly_threshold_sensitivity": str(config.paths.tables_dir / "anomaly_threshold_sensitivity.csv"),
        "interval_comparison": str(config.paths.tables_dir / "interval_comparison.csv"),
        "interval_comparison_by_price_band": str(config.paths.tables_dir / "interval_comparison_by_price_band.csv"),
        "synthetic_anomaly_recall_comparison": str(config.paths.tables_dir / "synthetic_anomaly_recall_comparison.csv"),
        "property_intelligence": str(config.paths.tables_dir / "property_intelligence_table.csv"),
        "model_flagged_cases": str(model_flagged_cases),
        "feature_importance_plot": str(importance_plot),
        "summary_report": str(summary_report),
        "trust_summary": str(trust_summary),
    }
    if shap_path:
        outputs["shap_summary"] = str(shap_path)
    outputs.update({name: str(path) for name, path in eda_figures.items()})
    outputs.update({name: str(path) for name, path in residual_figures.items()})
    LOGGER.info("Pipeline complete.")
    return outputs
