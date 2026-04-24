from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dc_reif.config import ProjectConfig
from dc_reif.diagnostics import build_diagnostics_artifacts
from dc_reif.utils import ensure_directory, format_float, utc_timestamp


class ReportResultsError(RuntimeError):
    """Raised when report-ready summaries cannot be assembled from frozen outputs."""


@dataclass(frozen=True)
class ReportResultsPaths:
    reports_dir: Path
    tables_dir: Path
    figures_dir: Path
    final_pack_dir: Path

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "ReportResultsPaths":
        return cls(
            reports_dir=config.paths.reports_dir,
            tables_dir=config.paths.tables_dir,
            figures_dir=config.paths.figures_dir,
            final_pack_dir=config.paths.outputs_dir / "final_report_pack",
        )


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise ReportResultsError(f"Missing required upstream output: {label} at {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return json.loads(_require_file(path, label).read_text(encoding="utf-8"))


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    return pd.read_csv(_require_file(path, label))


def _load_valuation_metrics(paths: ReportResultsPaths) -> pd.DataFrame:
    official_metrics_path = paths.tables_dir / "valuation_metrics.csv"
    return _read_csv(official_metrics_path, "valuation metrics table")


def _flatten_summary(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                visit(next_prefix, nested)
            return
        if isinstance(value, list):
            rows.append({"metric": prefix, "value": json.dumps(value)})
            return
        rows.append({"metric": prefix, "value": value})

    visit("", summary)
    flattened = pd.DataFrame(rows)
    flattened["section"] = flattened["metric"].str.split(".").str[0]
    return flattened[["section", "metric", "value"]]


def _build_case_examples(property_df: pd.DataFrame, n_per_extreme: int = 3) -> pd.DataFrame:
    columns = [
        "property_id",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "segment_label",
        "predicted_price_band",
        "anomaly_flag",
        "anomaly_score",
        "support_score",
        "slice_risk_level",
        "confidence_note",
        "why_flagged",
        "evidence_strength",
        "top_drivers",
        "data_quality_flag",
    ]
    available_columns = [column for column in columns if column in property_df.columns]
    working = property_df.copy()
    working["abs_anomaly_score"] = working["anomaly_score"].abs()

    over = (
        working.loc[working["anomaly_flag"] == "potentially_over_valued", available_columns + ["abs_anomaly_score"]]
        .sort_values("abs_anomaly_score", ascending=False)
        .head(n_per_extreme)
    )
    under = (
        working.loc[working["anomaly_flag"] == "potentially_under_valued", available_columns + ["abs_anomaly_score"]]
        .sort_values("abs_anomaly_score", ascending=False)
        .head(n_per_extreme)
    )
    within = (
        working.loc[working["anomaly_flag"] == "within_expected_range", available_columns + ["abs_anomaly_score"]]
        .sort_values("abs_anomaly_score", ascending=True)
        .head(1)
    )
    insufficient = (
        working.loc[working["anomaly_flag"] == "insufficient_history", available_columns + ["abs_anomaly_score"]]
        .head(1)
    )

    case_examples = pd.concat([over, under, within, insufficient], ignore_index=True)
    if case_examples.empty:
        raise ReportResultsError("No case examples could be selected from property_intelligence_table.csv")

    case_examples["valuation_interval"] = (
        case_examples["lower_bound"].round(2).astype(str) + " to " + case_examples["upper_bound"].round(2).astype(str)
    )
    case_examples["observed_price"] = case_examples["observed_price"].round(2)
    case_examples["fair_value_hat"] = case_examples["fair_value_hat"].round(2)
    case_examples["anomaly_score"] = case_examples["anomaly_score"].round(4)
    return case_examples.drop(columns=["abs_anomaly_score"])


def build_final_results_summary(config: ProjectConfig | None = None) -> dict[str, Any]:
    config = config or ProjectConfig.default()
    paths = ReportResultsPaths.from_config(config)

    valuation_metrics = _load_valuation_metrics(paths)
    cluster_profiles = _read_csv(paths.tables_dir / "cluster_profiles.csv", "cluster profiles table")
    feature_importance = _read_csv(paths.tables_dir / "feature_importance.csv", "feature importance table")
    property_intelligence = _read_csv(paths.tables_dir / "property_intelligence_table.csv", "property intelligence table")
    cluster_summary = _read_json(paths.reports_dir / "cluster_summary.json", "cluster summary report")
    uncertainty_metrics = _read_json(paths.reports_dir / "uncertainty_metrics.json", "uncertainty summary report")
    local_conformal_summary = _optional_json(paths.reports_dir / "local_conformal_calibration_summary.json")

    selected_row = valuation_metrics.iloc[0]

    top_features = feature_importance["feature"].head(5).tolist()
    anomaly_counts = property_intelligence["anomaly_flag"].value_counts().to_dict()

    figure_paths = [
        paths.figures_dir / "price_distribution.png",
        paths.figures_dir / "log_price_distribution.png",
        paths.figures_dir / "spatial_price_map.png",
        paths.figures_dir / "temporal_trend.png",
        paths.figures_dir / "feature_importance.png",
    ]
    optional_figure = paths.figures_dir / "shap_summary.png"
    if optional_figure.exists():
        figure_paths.append(optional_figure)

    table_paths = [
        paths.tables_dir / "valuation_metrics.csv",
        paths.tables_dir / "cluster_profiles.csv",
        paths.tables_dir / "feature_importance.csv",
        paths.tables_dir / "property_intelligence_table.csv",
    ]
    upstream_reports = [
        _relative_to_project(paths.reports_dir / "cluster_summary.json", config.project_root),
        _relative_to_project(paths.reports_dir / "uncertainty_metrics.json", config.project_root),
        _relative_to_project(paths.reports_dir / "pipeline_summary.md", config.project_root),
    ]
    for optional_name in [
        "xgboost_selection_summary.json",
        "segmentation_selection_summary.json",
        "local_conformal_calibration_summary.json",
    ]:
        optional_path = paths.reports_dir / optional_name
        if optional_path.exists():
            upstream_reports.append(_relative_to_project(optional_path, config.project_root))

    summary = {
        "project": {
            "title": "Beyond Price Prediction: DC-REIF for King County",
            "framework": "DC-REIF",
            "dataset": "King County House Sales",
            "generated_at_utc": utc_timestamp(),
            "freeze_status": "final_validated_results",
        },
        "core_valuation_metrics": {
            "selected_model": str(selected_row["model_name"]),
            "validation_rmse": format_float(float(selected_row["validation_rmse"]), digits=2),
            "test_rmse": format_float(float(selected_row["test_rmse"]), digits=2),
            "validation_mae": format_float(float(selected_row["validation_mae"]), digits=2),
            "test_mae": format_float(float(selected_row["test_mae"]), digits=2),
            "validation_r2": format_float(float(selected_row["validation_r2"]), digits=4),
            "test_r2": format_float(float(selected_row["test_r2"]), digits=4),
        },
        "segmentation_results": {
            "segment_count": int(cluster_summary["n_clusters"]),
            "silhouette_score": format_float(float(cluster_summary["silhouette_score"]), digits=4),
            "davies_bouldin_index": format_float(float(cluster_summary["davies_bouldin_index"]), digits=4),
            "segment_summary_reference": _relative_to_project(paths.tables_dir / "cluster_profiles.csv", config.project_root),
            "segment_profile_count": int(len(cluster_profiles)),
        },
        "uncertainty_results": {
            "interval_method": (
                local_conformal_summary.get("interval_method")
                if local_conformal_summary
                else "conformal_prediction_residual_quantile"
            ),
            "interval_coverage": format_float(float(uncertainty_metrics["empirical_coverage"]), digits=4),
            "average_interval_width": format_float(float(uncertainty_metrics["average_interval_width"]), digits=2),
            "conformal_qhat": format_float(float(uncertainty_metrics["q_hat"]), digits=2),
        },
        "pricing_anomaly_results": {
            "within_expected_range": int(anomaly_counts.get("within_expected_range", 0)),
            "potentially_over_valued": int(anomaly_counts.get("potentially_over_valued", 0)),
            "potentially_under_valued": int(anomaly_counts.get("potentially_under_valued", 0)),
            "insufficient_history": int(anomaly_counts.get("insufficient_history", 0)),
        },
        "explainability_results": {
            "top_features": top_features,
            "feature_importance_file": _relative_to_project(paths.tables_dir / "feature_importance.csv", config.project_root),
            "shap_summary_file": (
                _relative_to_project(optional_figure, config.project_root) if optional_figure.exists() else None
            ),
        },
        "output_references": {
            "key_figure_files": [_relative_to_project(path, config.project_root) for path in figure_paths],
            "key_table_files": [_relative_to_project(path, config.project_root) for path in table_paths],
            "final_property_intelligence_table": _relative_to_project(
                paths.tables_dir / "property_intelligence_table.csv", config.project_root
            ),
            "upstream_reports": upstream_reports,
        },
    }
    return summary


def markdown_summary_block(summary: dict[str, Any]) -> str:
    core = summary["core_valuation_metrics"]
    segmentation = summary["segmentation_results"]
    uncertainty = summary["uncertainty_results"]
    anomalies = summary["pricing_anomaly_results"]
    explainability = summary["explainability_results"]

    lines = [
        "# Final Results Master Summary",
        "",
        "## Core Valuation Metrics",
        f"- Selected model: `{core['selected_model']}`",
        f"- Validation RMSE: `{core['validation_rmse']}`",
        f"- Test RMSE: `{core['test_rmse']}`",
        f"- Validation MAE: `{core['validation_mae']}`",
        f"- Test MAE: `{core['test_mae']}`",
        f"- Validation R2: `{core['validation_r2']}`",
        f"- Test R2: `{core['test_r2']}`",
        "",
        "## Segmentation Results",
        f"- Segment count: `{segmentation['segment_count']}`",
        f"- Silhouette score: `{segmentation['silhouette_score']}`",
        f"- Davies-Bouldin index: `{segmentation['davies_bouldin_index']}`",
        f"- Segment profiles: `{segmentation['segment_summary_reference']}`",
        "",
        "## Uncertainty Results",
        f"- Interval method: `{uncertainty['interval_method']}`",
        f"- Interval coverage: `{uncertainty['interval_coverage']}`",
        f"- Average interval width: `{uncertainty['average_interval_width']}`",
        f"- Conformal q-hat: `{uncertainty['conformal_qhat']}`",
        "",
        "## Pricing Anomaly Results",
        f"- Within expected range: `{anomalies['within_expected_range']}`",
        f"- Potentially over-valued: `{anomalies['potentially_over_valued']}`",
        f"- Potentially under-valued: `{anomalies['potentially_under_valued']}`",
        f"- Insufficient history: `{anomalies['insufficient_history']}`",
        "",
        "## Explainability Results",
        f"- Top features: `{', '.join(explainability['top_features'])}`",
        f"- Feature importance file: `{explainability['feature_importance_file']}`",
    ]
    if explainability["shap_summary_file"]:
        lines.append(f"- SHAP summary file: `{explainability['shap_summary_file']}`")
    return "\n".join(lines)


def terminal_results_block(summary: dict[str, Any]) -> str:
    core = summary["core_valuation_metrics"]
    uncertainty = summary["uncertainty_results"]
    anomalies = summary["pricing_anomaly_results"]
    return "\n".join(
        [
            "DC-REIF Final Results",
            f"Selected model: {core['selected_model']}",
            f"Validation RMSE: {core['validation_rmse']}",
            f"Test RMSE: {core['test_rmse']}",
            f"Test interval coverage: {uncertainty['interval_coverage']}",
            (
                "Anomaly counts: "
                f"within={anomalies['within_expected_range']}, "
                f"over={anomalies['potentially_over_valued']}, "
                f"under={anomalies['potentially_under_valued']}, "
                f"insufficient={anomalies['insufficient_history']}"
            ),
        ]
    )


def latex_core_metrics_table(summary: dict[str, Any]) -> str:
    core = summary["core_valuation_metrics"]
    rows = [
        ("Selected model", core["selected_model"]),
        ("Validation RMSE", core["validation_rmse"]),
        ("Test RMSE", core["test_rmse"]),
        ("Validation MAE", core["validation_mae"]),
        ("Test MAE", core["test_mae"]),
        ("Validation R2", core["validation_r2"]),
        ("Test R2", core["test_r2"]),
    ]
    body = "\n".join(f"{label} & {value} \\\\" for label, value in rows)
    return (
        "\\begin{tabular}{ll}\n"
        "\\hline\n"
        "Metric & Value \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )


def _caption_bank() -> str:
    return "\n".join(
        [
            "# Caption Bank",
            "",
            "## Architecture Diagram",
            "- Title: DC-REIF Architecture for Real Estate Intelligence",
            "- Caption: Overview of the DC-REIF control plane, governance layer, market representation layer, valuation core, and trust-oriented decision support outputs.",
            "",
            "## Pipeline Diagram",
            "- Title: End-to-End DC-REIF Pipeline",
            "- Caption: Reproducible workflow from raw-data download and validation through segmentation, valuation, uncertainty estimation, pricing anomaly detection, and report-ready exports.",
            "",
            "## Price Distribution",
            "- Title: Distribution of King County Sale Prices",
            "- Caption: Histogram of observed sale prices used to characterize target skewness and overall market dispersion in the raw transaction data.",
            "",
            "## Log-Price Distribution",
            "- Title: Log-Transformed Sale Price Distribution",
            "- Caption: Log-scale visualization of sale prices showing a more compact target distribution for descriptive analysis and outlier inspection.",
            "",
            "## Spatial Price Map",
            "- Title: Spatial Distribution of Sale Prices",
            "- Caption: Latitude-longitude scatter plot illustrating geographic concentration and price heterogeneity across the King County housing market.",
            "",
            "## Temporal Trend",
            "- Title: Median Sale Price Over Time",
            "- Caption: Monthly median sale price trend highlighting temporal market movement and motivating time-aware train-validation-test splits.",
            "",
            "## Feature Importance",
            "- Title: Global Feature Importance for the Official Valuation Model",
            "- Caption: Ranked feature importance values from the selected valuation model, with structural quality, living area, and location-related variables contributing the strongest signal.",
            "",
            "## SHAP Summary",
            "- Title: SHAP Summary for the Selected Valuation Model",
            "- Caption: SHAP-based summary of global driver effects for the final valuation model, included as an explainability reference where runtime permits.",
            "",
            "## Valuation Performance Table",
            "- Title: Official Valuation Model Performance",
            "- Caption: Validation and holdout test metrics for the single official DC-REIF valuation model used in the final pricing anomaly workflow.",
            "",
            "## Segmentation Summary Table",
            "- Title: KMeans Contextual Market Grouping Summary",
            "- Caption: Cluster-level summary for the KMeans contextual market grouping used as market-context representation within the valuation workflow.",
            "",
            "## Anomaly Summary Table",
            "- Title: Pricing Anomaly Category Counts",
            "- Caption: Counts of sale transactions classified as within expected range, potentially over-valued, potentially under-valued, or insufficient history.",
            "",
            "## Property Intelligence Example Table",
            "- Title: Illustrative Property Intelligence Cases",
            "- Caption: Selected property-level examples showing observed sale price, model-implied fair value, uncertainty interval, segment context, anomaly label, and top drivers.",
        ]
    )


def _selected_figures_manifest(summary: dict[str, Any]) -> str:
    lines = ["# Selected Figures Manifest", ""]
    usage = {
        "price_distribution.png": "Descriptive EDA figure for target spread in the report body.",
        "log_price_distribution.png": "Supplementary descriptive figure for skewness discussion.",
        "spatial_price_map.png": "Spatial EDA figure for geographic market variation.",
        "temporal_trend.png": "Temporal EDA figure supporting chronological split rationale.",
        "feature_importance.png": "Primary explainability figure for report and slides.",
        "shap_summary.png": "Secondary explainability figure for appendix or advanced slides.",
    }
    for relative_path in summary["output_references"]["key_figure_files"]:
        name = Path(relative_path).name
        lines.append(f"- `{relative_path}`: {usage.get(name, 'Report-ready figure artifact.')}")
    return "\n".join(lines)


def _selected_tables_manifest(summary: dict[str, Any]) -> str:
    lines = ["# Selected Tables Manifest", ""]
    usage = {
        "valuation_metrics.csv": "Official validation and holdout performance table for the single active valuation model.",
        "cluster_profiles.csv": "Segmentation reference table for contextual market grouping.",
        "feature_importance.csv": "Explainability reference table for top global drivers.",
        "property_intelligence_table.csv": "Master property-level output table for anomaly analysis and case selection.",
    }
    for relative_path in summary["output_references"]["key_table_files"]:
        name = Path(relative_path).name
        lines.append(f"- `{relative_path}`: {usage.get(name, 'Report-ready table artifact.')}")
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _xgboost_parameters_block(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "# Selected XGBoost Parameters\n\nNo XGBoost selection summary was found."
    lines = [
        "# Selected XGBoost Parameters",
        "",
        f"- Selected model: `{payload.get('selected_model', 'xgboost')}`",
        f"- Target strategy: `{payload.get('target_strategy', 'raw')}`",
        f"- High-price sample weight: `{payload.get('high_price_weight', 1.0)}`",
        "",
        "## Parameters",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in payload.get("selected_parameters", {}).items())
    return "\n".join(lines)


def _segmentation_summary_block(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "# Segmentation Selection Summary\n\nNo segmentation selection summary was found."
    details = payload.get("selection_details", {})
    lines = [
        "# Segmentation Selection Summary",
        "",
        f"- Selected K: `{payload.get('selected_k')}`",
        f"- Silhouette score: `{format_float(float(details.get('silhouette_score', 0.0)), digits=4)}`",
        f"- Davies-Bouldin index: `{format_float(float(details.get('davies_bouldin_index', 0.0)), digits=4)}`",
        f"- Balance score: `{format_float(float(details.get('balance_score', 0.0)), digits=4)}`",
        f"- Small-cluster share: `{format_float(float(details.get('small_cluster_share', 0.0)), digits=4)}`",
    ]
    return "\n".join(lines)


def _local_conformal_block(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "# Local Conformal Calibration Summary\n\nNo local conformal calibration summary was found."
    lines = [
        "# Local Conformal Calibration Summary",
        "",
        f"- Interval method: `{payload.get('interval_method')}`",
        f"- Global q-hat: `{format_float(float(payload.get('global_q_hat', 0.0)), digits=2)}`",
        f"- Average localized q-hat: `{format_float(float(payload.get('average_local_q_hat', 0.0)), digits=2)}`",
        f"- Global empirical coverage: `{format_float(float(payload.get('global_empirical_coverage', 0.0)), digits=4)}`",
        f"- Q5 empirical coverage: `{format_float(float(payload.get('q5_empirical_coverage', 0.0)), digits=4)}`",
        f"- Global average interval width: `{format_float(float(payload.get('global_average_interval_width', 0.0)), digits=2)}`",
    ]
    return "\n".join(lines)


def build_report_results_pack(config: ProjectConfig | None = None) -> dict[str, Path]:
    config = config or ProjectConfig.default()
    paths = ReportResultsPaths.from_config(config)
    ensure_directory(paths.final_pack_dir)

    summary = build_final_results_summary(config)
    summary_json = paths.reports_dir / "final_results_master.json"
    summary_md = paths.reports_dir / "final_results_master.md"
    summary_csv = paths.reports_dir / "final_results_master.csv"

    _write_text(summary_json, json.dumps(summary, indent=2, sort_keys=True))
    _write_text(summary_md, markdown_summary_block(summary))
    _flatten_summary(summary).to_csv(summary_csv, index=False)

    property_intelligence = _read_csv(paths.tables_dir / "property_intelligence_table.csv", "property intelligence table")
    case_examples = _build_case_examples(property_intelligence)

    results_summary_rows = pd.DataFrame(
        [
            {"section": "valuation", "metric": "selected_model", "value": summary["core_valuation_metrics"]["selected_model"]},
            {"section": "valuation", "metric": "validation_rmse", "value": summary["core_valuation_metrics"]["validation_rmse"]},
            {"section": "valuation", "metric": "test_rmse", "value": summary["core_valuation_metrics"]["test_rmse"]},
            {"section": "valuation", "metric": "validation_mae", "value": summary["core_valuation_metrics"]["validation_mae"]},
            {"section": "valuation", "metric": "test_mae", "value": summary["core_valuation_metrics"]["test_mae"]},
            {"section": "valuation", "metric": "validation_r2", "value": summary["core_valuation_metrics"]["validation_r2"]},
            {"section": "valuation", "metric": "test_r2", "value": summary["core_valuation_metrics"]["test_r2"]},
            {"section": "segmentation", "metric": "segment_count", "value": summary["segmentation_results"]["segment_count"]},
            {"section": "segmentation", "metric": "silhouette_score", "value": summary["segmentation_results"]["silhouette_score"]},
            {
                "section": "segmentation",
                "metric": "davies_bouldin_index",
                "value": summary["segmentation_results"]["davies_bouldin_index"],
            },
            {"section": "uncertainty", "metric": "interval_method", "value": summary["uncertainty_results"]["interval_method"]},
            {"section": "uncertainty", "metric": "interval_coverage", "value": summary["uncertainty_results"]["interval_coverage"]},
            {
                "section": "uncertainty",
                "metric": "average_interval_width",
                "value": summary["uncertainty_results"]["average_interval_width"],
            },
            {"section": "uncertainty", "metric": "conformal_qhat", "value": summary["uncertainty_results"]["conformal_qhat"]},
        ]
    )
    anomaly_summary_rows = pd.DataFrame(
        [
            {"anomaly_flag": key, "count": value}
            for key, value in summary["pricing_anomaly_results"].items()
        ]
    )

    artifacts = {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "summary_csv": summary_csv,
        "pack_core_metrics": _write_text(paths.final_pack_dir / "01_core_metrics.md", markdown_summary_block(summary)),
        "pack_results_table": paths.final_pack_dir / "02_results_summary_table.csv",
        "pack_anomaly_table": paths.final_pack_dir / "03_anomaly_summary_table.csv",
        "pack_figures_manifest": _write_text(
            paths.final_pack_dir / "04_selected_figures_manifest.md", _selected_figures_manifest(summary)
        ),
        "pack_tables_manifest": _write_text(
            paths.final_pack_dir / "05_selected_tables_manifest.md", _selected_tables_manifest(summary)
        ),
        "pack_caption_bank": _write_text(paths.final_pack_dir / "06_caption_bank.md", _caption_bank()),
        "pack_case_examples": paths.final_pack_dir / "07_case_examples.csv",
        "pack_core_metrics_tex": _write_text(
            paths.final_pack_dir / "08_core_metrics_table.tex", latex_core_metrics_table(summary)
        ),
    }
    results_summary_rows.to_csv(artifacts["pack_results_table"], index=False)
    anomaly_summary_rows.to_csv(artifacts["pack_anomaly_table"], index=False)
    case_examples.to_csv(artifacts["pack_case_examples"], index=False)
    artifacts.update(build_diagnostics_artifacts(config, summary=summary))
    artifacts["pack_xgboost_parameters"] = _write_text(
        paths.final_pack_dir / "18_selected_xgboost_parameters.md",
        _xgboost_parameters_block(_optional_json(paths.reports_dir / "xgboost_selection_summary.json")),
    )
    artifacts["pack_segmentation_selection"] = _write_text(
        paths.final_pack_dir / "19_segmentation_selection_summary.md",
        _segmentation_summary_block(_optional_json(paths.reports_dir / "segmentation_selection_summary.json")),
    )
    artifacts["pack_local_conformal"] = _write_text(
        paths.final_pack_dir / "20_local_conformal_summary.md",
        _local_conformal_block(_optional_json(paths.reports_dir / "local_conformal_calibration_summary.json")),
    )
    return artifacts
