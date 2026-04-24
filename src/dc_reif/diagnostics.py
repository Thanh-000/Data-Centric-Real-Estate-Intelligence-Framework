from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dc_reif.config import ProjectConfig
from dc_reif.feature_engineering import add_safe_derived_features
from dc_reif.utils import ensure_directory


class DiagnosticsError(RuntimeError):
    """Raised when report diagnostics cannot be built from pipeline outputs."""


@dataclass(frozen=True)
class DiagnosticsPaths:
    feature_dataset: Path
    property_table: Path
    final_pack_dir: Path

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        feature_dataset_path: Path | None = None,
        property_table_path: Path | None = None,
    ) -> "DiagnosticsPaths":
        return cls(
            feature_dataset=feature_dataset_path or config.feature_dataset_path,
            property_table=property_table_path or (config.paths.tables_dir / "property_intelligence_table.csv"),
            final_pack_dir=config.paths.outputs_dir / "final_report_pack",
        )


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise DiagnosticsError(f"Missing required upstream output: {label} at {path}")
    return path


def _write_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def _price_band(series: pd.Series, n_bands: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="string")
    n_quantiles = min(n_bands, max(valid.nunique(), 1))
    labels = [f"Q{index}" for index in range(1, n_quantiles + 1)]
    ranked = valid.rank(method="first")
    bands = pd.qcut(ranked, q=n_quantiles, labels=labels)
    band_series = pd.Series(pd.NA, index=series.index, dtype="string")
    band_series.loc[valid.index] = bands.astype("string")
    return band_series


def load_diagnostic_frame(
    config: ProjectConfig,
    feature_dataset_path: Path | None = None,
    property_table_path: Path | None = None,
) -> pd.DataFrame:
    paths = DiagnosticsPaths.from_config(
        config,
        feature_dataset_path=feature_dataset_path,
        property_table_path=property_table_path,
    )
    property_df = pd.read_csv(_require_file(paths.property_table, "property intelligence table"))
    feature_df = pd.read_csv(_require_file(paths.feature_dataset, "feature dataset"), parse_dates=["date"])

    property_df["property_id"] = property_df["property_id"].astype(str)
    feature_df["property_id"] = feature_df[config.id_column].astype(str)
    feature_df = add_safe_derived_features(feature_df)

    context_columns = [
        "property_id",
        "date",
        "zipcode",
        "lat",
        "long",
        "sqft_living",
        "grade",
        "house_age",
        "renovated_flag",
        "living_to_lot_ratio",
        "basement_share",
        "bathrooms_per_bedroom",
        "relative_living_area",
        "relative_lot_size",
        "geo_cell",
        "lat_bin",
        "long_bin",
        "distance_to_seattle_core",
        "distance_to_bellevue_core",
        "grade_living_interaction",
        "waterfront_view_score",
        "location_grade_interaction",
    ]
    available_context = [
        column for column in context_columns if column in feature_df.columns and column not in property_df.columns
    ]
    merged = property_df.merge(feature_df[["property_id", *available_context]], on="property_id", how="left")

    if "sale_date" not in merged.columns and "date" in merged.columns:
        merged["sale_date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "zipcode" in merged.columns:
        merged["zipcode"] = merged["zipcode"].astype("string")

    merged["valuation_gap"] = merged["observed_price"] - merged["fair_value_hat"]
    merged["abs_error"] = merged["valuation_gap"].abs()
    merged["ape"] = merged["abs_error"] / merged["observed_price"].replace(0, np.nan) * 100
    merged["within_interval"] = (
        (merged["observed_price"] >= merged["lower_bound"]) & (merged["observed_price"] <= merged["upper_bound"])
    )
    merged["price_band"] = _price_band(merged["observed_price"])
    return merged


def error_summary(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    scored = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: group_value,
                "count": int(len(frame)),
                "rmse": float(np.sqrt(np.mean(np.square(frame["valuation_gap"])))),
                "mae": float(frame["abs_error"].mean()),
                "mape": float(frame["ape"].mean()),
                "median_abs_error": float(frame["abs_error"].median()),
                "mean_signed_error": float(frame["valuation_gap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", group_column], ascending=[False, True]).reset_index(drop=True)


def coverage_summary(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    scored = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for group_value, frame in scored.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: group_value,
                "count": int(len(frame)),
                "empirical_coverage": float(frame["within_interval"].mean()),
                "average_interval_width": float(frame["interval_width"].mean()),
                "median_interval_width": float(frame["interval_width"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", group_column], ascending=[False, True]).reset_index(drop=True)


def anomaly_distribution(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    counts = dataframe.groupby([group_column, "anomaly_flag"]).size().unstack(fill_value=0)
    for column in [
        "within_expected_range",
        "potentially_over_valued",
        "potentially_under_valued",
        "insufficient_history",
    ]:
        if column not in counts.columns:
            counts[column] = 0

    counts = counts[
        [
            "within_expected_range",
            "potentially_over_valued",
            "potentially_under_valued",
            "insufficient_history",
        ]
    ]
    counts["count"] = counts.sum(axis=1)
    counts["over_rate"] = counts["potentially_over_valued"] / counts["count"]
    counts["under_rate"] = counts["potentially_under_valued"] / counts["count"]
    counts["within_rate"] = counts["within_expected_range"] / counts["count"]
    counts["insufficient_rate"] = counts["insufficient_history"] / counts["count"]
    return counts.reset_index().sort_values("count", ascending=False).reset_index(drop=True)


def segment_profiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    profiled = dataframe.copy()
    profiled["is_over_valued"] = profiled["anomaly_flag"].eq("potentially_over_valued").astype(int)
    profiled["is_under_valued"] = profiled["anomaly_flag"].eq("potentially_under_valued").astype(int)
    profiled["is_within_range"] = profiled["anomaly_flag"].eq("within_expected_range").astype(int)
    profiled["is_insufficient_history"] = profiled["anomaly_flag"].eq("insufficient_history").astype(int)

    available_aggregations: dict[str, str] = {
        "property_id": "count",
        "observed_price": "median",
        "fair_value_hat": "median",
        "sqft_living": "median",
        "grade": "median",
        "house_age": "median",
        "lat": "median",
        "long": "median",
        "living_to_lot_ratio": "median",
        "basement_share": "median",
        "bathrooms_per_bedroom": "median",
        "relative_living_area": "median",
        "distance_to_seattle_core": "median",
        "distance_to_bellevue_core": "median",
        "renovated_flag": "mean",
        "is_over_valued": "mean",
        "is_under_valued": "mean",
        "is_within_range": "mean",
        "is_insufficient_history": "mean",
    }
    aggregations = {column: function for column, function in available_aggregations.items() if column in profiled.columns}
    summary = profiled.groupby("segment_label").agg(aggregations).reset_index()

    rename_map = {
        "property_id": "count",
        "observed_price": "median_observed_price",
        "fair_value_hat": "median_fair_value",
        "sqft_living": "median_sqft_living",
        "grade": "median_grade",
        "house_age": "median_house_age",
        "lat": "median_lat",
        "long": "median_long",
        "living_to_lot_ratio": "median_living_to_lot_ratio",
        "basement_share": "median_basement_share",
        "bathrooms_per_bedroom": "median_bathrooms_per_bedroom",
        "relative_living_area": "median_relative_living_area",
        "distance_to_seattle_core": "median_distance_to_seattle_core",
        "distance_to_bellevue_core": "median_distance_to_bellevue_core",
        "renovated_flag": "renovated_share",
        "is_over_valued": "over_rate",
        "is_under_valued": "under_rate",
        "is_within_range": "within_rate",
        "is_insufficient_history": "insufficient_rate",
    }
    summary = summary.rename(columns=rename_map)
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def anomaly_casebook(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "property_id",
        "sale_date",
        "zipcode",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "segment_label",
        "price_band",
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
    available = [column for column in display_columns if column in dataframe.columns]

    over = dataframe.loc[dataframe["anomaly_flag"] == "potentially_over_valued"].sort_values(
        "anomaly_score", ascending=False
    )
    under = dataframe.loc[dataframe["anomaly_flag"] == "potentially_under_valued"].sort_values(
        "anomaly_score", ascending=True
    )
    within = dataframe.loc[dataframe["anomaly_flag"] == "within_expected_range"].assign(
        abs_anomaly_score=lambda frame: frame["anomaly_score"].abs()
    )
    within = within.sort_values("abs_anomaly_score", ascending=True)
    insufficient = dataframe.loc[dataframe["anomaly_flag"] == "insufficient_history"].sort_values(
        "observed_price", ascending=False
    )

    casebook = pd.concat([over.head(3), under.head(3), within.head(2), insufficient.head(1)], ignore_index=True)
    if casebook.empty:
        raise DiagnosticsError("No anomaly casebook rows could be selected from property intelligence outputs.")
    return casebook.loc[:, available]


def collect_slice_diagnostics(
    config: ProjectConfig,
    feature_dataset_path: Path | None = None,
    property_table_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    diagnostic_frame = load_diagnostic_frame(
        config,
        feature_dataset_path=feature_dataset_path,
        property_table_path=property_table_path,
    )
    return {
        "diagnostic_frame": diagnostic_frame,
        "error_by_segment": error_summary(diagnostic_frame, "segment_label"),
        "error_by_price_band": error_summary(diagnostic_frame, "price_band"),
        "coverage_by_segment": coverage_summary(diagnostic_frame, "segment_label"),
        "coverage_by_price_band": coverage_summary(diagnostic_frame, "price_band"),
        "anomaly_by_segment": anomaly_distribution(diagnostic_frame, "segment_label"),
        "anomaly_by_price_band": anomaly_distribution(diagnostic_frame, "price_band"),
        "segment_profiles": segment_profiles(diagnostic_frame),
        "anomaly_casebook": anomaly_casebook(diagnostic_frame),
    }


def geospatial_feature_notes() -> str:
    return "\n".join(
        [
            "# Geospatial Feature Notes",
            "",
            "- `lat_bin` and `long_bin` provide coarse deterministic spatial grouping without fitting a global learned map.",
            "- `geo_cell` captures a higher-resolution latitude-longitude grid and remains leakage-safe because it is derived only from observed coordinates.",
            "- `distance_to_seattle_core` and `distance_to_bellevue_core` summarize proximity to major employment and price centers using fixed anchors.",
            "- `grade_living_interaction` and `location_grade_interaction` expose simple structural-location interactions without introducing price-derived information.",
            "- These spatial-context features are part of the official dataset-aligned feature policy for the final DC-REIF system.",
        ]
    )


def slice_interpretation_notes(
    summary: dict[str, Any],
    diagnostics: dict[str, pd.DataFrame],
) -> str:
    error_by_segment = diagnostics["error_by_segment"]
    error_by_price_band = diagnostics["error_by_price_band"]
    coverage_by_segment = diagnostics["coverage_by_segment"]
    coverage_by_price_band = diagnostics["coverage_by_price_band"]
    anomaly_by_segment = diagnostics["anomaly_by_segment"]
    anomaly_by_price_band = diagnostics["anomaly_by_price_band"]

    highest_mae_segment = error_by_segment.sort_values("mae", ascending=False).iloc[0]
    lowest_mae_segment = error_by_segment.sort_values("mae", ascending=True).iloc[0]
    weakest_price_band = error_by_price_band.sort_values("mae", ascending=False).iloc[0]
    weakest_coverage_price_band = coverage_by_price_band.sort_values("empirical_coverage", ascending=True).iloc[0]
    lowest_coverage_segment = coverage_by_segment.sort_values("empirical_coverage", ascending=True).iloc[0]
    anomaly_hotspot = anomaly_by_segment.sort_values("over_rate", ascending=False).iloc[0]
    price_band_hotspot = anomaly_by_price_band.sort_values("over_rate", ascending=False).iloc[0]

    lines = [
        "# Slice Interpretation Notes",
        "",
        "## Official System",
        (
            f"- The official valuation model is `{summary['core_valuation_metrics']['selected_model']}` with validation/test RMSE "
            f"{summary['core_valuation_metrics']['validation_rmse']} / {summary['core_valuation_metrics']['test_rmse']}."
        ),
        (
            f"- Highest segment MAE remains `{highest_mae_segment['segment_label']}` at {highest_mae_segment['mae']:.2f}; "
            f"lowest segment MAE is `{lowest_mae_segment['segment_label']}` at {lowest_mae_segment['mae']:.2f}."
        ),
        (
            f"- Weakest price band by MAE is `{weakest_price_band['price_band']}` at {weakest_price_band['mae']:.2f}."
        ),
        (
            f"- Weakest price-band coverage appears in `{weakest_coverage_price_band['price_band']}` at "
            f"{weakest_coverage_price_band['empirical_coverage']:.4f}."
        ),
        (
            f"- Lowest segment coverage appears in `{lowest_coverage_segment['segment_label']}` at "
            f"{lowest_coverage_segment['empirical_coverage']:.4f}."
        ),
        (
            f"- Highest over-valued anomaly rate by segment is `{anomaly_hotspot['segment_label']}` at "
            f"{anomaly_hotspot['over_rate']:.3f}; by price band it is `{price_band_hotspot['price_band']}` at "
            f"{price_band_hotspot['over_rate']:.3f}."
        ),
    ]
    lines.extend(
        [
            "",
            "These diagnostics support sale-price Pricing Anomaly Detection and should be read as valuation-gap evidence for realized transactions.",
        ]
    )
    return "\n".join(lines)


def build_diagnostics_artifacts(
    config: ProjectConfig | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    config = config or ProjectConfig.default()
    paths = DiagnosticsPaths.from_config(config)
    ensure_directory(paths.final_pack_dir)

    diagnostics = collect_slice_diagnostics(config)
    notes = slice_interpretation_notes(summary or {}, diagnostics) if summary else ""

    artifact_paths = {
        "pack_error_by_segment": paths.final_pack_dir / "09_error_by_segment.csv",
        "pack_error_by_price_band": paths.final_pack_dir / "10_error_by_price_band.csv",
        "pack_coverage_by_segment": paths.final_pack_dir / "11_coverage_by_segment.csv",
        "pack_coverage_by_price_band": paths.final_pack_dir / "12_coverage_by_price_band.csv",
        "pack_anomaly_by_segment": paths.final_pack_dir / "13_anomaly_by_segment.csv",
        "pack_anomaly_by_price_band": paths.final_pack_dir / "14_anomaly_by_price_band.csv",
        "pack_improved_segment_profiles": paths.final_pack_dir / "14_improved_segment_profiles.csv",
        "pack_anomaly_casebook": paths.final_pack_dir / "15_anomaly_casebook.csv",
        "pack_interpretation_notes": paths.final_pack_dir / "16_interpretation_notes.md",
        "pack_geospatial_notes": paths.final_pack_dir / "17_geospatial_feature_notes.md",
    }

    diagnostics["error_by_segment"].to_csv(artifact_paths["pack_error_by_segment"], index=False)
    diagnostics["error_by_price_band"].to_csv(artifact_paths["pack_error_by_price_band"], index=False)
    diagnostics["coverage_by_segment"].to_csv(artifact_paths["pack_coverage_by_segment"], index=False)
    diagnostics["coverage_by_price_band"].to_csv(artifact_paths["pack_coverage_by_price_band"], index=False)
    diagnostics["anomaly_by_segment"].to_csv(artifact_paths["pack_anomaly_by_segment"], index=False)
    diagnostics["anomaly_by_price_band"].to_csv(artifact_paths["pack_anomaly_by_price_band"], index=False)
    diagnostics["segment_profiles"].to_csv(artifact_paths["pack_improved_segment_profiles"], index=False)
    diagnostics["anomaly_casebook"].to_csv(artifact_paths["pack_anomaly_casebook"], index=False)
    _write_text(artifact_paths["pack_interpretation_notes"], notes)
    _write_text(artifact_paths["pack_geospatial_notes"], geospatial_feature_notes())
    return artifact_paths
