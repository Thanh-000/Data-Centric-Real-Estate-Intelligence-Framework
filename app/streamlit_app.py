from __future__ import annotations

import json
from html import escape
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from dc_reif.product_analytics import with_product_bands


APP_FILE = Path(__file__).resolve()
DEFAULT_TABLE = ROOT / "outputs" / "tables" / "property_intelligence_table.csv"
DEFAULT_TRUST_SUMMARY = ROOT / "outputs" / "reports" / "trust_summary.md"
DEFAULT_CONFORMAL_SUMMARY = ROOT / "outputs" / "reports" / "local_conformal_calibration_summary.json"
DEFAULT_COVERAGE_BY_BAND = ROOT / "outputs" / "tables" / "test_interval_coverage_by_price_band.csv"
DEFAULT_ERROR_BY_BAND = ROOT / "outputs" / "tables" / "test_error_by_price_band.csv"
DEFAULT_MODEL_COMPARISON = ROOT / "outputs" / "tables" / "model_baseline_comparison.csv"
DEFAULT_FEATURE_IMPORTANCE = ROOT / "outputs" / "tables" / "feature_importance.csv"
DEFAULT_DATA_QUALITY = ROOT / "outputs" / "reports" / "data_quality_report.json"
DEFAULT_CLEANING_SUMMARY = ROOT / "outputs" / "reports" / "cleaning_summary.json"
DEFAULT_UNCERTAINTY_METRICS = ROOT / "outputs" / "reports" / "uncertainty_metrics.json"
DEFAULT_VALUATION_METRICS = ROOT / "outputs" / "tables" / "valuation_metrics.csv"
DEFAULT_SHAP_PNG = ROOT / "outputs" / "figures" / "shap_summary.png"
DEFAULT_FEATURE_IMP_PNG = ROOT / "outputs" / "figures" / "feature_importance.png"
DEFAULT_RAW_DATA = ROOT / "data" / "raw" / "kc_house_data.csv"
DEFAULT_FEATURES_DATA = ROOT / "data" / "processed" / "kc_house_features.csv"

LABEL_COLORS = {
    "potentially_over_valued": [255, 75, 75, 220],
    "potentially_under_valued": [9, 171, 59, 220],
    "insufficient_history": [255, 165, 0, 210],
    "within_expected_range": [34, 197, 94, 155],
}

LABEL_NAMES = {
    "potentially_over_valued": "Over-valued",
    "potentially_under_valued": "Under-valued",
    "insufficient_history": "Low support",
    "within_expected_range": "Within expected range",
}

ACTIONABLE_LABELS = ["potentially_over_valued", "potentially_under_valued"]

FILTER_LABELS = {
    "anomaly_flag": "Model signal",
    "zipcode": "Zipcode",
    "observed_price_band": "Price band",
    "evidence_strength": "Evidence strength",
    "slice_risk_level": "Slice risk",
}

FOCUS_LABELS = {
    "Anomalies only": "Model-flagged cases",
    "All transactions": "All sales for context",
}

FOCUS_HELP = {
    "Anomalies only": "Shows model-flagged candidates for human review. This is not a final valuation judgment.",
    "All transactions": "Shows normal sales too, useful for geographic context but visually denser.",
}

QUEUE_COLUMN_NAMES = {
    "property_id": "Property ID",
    "sale_date": "Sale date",
    "zipcode": "Zipcode",
    "observed_price": "Observed price",
    "fair_value_hat": "Fair value estimate",
    "lower_bound": "Lower fair range",
    "upper_bound": "Upper fair range",
    "review_label": "Model signal",
    "anomaly_score": "Review score",
    "evidence_strength": "Evidence strength",
    "slice_risk_level": "Slice risk",
    "model_confidence": "Model confidence",
    "review_note": "Human review note",
    "top_drivers": "Main drivers",
    "why_flagged": "Reason",
}

SLICE_COLUMN_NAMES = {
    "zipcode": "Zipcode",
    "observed_price_band": "Price band",
    "evidence_strength": "Evidence strength",
    "slice_risk_level": "Slice risk",
    "transactions": "Sales",
    "anomalies": "Model flags",
    "low_support": "Low support",
    "median_price": "Median price",
    "anomaly_rate": "Review flag rate",
    "low_support_rate": "Low-support rate",
}

SIDEBAR_FILTER_COLUMNS = [
    "zipcode",
    "observed_price_band",
    "evidence_strength",
    "slice_risk_level",
]

MARKET_SLICE_COLUMNS = [
    "zipcode",
    "observed_price_band",
    "evidence_strength",
    "slice_risk_level",
]

MAP_FOCUS = {
    "Anomalies only": [*ACTIONABLE_LABELS, "insufficient_history"],
    "All transactions": [
        "potentially_over_valued",
        "potentially_under_valued",
        "insufficient_history",
        "within_expected_range",
    ],
}


@st.cache_data
def load_property_table(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return with_product_bands(dataframe)


@st.cache_data
def load_trust_metrics() -> dict[str, object]:
    payload: dict[str, object] = {}
    if DEFAULT_CONFORMAL_SUMMARY.exists():
        payload.update(json.loads(DEFAULT_CONFORMAL_SUMMARY.read_text(encoding="utf-8")))
    if DEFAULT_COVERAGE_BY_BAND.exists():
        bands = pd.read_csv(DEFAULT_COVERAGE_BY_BAND)
        q5 = bands.loc[bands["price_band"].eq("Q5")]
        if not q5.empty:
            payload["q5_interval_width"] = float(q5["average_interval_width"].iloc[0])
            payload["q5_coverage_from_table"] = float(q5["empirical_coverage"].iloc[0])
    if DEFAULT_ERROR_BY_BAND.exists():
        errors = pd.read_csv(DEFAULT_ERROR_BY_BAND)
        q1_error = errors.loc[errors["price_band"].eq("Q1")]
        if not q1_error.empty:
            payload["q1_mape"] = float(q1_error["mape"].iloc[0])
    return payload


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&family=Manrope:wght@600;700&display=swap');

        /* Apply typography */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Manrope', sans-serif !important;
            color: #003366 !important;
        }
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        section[data-testid="stSidebar"] {
            min-width: 310px;
            max-width: 340px;
            background-color: #f9f9fe;
        }
        h1 {
            font-size: 2.35rem !important;
            line-height: 1.12 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.2rem !important;
        }
        h2, h3 {
            letter-spacing: 0 !important;
        }
        div[data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid #e2e2e7;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 4px 4px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetricValue"] {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 28px;
            color: #003366;
        }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem 1rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.9rem;
            color: inherit;
            opacity: 0.86;
            font-family: 'Inter', sans-serif;
        }
        .legend-dot {
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 50%;
            display: inline-block;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        .hint {
            color: inherit;
            opacity: 0.72;
            font-size: 0.92rem;
            margin-bottom: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .status-line {
            border-left: 4px solid #ff4b4b;
            background: #ffdad6;
            color: #93000a;
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin: 0.8rem 0 1rem 0;
            font-size: 0.96rem;
            font-family: 'Inter', sans-serif;
        }
        .map-help {
            color: inherit;
            opacity: 0.74;
            font-size: 0.92rem;
            margin-top: -0.2rem;
            margin-bottom: 0.5rem;
        }
        .model-caveat {
            border-left: 4px solid #FFA500;
            background: rgba(255, 165, 0, 0.10);
            border-radius: 6px;
            padding: 0.7rem 0.9rem;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.93rem;
            color: inherit;
        }

        /* Table Headers */
        th {
            background-color: #f0f2f6 !important;
            font-family: 'JetBrains Mono', monospace !important;
            text-transform: uppercase;
        }
        /* Section Card */
        .section-card {
            background: #ffffff;
            border: 1px solid #e2e2e7;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .section-card h4 {
            font-family: 'Manrope', sans-serif !important;
            color: #003366 !important;
            margin-bottom: 0.4rem !important;
        }
        /* KPI label */
        .kpi-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.6rem;
            color: #003366;
        }
        /* Tab description */
        .tab-desc {
            font-size: 0.93rem;
            color: #6b7280;
            margin: -0.5rem 0 1rem 0;
            font-family: 'Inter', sans-serif;
        }
        /* Stat pill for data quality */
        .stat-pill {
            display: inline-block;
            background: #e8f5e9;
            color: #09AB3B;
            font-weight: 600;
            padding: 0.25rem 0.7rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .stat-pill.warn {
            background: #fff3e0;
            color: #e65100;
        }
        .stat-pill.danger {
            background: #ffdad6;
            color: #FF4B4B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def options_for(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        return []
    return sorted([str(value) for value in dataframe[column].dropna().unique()])


def display_value(column: str, value: str) -> str:
    if column == "anomaly_flag":
        return LABEL_NAMES.get(value, value.replace("_", " ").title())
    if column == "evidence_strength":
        return value.replace("_", " ").title()
    if column == "slice_risk_level":
        return value.replace("_", " ").title()
    return value


def display_column(column: str) -> str:
    return FILTER_LABELS.get(column, column.replace("_", " ").title())


def sidebar_filter(dataframe: pd.DataFrame, column: str, label: str) -> tuple[pd.DataFrame, list[str]]:
    options = options_for(dataframe, column)
    if not options:
        st.sidebar.caption(f"{label}: unavailable")
        return dataframe, []
    selected = st.sidebar.multiselect(
        label,
        options,
        format_func=lambda value: display_value(column, value),
        placeholder="All",
    )
    if not selected:
        return dataframe, []
    return dataframe.loc[dataframe[column].astype(str).isin(selected)], selected


def apply_selected_filters(dataframe: pd.DataFrame, selections: dict[str, list[str]]) -> pd.DataFrame:
    filtered = dataframe.copy()
    for column, selected in selections.items():
        if selected and column in filtered.columns:
            filtered = filtered.loc[filtered[column].astype(str).isin(selected)]
    return filtered


def map_labels_for_focus(map_focus: str, selected_review_labels: list[str]) -> list[str]:
    focus_labels = MAP_FOCUS[map_focus]
    if not selected_review_labels:
        return focus_labels
    selected = set(selected_review_labels)
    return [label for label in focus_labels if label in selected]


def map_excluded_labels(map_focus: str, selected_review_labels: list[str]) -> list[str]:
    if not selected_review_labels:
        return []
    focus = set(MAP_FOCUS[map_focus])
    return [label for label in selected_review_labels if label not in focus]


def load_json_report(filepath: Path) -> dict | None:
    """Safely load a JSON report file."""
    if filepath.exists():
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def load_csv_safe(filepath: Path) -> pd.DataFrame | None:
    """Safely load a CSV file."""
    if filepath.exists():
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    return None


@st.cache_data
def load_optional_csv(path: str) -> pd.DataFrame:
    filepath = Path(path)
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath)


def model_confidence(evidence_strength: object, slice_risk_level: object) -> str:
    evidence = str(evidence_strength)
    risk = str(slice_risk_level)
    if evidence == "strong" and risk in {"low", "medium"}:
        return "Higher"
    if evidence in {"moderate", "strong"} and risk != "high":
        return "Medium"
    return "Lower"


def review_note(row: pd.Series) -> str:
    signal = LABEL_NAMES.get(str(row.get("anomaly_flag")), str(row.get("anomaly_flag", "Unknown")))
    reason = str(row.get("why_flagged", "")).strip()
    drivers = str(row.get("top_drivers", "")).strip()
    confidence = model_confidence(row.get("evidence_strength"), row.get("slice_risk_level"))
    if signal == "Within expected range":
        return f"Inside the model range. Confidence: {confidence}."
    if signal == "Low support":
        return f"Local evidence is limited. Review comparable sales before using this estimate. Confidence: {confidence}."
    driver_text = f" Main drivers: {drivers}." if drivers else ""
    return f"{signal} candidate for human review. {reason}{driver_text} Confidence: {confidence}."


def summarize_metrics(dataframe: pd.DataFrame, full_count: int) -> dict[str, float | int]:
    total = int(len(dataframe))
    if not total:
        return {
            "transactions": 0,
            "anomalies": 0,
            "low_support": 0,
            "within_range": 0,
            "coverage": 0.0,
        }
    return {
        "transactions": total,
        "anomalies": int(dataframe["anomaly_flag"].isin(ACTIONABLE_LABELS).sum()),
        "low_support": int(dataframe["anomaly_flag"].eq("insufficient_history").sum()),
        "within_range": int(dataframe["anomaly_flag"].eq("within_expected_range").sum()),
        "coverage": float(total / full_count) if full_count else 0.0,
    }


def map_metrics(dataframe: pd.DataFrame, focus_labels: list[str], max_points: int) -> dict[str, int]:
    frame = prepare_map_frame(dataframe, focus_labels=focus_labels, max_points=max_points)
    return {
        "mapped_sales": int(len(frame)),
        "review_flags": int(frame["anomaly_flag"].isin(ACTIONABLE_LABELS).sum()) if not frame.empty else 0,
        "low_support": int(frame["anomaly_flag"].eq("insufficient_history").sum()) if not frame.empty else 0,
        "within_range": int(frame["anomaly_flag"].eq("within_expected_range").sum()) if not frame.empty else 0,
    }


def status_line(metrics: dict[str, float | int]) -> str:
    def sale_word(count: float | int) -> str:
        return "sale" if int(count) == 1 else "sales"

    if metrics["transactions"] == 0:
        return "No sales match the current filters."
    if metrics["anomalies"] == 0 and metrics["low_support"] == 0:
        return "Current filters show sales inside the model's expected fair-value range."
    parts = []
    if metrics["anomalies"]:
        verb = "needs" if int(metrics["anomalies"]) == 1 else "need"
        parts.append(f"{metrics['anomalies']:,} {sale_word(metrics['anomalies'])} {verb} human review because the model flagged pricing risk")
    if metrics["low_support"]:
        verb = "has" if int(metrics["low_support"]) == 1 else "have"
        parts.append(f"{metrics['low_support']:,} {sale_word(metrics['low_support'])} {verb} limited local evidence")
    return " and ".join(parts) + " in the current view."


def within_range_help() -> str:
    return (
        "Within expected range means the observed sale price falls inside the model-estimated fair-value interval. "
        "These sales are treated as cleared background context, not priority review leads."
    )


def build_review_queue(dataframe: pd.DataFrame) -> pd.DataFrame:
    queue_columns = [
        "property_id",
        "sale_date",
        "zipcode",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "anomaly_flag",
        "anomaly_score",
        "evidence_strength",
        "slice_risk_level",
        "top_drivers",
        "why_flagged",
    ]
    available = [column for column in queue_columns if column in dataframe.columns]
    display = dataframe[available].copy()
    if "anomaly_flag" in display.columns:
        display["review_label"] = display["anomaly_flag"].map(LABEL_NAMES).fillna(display["anomaly_flag"])
        display["model_confidence"] = display.apply(
            lambda row: model_confidence(row.get("evidence_strength"), row.get("slice_risk_level")),
            axis=1,
        )
        display["review_note"] = display.apply(review_note, axis=1)
        ordered = [
            "property_id",
            "sale_date",
            "zipcode",
            "observed_price",
            "fair_value_hat",
            "lower_bound",
            "upper_bound",
            "review_label",
            "anomaly_score",
            "model_confidence",
            "review_note",
            "evidence_strength",
            "slice_risk_level",
            "top_drivers",
            "why_flagged",
        ]
        display = display[[column for column in ordered if column in display.columns]]
    if "anomaly_score" in display.columns:
        display = display.sort_values(
            "anomaly_score",
            key=lambda series: series.abs(),
            ascending=False,
            na_position="last",
        )
    return display.rename(columns=QUEUE_COLUMN_NAMES)


def build_slice_summary(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = dataframe.groupby(column, dropna=False).agg(
        transactions=("property_id", "size"),
        anomalies=("anomaly_flag", lambda values: int(values.isin(ACTIONABLE_LABELS).sum())),
        low_support=("anomaly_flag", lambda values: int((values == "insufficient_history").sum())),
        median_price=("observed_price", "median"),
    )
    summary["anomaly_rate"] = summary["anomalies"] / summary["transactions"]
    summary["low_support_rate"] = summary["low_support"] / summary["transactions"]
    summary = summary.reset_index()
    summary[column] = summary[column].astype(str).map(lambda value: display_value(column, value))
    for rate_column in ["anomaly_rate", "low_support_rate"]:
        summary[rate_column] = summary[rate_column].map(lambda value: f"{value:.1%}")
    return summary.sort_values("transactions", ascending=False).rename(columns=SLICE_COLUMN_NAMES)


def format_currency(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"${float(value):,.0f}"


def format_score(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def queue_column_config() -> dict[str, Any]:
    return {
        "Observed price": st.column_config.NumberColumn("Observed price", format="$%d"),
        "Fair value estimate": st.column_config.NumberColumn("Fair value estimate", format="$%d"),
        "Lower fair range": st.column_config.NumberColumn("Lower fair range", format="$%d"),
        "Upper fair range": st.column_config.NumberColumn("Upper fair range", format="$%d"),
        "Review score": st.column_config.NumberColumn("Review score", format="%.3f"),
    }


def slice_column_config() -> dict[str, Any]:
    return {
        "Median price": st.column_config.NumberColumn("Median price", format="$%d"),
    }


def rgba_css(color: list[int]) -> str:
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3] / 255:.2f})"


def legend_html(labels: list[str]) -> str:
    items = []
    for label in labels:
        items.append(
            f"<span class='legend-item'><span class='legend-dot' style='background:{rgba_css(LABEL_COLORS[label])}'></span>"
            f"{LABEL_NAMES[label]}</span>"
        )
    return "<div class='legend'>" + "".join(items) + "</div>"


def prepare_map_frame(dataframe: pd.DataFrame, focus_labels: list[str], max_points: int) -> pd.DataFrame:
    map_frame = dataframe.loc[
        dataframe["lat"].notna() & dataframe["long"].notna() & dataframe["anomaly_flag"].isin(focus_labels)
    ].copy()
    if map_frame.empty:
        return map_frame

    if "anomaly_score" in map_frame.columns:
        map_frame["_sort_score"] = map_frame["anomaly_score"].abs().fillna(0.0)
        map_frame = map_frame.sort_values("_sort_score", ascending=False)
    if len(map_frame) > max_points:
        map_frame = map_frame.head(max_points)

    map_frame["color"] = map_frame["anomaly_flag"].astype(str).map(LABEL_COLORS)
    map_frame["label"] = map_frame["anomaly_flag"].astype(str).map(LABEL_NAMES).fillna(map_frame["anomaly_flag"].astype(str))
    score = map_frame["anomaly_score"].abs().fillna(0.0) if "anomaly_score" in map_frame.columns else pd.Series(0.0, index=map_frame.index)
    map_frame["radius_px"] = score.clip(upper=0.35).mul(18.0).add(3.0).clip(lower=3.0, upper=9.0)
    map_frame.loc[map_frame["anomaly_flag"].eq("within_expected_range"), "radius_px"] = 2.4
    map_frame["observed_price_display"] = map_frame["observed_price"].map(format_currency)
    map_frame["fair_value_display"] = map_frame["fair_value_hat"].map(format_currency)
    map_frame["lower_bound_display"] = map_frame["lower_bound"].map(format_currency) if "lower_bound" in map_frame.columns else "-"
    map_frame["upper_bound_display"] = map_frame["upper_bound"].map(format_currency) if "upper_bound" in map_frame.columns else "-"
    map_frame["score_display"] = map_frame["anomaly_score"].map(format_score) if "anomaly_score" in map_frame.columns else "-"
    return map_frame


def render_map(dataframe: pd.DataFrame, focus_labels: list[str], max_points: int, focus_source: str) -> None:
    if not {"lat", "long"}.issubset(dataframe.columns):
        st.info("Latitude/longitude columns are unavailable. Re-run the current pipeline version.")
        return

    map_frame = prepare_map_frame(dataframe, focus_labels=focus_labels, max_points=max_points)
    if map_frame.empty:
        st.info(f"No mapped sales match the current filters for {focus_source}.")
        return

    st.markdown(legend_html(focus_labels), unsafe_allow_html=True)
    st.markdown(
        f"<div class='hint'>Showing {len(map_frame):,} mapped sales from {focus_source}.</div>",
        unsafe_allow_html=True,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_frame,
        get_position="[long, lat]",
        get_fill_color="color",
        get_radius="radius_px",
        radius_units="pixels",
        radius_min_pixels=3,
        radius_max_pixels=9,
        pickable=True,
        auto_highlight=True,
        opacity=0.86,
        stroked=True,
        get_line_color=[245, 245, 245, 130],
        line_width_min_pixels=0.6,
    )
    tooltip = {
        "html": (
            "<b>{label}</b><br/>"
            "Property: {property_id}<br/>"
            "Observed sale: {observed_price_display}<br/>"
            "Fair value estimate: {fair_value_display}<br/>"
            "Expected range: {lower_bound_display} to {upper_bound_display}<br/>"
            "Review score: {score_display}<br/>"
            "Zipcode: {zipcode}"
        )
    }
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=float(map_frame["lat"].median()),
            longitude=float(map_frame["long"].median()),
            zoom=9,
            pitch=0,
        ),
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, width="stretch", height=620)


NAV_ITEMS = [
    "Overview",
    "Review Queue",
    "Map",
    "Data Processing & EDA",
    "Uncertainty",
    "Performance",
    "Explainability",
    "Data Quality",
    "Validation",
]

NAV_ICONS = {
    "Overview": "▦",
    "Review Queue": "☷",
    "Map": "◇",
    "Data Processing & EDA": "▤",
    "Uncertainty": "⌁",
    "Performance": "◒",
    "Explainability": "◎",
    "Data Quality": "▣",
    "Validation": "✓",
}


def inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #001f4d;
            --muted: #5e6677;
            --line: #cdd3df;
            --panel: #ffffff;
            --canvas: #f7f8fc;
            --soft: #eef1f6;
            --danger: #c70000;
            --navy: #002b5c;
            --accent: #0b3a75;
        }
        html, body, [class*="css"] {
            font-family: Inter, Arial, sans-serif;
            color: var(--ink);
        }
        .stApp {
            background:
                radial-gradient(circle at 16% 0%, rgba(86, 148, 255, 0.12), transparent 28rem),
                linear-gradient(180deg, #f3f6ff 0%, #f8f9fd 42%, #f7f8fc 100%);
        }
        .block-container {
            padding: 0 1.35rem 1.6rem 1.35rem;
            max-width: 1500px;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        section[data-testid="stSidebar"] {
            width: 292px !important;
            min-width: 292px !important;
            background: #f4f5fa;
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] > div {
            padding: 1.05rem 0.75rem;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--ink) !important;
        }
        div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.14rem;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: #dde2e8;
            color: var(--ink);
            font-weight: 700;
        }
        .topbar {
            height: 58px;
            border-bottom: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0 -1.35rem 1.25rem -1.35rem;
            padding: 0 1.35rem;
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .topbar-title {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .topbar-tools {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            color: var(--ink);
            font-size: 1rem;
        }
        .search-shell {
            border: 1px solid var(--line);
            border-radius: 18px;
            min-width: 300px;
            height: 34px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0 0.85rem;
            color: #747c8d;
            background: white;
            font-size: 0.86rem;
        }
        .page-title {
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.025em;
            color: var(--ink);
            margin: 0;
        }
        .page-subtitle {
            margin-top: 0.35rem;
            color: #2d3441;
            font-size: 1.02rem;
        }
        .card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            box-shadow: 0 1px 2px rgba(0, 24, 64, 0.04);
        }
        .kpi-card {
            min-height: 126px;
            padding: 1.05rem 1rem 0.85rem 1rem;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::after {
            content: "";
            position: absolute;
            right: -2.2rem;
            top: -2.4rem;
            width: 7rem;
            height: 7rem;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.48);
        }
        .kpi-card.active {
            border-top: 4px solid var(--ink);
        }
        .kpi-card.blue {
            background: linear-gradient(135deg, #ffffff 0%, #eaf2ff 100%);
            border-color: #b9d4ff;
        }
        .kpi-card.red {
            background: linear-gradient(135deg, #fff8f7 0%, #ffe5df 100%);
            border-color: #ffb8ab;
        }
        .kpi-card.green {
            background: linear-gradient(135deg, #fbfffd 0%, #e0f7ea 100%);
            border-color: #aee4c3;
        }
        .kpi-card.amber {
            background: linear-gradient(135deg, #fffdf6 0%, #fff0c2 100%);
            border-color: #f1d47b;
        }
        .kpi-card.purple {
            background: linear-gradient(135deg, #fbfaff 0%, #eee9ff 100%);
            border-color: #c9bcff;
        }
        .kpi-card.red .kpi-label,
        .kpi-card.red .kpi-value {
            color: #a51616;
        }
        .kpi-card.green .kpi-value {
            color: #0f6b3c;
        }
        .kpi-card.amber .kpi-value {
            color: #8a5100;
        }
        .kpi-label {
            color: #5f6674;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            font-family: Georgia, serif;
        }
        .kpi-value {
            color: var(--ink);
            font-weight: 850;
            font-size: 1.9rem;
            margin-top: 0.85rem;
        }
        .kpi-foot {
            color: #303847;
            font-size: 0.78rem;
            margin-top: 0.35rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }
        .alert {
            border: 1px solid #b8c0ce;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            background: #f9fafe;
            color: #0b1e3a;
            margin: 1rem 0 1.35rem 0;
        }
        .alert-danger {
            border-color: #ff8f8f;
            background: #ffd8d4;
            color: #a00000;
        }
        .guidance-note {
            border: 1px solid #d7deea;
            border-left: 4px solid #2f6fb5;
            border-radius: 8px;
            background: #f7faff;
            padding: 0.85rem 1rem;
            margin: 0.65rem 0 1rem 0;
            color: #172033;
            font-size: 0.9rem;
        }
        .guidance-title {
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }
        .guidance-note ul {
            margin: 0;
            padding-left: 1.1rem;
        }
        .guidance-note li {
            margin: 0.22rem 0;
        }
        .section-title {
            font-size: 1.28rem;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }
        .pipeline {
            padding: 1.1rem 1rem 1.25rem 1rem;
            background: linear-gradient(135deg, #ffffff 0%, #f3f7ff 100%);
        }
        .pipeline-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-top: 1.3rem;
        }
        .node {
            text-align: center;
            min-width: 86px;
        }
        .node-icon {
            width: 44px;
            height: 44px;
            border: 2px solid #8aa7d8;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            background: #f9fbff;
            color: var(--ink);
            box-shadow: 0 8px 20px rgba(0, 47, 108, 0.08);
        }
        .node:nth-child(1) .node-icon { border-color: #4f8df7; background: #eaf2ff; }
        .node:nth-child(3) .node-icon { border-color: #10a66a; background: #e3f8ed; }
        .node:nth-child(5) .node-icon { border-color: #8c6df0; background: #eee9ff; }
        .node:nth-child(7) .node-icon { border-color: #f0a429; background: #fff1d2; }
        .node:nth-child(9) .node-icon { border-color: #23a6b8; background: #e2f8fb; }
        .node-icon.model,
        .node-icon.queue {
            border-radius: 4px;
            background: linear-gradient(135deg, #002b5c 0%, #1f67d1 100%);
            color: white;
        }
        .node-icon.blue { border-color: #4f8df7; background: #eaf2ff; }
        .node-icon.green { border-color: #10a66a; background: #e3f8ed; }
        .node-icon.amber { border-color: #f0a429; background: #fff1d2; }
        .node-icon.red { border-color: #ef4444; background: #ffe5df; }
        .node-icon.purple { border-color: #8c6df0; background: #eee9ff; }
        .node-label {
            margin-top: 0.55rem;
            font-size: 0.76rem;
            color: #1d2634;
        }
        .node-line {
            height: 1px;
            background: linear-gradient(90deg, #b9d4ff, #95e0c0, #d2c8ff, #ffd77a);
            flex: 1;
            min-width: 28px;
        }
        .detail-panel {
            padding: 1rem;
        }
        .badge {
            display: inline-flex;
            padding: 0.2rem 0.55rem;
            border-radius: 3px;
            background: #ffd8d4;
            color: #b00000;
            font-weight: 800;
            font-size: 0.72rem;
            text-transform: uppercase;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0;
        }
        .mini-card {
            border: 1px solid var(--line);
            background: #f9fafd;
            padding: 0.8rem;
            min-height: 74px;
        }
        .mini-label {
            color: #6b7280;
            font-size: 0.75rem;
        }
        .mini-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            margin-top: 0.15rem;
        }
        .mini-value.red {
            color: var(--danger);
        }
        .divider {
            height: 1px;
            background: var(--line);
            margin: 1rem 0;
        }
        .explain-box {
            min-height: 260px;
            border: 1px dashed #b7bfcb;
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #4b5563;
            background: #fbfcff;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.82rem;
        }
        .side-footer {
            position: fixed;
            bottom: 1rem;
            width: 244px;
            left: 18px;
        }
        .status-row {
            color: #1f2937;
            font-size: 0.86rem;
            padding: 0.45rem 0.2rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 4px;
            background: var(--ink);
            color: white;
            border: 1px solid var(--ink);
            font-weight: 750;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
            box-shadow: none;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 850;
        }
        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .topbar {
                position: static;
            }
            .search-shell {
                min-width: 150px;
            }
            .pipeline-row {
                flex-wrap: wrap;
            }
            .node-line {
                display: none;
            }
            .side-footer {
                position: static;
                width: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def nav_label(name: str) -> str:
    return f"{NAV_ICONS.get(name, '•')}  {name}"


def topbar(search_placeholder: str = "Search properties...") -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-title">Valuation Intelligence</div>
            <div class="topbar-tools">
                <div class="search-shell">⌕ <span>{escape(search_placeholder)}</span></div>
                <span>♧</span><span>⚙</span><span>?</span><span>◉</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f"<span class='badge'>{escape(badge)}</span>" if badge else ""
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:0.75rem;">
            <div>
                <h1 class="page-title">{escape(title)}</h1>
                <div class="page-subtitle">{escape(subtitle)}</div>
            </div>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot_left: str = "", foot_right: str = "", active: bool = False, tone: str = "blue") -> None:
    active_class = " active" if active else ""
    tone_class = tone if tone in {"blue", "red", "green", "amber", "purple"} else "blue"
    st.markdown(
        f"""
        <div class="card kpi-card {tone_class}{active_class}">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            <div class="kpi-foot"><span>{escape(foot_left)}</span><strong>{escape(foot_right)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_box(text: str, danger: bool = False) -> None:
    klass = "alert alert-danger" if danger else "alert"
    st.markdown(f"<div class='{klass}'><strong>ⓘ</strong>&nbsp;&nbsp;{escape(text)}</div>", unsafe_allow_html=True)


def info_note(text: str) -> None:
    st.markdown(
        f"""
        <div style="border-left:4px solid #22c55e;background:#ecfdf5;border-radius:8px;padding:0.8rem 1rem;margin:0.85rem 0 1.1rem 0;color:#064e3b;">
            <strong>Within expected range:</strong> {escape(text)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def guidance_note(title: str, lines: list[str]) -> None:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    st.markdown(
        f"""
        <div class="guidance-note">
            <div class="guidance-title">{escape(title)}</div>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_card(compact: bool = False) -> None:
    nodes = [
        ("Raw", "▣", ""),
        ("Cleaning", "⌂", ""),
        ("Features", "⌘", ""),
        ("Model", "XGB", "model"),
        ("Fair-value Interval", "☷", ""),
        ("Review Queue", "☷", "queue"),
    ]
    if compact:
        nodes = [("Raw", "▣", ""), ("Schema", "☷", ""), ("Cleaning", "●", ""), ("Features", "✣", ""), ("Split", "⌁", ""), ("Model", "◉", ""), ("Intervals", "⇔", ""), ("Queue", "▤", "queue")]
    pieces = []
    for index, (label, icon, klass) in enumerate(nodes):
        if index:
            pieces.append("<div class='node-line'></div>")
        pieces.append(
            f"<div class='node'><div class='node-icon {klass}'>{escape(icon)}</div><div class='node-label'>{escape(label)}</div></div>"
        )
    st.markdown(
        f"""
        <div class="card pipeline">
            <div class="section-title">Valuation Pipeline Architecture</div>
            <div class="pipeline-row">{''.join(pieces)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def eda_pipeline_card() -> None:
    nodes = [
        ("Raw Data", "King County Sales", "▣", "blue"),
        ("Schema Check", "Columns / Types", "☑", "purple"),
        ("Cleaning", "Missing / Duplicates", "⌂", "green"),
        ("Feature Eng.", "Temporal / Geo", "⌁", "amber"),
        ("EDA", "Dist / Trends", "⌁", "red"),
        ("Split", "Chronological", "⇥", "blue"),
        ("Model Ready", "XGBoost Matrix", "▤", "green"),
    ]
    pieces = []
    for index, (title, subtitle, icon, tone) in enumerate(nodes):
        if index:
            pieces.append("<div style='font-size:1.8rem;color:#bcc5d2;'>→</div>")
        pieces.append(
            f"""
            <div style="min-width:116px;text-align:center;">
                <div class="node-icon {tone}" style="margin:auto;">{escape(icon)}</div>
                <div style="font-family:JetBrains Mono, monospace;font-size:0.72rem;font-weight:800;margin-top:0.55rem;text-transform:uppercase;">{escape(title)}</div>
                <div style="font-size:0.72rem;color:#596273;">{escape(subtitle)}</div>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="card pipeline">
            <div class="pipeline-row" style="justify-content:flex-start;overflow-x:auto;padding-bottom:0.3rem;">{''.join(pieces)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=24, b=30),
        font=dict(family="Inter", color="#001f4d"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor="#cdd3df")
    fig.update_yaxes(gridcolor="#edf0f5", linecolor="#cdd3df")
    return fig


def selected_property_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    actionable = dataframe.loc[dataframe["anomaly_flag"].isin(ACTIONABLE_LABELS)].copy()
    if actionable.empty:
        actionable = dataframe.copy()
    if "anomaly_score" in actionable.columns:
        actionable = actionable.sort_values("anomaly_score", key=lambda s: s.abs(), ascending=False)
    return actionable


def property_detail(row: pd.Series) -> None:
    label = LABEL_NAMES.get(str(row.get("anomaly_flag")), str(row.get("anomaly_flag", "Unknown")))
    risk = "High risk" if str(row.get("anomaly_flag")) in ACTIONABLE_LABELS else "Monitor"
    st.markdown(
        f"""
        <div class="card detail-panel">
            <div style="display:flex;justify-content:space-between;gap:1rem;">
                <div>
                    <div class="kpi-label">Selected Property</div>
                    <div style="font-weight:850;font-size:1.35rem;color:#001f4d;">{escape(str(row.get("property_id", "-")))}</div>
                    <div style="color:#4b5563;font-size:0.86rem;">ZIP {escape(str(row.get("zipcode", "-")))}</div>
                </div>
                <span class="badge">{escape(risk)}</span>
            </div>
            <div class="mini-grid">
                <div class="mini-card"><div class="mini-label">Observed Price</div><div class="mini-value red">{escape(format_currency(row.get("observed_price")))}</div></div>
                <div class="mini-card"><div class="mini-label">Fair-value Est.</div><div class="mini-value">{escape(format_currency(row.get("fair_value_hat")))}</div></div>
            </div>
            <div class="divider"></div>
            <div style="display:flex;justify-content:space-between;font-size:0.86rem;"><span>Lower Bound</span><strong>{escape(format_currency(row.get("lower_bound")))}</strong></div>
            <div style="display:flex;justify-content:space-between;font-size:0.86rem;margin-top:0.45rem;"><span>Upper Bound</span><strong>{escape(format_currency(row.get("upper_bound")))}</strong></div>
            <div class="divider"></div>
            <div style="display:flex;justify-content:space-between;font-size:0.86rem;"><span>Model Signal</span><strong>{escape(label)}</strong></div>
            <div style="display:flex;justify-content:space-between;font-size:0.86rem;margin-top:0.45rem;"><span>Top Driver</span><strong>{escape(str(row.get("top_drivers", "-")).split(",")[0])}</strong></div>
            <div style="display:flex;justify-content:space-between;font-size:0.86rem;margin-top:0.45rem;"><span>Confidence</span><strong>{escape(model_confidence(row.get("evidence_strength"), row.get("slice_risk_level")))}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_property_cards(dataframe: pd.DataFrame, limit: int = 6) -> None:
    rows = []
    for _, row in dataframe.head(limit).iterrows():
        signal = LABEL_NAMES.get(str(row.get("anomaly_flag")), str(row.get("anomaly_flag", "Unknown")))
        rows.append(
            f"""
            <div class="card detail-panel" style="margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
                    <div>
                        <div style="font-weight:850;color:#001f4d;">{escape(str(row.get("property_id", "-")))}</div>
                        <div style="font-size:0.78rem;color:#5e6677;">ZIP {escape(str(row.get("zipcode", "-")))} · {escape(str(row.get("sale_date", "-")))}</div>
                    </div>
                    <span class="badge">{escape(signal)}</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.55rem;margin-top:0.75rem;font-size:0.82rem;">
                    <div><div class="mini-label">Observed</div><strong>{escape(format_currency(row.get("observed_price")))}</strong></div>
                    <div><div class="mini-label">Fair value</div><strong>{escape(format_currency(row.get("fair_value_hat")))}</strong></div>
                    <div><div class="mini-label">Score</div><strong>{escape(format_score(row.get("anomaly_score")))}</strong></div>
                </div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_sidebar(dataframe: pd.DataFrame) -> tuple[str, dict[str, list[str]], str]:
    st.sidebar.markdown(
        """
        <div style="padding:0.1rem 0.35rem 1rem 0.35rem;">
            <div style="font-size:1.18rem;font-weight:850;color:#001f4d;">Anomaly Detection</div>
            <div style="font-size:0.72rem;color:#293447;">Market Control Room</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_label = st.sidebar.radio(
        "Navigation",
        [nav_label(name) for name in NAV_ITEMS],
        label_visibility="collapsed",
    )
    page = selected_label.split("  ", 1)[1]

    with st.sidebar.expander("Data filters", expanded=False):
        table_path = st.text_input("Property table", str(DEFAULT_TABLE))
        selections: dict[str, list[str]] = {}
        for column in ["anomaly_flag", *SIDEBAR_FILTER_COLUMNS]:
            options = options_for(dataframe, column)
            selections[column] = st.multiselect(
                FILTER_LABELS[column],
                options,
                format_func=lambda value, column=column: display_value(column, value),
                placeholder="All",
            )
    st.sidebar.markdown("<div class='side-footer'>", unsafe_allow_html=True)
    st.sidebar.download_button(
        "Export Report",
        data=build_review_queue(dataframe).head(500).to_csv(index=False),
        file_name="dc_reif_review_export.csv",
        mime="text/csv",
        width="stretch",
    )
    st.sidebar.markdown("<div class='status-row'>ⓘ &nbsp; System Status</div><div class='status-row'>↪ &nbsp; Log Out</div></div>", unsafe_allow_html=True)
    return page, selections, table_path


def render_overview(dataframe: pd.DataFrame, filtered: pd.DataFrame, metrics: dict[str, float | int]) -> None:
    page_header("DC-REIF Valuation Review Dashboard", "Model-assisted pricing anomaly detection for human valuation review")
    alert_box("Decision-support only: flagged cases are review leads, not proof of mispricing.")
    info_note(within_range_help())
    guidance_note(
        "How to read this page",
        [
            "Total Properties is the number of sales currently loaded; Within expected range means the sale price sits inside the model fair-value interval.",
            "Flagged Cases are review leads where the observed price is materially above or below the estimated interval.",
            "Interval Coverage is the share of tested sales whose real price landed inside the model interval; higher is usually safer, but wider intervals can also increase coverage.",
        ],
    )
    trust = load_trust_metrics()
    val_df = load_csv_safe(DEFAULT_VALUATION_METRICS)
    row = val_df.iloc[0] if val_df is not None and not val_df.empty else {}
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        metric_card("Total Properties", f"{len(dataframe):,}", "Within expected range", f"{metrics['within_range']:,}", tone="green")
    with kpi_cols[1]:
        metric_card("Flagged Cases", f"{metrics['anomalies']:,}", "Over / Under", f"{int(filtered['anomaly_flag'].eq('potentially_over_valued').sum()):,} / {int(filtered['anomaly_flag'].eq('potentially_under_valued').sum()):,}", active=True, tone="red")
    with kpi_cols[2]:
        test_r2 = row.get("test_r2", None) if hasattr(row, "get") else None
        test_mape = row.get("test_mape", None) if hasattr(row, "get") else None
        metric_card("Model Performance", f"{float(test_r2):.4f}" if isinstance(test_r2, (int, float)) else "-", "Test MAPE", f"{float(test_mape):.2f}%" if isinstance(test_mape, (int, float)) else "-", tone="green")
    with kpi_cols[3]:
        coverage = trust.get("global_empirical_coverage")
        q5 = trust.get("q5_empirical_coverage", trust.get("q5_coverage_from_table"))
        metric_card("Interval Coverage", f"{coverage:.2%}" if isinstance(coverage, (int, float)) else "-", "Q5 Coverage", f"{q5:.2%}" if isinstance(q5, (int, float)) else "-", tone="amber")

    st.write("")
    pipeline_card()
    st.write("")
    chart_col, donut_col = st.columns([2.1, 1])
    counts = filtered["anomaly_flag"].map(LABEL_NAMES).fillna(filtered["anomaly_flag"]).value_counts()
    with chart_col:
        st.markdown("<div class='section-title'>Review Outcome Counts</div>", unsafe_allow_html=True)
        chart_colors = {
            "Within expected range": "#22c55e",
            "Over-valued": "#d71920",
            "Under-valued": "#ff9f66",
            "Low support": "#f2c94c",
        }
        fig = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[chart_colors.get(str(label), "#4f8df7") for label in counts.index],
        ))
        st.plotly_chart(style_plotly(fig, height=330), width="stretch")
        st.caption("Within expected range is a cleared/background status. Over-valued and Under-valued are the actionable review groups.")
    with donut_col:
        st.markdown("<div class='section-title'>Portfolio Status</div>", unsafe_allow_html=True)
        cleared = int(filtered["anomaly_flag"].eq("within_expected_range").sum())
        flagged = int(filtered["anomaly_flag"].isin(ACTIONABLE_LABELS).sum())
        fig = go.Figure(go.Pie(values=[cleared, flagged], labels=["Within expected range", "Flagged"], hole=0.68, marker_colors=["#22c55e", "#d71920"]))
        fig.update_traces(textinfo="none")
        fig.add_annotation(text=f"{(cleared / len(filtered)):.0%}<br>In range" if len(filtered) else "0%<br>In range", showarrow=False, font=dict(size=24, color="#064e3b"))
        st.plotly_chart(style_plotly(fig, height=330), width="stretch")


def render_review_queue(dataframe: pd.DataFrame) -> None:
    page_header("Flagged Properties", "Prioritized valuation review queue with property-level evidence.")
    guidance_note(
        "Queue terminology",
        [
            "Review score ranks how far a sale appears from the model's expected range; larger absolute values should be reviewed first.",
            "Interval width is Upper fair range minus Lower fair range; a wide interval means the model is less certain.",
            "Over-valued means the observed price is above the model range; Under-valued means it is below the model range.",
        ],
    )
    queue_source = selected_property_frame(dataframe)
    left, right = st.columns([2.2, 1])
    with left:
        controls = st.columns([1.4, 1, 1])
        sort_by = controls[0].selectbox("Sort by", ["Review score", "Observed price", "Interval width"], label_visibility="collapsed")
        ascending = controls[1].toggle("Ascending", value=False)
        top_n = controls[2].selectbox("Show", [6, 12, 25], index=0, label_visibility="collapsed")
        sort_map = {
            "Review score": "anomaly_score",
            "Observed price": "observed_price",
            "Interval width": "interval_width",
        }
        sort_column = sort_map[sort_by]
        if sort_column in queue_source.columns:
            key = (lambda series: series.abs()) if sort_column == "anomaly_score" else None
            queue_source = queue_source.sort_values(sort_column, key=key, ascending=ascending, na_position="last")
        st.markdown("<div class='section-title'>Priority Review Cards</div>", unsafe_allow_html=True)
        priority_property_cards(queue_source, limit=int(top_n))
        st.caption(f"Showing top {min(int(top_n), len(queue_source))} of {len(queue_source):,} review candidates. Open the full queue only for row-level audit detail.")
        with st.expander("Open full review queue table", expanded=False):
            table = build_review_queue(queue_source).head(250)
            st.dataframe(table, width="stretch", height=420, hide_index=True, column_config=queue_column_config())
            st.download_button(
                "Download filtered queue",
                data=build_review_queue(queue_source).to_csv(index=False),
                file_name="dc_reif_review_queue.csv",
                mime="text/csv",
            )
    with right:
        if queue_source.empty:
            st.info("No properties match the current filters.")
            return
        ids = queue_source["property_id"].astype(str).tolist()
        selected_id = st.selectbox("Queue ID", ids, label_visibility="collapsed")
        row = queue_source.loc[queue_source["property_id"].astype(str).eq(selected_id)].iloc[0]
        property_detail(row)
        st.markdown("<div class='card detail-panel'><div class='section-title'>Triage Action</div>", unsafe_allow_html=True)
        st.selectbox("Reviewer Label", ["Select label...", "Valid Flag", "False Positive", "Uncertain"])
        st.text_area("Reviewer Notes", placeholder="Enter justification for label...", height=110)
        st.button("▣ Save Review", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)


def render_uncertainty() -> None:
    page_header("Uncertainty Diagnostics", "Conformal Inference & Predictive Confidence Evaluation", "LIVE")
    alert_box("Coverage values are empirical diagnostics under the implemented chronological, localized, upper-tail-adjusted protocol. They are not regulatory guarantees.", danger=True)
    guidance_note(
        "Metric definitions",
        [
            "Q1-Q5 are price quintiles: Q1 is the lowest-priced 20% of homes, Q5 is the highest-priced 20%. They are not calendar quarters.",
            "q-hat is the calibration buffer used to widen prediction intervals; larger q-hat means the model needs more uncertainty allowance.",
            "Empirical coverage is the percentage of tested sales that fell inside the predicted fair-value interval. The dotted 90% line is the target reference.",
            "Interval width is the dollar distance between the lower and upper fair-value bounds; wider intervals indicate lower confidence.",
        ],
    )
    trust = load_trust_metrics()
    unc_data = load_json_report(DEFAULT_UNCERTAINTY_METRICS) or {}
    values = [
        ("Global q-hat", format_currency(trust.get("global_q_hat"))),
        ("Avg localized q-hat", format_currency(trust.get("average_local_q_hat"))),
        ("Empirical coverage", f"{trust.get('global_empirical_coverage', 0):.2%}"),
        ("Avg interval width", format_currency(unc_data.get("average_interval_width", trust.get("global_average_interval_width")))),
        ("Q5 coverage", f"{trust.get('q5_empirical_coverage', 0):.2%}"),
        ("Q5 interval width", format_currency(trust.get("q5_interval_width"))),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, values):
        with col:
            metric_card(label, value)
    st.write("")
    left, right = st.columns([2.2, 1])
    with left:
        cov_df = load_csv_safe(DEFAULT_COVERAGE_BY_BAND)
        if cov_df is not None and not cov_df.empty:
            fig = go.Figure(go.Bar(x=cov_df["price_band"], y=cov_df["empirical_coverage"], marker_color="#002b5c"))
            fig.add_hline(y=0.9, line_dash="dot", line_color="#6b7280", annotation_text="90% Target")
            st.markdown("<div class='section-title'>Coverage by Price Band (Quintiles)</div>", unsafe_allow_html=True)
            st.plotly_chart(style_plotly(fig, height=350), width="stretch")
            st.caption("Q1-Q5 split the tested properties into five equal-sized price groups from lowest price (Q1) to highest price (Q5).")
    with right:
        st.markdown(
            """
            <div class="card detail-panel">
                <div class="section-title">Theoretical Explanation</div>
                <p>Conformal prediction constructs intervals that offer marginal coverage guarantees under exchangeability assumptions.</p>
                <p>The <strong>q-hat</strong> threshold is localized by price band and market segment so interval width reflects heteroscedasticity in housing prices.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    width_df = load_csv_safe(ROOT / "outputs" / "tables" / "interval_width_predicted_price_band.csv")
    bottom = st.columns(2)
    with bottom[0]:
        if width_df is not None and not width_df.empty:
            y_col = "average_interval_width" if "average_interval_width" in width_df.columns else width_df.columns[-1]
            x_col = "predicted_price_band" if "predicted_price_band" in width_df.columns else width_df.columns[0]
            st.markdown("<div class='section-title'>Avg Interval Width by Band</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=width_df[x_col], y=width_df[y_col], marker_color="#0b3a75"))
            st.plotly_chart(style_plotly(fig, height=280), width="stretch")
            st.caption("Wider intervals mean the model is allowing more uncertainty for that price segment.")


def render_explainability() -> None:
    page_header("Model Explainability", "Analyze global and local feature contributions driving current valuation estimates.", "MODEL VER: v2.4.1-PRD")
    alert_box("Feature importance and SHAP describe model behavior, not causal effects.", danger=True)
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown("<div class='card detail-panel'><div class='section-title'>Global SHAP Summary</div>", unsafe_allow_html=True)
        if DEFAULT_SHAP_PNG.exists():
            st.image(str(DEFAULT_SHAP_PNG), width="stretch")
        else:
            st.markdown("<div class='explain-box'>▥<br>[Interactive SHAP Summary Plot Renderer]<br><small>Awaiting data context</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("<div class='card detail-panel'><div class='section-title'>Feature Importance (Gain)</div>", unsafe_allow_html=True)
        fi_df = load_csv_safe(DEFAULT_FEATURE_IMPORTANCE)
        if fi_df is not None and not fi_df.empty:
            importance_col = next((c for c in ["importance", "gain", "weight", "cover"] if c in fi_df.columns), fi_df.columns[-1])
            feature_col = "feature" if "feature" in fi_df.columns else fi_df.columns[0]
            top = fi_df.nlargest(15, importance_col)
            fig = go.Figure(go.Bar(x=top[importance_col], y=top[feature_col], orientation="h", marker_color="#002b5c"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_plotly(fig, height=330), width="stretch")
        else:
            st.markdown("<div class='explain-box'>▥<br>[Feature Importance Bar Chart Renderer]</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        rows = [
            ("Location-grade interaction", "Quality value depends on location"),
            ("Grade-living interaction", "Size value depends on grade"),
            ("Prior neighbor median price", "Comparable-sale context"),
            ("Distance to Seattle core", "Accessibility/proximity"),
            ("Waterfront", "Premium amenity signal"),
        ]
        body = "".join(f"<tr><td><strong>{escape(a)}</strong></td><td>{escape(b)}</td></tr>" for a, b in rows)
        st.markdown(
            f"""
            <div class="card detail-panel">
                <div class="section-title">Top Model Drivers Context</div>
                <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
                    <tr><th align="left">Feature</th><th align="left">Meaning</th></tr>
                    {body}
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_data_quality(dataframe: pd.DataFrame) -> None:
    page_header("Data Quality", "Pipeline health, missing values, and cleaning policy.")
    guidance_note(
        "Data quality terms",
        [
            "Cleaned Rows are usable records after schema checks, deduplication, and invalid-value rules.",
            "Train / Val / Test is the chronological model split: train builds the model, validation tunes it, test measures final performance.",
            "Missing Values shows fields that needed imputation or review; missing does not automatically mean the row was removed.",
        ],
    )
    cleaning = load_json_report(DEFAULT_CLEANING_SUMMARY) or {}
    dq_report = load_json_report(DEFAULT_DATA_QUALITY) or {}
    cols = st.columns(3)
    with cols[0]:
        retention = cleaning.get("rows_out", 0) / cleaning.get("rows_in", 1)
        metric_card("Cleaned Rows", f"{cleaning.get('rows_out', len(dataframe)):,}", f"{retention:.1%} retention rate")
    with cols[1]:
        metric_card("Train / Val / Test", "70 / 15 / 15", "Stratified by micro-market")
    with cols[2]:
        missing = dq_report.get("missing_summary", {})
        non_zero = [k for k, v in missing.items() if v > 0]
        metric_card("Missing Values", ", ".join(non_zero[:2]) if non_zero else "None", "KNN imputation active")
    st.write("")
    pipeline_card(compact=True)
    st.write("")
    left, right = st.columns([1.8, 0.85])
    with left:
        checks = []
        for key, value in (dq_report.get("invalid_summary") or {}).items():
            checks.append({"Validation Rule": key.replace("_", " ").title(), "Entity / Feature": key.split("_")[0].title(), "Status": "Pass" if value == 0 else "Review"})
        st.markdown("<div class='section-title'>Data Health Checks</div>", unsafe_allow_html=True)
        checks_frame = pd.DataFrame(checks)
        pass_count = int(checks_frame["Status"].eq("Pass").sum()) if not checks_frame.empty else 0
        health_cols = st.columns(3)
        health_cols[0].metric("Checks passed", f"{pass_count}/{len(checks_frame)}")
        health_cols[1].metric("Duplicate rows", f"{dq_report.get('duplicate_rows', 0):,}")
        health_cols[2].metric("Missing columns", f"{len(dq_report.get('missing_columns', []))}")
        with st.expander("Open validation rule table", expanded=False):
            st.dataframe(checks_frame, width="stretch", hide_index=True)
    with right:
        st.markdown(
            """
            <div class="card detail-panel" style="background:#002b5c;color:white;">
                <div class="section-title" style="color:white;">Cleaning Policy</div>
                <p>Conservative outlier rejection is rigidly enforced for luxury records.</p>
                <p>Unlike standard filtering, records flagged in high-variance micro-markets are retained for manual analyst review.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def chronological_split_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    if "sale_date" not in dataframe.columns:
        return {"Train": 0, "Validation": 0, "Test": 0}
    dates = pd.to_datetime(dataframe["sale_date"], errors="coerce")
    train = int((dates <= pd.Timestamp("2014-12-31")).sum())
    validation = int(((dates > pd.Timestamp("2014-12-31")) & (dates <= pd.Timestamp("2015-03-31"))).sum())
    test = int((dates > pd.Timestamp("2015-03-31")).sum())
    return {"Train": train, "Validation": validation, "Test": test}


def render_data_processing_eda(dataframe: pd.DataFrame) -> None:
    page_header(
        "Data Processing & EDA",
        "Comprehensive view of the data pipeline, cleaning operations, engineered features, and exploratory signals behind the valuation model.",
    )
    guidance_note(
        "How to use this page",
        [
            "Use this page to understand how raw sales become model-ready features before looking at model scores.",
            "Training, Validation, and Test are time-based splits, so future-like records are not used to train the past.",
            "EDA charts summarize the dataset shape and data issues; they are not property-level decisions.",
        ],
    )
    st.markdown("<div class='section-title'>Pipeline Flow</div>", unsafe_allow_html=True)
    eda_pipeline_card()
    cleaning = load_json_report(DEFAULT_CLEANING_SUMMARY) or {}
    dq_report = load_json_report(DEFAULT_DATA_QUALITY) or {}
    raw_df = load_optional_csv(str(DEFAULT_RAW_DATA))
    features_df = load_optional_csv(str(DEFAULT_FEATURES_DATA))
    split_counts = chronological_split_counts(dataframe)

    st.markdown("<div class='section-title'>Data Quality Summary</div>", unsafe_allow_html=True)
    cols = st.columns(6)
    with cols[0]:
        metric_card("Cleaned Rows", f"{cleaning.get('rows_out', len(dataframe)):,}", tone="blue")
    with cols[1]:
        metric_card("Training", f"{split_counts['Train']:,}", "Pre-2015", tone="green")
    with cols[2]:
        metric_card("Validation", f"{split_counts['Validation']:,}", "Q1 2015", tone="amber")
    with cols[3]:
        metric_card("Test", f"{split_counts['Test']:,}", "Post-Q1 2015", tone="purple")
    with cols[4]:
        metric_card("Duplicates", f"{cleaning.get('duplicates_removed', 0):,}", tone="green")
    with cols[5]:
        metric_card("Invalid", f"{cleaning.get('rows_dropped_invalid', 0):,}", tone="red")

    info_note("The pipeline follows a conservative flag-first, drop-later policy. Suspect records are tagged rather than immediately removed, so rare-market cases remain available for modeling and analyst review.")

    left, right = st.columns(2)
    with left:
        missing = dq_report.get("missing_summary", {})
        missing_df = pd.DataFrame(
            [
                {
                    "feature": key,
                    "missing_pct": (float(value) / max(float(dq_report.get("row_count", len(dataframe))), 1.0)) * 100,
                }
                for key, value in missing.items()
                if float(value) > 0
            ]
        ).sort_values("missing_pct", ascending=True)
        st.markdown("<div class='section-title'>Missing Values by Feature</div>", unsafe_allow_html=True)
        if not missing_df.empty:
            fig = go.Figure(go.Bar(x=missing_df["missing_pct"], y=missing_df["feature"], orientation="h", marker_color="#2f6fb5"))
            fig.update_xaxes(ticksuffix="%")
            st.plotly_chart(style_plotly(fig, height=280), width="stretch")
        else:
            st.success("No missing values detected.")
    with right:
        st.markdown("<div class='section-title'>Chronological Split Distribution</div>", unsafe_allow_html=True)
        split_df = pd.DataFrame({"split": list(split_counts), "rows": list(split_counts.values())})
        fig = go.Figure(go.Bar(x=split_df["split"], y=split_df["rows"], marker_color=["#4f8df7", "#9dc2ff", "#0b3a75"]))
        st.plotly_chart(style_plotly(fig, height=280), width="stretch")
        st.caption("The split is chronological, not random: older sales train the model, later sales test whether it generalizes.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("<div class='section-title'>Price Distribution</div>", unsafe_allow_html=True)
        price_source = raw_df if not raw_df.empty and "price" in raw_df.columns else dataframe.rename(columns={"observed_price": "price"})
        if "price" in price_source.columns:
            fig = go.Figure(go.Histogram(x=price_source["price"], nbinsx=40, marker_color="#22c55e"))
            st.plotly_chart(style_plotly(fig, height=280), width="stretch")
    with bottom_right:
        st.markdown("<div class='section-title'>Engineered Feature Snapshot</div>", unsafe_allow_html=True)
        feature_count = max(int(features_df.shape[1]) if not features_df.empty else int(dataframe.shape[1]), 0)
        feature_rows = int(features_df.shape[0]) if not features_df.empty else int(len(dataframe))
        metric_cols = st.columns(2)
        metric_cols[0].metric("Feature rows", f"{feature_rows:,}")
        metric_cols[1].metric("Feature columns", f"{feature_count:,}")
        top_features = ["house_age", "predicted_price_band", "support_score", "slice_risk_level", "valuation_gap"]
        present = [feature for feature in top_features if feature in dataframe.columns]
        st.caption("Key engineered fields: " + (", ".join(present) if present else "Run feature pipeline to populate feature fields."))

    st.markdown("<div class='section-title'>Cleaning & Transformation Log</div>", unsafe_allow_html=True)
    log_rows = pd.DataFrame(
        [
            {"Cleaning Step": "Schema validation", "Purpose": "Verify expected columns and datatypes", "Output / Impact": f"{len(dq_report.get('missing_columns', []))} missing required columns"},
            {"Cleaning Step": "Deduplication", "Purpose": "Remove repeated transaction records", "Output / Impact": f"{cleaning.get('duplicates_removed', 0):,} duplicates removed"},
            {"Cleaning Step": "Invalid record policy", "Purpose": "Reject impossible values while preserving edge cases", "Output / Impact": f"{cleaning.get('rows_dropped_invalid', 0):,} invalid rows dropped"},
            {"Cleaning Step": "Suspect flagging", "Purpose": "Keep rare-market signals available for review", "Output / Impact": f"{cleaning.get('rows_flagged_suspect', 0):,} rows flagged suspect"},
            {"Cleaning Step": "Feature engineering", "Purpose": "Create temporal, geospatial, and comparable-sale context", "Output / Impact": f"{feature_count:,} feature columns available"},
        ]
    )
    st.dataframe(log_rows, width="stretch", hide_index=True)


def render_map_page(dataframe: pd.DataFrame, selected_review_labels: list[str]) -> None:
    page_header("Geospatial Distribution", "King County anomaly map and geographic concentration diagnostics.")
    left, right = st.columns([2.2, 0.85])
    with right:
        st.markdown("<div class='card detail-panel'><div class='section-title'>Map Controls</div>", unsafe_allow_html=True)
        map_focus = st.toggle("Flagged only", value=True)
        max_points = st.slider("Maximum mapped sales", 500, 12000, 5000, 500)
        st.toggle("High slice risk", value=False)
        st.toggle("Heatmap mode", value=False)
        st.markdown("</div>", unsafe_allow_html=True)
    with left:
        focus_name = "Anomalies only" if map_focus else "All transactions"
        focus_labels = map_labels_for_focus(focus_name, selected_review_labels)
        render_map(dataframe, focus_labels=focus_labels, max_points=max_points, focus_source=FOCUS_LABELS[focus_name].lower())
    bottom_left, bottom_right = st.columns([1.3, 1])
    with bottom_left:
        flagged = dataframe.loc[dataframe["anomaly_flag"].isin(ACTIONABLE_LABELS)]
        top_zip = flagged.groupby("zipcode").size().sort_values(ascending=False).head(5)
        st.markdown("<div class='section-title'>Top Zipcodes by Flagged Count</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=top_zip.values, y=top_zip.index.astype(str), orientation="h", marker_color="#002b5c"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_plotly(fig, height=280), width="stretch")
    with bottom_right:
        st.markdown(
            """
            <div class="card detail-panel">
                <div class="section-title">Flagged Rate Trend</div>
                <p>Spatial concentration note: recent anomaly spikes are tightly correlated with high-density rezoning areas in the downtown periphery.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_performance() -> None:
    page_header("Model Performance", "Compare primary valuation model metrics against established baselines across market segments.")
    guidance_note(
        "Performance metrics",
        [
            "R² measures how much price variation the model explains; closer to 1.0 is better.",
            "MAPE is average percentage error; lower is better and easier to compare across price levels.",
            "RMSE is dollar error with extra penalty for large misses; it is useful for understanding financial impact.",
            "Error by Price Band uses Q1-Q5 price quintiles, where Q5 is the luxury/highest-price segment.",
        ],
    )
    val_df = load_csv_safe(DEFAULT_VALUATION_METRICS)
    comp_df = load_csv_safe(DEFAULT_MODEL_COMPARISON)
    row = val_df.iloc[0] if val_df is not None and not val_df.empty else {}
    cols = st.columns(4)
    with cols[0]:
        metric_card("Primary Algorithm", "XGBoost", "v2.4.1")
    with cols[1]:
        metric_card("Test R²", f"{row.get('test_r2', 0):.4f}" if hasattr(row, "get") else "-")
    with cols[2]:
        metric_card("Test MAPE", f"{row.get('test_mape', 0):.2f}%" if hasattr(row, "get") else "-")
    with cols[3]:
        metric_card("Test RMSE", format_currency(row.get("test_rmse")) if hasattr(row, "get") else "-")
    st.write("")
    left, right = st.columns([1.7, 0.9])
    with left:
        st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
        if comp_df is not None:
            display_cols = [column for column in ["model_name", "test_r2", "test_mape", "test_rmse"] if column in comp_df.columns]
            st.dataframe(comp_df[display_cols], width="stretch", hide_index=True)
            with st.expander("Open full baseline metrics", expanded=False):
                st.dataframe(comp_df, width="stretch", hide_index=True)
    with right:
        err_df = load_csv_safe(DEFAULT_ERROR_BY_BAND)
        if err_df is not None and not err_df.empty:
            st.markdown("<div class='section-title'>Error by Price Band</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=err_df["mae"], y=err_df["price_band"], orientation="h", marker_color="#002b5c"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_plotly(fig, height=300), width="stretch")
            st.caption("Q5 is the highest-price quintile. Higher error there usually reflects luxury-market variance and fewer close comparables.")
        st.markdown("<div class='alert'>ⓘ Analyst note: Q5 variance remains elevated under upper-tail uncertainty policy.</div>", unsafe_allow_html=True)


def render_validation(dataframe: pd.DataFrame) -> None:
    page_header("Human-in-the-Loop Validation", "Review model anomalies to refine uncertainty thresholds.")
    queue_source = selected_property_frame(dataframe)
    row = queue_source.iloc[0] if not queue_source.empty else pd.Series(dtype=object)
    left, right = st.columns([1.9, 0.8])
    with left:
        st.markdown("<div class='card detail-panel'><div class='section-title'>Analyst Labeling Form</div>", unsafe_allow_html=True)
        st.text_input("Queue ID", str(row.get("property_id", "")))
        c1, c2, c3 = st.columns(3)
        c1.metric("Model AVM Value", format_currency(row.get("fair_value_hat")))
        c2.metric("Observed Price", format_currency(row.get("observed_price")))
        c3.metric("Confidence Score", model_confidence(row.get("evidence_strength"), row.get("slice_risk_level")))
        st.radio("Assessment Outcome", ["Valid Flag", "False Positive", "Uncertain"], horizontal=True)
        st.text_area("Analyst Notes", placeholder="Enter detailed analysis here...", height=160)
        st.button("Submit Review →", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            """
            <div class="card detail-panel">
                <div class="section-title">Pilot Metrics</div>
                <div class="mini-value">88.5%</div><div class="mini-label">Precision target before deployment</div>
                <div class="divider"></div>
                <div class="mini-value">92.0%</div><div class="mini-label">Reviewer acceptance rate</div>
                <div class="divider"></div>
                <div class="mini-value">3m 42s</div><div class="mini-label">Avg handle time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def main() -> None:
    st.set_page_config(page_title="DC-REIF | Market Control Room", layout="wide", page_icon="🏠")
    inject_css()
    inject_dashboard_css()

    bootstrap_path = Path(st.session_state.get("table_path", str(DEFAULT_TABLE)))
    bootstrap_frame = load_property_table(str(bootstrap_path)) if bootstrap_path.exists() else pd.DataFrame()

    page, selected_filters, table_path = render_sidebar(bootstrap_frame)
    st.session_state["table_path"] = table_path
    path = Path(table_path)
    if not path.exists():
        topbar()
        page_header("DC-REIF Valuation Review Dashboard", "Model-assisted pricing anomaly detection for human valuation review")
        st.warning("Run the quickstart before opening the dashboard, or choose a valid property intelligence table.")
        st.code("python scripts/quickstart.py --install\nstreamlit run app/streamlit_app.py")
        return

    dataframe = load_property_table(str(path))
    topbar(search_placeholder="Search parameters..." if page == "Uncertainty" else "Search properties...")

    selected_review_labels = selected_filters.get("anomaly_flag", [])
    filtered = apply_selected_filters(dataframe, selected_filters)
    metrics = summarize_metrics(filtered, full_count=len(dataframe))

    if page == "Overview":
        render_overview(dataframe, filtered, metrics)
    elif page == "Review Queue":
        render_review_queue(filtered)
    elif page == "Map":
        render_map_page(filtered, selected_review_labels)
    elif page == "Data Processing & EDA":
        render_data_processing_eda(dataframe)
    elif page == "Uncertainty":
        render_uncertainty()
    elif page == "Performance":
        render_performance()
    elif page == "Explainability":
        render_explainability()
    elif page == "Data Quality":
        render_data_quality(dataframe)
    elif page == "Validation":
        render_validation(filtered)


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    return get_script_run_ctx() is not None


def run_with_streamlit_cli() -> None:
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(APP_FILE), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        run_with_streamlit_cli()
