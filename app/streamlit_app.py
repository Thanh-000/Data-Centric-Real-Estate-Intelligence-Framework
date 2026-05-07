from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import pydeck as pdk
import streamlit as st

from dc_reif.product_analytics import with_product_bands


DEFAULT_TABLE = ROOT / "outputs" / "tables" / "property_intelligence_table.csv"

LABEL_COLORS = {
    "potentially_over_valued": [222, 82, 70, 220],
    "potentially_under_valued": [43, 138, 174, 220],
    "insufficient_history": [240, 180, 64, 210],
    "within_expected_range": [84, 150, 92, 42],
}

LABEL_NAMES = {
    "potentially_over_valued": "Over-valued",
    "potentially_under_valued": "Under-valued",
    "insufficient_history": "Low support",
    "within_expected_range": "Within range",
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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        section[data-testid="stSidebar"] {
            min-width: 310px;
            max-width: 340px;
        }
        h1 {
            font-size: 2.35rem !important;
            line-height: 1.12 !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.2rem !important;
        }
        h2, h3 {
            letter-spacing: 0 !important;
        }
        div[data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 8px;
            background: rgba(128, 128, 128, 0.06);
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
        }
        .status-line {
            border-left: 4px solid #ff4b4b;
            background: rgba(128, 128, 128, 0.08);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin: 0.8rem 0 1rem 0;
            font-size: 0.96rem;
        }
        .map-help {
            color: inherit;
            opacity: 0.74;
            font-size: 0.92rem;
            margin-top: -0.2rem;
            margin-bottom: 0.5rem;
        }
        .model-caveat {
            border-left: 4px solid #f0b429;
            background: rgba(240, 180, 41, 0.10);
            border-radius: 6px;
            padding: 0.7rem 0.9rem;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.93rem;
            color: inherit;
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
    if signal == "Within range":
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


def format_currency(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"${float(value):,.0f}"


def format_score(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def queue_column_config() -> dict[str, object]:
    return {
        "Observed price": st.column_config.NumberColumn("Observed price", format="$%d"),
        "Fair value estimate": st.column_config.NumberColumn("Fair value estimate", format="$%d"),
        "Lower fair range": st.column_config.NumberColumn("Lower fair range", format="$%d"),
        "Upper fair range": st.column_config.NumberColumn("Upper fair range", format="$%d"),
        "Review score": st.column_config.NumberColumn("Review score", format="%.3f"),
    }


def slice_column_config() -> dict[str, object]:
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
    map_frame["radius_px"] = (3.0 + score.clip(upper=0.35) * 18).clip(lower=3.0, upper=9.0)
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
    st.pydeck_chart(deck, use_container_width=True, height=620)


def main() -> None:
    st.set_page_config(page_title="DC-REIF Review Dashboard", layout="wide")
    inject_css()

    st.sidebar.title("Filters")
    table_path = st.sidebar.text_input("Property table", str(DEFAULT_TABLE))
    path = Path(table_path)
    if not path.exists():
        st.warning("Run the quickstart before opening the dashboard.")
        st.code("python scripts/quickstart.py --install\nstreamlit run app/streamlit_app.py")
        return

    dataframe = load_property_table(str(path))

    selected_filters: dict[str, list[str]] = {}
    filtered = dataframe.copy()
    filtered, selected_review_labels = sidebar_filter(filtered, "anomaly_flag", FILTER_LABELS["anomaly_flag"])
    selected_filters["anomaly_flag"] = selected_review_labels
    for column in SIDEBAR_FILTER_COLUMNS:
        filtered, selected_filters[column] = sidebar_filter(filtered, column, FILTER_LABELS[column])
    context_filters = {column: values for column, values in selected_filters.items() if column != "anomaly_flag"}
    slice_context = apply_selected_filters(dataframe, context_filters)

    metrics = summarize_metrics(filtered, full_count=len(dataframe))

    st.title("Real Estate Pricing Review")
    st.caption("Sales are grouped by whether the observed price falls inside or outside the model-estimated fair-value range.")
    st.markdown(
        "<div class='model-caveat'>This dashboard is a triage aid. A model flag means a sale should be reviewed with local market context; it is not proof that the price is wrong.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='status-line'>{status_line(metrics)}</div>", unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sales in view", f"{metrics['transactions']:,}", f"{metrics['coverage']:.1%} of dataset")
    metric_cols[1].metric("Model-flagged", f"{metrics['anomalies']:,}")
    metric_cols[2].metric("Limited evidence", f"{metrics['low_support']:,}")
    metric_cols[3].metric("Within range", f"{metrics['within_range']:,}")

    map_tab, queue_tab, slice_tab = st.tabs(["Map", "Sales list", "Market slices"])

    with map_tab:
        control_cols = st.columns([1.4, 1])
        map_focus = control_cols[0].radio(
            "Map display",
            list(MAP_FOCUS),
            index=1,
            format_func=lambda value: FOCUS_LABELS[value],
            horizontal=True,
        )
        max_points = control_cols[1].slider("Maximum mapped sales", min_value=500, max_value=12000, value=5000, step=500)
        st.markdown(f"<div class='map-help'>{FOCUS_HELP[map_focus]}</div>", unsafe_allow_html=True)
        focus_labels = map_labels_for_focus(map_focus, selected_review_labels)
        active_names = ", ".join(display_value("anomaly_flag", label) for label in focus_labels)
        focus_source = f"{FOCUS_LABELS[map_focus].lower()} ({active_names})" if active_names else FOCUS_LABELS[map_focus].lower()
        if selected_review_labels:
            excluded_labels = map_excluded_labels(map_focus, selected_review_labels)
            if excluded_labels:
                excluded_names = ", ".join(display_value("anomaly_flag", label) for label in excluded_labels)
                st.caption(f"Map display excludes {excluded_names}. Use All sales for context to include them on the map.")
        if not focus_labels:
            st.info("The selected model signal is outside the current map view. Switch to All sales or adjust Model signal.")
        else:
            mapped = map_metrics(filtered, focus_labels=focus_labels, max_points=max_points)
            map_metric_cols = st.columns(4)
            map_metric_cols[0].metric("Mapped sales", f"{mapped['mapped_sales']:,}")
            map_metric_cols[1].metric("Model-flagged", f"{mapped['review_flags']:,}")
            map_metric_cols[2].metric("Limited evidence", f"{mapped['low_support']:,}")
            map_metric_cols[3].metric("Within range", f"{mapped['within_range']:,}")
            render_map(filtered, focus_labels=focus_labels, max_points=max_points, focus_source=focus_source)

    with queue_tab:
        display = build_review_queue(filtered)
        st.dataframe(
            display,
            use_container_width=True,
            height=560,
            hide_index=True,
            column_config=queue_column_config(),
        )
        st.download_button(
            "Download sales list",
            data=display.to_csv(index=False),
            file_name="dc_reif_review_queue.csv",
            mime="text/csv",
        )

    with slice_tab:
        slice_scope_options = ["All review outcomes in current filters", "Selected outcome only"]
        slice_scope = st.radio(
            "Slice summary scope",
            slice_scope_options,
            horizontal=True,
            label_visibility="collapsed",
        )
        slice_frame = slice_context if slice_scope == slice_scope_options[0] else filtered
        if selected_review_labels and slice_scope == slice_scope_options[0]:
            st.caption("Market slices keep every review outcome visible inside the selected market filters.")

        label_counts = slice_frame["anomaly_flag"].map(LABEL_NAMES).fillna(slice_frame["anomaly_flag"]).value_counts().rename_axis("label").reset_index(name="transactions")
        st.bar_chart(label_counts.set_index("label"))
        for column in MARKET_SLICE_COLUMNS:
            if column in slice_frame.columns:
                st.subheader(display_column(column))
                st.dataframe(
                    build_slice_summary(slice_frame, column),
                    use_container_width=True,
                    hide_index=True,
                    column_config=slice_column_config(),
                )


if __name__ == "__main__":
    main()
