from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from dc_reif.product_analytics import with_product_bands


DEFAULT_TABLE = ROOT / "outputs" / "tables" / "property_intelligence_table.csv"


@st.cache_data
def load_property_table(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return with_product_bands(dataframe)


def filter_multiselect(dataframe: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in dataframe.columns:
        return dataframe
    options = sorted([str(value) for value in dataframe[column].dropna().unique()])
    selected = st.sidebar.multiselect(label, options)
    if not selected:
        return dataframe
    return dataframe.loc[dataframe[column].astype(str).isin(selected)]


def main() -> None:
    st.set_page_config(page_title="DC-REIF Review Dashboard", layout="wide")
    st.title("DC-REIF Pricing Anomaly Review")

    table_path = st.sidebar.text_input("Property table", str(DEFAULT_TABLE))
    path = Path(table_path)
    if not path.exists():
        st.warning("Run `python scripts/quickstart.py --install` before opening the dashboard.")
        st.code("python scripts/quickstart.py --install\nstreamlit run app/streamlit_app.py")
        return

    dataframe = load_property_table(str(path))

    filtered = dataframe.copy()
    filtered = filter_multiselect(filtered, "anomaly_flag", "Anomaly label")
    filtered = filter_multiselect(filtered, "zipcode", "Zipcode")
    filtered = filter_multiselect(filtered, "segment_label", "Segment")
    filtered = filter_multiselect(filtered, "predicted_price_band", "Predicted price band")
    filtered = filter_multiselect(filtered, "observed_price_band", "Observed price band")
    filtered = filter_multiselect(filtered, "evidence_strength", "Evidence strength")
    filtered = filter_multiselect(filtered, "slice_risk_level", "Slice risk")

    total = len(filtered)
    abstained = int(filtered["anomaly_flag"].eq("insufficient_history").sum()) if total else 0
    anomalies = int(filtered["anomaly_flag"].isin(["potentially_over_valued", "potentially_under_valued"]).sum()) if total else 0
    coverage = total / len(dataframe) if len(dataframe) else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered transactions", f"{total:,}", f"{coverage:.1%} of all")
    metric_cols[1].metric("Potential anomalies", f"{anomalies:,}")
    metric_cols[2].metric("Insufficient history", f"{abstained:,}")
    metric_cols[3].metric("Abstention rate", f"{(abstained / total):.1%}" if total else "n/a")

    map_tab, queue_tab, slice_tab = st.tabs(["Map", "Review queue", "Slice summary"])

    with map_tab:
        if {"lat", "long"}.issubset(filtered.columns):
            map_frame = filtered.loc[filtered["lat"].notna() & filtered["long"].notna(), ["lat", "long"]].copy()
            st.map(map_frame.rename(columns={"long": "lon"}), latitude="lat", longitude="lon", zoom=8)
        else:
            st.info("Latitude/longitude columns are unavailable. Re-run the current pipeline version.")

    with queue_tab:
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
        available = [column for column in queue_columns if column in filtered.columns]
        if "anomaly_score" in filtered.columns:
            display = filtered[available].sort_values(
                "anomaly_score",
                key=lambda series: series.abs(),
                ascending=False,
                na_position="last",
            )
        else:
            display = filtered[available]
        st.dataframe(display)
        st.download_button(
            "Download filtered review queue",
            data=filtered[available].to_csv(index=False),
            file_name="dc_reif_review_queue.csv",
            mime="text/csv",
        )

    with slice_tab:
        if "anomaly_flag" in filtered.columns:
            st.subheader("Anomaly labels")
            st.bar_chart(filtered["anomaly_flag"].value_counts())
        for column in ["zipcode", "segment_label", "observed_price_band", "evidence_strength"]:
            if column in filtered.columns:
                st.subheader(column)
                summary = filtered.groupby(column, dropna=False).agg(
                    transactions=("property_id", "size"),
                    abstentions=("anomaly_flag", lambda values: int((values == "insufficient_history").sum())),
                    median_price=("observed_price", "median"),
                )
                summary["abstention_rate"] = summary["abstentions"] / summary["transactions"]
                st.dataframe(summary.sort_values("transactions", ascending=False))


if __name__ == "__main__":
    main()
