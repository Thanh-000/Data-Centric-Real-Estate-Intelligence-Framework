import pandas as pd

from app.streamlit_app import (
    FILTER_LABELS,
    MARKET_SLICE_COLUMNS,
    SIDEBAR_FILTER_COLUMNS,
    build_review_queue,
    build_slice_summary,
    map_excluded_labels,
    map_labels_for_focus,
    map_metrics,
    prepare_map_frame,
    summarize_metrics,
    status_line,
)


def _dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "property_id": ["1", "2", "3", "4"],
            "sale_date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
            "zipcode": ["98001", "98001", "98002", "98002"],
            "segment_label": ["segment_zipcode_98001", "segment_zipcode_98001", "segment_zipcode_98002", "segment_zipcode_98002"],
            "observed_price_band": ["Q1", "Q1", "Q5", "Q5"],
            "evidence_strength": ["strong", "limited", "moderate", "strong"],
            "slice_risk_level": ["low", "high", "medium", "low"],
            "lat": [47.1, 47.2, 47.3, 47.4],
            "long": [-122.1, -122.2, -122.3, -122.4],
            "observed_price": [300000, 450000, 700000, 900000],
            "fair_value_hat": [340000, 390000, 800000, 880000],
            "lower_bound": [250000, 300000, 650000, 760000],
            "upper_bound": [430000, 480000, 950000, 1000000],
            "anomaly_flag": [
                "within_expected_range",
                "potentially_over_valued",
                "potentially_under_valued",
                "insufficient_history",
            ],
            "anomaly_score": [0.05, 0.42, -0.55, 0.0],
            "top_drivers": ["grade", "location", "sqft", "support"],
            "why_flagged": ["inside", "above", "below", "low support"],
        }
    )


def test_map_focus_is_authoritative_when_review_labels_include_within_range():
    selected = ["within_expected_range", "insufficient_history"]

    assert map_labels_for_focus("Anomalies only", selected) == ["insufficient_history"]
    assert map_excluded_labels("Anomalies only", selected) == ["within_expected_range"]
    assert map_labels_for_focus("All transactions", selected) == ["insufficient_history", "within_expected_range"]


def test_dashboard_hides_technical_zipcode_segment_from_beginner_filters():
    assert "segment_label" not in FILTER_LABELS
    assert "segment_label" not in SIDEBAR_FILTER_COLUMNS
    assert "segment_label" not in MARKET_SLICE_COLUMNS
    assert "zipcode" in SIDEBAR_FILTER_COLUMNS
    assert "zipcode" in MARKET_SLICE_COLUMNS


def test_prepare_map_frame_filters_to_focus_labels_and_limits_rows():
    frame = _dashboard_frame()

    map_frame = prepare_map_frame(frame, ["potentially_over_valued", "potentially_under_valued"], max_points=1)

    assert len(map_frame) == 1
    assert map_frame["anomaly_flag"].iloc[0] == "potentially_under_valued"
    assert "within_expected_range" not in set(map_frame["anomaly_flag"])


def test_prepare_map_frame_renders_within_range_as_background_points():
    frame = _dashboard_frame()

    map_frame = prepare_map_frame(frame, ["within_expected_range"], max_points=10)

    assert len(map_frame) == 1
    assert map_frame["radius_px"].iloc[0] == 2.4
    assert map_frame["label"].iloc[0] == "Within expected range"


def test_map_metrics_counts_what_is_actually_mapped():
    frame = _dashboard_frame()

    metrics = map_metrics(frame, ["potentially_over_valued", "potentially_under_valued"], max_points=10)

    assert metrics == {
        "mapped_sales": 2,
        "review_flags": 2,
        "low_support": 0,
        "within_range": 0,
    }


def test_summarize_metrics_counts_actionable_low_support_and_within_range():
    metrics = summarize_metrics(_dashboard_frame(), full_count=8)

    assert metrics["transactions"] == 4
    assert metrics["anomalies"] == 2
    assert metrics["low_support"] == 1
    assert metrics["within_range"] == 1
    assert metrics["coverage"] == 0.5
    assert (
        status_line(metrics)
        == "2 sales need human review because the model flagged pricing risk and 1 sale has limited local evidence in the current view."
    )


def test_build_review_queue_uses_friendly_label_and_sorts_by_absolute_score():
    queue = build_review_queue(_dashboard_frame())

    assert "Model signal" in queue.columns
    assert "Model confidence" in queue.columns
    assert "Human review note" in queue.columns
    assert queue["Property ID"].tolist()[:2] == ["3", "2"]
    assert queue.loc[queue["Property ID"].eq("2"), "Model signal"].iloc[0] == "Over-valued"
    assert "candidate for human review" in queue.loc[queue["Property ID"].eq("2"), "Human review note"].iloc[0]


def test_build_slice_summary_keeps_rates_visible_across_labels():
    summary = build_slice_summary(_dashboard_frame(), "observed_price_band")
    q1 = summary.loc[summary["Price band"].eq("Q1")].iloc[0]
    q5 = summary.loc[summary["Price band"].eq("Q5")].iloc[0]

    assert q1["Sales"] == 2
    assert q1["Model flags"] == 1
    assert q1["Review flag rate"] == "50.0%"
    assert q5["Sales"] == 2
    assert q5["Low support"] == 1
    assert q5["Low-support rate"] == "50.0%"
