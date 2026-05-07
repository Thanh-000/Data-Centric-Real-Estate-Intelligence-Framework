from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"
FIGURES = Path(__file__).resolve().parent / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def model_comparison() -> None:
    frame = pd.read_csv(TABLES / "model_baseline_comparison.csv").sort_values("test_rmse", ascending=True)
    colors = ["#2f6f4e" if name == "xgboost_selected" else "#7794aa" for name in frame["model_name"]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(frame["model_name"], frame["test_rmse"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (USD)")
    ax.set_title("Held-out Test RMSE by Model")
    for index, value in enumerate(frame["test_rmse"]):
        ax.text(value + 8000, index, f"${value:,.0f}", va="center", fontsize=9)
    _save(fig, "report_model_comparison_rmse.png")


def coverage_and_width() -> None:
    frame = pd.read_csv(TABLES / "test_interval_coverage_by_price_band.csv")
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(frame["price_band"], frame["empirical_coverage"] * 100, color="#4c78a8", alpha=0.78)
    ax1.axhline(90, color="#d95f02", linestyle="--", linewidth=1.5, label="90% target")
    ax2.plot(
        frame["price_band"],
        frame["average_interval_width"] / 1000,
        color="#2f6f4e",
        marker="o",
        linewidth=2.2,
        label="Average interval width",
    )
    ax1.set_ylim(80, 100)
    ax1.set_ylabel("Empirical coverage (%)")
    ax2.set_ylabel("Average interval width (USD thousands)")
    ax1.set_xlabel("Observed price quintile")
    ax1.set_title("Held-out Coverage and Interval Width by Observed Price Quintile")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left")
    _save(fig, "report_coverage_interval_width.png")


def error_by_band() -> None:
    frame = pd.read_csv(TABLES / "test_error_by_price_band.csv")
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(frame["price_band"], frame["mae"] / 1000, color="#7aa6c2", alpha=0.82, label="MAE")
    ax2.plot(frame["price_band"], frame["mape"], color="#c44e52", marker="o", linewidth=2.2, label="MAPE")
    ax1.set_ylabel("MAE (USD thousands)")
    ax2.set_ylabel("MAPE (%)")
    ax1.set_xlabel("Observed price quintile")
    ax1.set_title("Held-out Error Profile by Observed Price Quintile")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    _save(fig, "report_error_by_price_band.png")


def decision_mix() -> None:
    ledger = pd.read_csv(TABLES / "property_intelligence_table.csv")
    counts = ledger["anomaly_flag"].value_counts()
    labels = ["Within range", "Over-valued", "Under-valued"]
    values = [
        int(counts.get("within_expected_range", 0)),
        int(counts.get("potentially_over_valued", 0)),
        int(counts.get("potentially_under_valued", 0)),
    ]
    colors = ["#8ab17d", "#d65f5f", "#3f8fb5"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Transactions")
    ax.set_title("Full-portfolio Decision Layer Composition")
    for index, value in enumerate(values):
        ax.text(index, value + max(values) * 0.015, f"{value:,}", ha="center", fontsize=9)
    _save(fig, "report_decision_mix.png")


def synthetic_recall() -> None:
    frame = pd.read_csv(TABLES / "synthetic_anomaly_recall.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for scenario, group in frame.loc[frame["scenario"] != "overall"].groupby("scenario"):
        label = "Over-valued shock" if "over" in scenario else "Under-valued shock"
        ax.plot(group["shock"] * 100, group["recall"] * 100, marker="o", linewidth=2.2, label=label)
    overall = frame.loc[frame["scenario"].eq("overall")]
    ax.plot(overall["shock"] * 100, overall["recall"] * 100, color="#222222", marker="s", linewidth=2.4, label="Overall")
    ax.set_xlabel("Injected price shock (%)")
    ax.set_ylabel("Synthetic recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Synthetic Anomaly Recall Sanity Check")
    ax.legend()
    _save(fig, "report_synthetic_recall.png")


def top_features() -> None:
    frame = pd.read_csv(TABLES / "feature_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(frame["feature"], frame["importance"], color="#6c8ebf")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Model Feature Importance Values")
    _save(fig, "report_top_feature_importance.png")


def correlation_heatmap() -> None:
    frame = pd.read_csv(PROCESSED / "kc_house_data_clean.csv")
    columns = [
        "price",
        "sqft_living",
        "grade",
        "bathrooms",
        "bedrooms",
        "view",
        "waterfront",
        "condition",
        "sqft_above",
        "sqft_lot",
        "lat",
        "long",
    ]
    available = [column for column in columns if column in frame.columns]
    corr = frame[available].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Heatmap of Core Housing Variables")
    fig.colorbar(image, ax=ax, shrink=0.78)
    _save(fig, "report_correlation_heatmap.png")


def price_by_grade() -> None:
    frame = pd.read_csv(PROCESSED / "kc_house_data_clean.csv")
    grouped = frame.groupby("grade")["price"].agg(["median", "count"]).reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(grouped["grade"], grouped["median"] / 1000, color="#4c78a8", alpha=0.82)
    ax2.plot(grouped["grade"], grouped["count"], color="#d95f02", marker="o", linewidth=1.8)
    ax1.set_xlabel("Property grade")
    ax1.set_ylabel("Median sale price (USD thousands)")
    ax2.set_ylabel("Transaction count")
    ax1.set_title("Median Sale Price and Volume by Property Grade")
    _save(fig, "report_price_by_grade.png")


def price_by_waterfront_view() -> None:
    frame = pd.read_csv(PROCESSED / "kc_house_data_clean.csv")
    waterfront = frame.assign(
        waterfront_label=frame["waterfront"].fillna(0).astype(int).map({0: "No waterfront", 1: "Waterfront"})
    )
    waterfront_group = waterfront.groupby("waterfront_label")["price"].median().reindex(["No waterfront", "Waterfront"])
    view_group = frame.assign(view=frame["view"].fillna(0).astype(int)).groupby("view")["price"].median()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(waterfront_group.index, waterfront_group.values / 1000, color=["#7aa6c2", "#d65f5f"])
    axes[0].set_ylabel("Median sale price (USD thousands)")
    axes[0].set_title("Price by Waterfront")
    axes[1].bar(view_group.index.astype(str), view_group.values / 1000, color="#8ab17d")
    axes[1].set_xlabel("View rating")
    axes[1].set_title("Price by View Rating")
    _save(fig, "report_price_by_waterfront_view.png")


def zipcode_distribution() -> None:
    frame = pd.read_csv(PROCESSED / "kc_house_data_clean.csv")
    counts = frame["zipcode"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(counts.index.astype(str), counts.values, color="#6c8ebf")
    ax.set_xlabel("Transactions")
    ax.set_ylabel("Zipcode")
    ax.set_title("Top 15 Zipcodes by Transaction Count")
    for index, value in enumerate(counts.values):
        ax.text(value + 10, index, f"{value:,}", va="center", fontsize=8)
    _save(fig, "report_zipcode_distribution.png")


def main() -> None:
    model_comparison()
    coverage_and_width()
    error_by_band()
    decision_mix()
    synthetic_recall()
    top_features()
    correlation_heatmap()
    price_by_grade()
    price_by_waterfront_view()
    zipcode_distribution()
    print(f"saved report figures to {FIGURES}")


if __name__ == "__main__":
    main()
