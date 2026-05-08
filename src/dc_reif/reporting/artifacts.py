from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dc_reif.utils import ensure_directory


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    dataframe.to_csv(path, index=False)
    return path


def save_json(payload: dict[str, object], path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def create_eda_figures(dataframe: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    ensure_directory(figures_dir)
    outputs: dict[str, Path] = {}

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(dataframe["price"], bins=40, ax=ax, color="#33658a")
    ax.set_title("Sale Price Distribution")
    fig.tight_layout()
    outputs["price_distribution"] = figures_dir / "price_distribution.png"
    fig.savefig(outputs["price_distribution"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    log_price = np.log(dataframe.loc[dataframe["price"] > 0, "price"])
    sns.histplot(log_price, bins=40, ax=ax, color="#86bbd8")
    ax.set_title("Log Price Distribution")
    fig.tight_layout()
    outputs["log_price_distribution"] = figures_dir / "log_price_distribution.png"
    fig.savefig(outputs["log_price_distribution"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(dataframe["long"], dataframe["lat"], c=dataframe["price"], s=8, cmap="viridis", alpha=0.5)
    fig.colorbar(scatter, ax=ax, label="Sale Price")
    ax.set_title("Spatial Sale Price Pattern")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    outputs["spatial_price_map"] = figures_dir / "spatial_price_map.png"
    fig.savefig(outputs["spatial_price_map"], dpi=150)
    plt.close(fig)

    trend = (
        dataframe.assign(sale_period=dataframe["date"].dt.to_period("M").astype(str))
        .groupby("sale_period")["price"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=trend, x="sale_period", y="price", marker="o", ax=ax, color="#758e4f")
    ax.set_title("Median Sale Price by Month")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    outputs["temporal_trend"] = figures_dir / "temporal_trend.png"
    fig.savefig(outputs["temporal_trend"], dpi=150)
    plt.close(fig)

    return outputs


def create_residual_diagnostics(dataframe: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    ensure_directory(figures_dir)
    outputs: dict[str, Path] = {}
    scored = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    if scored.empty:
        return outputs

    scored["residual"] = scored["observed_price"] - scored["fair_value_hat"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(scored["fair_value_hat"], scored["residual"], s=10, alpha=0.45, color="#33658a")
    ax.axhline(0, color="#2f4858", linewidth=1)
    ax.set_title("Residuals vs Predicted Fair Value")
    ax.set_xlabel("Predicted fair value")
    ax.set_ylabel("Observed minus predicted")
    fig.tight_layout()
    outputs["residual_vs_predicted"] = figures_dir / "residual_vs_predicted.png"
    fig.savefig(outputs["residual_vs_predicted"], dpi=150)
    plt.close(fig)

    binned = scored.copy()
    ranked = binned["fair_value_hat"].rank(method="first")
    binned["predicted_value_bin"] = pd.qcut(ranked, q=min(10, binned["fair_value_hat"].nunique()), duplicates="drop")
    variance_summary = (
        binned.groupby("predicted_value_bin", observed=True)
        .agg(
            predicted_midpoint=("fair_value_hat", "median"),
            residual_std=("residual", "std"),
            mean_abs_residual=("residual", lambda series: float(series.abs().mean())),
            count=("residual", "size"),
        )
        .dropna(subset=["predicted_midpoint", "residual_std"])
        .reset_index(drop=True)
    )
    if not variance_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            variance_summary["predicted_midpoint"],
            variance_summary["residual_std"],
            marker="o",
            color="#d95f02",
            label="Residual standard deviation",
        )
        ax.plot(
            variance_summary["predicted_midpoint"],
            variance_summary["mean_abs_residual"],
            marker="s",
            color="#33658a",
            label="Mean absolute residual",
        )
        ax.set_title("Residual Variance by Predicted Fair Value Bin")
        ax.set_xlabel("Median predicted fair value in bin")
        ax.set_ylabel("Error magnitude")
        ax.legend()
        fig.tight_layout()
        outputs["residual_variance_by_predicted_bin"] = figures_dir / "residual_variance_by_predicted_bin.png"
        fig.savefig(outputs["residual_variance_by_predicted_bin"], dpi=150)
        plt.close(fig)

    if {"lat", "long"}.issubset(scored.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = scored["residual"].abs().quantile(0.95)
        scatter = ax.scatter(
            scored["long"],
            scored["lat"],
            c=scored["residual"].clip(lower=-limit, upper=limit),
            s=8,
            cmap="coolwarm",
            alpha=0.55,
        )
        fig.colorbar(scatter, ax=ax, label="Residual")
        ax.set_title("Spatial Residual Pattern")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        outputs["spatial_residual_map"] = figures_dir / "spatial_residual_map.png"
        fig.savefig(outputs["spatial_residual_map"], dpi=150)
        plt.close(fig)

    return outputs


def write_summary_report(summary_lines: list[str], path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text("\n".join(summary_lines), encoding="utf-8")
    return path


def write_trust_summary(
    *,
    valuation_metrics: pd.DataFrame,
    interval_metrics: dict[str, float],
    q5_coverage: float,
    q5_interval_width: float,
    property_ledger: pd.DataFrame,
    output_path: Path,
    interval_comparison: pd.DataFrame | None = None,
    selected_interval_method: str = "conformal",
    quantile_rejection_reasons: list[str] | None = None,
) -> Path:
    row = valuation_metrics.iloc[0].to_dict() if not valuation_metrics.empty else {}
    counts = property_ledger["anomaly_flag"].value_counts().to_dict()
    model_flagged = int(
        property_ledger["anomaly_flag"].isin(["potentially_over_valued", "potentially_under_valued"]).sum()
    )
    comparison_lines: list[str] = []
    if interval_comparison is not None and not interval_comparison.empty:
        comparison_lines.extend(
            [
                "",
                "### Interval Method Comparison",
                "",
                f"- Selected decision-layer interval method: `{selected_interval_method}`",
            ]
        )
        for method in ["conformal", "quantile_xgb"]:
            subset = interval_comparison.loc[interval_comparison["method"].eq(method)]
            if subset.empty:
                continue
            method_row = subset.iloc[0]
            comparison_lines.append(
                f"- {method}: coverage {float(method_row.get('coverage', float('nan'))):.1%}, "
                f"Q5 coverage {float(method_row.get('q5_coverage', float('nan'))):.1%}, "
                f"average width ${float(method_row.get('avg_width', float('nan'))):,.0f}, "
                f"30% synthetic recall {float(method_row.get('recall_30pct', float('nan'))):.1%}"
            )
        if selected_interval_method != "quantile_xgb" and quantile_rejection_reasons:
            comparison_lines.append(
                "- Quantile XGBoost was not selected because "
                + "; ".join(quantile_rejection_reasons)
                + "."
            )
        elif selected_interval_method == "quantile_xgb":
            comparison_lines.append(
                "- Quantile XGBoost was selected because it met the coverage, Q5 coverage, width, and synthetic-recall gates."
            )

    lines = [
        "# Trust Summary",
        "",
        "## Intended Use",
        "",
        "This system is a model-assisted triage tool for realized sale-price review. It flags candidates for human review; it is not an automated appraisal engine or a final pricing authority.",
        "",
        "## Model Performance",
        "",
        f"- Test R2: {float(row.get('test_r2', float('nan'))):.3f}",
        f"- Test MAPE: {float(row.get('test_mape', float('nan'))):.2f}%",
        f"- Test RMSE: ${float(row.get('test_rmse', float('nan'))):,.0f}",
        f"- Test MAE: ${float(row.get('test_mae', float('nan'))):,.0f}",
        "",
        "## Uncertainty",
        "",
        "Coverage values are empirical diagnostics under the current chronological, localized, upper-tail-adjusted protocol; they are not theoretical guarantees of standard split conformal prediction.",
        "",
        f"- Global empirical coverage: {interval_metrics.get('empirical_coverage', float('nan')):.1%}",
        f"- High-price Q5 empirical coverage: {q5_coverage:.1%}",
        f"- Average interval width: ${interval_metrics.get('average_interval_width', float('nan')):,.0f}",
        f"- High-price Q5 average interval width: ${q5_interval_width:,.0f}",
        *comparison_lines,
        "",
        "## Decision Layer",
        "",
        f"- Model-flagged cases: {model_flagged:,}",
        f"- Potentially over-valued: {int(counts.get('potentially_over_valued', 0)):,}",
        f"- Potentially under-valued: {int(counts.get('potentially_under_valued', 0)):,}",
        f"- Withheld insufficient-history cases: {int(counts.get('insufficient_history', 0)):,}",
        f"- Within model range: {int(counts.get('within_expected_range', 0)):,}",
        "",
        "## Known Limitations",
        "",
        "- The data is the static King County 2014-2015 transaction dataset.",
        "- High-price properties require wider intervals, so flags are less sharp in that segment.",
        "- Feature importance and SHAP describe model behavior, not causal proof.",
        "- All model-flagged cases should be reviewed with local market context before business use.",
    ]
    return write_summary_report(lines, output_path)
