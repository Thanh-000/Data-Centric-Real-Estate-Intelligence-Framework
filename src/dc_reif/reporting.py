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


def write_summary_report(summary_lines: list[str], path: Path) -> Path:
    ensure_directory(path.parent)
    path.write_text("\n".join(summary_lines), encoding="utf-8")
    return path
