from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dc_reif.utils import ensure_directory


ABSTENTION_LABEL = "insufficient_history"


def price_band(series: pd.Series, n_bands: int = 5) -> pd.Series:
    valid = series.dropna()
    output = pd.Series(pd.NA, index=series.index, dtype="string")
    if valid.empty:
        return output
    n_quantiles = min(n_bands, max(valid.nunique(), 1))
    labels = [f"Q{index}" for index in range(1, n_quantiles + 1)]
    ranked = valid.rank(method="first")
    bands = pd.qcut(ranked, q=n_quantiles, labels=labels)
    output.loc[valid.index] = bands.astype("string")
    return output


def with_product_bands(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if "observed_price" in frame.columns and "observed_price_band" not in frame.columns:
        frame["observed_price_band"] = price_band(frame["observed_price"])
    if "house_age" in frame.columns and "house_age_band" not in frame.columns:
        frame["house_age_band"] = pd.cut(
            frame["house_age"],
            bins=[-np.inf, 10, 30, 60, np.inf],
            labels=["0-10", "11-30", "31-60", "61+"],
        ).astype("string")
    if "grade" in frame.columns and "grade_band" not in frame.columns:
        frame["grade_band"] = pd.cut(
            frame["grade"],
            bins=[-np.inf, 6, 8, 10, np.inf],
            labels=["low", "mid", "high", "luxury"],
        ).astype("string")
    return frame


def abstention_summary(dataframe: pd.DataFrame, group_columns: Iterable[str]) -> dict[str, pd.DataFrame]:
    frame = with_product_bands(dataframe)
    summaries: dict[str, pd.DataFrame] = {}
    for column in group_columns:
        if column not in frame.columns:
            continue
        grouped = frame.groupby(column, dropna=False)
        summary = grouped.agg(
            transaction_count=("anomaly_flag", "size"),
            abstention_count=("anomaly_flag", lambda values: int((values == ABSTENTION_LABEL).sum())),
            median_observed_price=("observed_price", "median"),
        ).reset_index()
        summary["abstention_rate"] = summary["abstention_count"] / summary["transaction_count"]
        summaries[column] = summary.sort_values(
            ["abstention_rate", "transaction_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return summaries


def slice_metrics(dataframe: pd.DataFrame, group_columns: Iterable[str]) -> dict[str, pd.DataFrame]:
    frame = with_product_bands(dataframe)
    frame["is_abstained"] = frame["anomaly_flag"].eq(ABSTENTION_LABEL)
    frame["is_anomaly"] = frame["anomaly_flag"].isin(["potentially_over_valued", "potentially_under_valued"])
    frame["abs_error"] = (frame["observed_price"] - frame["fair_value_hat"]).abs()
    frame["squared_error"] = np.square(frame["observed_price"] - frame["fair_value_hat"])
    frame["ape"] = frame["abs_error"] / frame["observed_price"].replace(0, np.nan)
    frame["within_interval"] = (
        (frame["observed_price"] >= frame["lower_bound"]) & (frame["observed_price"] <= frame["upper_bound"])
    )

    outputs: dict[str, pd.DataFrame] = {}
    for column in group_columns:
        if column not in frame.columns:
            continue
        rows: list[dict[str, object]] = []
        for value, group in frame.groupby(column, dropna=False):
            scored = group.loc[group["fair_value_hat"].notna()].copy()
            rows.append(
                {
                    column: value,
                    "transaction_count": int(len(group)),
                    "scored_count": int(len(scored)),
                    "abstention_count": int(group["is_abstained"].sum()),
                    "abstention_rate": float(group["is_abstained"].mean()),
                    "anomaly_count": int(group["is_anomaly"].sum()),
                    "anomaly_rate": float(group["is_anomaly"].mean()),
                    "mae": float(scored["abs_error"].mean()) if not scored.empty else np.nan,
                    "rmse": float(np.sqrt(scored["squared_error"].mean())) if not scored.empty else np.nan,
                    "mape": float(scored["ape"].mean()) if not scored.empty else np.nan,
                    "interval_coverage": float(scored["within_interval"].mean()) if not scored.empty else np.nan,
                    "average_interval_width": float(scored["interval_width"].mean()) if not scored.empty else np.nan,
                    "median_observed_price": float(group["observed_price"].median()) if "observed_price" in group else np.nan,
                }
            )
        outputs[column] = pd.DataFrame(rows).sort_values("transaction_count", ascending=False).reset_index(drop=True)
    return outputs


def threshold_sensitivity(dataframe: pd.DataFrame, thresholds: Iterable[float] = (0.05, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    frame = dataframe.loc[dataframe["fair_value_hat"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["threshold", "scored_count", "overvalued_count", "undervalued_count", "total_flagged", "flagged_rate"]
        )
    relative_gap = (frame["observed_price"] - frame["fair_value_hat"]) / frame["fair_value_hat"].replace(0, np.nan)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        overvalued = relative_gap >= threshold
        undervalued = relative_gap <= -threshold
        total_flagged = int((overvalued | undervalued).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "scored_count": int(relative_gap.notna().sum()),
                "overvalued_count": int(overvalued.sum()),
                "undervalued_count": int(undervalued.sum()),
                "total_flagged": total_flagged,
                "flagged_rate": float(total_flagged / max(relative_gap.notna().sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def interval_width_summary(dataframe: pd.DataFrame, group_columns: Iterable[str]) -> dict[str, pd.DataFrame]:
    frame = with_product_bands(dataframe.loc[dataframe["interval_width"].notna()].copy())
    outputs: dict[str, pd.DataFrame] = {}
    for column in group_columns:
        if column not in frame.columns:
            continue
        summary = (
            frame.groupby(column, dropna=False)["interval_width"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .reset_index()
            .rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
        )
        outputs[column] = summary.sort_values("count", ascending=False).reset_index(drop=True)
    return outputs


def write_analysis_tables(tables: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> dict[str, Path]:
    ensure_directory(output_dir)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
