from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LocalCalibrationArtifacts:
    global_q_hat: float
    price_band_edges: list[float]
    price_band_summary: pd.DataFrame
    segment_summary: pd.DataFrame
    calibration_summary: dict[str, Any]


def conformal_quantile(residuals: pd.Series, alpha: float = 0.1) -> float:
    absolute_residuals = residuals.abs().dropna().sort_values().to_numpy()
    if absolute_residuals.size == 0:
        raise ValueError("Residuals are required to estimate uncertainty intervals.")
    quantile_rank = math.ceil((absolute_residuals.size + 1) * (1 - alpha)) / absolute_residuals.size
    quantile_rank = min(max(quantile_rank, 0.0), 1.0)
    return float(np.quantile(absolute_residuals, quantile_rank, method="higher"))


def _prediction_band_edges(predictions: pd.Series, n_bands: int = 5) -> list[float]:
    valid = predictions.dropna().astype(float)
    if valid.empty:
        return []
    quantiles = valid.quantile(np.linspace(0, 1, n_bands + 1)).to_numpy(dtype=float)
    unique_edges = np.unique(quantiles)
    if unique_edges.size <= 2:
        return []
    return unique_edges[1:-1].tolist()


def assign_prediction_bands(predictions: pd.Series, edges: list[float]) -> pd.Series:
    if not edges:
        return pd.Series("Q1", index=predictions.index, dtype="string")
    labels = [f"Q{index}" for index in range(1, len(edges) + 2)]
    bins = [-np.inf, *edges, np.inf]
    bands = pd.cut(predictions.astype(float), bins=bins, labels=labels, include_lowest=True)
    return bands.astype("string")


def _smoothed_q_hat(residuals: pd.Series, alpha: float, global_q_hat: float, min_samples: int) -> tuple[float, int, float]:
    sample_count = int(residuals.dropna().shape[0])
    if sample_count == 0:
        return global_q_hat, 0, 0.0
    local_q_hat = conformal_quantile(residuals, alpha=alpha)
    shrink_weight = min(1.0, sample_count / max(min_samples, 1))
    blended_q_hat = shrink_weight * local_q_hat + (1 - shrink_weight) * global_q_hat
    return float(max(global_q_hat, blended_q_hat)), sample_count, float(shrink_weight)


def calibrate_local_conformal(
    calibration_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    alpha: float = 0.1,
    min_price_band_samples: int = 300,
    min_segment_samples: int = 200,
) -> tuple[pd.DataFrame, LocalCalibrationArtifacts]:
    calibration = calibration_frame.dropna(subset=["fair_value_hat", "observed_price"]).copy()
    if calibration.empty:
        raise ValueError("Calibration frame must contain observed price and OOF fair-value predictions.")

    calibration["abs_residual"] = (calibration["observed_price"] - calibration["fair_value_hat"]).abs()
    global_q_hat = conformal_quantile(calibration["observed_price"] - calibration["fair_value_hat"], alpha=alpha)
    price_band_edges = _prediction_band_edges(calibration["fair_value_hat"])
    calibration["predicted_price_band"] = assign_prediction_bands(calibration["fair_value_hat"], price_band_edges)

    band_rows: list[dict[str, Any]] = []
    for band, frame in calibration.groupby("predicted_price_band", dropna=False):
        blended_q_hat, sample_count, shrink_weight = _smoothed_q_hat(
            frame["observed_price"] - frame["fair_value_hat"],
            alpha=alpha,
            global_q_hat=global_q_hat,
            min_samples=min_price_band_samples,
        )
        band_rows.append(
            {
                "predicted_price_band": band,
                "count": sample_count,
                "shrink_weight": shrink_weight,
                "local_q_hat": blended_q_hat,
                "mean_abs_residual": float(frame["abs_residual"].mean()),
            }
        )
    price_band_summary = pd.DataFrame(band_rows).sort_values("predicted_price_band").reset_index(drop=True)

    segment_rows: list[dict[str, Any]] = []
    for segment_label, frame in calibration.groupby("segment_label", dropna=False):
        blended_q_hat, sample_count, shrink_weight = _smoothed_q_hat(
            frame["observed_price"] - frame["fair_value_hat"],
            alpha=alpha,
            global_q_hat=global_q_hat,
            min_samples=min_segment_samples,
        )
        segment_rows.append(
            {
                "segment_label": segment_label,
                "count": sample_count,
                "shrink_weight": shrink_weight,
                "local_q_hat": blended_q_hat,
                "mean_abs_residual": float(frame["abs_residual"].mean()),
            }
        )
    segment_summary = pd.DataFrame(segment_rows).sort_values("segment_label").reset_index(drop=True)

    prediction_rows = prediction_frame.copy()
    prediction_rows["predicted_price_band"] = assign_prediction_bands(prediction_rows["fair_value_hat"], price_band_edges)
    band_q_map = price_band_summary.set_index("predicted_price_band")["local_q_hat"].to_dict()
    band_count_map = price_band_summary.set_index("predicted_price_band")["count"].to_dict()
    segment_q_map = segment_summary.set_index("segment_label")["local_q_hat"].to_dict()
    segment_count_map = segment_summary.set_index("segment_label")["count"].to_dict()

    prediction_rows["price_band_q_hat"] = prediction_rows["predicted_price_band"].map(band_q_map).fillna(global_q_hat)
    prediction_rows["segment_q_hat"] = prediction_rows["segment_label"].map(segment_q_map).fillna(global_q_hat)
    prediction_rows["price_band_support_n"] = prediction_rows["predicted_price_band"].map(band_count_map).fillna(0).astype(int)
    prediction_rows["segment_support_n"] = prediction_rows["segment_label"].map(segment_count_map).fillna(0).astype(int)
    prediction_rows["q_hat"] = prediction_rows[["price_band_q_hat", "segment_q_hat"]].max(axis=1)
    prediction_rows["q_hat"] = prediction_rows["q_hat"].fillna(global_q_hat).clip(lower=0.9 * global_q_hat)

    calibration_summary = {
        "interval_method": "conformal_prediction_residual_quantile_localized",
        "global_q_hat": float(global_q_hat),
        "price_band_edges": [float(edge) for edge in price_band_edges],
        "min_price_band_samples": int(min_price_band_samples),
        "min_segment_samples": int(min_segment_samples),
        "average_local_q_hat": float(prediction_rows["q_hat"].mean()),
    }
    artifacts = LocalCalibrationArtifacts(
        global_q_hat=float(global_q_hat),
        price_band_edges=[float(edge) for edge in price_band_edges],
        price_band_summary=price_band_summary,
        segment_summary=segment_summary,
        calibration_summary=calibration_summary,
    )
    return prediction_rows, artifacts


def build_prediction_intervals(predictions: pd.Series, q_hat: float | pd.Series) -> pd.DataFrame:
    q_hat_series = pd.Series(q_hat, index=predictions.index, dtype=float) if np.isscalar(q_hat) else pd.Series(q_hat)
    q_hat_series = q_hat_series.reindex(predictions.index)
    intervals = pd.DataFrame(index=predictions.index)
    intervals["fair_value_hat"] = predictions
    intervals["q_hat"] = q_hat_series
    intervals["lower_bound"] = predictions - q_hat_series
    intervals["upper_bound"] = predictions + q_hat_series
    intervals["interval_width"] = intervals["upper_bound"] - intervals["lower_bound"]
    return intervals


def evaluate_interval_quality(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> dict[str, float]:
    coverage = ((actual >= lower) & (actual <= upper)).mean()
    average_width = (upper - lower).mean()
    return {
        "empirical_coverage": float(coverage),
        "average_interval_width": float(average_width),
    }
