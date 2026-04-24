from __future__ import annotations

import math

import numpy as np
import pandas as pd


def conformal_quantile(residuals: pd.Series, alpha: float = 0.1) -> float:
    absolute_residuals = residuals.abs().dropna().sort_values().to_numpy()
    if absolute_residuals.size == 0:
        raise ValueError("Residuals are required to estimate uncertainty intervals.")
    quantile_rank = math.ceil((absolute_residuals.size + 1) * (1 - alpha)) / absolute_residuals.size
    quantile_rank = min(max(quantile_rank, 0.0), 1.0)
    return float(np.quantile(absolute_residuals, quantile_rank, method="higher"))


def build_prediction_intervals(predictions: pd.Series, q_hat: float) -> pd.DataFrame:
    intervals = pd.DataFrame(index=predictions.index)
    intervals["fair_value_hat"] = predictions
    intervals["lower_bound"] = predictions - q_hat
    intervals["upper_bound"] = predictions + q_hat
    intervals["interval_width"] = intervals["upper_bound"] - intervals["lower_bound"]
    return intervals


def evaluate_interval_quality(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> dict[str, float]:
    coverage = ((actual >= lower) & (actual <= upper)).mean()
    average_width = (upper - lower).mean()
    return {
        "empirical_coverage": float(coverage),
        "average_interval_width": float(average_width),
    }

