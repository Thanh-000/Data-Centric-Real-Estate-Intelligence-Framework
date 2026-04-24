from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pricing_anomalies(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["valuation_gap"] = df["observed_price"] - df["fair_value_hat"]
    safe_width = df["interval_width"].replace(0, np.nan)
    df["anomaly_score"] = df["valuation_gap"] / safe_width

    df["anomaly_flag"] = np.select(
        [
            df["fair_value_hat"].isna(),
            df["observed_price"] < df["lower_bound"],
            df["observed_price"] > df["upper_bound"],
        ],
        [
            "insufficient_history",
            "potentially_under_valued",
            "potentially_over_valued",
        ],
        default="within_expected_range",
    )
    return df


def enrich_pricing_anomalies(
    dataframe: pd.DataFrame,
    global_q_hat: float,
    min_segment_support: int,
    min_price_band_support: int,
) -> pd.DataFrame:
    df = dataframe.copy()
    support_components = []
    if "segment_support_n" in df.columns:
        support_components.append((df["segment_support_n"] / max(min_segment_support, 1)).clip(upper=1.0))
    if "price_band_support_n" in df.columns:
        support_components.append((df["price_band_support_n"] / max(min_price_band_support, 1)).clip(upper=1.0))
    if support_components:
        df["support_score"] = pd.concat(support_components, axis=1).mean(axis=1).round(3)
    else:
        df["support_score"] = 0.0

    local_ratio = (df.get("q_hat", global_q_hat) / max(global_q_hat, 1e-9)).astype(float)
    df["slice_risk_level"] = np.select(
        [
            df["anomaly_flag"].eq("insufficient_history"),
            (local_ratio >= 1.2) | (df["support_score"] < 0.55),
            (local_ratio >= 1.05) | (df["support_score"] < 0.75),
        ],
        [
            "high",
            "high",
            "medium",
        ],
        default="low",
    )

    df["evidence_strength"] = np.select(
        [
            df["anomaly_flag"].eq("insufficient_history"),
            (df["anomaly_score"].abs() >= 0.18) & (df["support_score"] >= 0.75),
            df["anomaly_score"].abs() >= 0.10,
        ],
        [
            "insufficient_history",
            "strong",
            "moderate",
        ],
        default="limited",
    )

    breach_low = (df["lower_bound"] - df["observed_price"]).clip(lower=0).fillna(0.0)
    breach_high = (df["observed_price"] - df["upper_bound"]).clip(lower=0).fillna(0.0)

    confidence_notes: list[str] = []
    why_flagged: list[str] = []
    for row in df.itertuples(index=False):
        if row.anomaly_flag == "insufficient_history":
            confidence_notes.append("Fair value is withheld because train-fold support is insufficient for a stable anomaly assessment.")
            why_flagged.append("Insufficient historical support prevented a reliable fair-value estimate.")
            continue

        if row.anomaly_flag == "potentially_over_valued":
            breach_value = float(getattr(row, "observed_price") - getattr(row, "upper_bound"))
            why_flagged.append(
                f"Observed sale price exceeds the upper expected range by {breach_value:,.0f} within the modeled uncertainty band."
            )
        elif row.anomaly_flag == "potentially_under_valued":
            breach_value = float(getattr(row, "lower_bound") - getattr(row, "observed_price"))
            why_flagged.append(
                f"Observed sale price falls below the lower expected range by {breach_value:,.0f} within the modeled uncertainty band."
            )
        else:
            why_flagged.append("Observed sale price remains inside the expected valuation interval.")

        if row.slice_risk_level == "high":
            confidence_notes.append("Interpret this flag cautiously because the local slice remains difficult or lightly supported.")
        elif row.slice_risk_level == "medium":
            confidence_notes.append("Interpret this flag with moderate caution because the local slice is more variable than the global market average.")
        else:
            confidence_notes.append("Local support is comparatively stable for this slice, so the anomaly signal is relatively well supported.")

    df["confidence_note"] = confidence_notes
    df["why_flagged"] = why_flagged
    df["support_score"] = df["support_score"].fillna(0.0)
    return df
