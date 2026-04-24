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

