from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dc_reif.config import TARGET_DERIVED_FEATURES


@dataclass
class FeatureSet:
    dataframe: pd.DataFrame
    predictive_features: list[str]
    descriptive_features: list[str]


def build_feature_matrix(dataframe: pd.DataFrame) -> FeatureSet:
    df = dataframe.copy()
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["house_age"] = (df["sale_year"] - df["yr_built"]).clip(lower=0)
    df["renovated_flag"] = (df["yr_renovated"].fillna(0) > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["renovated_flag"].eq(1),
        (df["sale_year"] - df["yr_renovated"]).clip(lower=0),
        np.nan,
    )
    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, np.nan)

    predictive_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
        "sale_year",
        "sale_month",
        "sale_quarter",
        "house_age",
        "renovated_flag",
        "years_since_renovation",
    ]
    descriptive_features = ["price_per_sqft"]
    return FeatureSet(dataframe=df, predictive_features=predictive_features, descriptive_features=descriptive_features)


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(TARGET_DERIVED_FEATURES.intersection(feature_columns))
    if leaked:
        raise ValueError(f"Target-derived features detected in predictive branch: {leaked}")

