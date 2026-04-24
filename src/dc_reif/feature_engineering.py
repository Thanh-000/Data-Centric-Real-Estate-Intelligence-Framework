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


BASE_PREDICTIVE_FEATURES = [
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

ENHANCED_PREDICTIVE_FEATURES = [
    "total_sqft",
    "living_to_lot_ratio",
    "basement_share",
    "bathrooms_per_bedroom",
    "sqft_per_floor",
    "relative_living_area",
    "relative_lot_size",
    "sale_month_sin",
    "sale_month_cos",
    "build_decade",
    "renovation_recency",
    "lat_bin",
    "long_bin",
    "geo_cell",
    "distance_to_seattle_core",
    "distance_to_bellevue_core",
    "grade_living_interaction",
    "waterfront_view_score",
    "location_grade_interaction",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _series_or_default(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in dataframe.columns:
        return dataframe[column]
    return pd.Series(default, index=dataframe.index, dtype=float)


def _haversine_distance_km(
    lat_series: pd.Series,
    lon_series: pd.Series,
    anchor_lat: float,
    anchor_lon: float,
) -> pd.Series:
    earth_radius_km = 6371.0
    lat1 = np.radians(lat_series.astype(float))
    lon1 = np.radians(lon_series.astype(float))
    lat2 = np.radians(anchor_lat)
    lon2 = np.radians(anchor_lon)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius_km * c


def add_safe_derived_features(dataframe: pd.DataFrame) -> pd.DataFrame:
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
    df["total_sqft"] = df["sqft_above"].fillna(0) + df["sqft_basement"].fillna(0)
    df["living_to_lot_ratio"] = _safe_ratio(df["sqft_living"], df["sqft_lot"])
    df["basement_share"] = _safe_ratio(df["sqft_basement"], df["total_sqft"])
    df["bathrooms_per_bedroom"] = _safe_ratio(df["bathrooms"], df["bedrooms"])
    df["sqft_per_floor"] = _safe_ratio(df["sqft_living"], df["floors"])
    df["relative_living_area"] = _safe_ratio(df["sqft_living"], df["sqft_living15"])
    df["relative_lot_size"] = _safe_ratio(df["sqft_lot"], df["sqft_lot15"])
    month_angle = 2 * np.pi * (df["sale_month"] - 1) / 12
    df["sale_month_sin"] = np.sin(month_angle)
    df["sale_month_cos"] = np.cos(month_angle)
    df["build_decade"] = ((df["yr_built"] // 10) * 10).astype("Int64").astype("string")
    df["renovation_recency"] = np.where(
        df["renovated_flag"].eq(1),
        (df["sale_year"] - df["yr_renovated"]).clip(lower=0),
        df["house_age"],
    )
    df["lat_bin"] = df["lat"].round(1).astype("string")
    df["long_bin"] = df["long"].round(1).astype("string")
    df["geo_cell"] = df["lat"].round(2).astype("string") + "_" + df["long"].round(2).astype("string")
    df["distance_to_seattle_core"] = _haversine_distance_km(df["lat"], df["long"], 47.6062, -122.3321)
    df["distance_to_bellevue_core"] = _haversine_distance_km(df["lat"], df["long"], 47.6101, -122.2015)
    df["grade_living_interaction"] = _series_or_default(df, "grade") * _series_or_default(df, "sqft_living")
    df["waterfront_view_score"] = _series_or_default(df, "waterfront").fillna(0) + _series_or_default(df, "view").fillna(0)
    df["location_grade_interaction"] = _series_or_default(df, "grade") * _series_or_default(df, "lat")
    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, np.nan)
    return df


def build_feature_matrix(dataframe: pd.DataFrame, include_enhanced_features: bool = False) -> FeatureSet:
    df = add_safe_derived_features(dataframe)
    predictive_features = BASE_PREDICTIVE_FEATURES.copy()
    if include_enhanced_features:
        predictive_features.extend(ENHANCED_PREDICTIVE_FEATURES)
    descriptive_features = ["price_per_sqft"]
    return FeatureSet(dataframe=df, predictive_features=predictive_features, descriptive_features=descriptive_features)


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(TARGET_DERIVED_FEATURES.intersection(feature_columns))
    if leaked:
        raise ValueError(f"Target-derived features detected in predictive branch: {leaked}")
