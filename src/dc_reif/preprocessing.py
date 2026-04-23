from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingSpec:
    numeric_features: list[str]
    categorical_features: list[str]
    transformer: ColumnTransformer


def infer_feature_types(dataframe: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)
    return numeric_features, categorical_features


def build_preprocessor(dataframe: pd.DataFrame, feature_columns: list[str], scale_numeric: bool = False) -> PreprocessingSpec:
    numeric_features, categorical_features = infer_feature_types(dataframe, feature_columns)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return PreprocessingSpec(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        transformer=transformer,
    )

