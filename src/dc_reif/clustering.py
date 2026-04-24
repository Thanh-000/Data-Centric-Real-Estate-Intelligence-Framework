from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    pipeline: Pipeline
    feature_columns: list[str]
    n_clusters: int
    min_keep_cluster: int
    min_local_cluster: int
    cluster_mapping: dict[int, str]
    silhouette: float
    davies_bouldin: float
    cluster_profiles: pd.DataFrame
    selection_summary: pd.DataFrame
    selection_details: dict[str, Any]


def _segmentation_features(include_enhanced_features: bool = False) -> list[str]:
    features = [
        "sqft_living",
        "sqft_lot",
        "bedrooms",
        "bathrooms",
        "floors",
        "grade",
        "condition",
        "house_age",
        "lat",
        "long",
        "sqft_living15",
    ]
    if include_enhanced_features:
        features.extend(
            [
                "total_sqft",
                "living_to_lot_ratio",
                "bathrooms_per_bedroom",
                "relative_living_area",
                "renovated_flag",
                "distance_to_seattle_core",
                "distance_to_bellevue_core",
                "waterfront_view_score",
                "grade_living_interaction",
            ]
        )
    return features


def _selection_score(
    silhouette: float,
    davies_bouldin: float,
    balance_score: float,
    small_cluster_share: float,
    k: int,
) -> float:
    return silhouette - 0.12 * davies_bouldin + 0.20 * balance_score - 0.10 * small_cluster_share


def fit_submarket_clustering(
    train_df: pd.DataFrame,
    random_state: int = 42,
    include_enhanced_features: bool = False,
) -> ClusteringArtifacts:
    feature_columns = [column for column in _segmentation_features(include_enhanced_features) if column in train_df.columns]
    n_train = len(train_df)
    min_keep_cluster = max(30, math.ceil(0.01 * n_train))
    min_local_cluster = max(80, math.ceil(0.03 * n_train))

    best_details: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None
    best_labels: pd.Series | None = None
    selection_rows: list[dict[str, Any]] = []

    for k in range(3, 11):
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, n_init=30, random_state=random_state)),
            ]
        )
        imputed = pipeline.named_steps["imputer"].fit_transform(train_df[feature_columns])
        transformed = pipeline.named_steps["scaler"].fit_transform(imputed)
        labels = pipeline.named_steps["kmeans"].fit_predict(transformed)

        if len(set(labels)) < 2:
            continue

        label_series = pd.Series(labels, index=train_df.index)
        counts = label_series.value_counts().sort_values(ascending=False)
        silhouette = float(silhouette_score(transformed, labels))
        dbi = float(davies_bouldin_score(transformed, labels))
        min_cluster_size = int(counts.min())
        max_cluster_size = int(counts.max())
        balance_score = float(min_cluster_size / max_cluster_size) if max_cluster_size else 0.0
        small_cluster_share = float(counts.loc[counts < min_keep_cluster].sum() / len(train_df))
        score = _selection_score(silhouette, dbi, balance_score, small_cluster_share, k)

        selection_rows.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "davies_bouldin_index": dbi,
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size,
                "balance_score": balance_score,
                "small_cluster_share": small_cluster_share,
                "selection_score": score,
            }
        )

        details = {
            "k": k,
            "silhouette_score": silhouette,
            "davies_bouldin_index": dbi,
            "balance_score": balance_score,
            "small_cluster_share": small_cluster_share,
            "selection_score": score,
        }
        if best_details is None or score > best_details["selection_score"]:
            best_details = details
            best_pipeline = Pipeline(
                steps=[
                    ("imputer", pipeline.named_steps["imputer"]),
                    ("scaler", pipeline.named_steps["scaler"]),
                    ("kmeans", pipeline.named_steps["kmeans"]),
                ]
            )
            best_labels = label_series

    if best_pipeline is None or best_labels is None or best_details is None:
        raise ValueError("Clustering failed to produce a usable segmentation.")

    selection_summary = pd.DataFrame(selection_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    counts = best_labels.value_counts()
    cluster_mapping = {
        int(label): (f"segment_{int(label)}" if size >= min_keep_cluster else "segment_other")
        for label, size in counts.items()
    }

    profiled = train_df.copy()
    profiled["segment_label"] = best_labels.map(cluster_mapping)
    profile_columns = [
        "price",
        "sqft_living",
        "sqft_lot",
        "grade",
        "bathrooms",
        "bedrooms",
        "condition",
        "house_age",
        "lat",
        "long",
        "distance_to_seattle_core",
        "distance_to_bellevue_core",
        "waterfront_view_score",
    ]
    available_profile_columns = [column for column in profile_columns if column in profiled.columns]
    cluster_profiles = (
        profiled.groupby("segment_label")[available_profile_columns]
        .median()
        .assign(count=profiled.groupby("segment_label").size())
        .assign(share=lambda frame: frame["count"] / frame["count"].sum())
        .reset_index()
        .rename(columns={"price": "median_observed_price"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return ClusteringArtifacts(
        pipeline=best_pipeline,
        feature_columns=feature_columns,
        n_clusters=int(best_details["k"]),
        min_keep_cluster=min_keep_cluster,
        min_local_cluster=min_local_cluster,
        cluster_mapping=cluster_mapping,
        silhouette=float(best_details["silhouette_score"]),
        davies_bouldin=float(best_details["davies_bouldin_index"]),
        cluster_profiles=cluster_profiles,
        selection_summary=selection_summary,
        selection_details=best_details,
    )


def assign_submarket_segments(dataframe: pd.DataFrame, artifacts: ClusteringArtifacts) -> pd.Series:
    numeric = artifacts.pipeline.named_steps["imputer"].transform(dataframe[artifacts.feature_columns])
    scaled = artifacts.pipeline.named_steps["scaler"].transform(numeric)
    labels = artifacts.pipeline.named_steps["kmeans"].predict(scaled)
    mapped = pd.Series(labels, index=dataframe.index).map(artifacts.cluster_mapping).fillna("segment_other")
    return mapped.astype("string")
