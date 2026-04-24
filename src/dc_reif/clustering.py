from __future__ import annotations

import math
from dataclasses import dataclass

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


def _cluster_features(include_enhanced_features: bool = False) -> list[str]:
    feature_columns = [
        "sqft_living",
        "sqft_lot",
        "bedrooms",
        "bathrooms",
        "grade",
        "condition",
        "house_age",
        "lat",
        "long",
    ]
    if include_enhanced_features:
        feature_columns.extend(
            [
                "living_to_lot_ratio",
                "basement_share",
                "bathrooms_per_bedroom",
                "relative_living_area",
                "renovated_flag",
                "distance_to_seattle_core",
                "distance_to_bellevue_core",
                "waterfront_view_score",
            ]
        )
    return feature_columns


def fit_submarket_clustering(
    train_df: pd.DataFrame,
    random_state: int = 42,
    include_enhanced_features: bool = False,
) -> ClusteringArtifacts:
    feature_columns = [column for column in _cluster_features(include_enhanced_features) if column in train_df.columns]
    n_train = len(train_df)
    min_keep_cluster = max(30, math.ceil(0.01 * n_train))
    min_local_cluster = max(80, math.ceil(0.03 * n_train))

    best_score = float("-inf")
    best_pipeline: Pipeline | None = None
    best_labels: pd.Series | None = None
    best_k = 3
    best_davies_bouldin = float("inf")

    for k in range(3, 7):
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
            ]
        )
        transformed = pipeline.named_steps["scaler"].fit_transform(
            pipeline.named_steps["imputer"].fit_transform(train_df[feature_columns])
        )
        labels = pipeline.named_steps["kmeans"].fit_predict(transformed)

        if len(set(labels)) < 2:
            continue

        silhouette = silhouette_score(transformed, labels)
        dbi = davies_bouldin_score(transformed, labels)
        if silhouette > best_score:
            best_score = silhouette
            best_davies_bouldin = dbi
            best_pipeline = Pipeline(
                steps=[
                    ("imputer", pipeline.named_steps["imputer"]),
                    ("scaler", pipeline.named_steps["scaler"]),
                    ("kmeans", pipeline.named_steps["kmeans"]),
                ]
            )
            best_labels = pd.Series(labels, index=train_df.index)
            best_k = k

    if best_pipeline is None or best_labels is None:
        raise ValueError("Clustering failed to produce a usable segmentation.")

    counts = best_labels.value_counts()
    cluster_mapping = {
        int(label): (f"segment_{int(label)}" if size >= min_keep_cluster else "segment_other")
        for label, size in counts.items()
    }

    profiled = train_df.copy()
    profiled["segment_label"] = best_labels.map(cluster_mapping)
    cluster_profiles = (
        profiled.groupby("segment_label")[
            ["sqft_living", "grade", "condition", "house_age", "lat", "long"]
        ]
        .median()
        .assign(count=profiled.groupby("segment_label").size())
        .reset_index()
        .sort_values("count", ascending=False)
    )

    return ClusteringArtifacts(
        pipeline=best_pipeline,
        feature_columns=feature_columns,
        n_clusters=best_k,
        min_keep_cluster=min_keep_cluster,
        min_local_cluster=min_local_cluster,
        cluster_mapping=cluster_mapping,
        silhouette=float(best_score),
        davies_bouldin=float(best_davies_bouldin),
        cluster_profiles=cluster_profiles,
    )


def assign_submarket_segments(dataframe: pd.DataFrame, artifacts: ClusteringArtifacts) -> pd.Series:
    numeric = artifacts.pipeline.named_steps["imputer"].transform(dataframe[artifacts.feature_columns])
    scaled = artifacts.pipeline.named_steps["scaler"].transform(numeric)
    labels = artifacts.pipeline.named_steps["kmeans"].predict(scaled)
    mapped = pd.Series(labels, index=dataframe.index).map(artifacts.cluster_mapping).fillna("segment_other")
    return mapped.astype("string")
