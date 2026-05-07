from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    pipeline: object | None
    feature_columns: list[str]
    n_clusters: int
    min_keep_cluster: int
    min_local_cluster: int
    cluster_mapping: dict[str, str]
    silhouette: float
    davies_bouldin: float
    cluster_profiles: pd.DataFrame
    selection_summary: pd.DataFrame
    selection_details: dict[str, Any]
    segmentation_method: str = "zipcode_market"


def _zipcode_labels(dataframe: pd.DataFrame, min_keep_cluster: int) -> pd.Series:
    zipcodes = dataframe["zipcode"].astype(str)
    counts = zipcodes.value_counts()
    if (counts >= min_keep_cluster).any():
        kept = set(counts.loc[counts >= min_keep_cluster].index)
        labels = zipcodes.where(zipcodes.isin(kept), "other")
    else:
        labels = zipcodes
    return ("segment_zipcode_" + labels).astype("string")


def _spatial_validation_scores(dataframe: pd.DataFrame, labels: pd.Series) -> tuple[float, float]:
    if labels.nunique(dropna=True) < 2 or len(dataframe) <= labels.nunique(dropna=True):
        return float("nan"), float("nan")
    coordinates = dataframe[["lat", "long"]].astype(float)
    scaled = StandardScaler().fit_transform(coordinates)
    try:
        silhouette = float(silhouette_score(scaled, labels.astype(str)))
        davies_bouldin = float(davies_bouldin_score(scaled, labels.astype(str)))
    except ValueError:
        return float("nan"), float("nan")
    return silhouette, davies_bouldin


def fit_submarket_clustering(
    train_df: pd.DataFrame,
    random_state: int = 42,
    include_enhanced_features: bool = False,
) -> ClusteringArtifacts:
    _ = random_state, include_enhanced_features
    n_train = len(train_df)
    min_keep_cluster = max(10, math.ceil(0.01 * n_train))
    min_local_cluster = max(50, math.ceil(0.03 * n_train))

    labels = _zipcode_labels(train_df, min_keep_cluster=min_keep_cluster)
    counts = labels.value_counts().sort_values(ascending=False)
    silhouette, dbi = _spatial_validation_scores(train_df, labels)
    balance_score = float(counts.min() / counts.max()) if len(counts) and counts.max() else 0.0
    small_cluster_share = float(counts.loc[counts < min_keep_cluster].sum() / n_train) if n_train else 0.0
    selection_score = (
        (0.0 if np.isnan(silhouette) else silhouette)
        - 0.12 * (0.0 if np.isnan(dbi) else dbi)
        + 0.20 * balance_score
        - 0.10 * small_cluster_share
    )

    selection_summary = pd.DataFrame(
        [
            {
                "segmentation_method": "zipcode_market",
                "k": int(labels.nunique()),
                "silhouette_score": silhouette,
                "davies_bouldin_index": dbi,
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "balance_score": balance_score,
                "small_cluster_share": small_cluster_share,
                "selection_score": float(selection_score),
            }
        ]
    )

    profiled = train_df.copy()
    profiled["segment_label"] = labels
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
        "prior_zipcode_median_price",
        "prior_neighbor_median_price",
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
    zipcode_to_segment = train_df.assign(_zipcode=train_df["zipcode"].astype(str), _segment=labels).set_index("_zipcode")[
        "_segment"
    ].to_dict()

    return ClusteringArtifacts(
        pipeline=None,
        feature_columns=["zipcode", "lat", "long"],
        n_clusters=int(labels.nunique()),
        min_keep_cluster=min_keep_cluster,
        min_local_cluster=min_local_cluster,
        cluster_mapping={str(key): str(value) for key, value in zipcode_to_segment.items()},
        silhouette=silhouette,
        davies_bouldin=dbi,
        cluster_profiles=cluster_profiles,
        selection_summary=selection_summary,
        selection_details={
            "segmentation_method": "zipcode_market",
            "k": int(labels.nunique()),
            "silhouette_score": silhouette,
            "davies_bouldin_index": dbi,
            "balance_score": balance_score,
            "small_cluster_share": small_cluster_share,
            "selection_score": float(selection_score),
            "rationale": "Zipcode market grouping replaces KMeans because residential submarkets are spatial and administrative.",
        },
    )


def assign_submarket_segments(dataframe: pd.DataFrame, artifacts: ClusteringArtifacts) -> pd.Series:
    mapped = dataframe["zipcode"].astype(str).map(artifacts.cluster_mapping).fillna("segment_zipcode_other")
    return mapped.astype("string")
