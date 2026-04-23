from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dc_reif.utils import get_logger

LOGGER = get_logger(__name__)


def _feature_names(model_pipeline) -> list[str]:
    names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [name.replace("numeric__", "").replace("categorical__", "") for name in names]


def global_feature_importance(model_pipeline, model_name: str) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    names = _feature_names(model_pipeline)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:
        raise ValueError(f"Model {model_name} does not expose supported importance attributes.")

    importance_df = (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> Path:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"], color="#2f4858")
    ax.set_title("DC-REIF Global Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def shap_explanations(
    model_pipeline,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    local_sample_ids: list[str] | None = None,
    id_column: str = "id",
    sample_size: int = 300,
) -> tuple[Path | None, dict[str, str]]:
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SHAP not available: %s", exc)
        return None, {}

    if not hasattr(model_pipeline.named_steps["model"], "feature_importances_"):
        return None, {}

    sample_df = dataset.copy()
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample_df[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    names = _feature_names(model_pipeline)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    plt.figure()
    shap.summary_plot(shap_values, transformed, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    local_map: dict[str, str] = {}
    if local_sample_ids:
        lookup = dataset[dataset[id_column].astype(str).isin(local_sample_ids)].copy()
        if not lookup.empty:
            local_transformed = preprocessor.transform(lookup[feature_columns])
            if hasattr(local_transformed, "toarray"):
                local_transformed = local_transformed.toarray()
            local_values = explainer.shap_values(local_transformed)
            for row_index, property_id in enumerate(lookup[id_column].astype(str)):
                contributions = pd.Series(np.abs(local_values[row_index]), index=names).sort_values(ascending=False)
                local_map[property_id] = ", ".join(contributions.head(3).index.tolist())

    return output_path, local_map


def build_top_driver_map(
    dataframe: pd.DataFrame,
    id_column: str,
    importance_df: pd.DataFrame,
    local_driver_map: dict[str, str] | None = None,
) -> pd.Series:
    default_drivers = ", ".join(importance_df["feature"].head(3).tolist())
    series = pd.Series(default_drivers, index=dataframe.index, name="top_drivers")
    if local_driver_map:
        for idx, property_id in dataframe[id_column].astype(str).items():
            if property_id in local_driver_map:
                series.loc[idx] = local_driver_map[property_id]
    return series

