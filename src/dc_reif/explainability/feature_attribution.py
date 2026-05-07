from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from dc_reif.utils import get_logger

LOGGER = get_logger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
