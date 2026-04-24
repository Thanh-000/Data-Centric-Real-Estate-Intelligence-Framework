from __future__ import annotations

from dc_reif.config import DESCRIPTIVE_ONLY_FEATURES


def descriptive_only_features() -> list[str]:
    return DESCRIPTIVE_ONLY_FEATURES.copy()


def predictive_feature_policy() -> dict[str, object]:
    return {
        "target_derived_features_forbidden": True,
        "train_only_preprocessing_required": True,
        "descriptive_only_features": descriptive_only_features(),
    }

