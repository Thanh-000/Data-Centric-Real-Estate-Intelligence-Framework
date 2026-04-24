from __future__ import annotations

from dc_reif.config import REQUIRED_COLUMNS


REQUIRED_SCHEMA_CONTRACT = {
    "dataset": "King County House Sales",
    "required_columns": REQUIRED_COLUMNS,
    "target_column": "price",
    "date_column": "date",
    "policy": {
        "target_derived_features_forbidden_in_predictive_path": True,
        "train_only_preprocessing_required": True,
        "oof_fair_values_required_for_anomaly_scoring": True,
    },
    "notes": [
        "Sale-price dataset only; do not relabel outputs as strict asking-price mispricing.",
        "Target-derived variables are excluded from the predictive branch.",
    ],
}


def build_schema_contract() -> dict[str, object]:
    return REQUIRED_SCHEMA_CONTRACT.copy()
