from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PropertyLedgerRecord:
    """Current student-scale property ledger record.

    The implemented ledger is the property-level output table produced by the
    frozen DC-REIF workflow. Future ledger extensions may add cross-source
    history, revision tracking, and richer entity linkage.
    """

    property_id: str
    observed_price: float
    segment_label: str
    anomaly_flag: str


def build_property_ledger(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the current property-ledger view from an existing property table."""

    expected_columns = [
        "property_id",
        "sale_date",
        "zipcode",
        "observed_price",
        "fair_value_hat",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "segment_label",
        "sqft_living",
        "grade",
        "house_age",
        "valuation_gap",
        "anomaly_flag",
        "anomaly_score",
        "top_drivers",
        "data_quality_flag",
    ]
    return dataframe.loc[:, [column for column in expected_columns if column in dataframe.columns]].copy()


def load_property_ledger_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing property-ledger snapshot at {path}")
    return build_property_ledger(pd.read_csv(path))
