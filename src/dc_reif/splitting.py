from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    train_validation_df: pd.DataFrame


def chronological_split(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitBundle:
    if round(train_fraction + validation_fraction + test_fraction, 5) != 1.0:
        raise ValueError("train, validation, and test fractions must sum to 1.0")

    ordered = dataframe.sort_values([date_column, "id"]).reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)

    train_df = ordered.iloc[:train_end].copy()
    validation_df = ordered.iloc[train_end:validation_end].copy()
    test_df = ordered.iloc[validation_end:].copy()
    train_validation_df = ordered.iloc[:validation_end].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    return SplitBundle(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        train_validation_df=train_validation_df,
    )


def make_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)

