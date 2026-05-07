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
    train_end_date: str | pd.Timestamp | None = None,
    validation_end_date: str | pd.Timestamp | None = None,
) -> SplitBundle:
    if round(train_fraction + validation_fraction + test_fraction, 5) != 1.0:
        raise ValueError("train, validation, and test fractions must sum to 1.0")

    ordered = dataframe.sort_values([date_column, "id"]).reset_index(drop=True)
    if train_end_date is not None and validation_end_date is not None:
        train_cutoff = pd.Timestamp(train_end_date)
        validation_cutoff = pd.Timestamp(validation_end_date)
        if train_cutoff >= validation_cutoff:
            raise ValueError("train_end_date must be earlier than validation_end_date")

        train_df = ordered.loc[ordered[date_column] <= train_cutoff].copy()
        validation_df = ordered.loc[(ordered[date_column] > train_cutoff) & (ordered[date_column] <= validation_cutoff)].copy()
        test_df = ordered.loc[ordered[date_column] > validation_cutoff].copy()
        train_validation_df = ordered.loc[ordered[date_column] <= validation_cutoff].copy()

        if not train_df.empty and not validation_df.empty and not test_df.empty:
            return SplitBundle(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                train_validation_df=train_validation_df,
            )

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
