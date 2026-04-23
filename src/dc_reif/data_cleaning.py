from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CRITICAL_COLUMNS = ["id", "date", "price", "sqft_living", "lat", "long"]
NUMERIC_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

WATERFRONT_MAP = {"NO": 0, "YES": 1}
VIEW_MAP = {"NONE": 0, "FAIR": 1, "AVERAGE": 2, "GOOD": 3, "EXCELLENT": 4}
CONDITION_MAP = {"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5}


def _coerce_numeric_or_map(series: pd.Series, mapping: dict[str, int], uppercase: bool = True) -> pd.Series:
    stringified = series.astype("string").str.strip()
    if uppercase:
        mapped = stringified.str.upper().map(mapping)
    else:
        mapped = stringified.str.title().map(mapping)
    numeric = pd.to_numeric(stringified, errors="coerce")
    return numeric.fillna(mapped)


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame
    summary: dict[str, int | float]


def _iqr_flag(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def clean_king_county_data(dataframe: pd.DataFrame) -> CleaningResult:
    df = dataframe.copy()
    df.columns = [column.strip().lower() for column in df.columns]

    if "waterfront" in df.columns:
        df["waterfront"] = _coerce_numeric_or_map(df["waterfront"], WATERFRONT_MAP, uppercase=True)
    if "view" in df.columns:
        df["view"] = _coerce_numeric_or_map(df["view"], VIEW_MAP, uppercase=True)
    if "condition" in df.columns:
        df["condition"] = _coerce_numeric_or_map(df["condition"], CONDITION_MAP, uppercase=False)
    if "grade" in df.columns:
        grade_string = df["grade"].astype("string")
        numeric_grade = pd.to_numeric(grade_string, errors="coerce")
        extracted_grade = grade_string.str.extract(r"(\d+)", expand=False)
        df["grade"] = numeric_grade.fillna(pd.to_numeric(extracted_grade, errors="coerce"))
    if "sqft_basement" in df.columns:
        df["sqft_basement"] = df["sqft_basement"].replace({"?": np.nan})

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype("string")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_duplicates - len(df)

    df["is_missing_critical"] = df[CRITICAL_COLUMNS].isna().any(axis=1)
    df["is_invalid_price"] = df["price"].le(0) | df["price"].isna()
    df["is_invalid_area"] = df["sqft_living"].le(0) | df["sqft_living"].isna()
    df["is_invalid_geo"] = df["lat"].isna() | df["long"].isna()
    df["is_invalid_record"] = (
        df["is_missing_critical"] | df["is_invalid_price"] | df["is_invalid_area"] | df["is_invalid_geo"]
    )

    df["is_outlier_price"] = _iqr_flag(df["price"])
    df["is_outlier_area"] = _iqr_flag(df["sqft_living"])
    df["is_suspect_record"] = df["is_outlier_price"] | df["is_outlier_area"]
    df["data_quality_flag"] = np.where(df["is_invalid_record"], "invalid", np.where(df["is_suspect_record"], "suspect", "ok"))

    cleaned = df.loc[~df["is_invalid_record"]].copy()
    cleaned = cleaned.sort_values(["date", "id"]).reset_index(drop=True)

    summary = {
        "rows_in": int(len(dataframe)),
        "rows_after_deduplication": int(len(df)),
        "duplicates_removed": int(duplicates_removed),
        "rows_dropped_invalid": int(df["is_invalid_record"].sum()),
        "rows_flagged_suspect": int(cleaned["is_suspect_record"].sum()),
        "rows_out": int(len(cleaned)),
    }
    return CleaningResult(dataframe=cleaned, summary=summary)
