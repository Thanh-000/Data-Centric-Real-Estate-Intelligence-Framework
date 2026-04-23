from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dc_reif.utils import format_float


def _normalize_numeric_like(series: pd.Series, mapping: dict[str, int], uppercase: bool = True) -> pd.Series:
    stringified = series.astype("string").str.strip()
    if uppercase:
        mapped = stringified.str.upper().map(mapping)
    else:
        mapped = stringified.str.title().map(mapping)
    numeric = pd.to_numeric(stringified, errors="coerce")
    return numeric.fillna(mapped)


@dataclass
class ValidationReport:
    row_count: int
    column_count: int
    missing_columns: list[str]
    duplicate_rows: int
    missing_summary: dict[str, int]
    invalid_summary: dict[str, int]
    type_summary: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_columns": self.missing_columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_summary": self.missing_summary,
            "invalid_summary": self.invalid_summary,
            "type_summary": self.type_summary,
        }


def validate_schema(dataframe: pd.DataFrame, required_columns: list[str]) -> ValidationReport:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    duplicate_rows = int(dataframe.duplicated().sum())
    missing_summary = dataframe.isna().sum().sort_values(ascending=False).astype(int).to_dict()
    normalized = dataframe.copy()

    if "waterfront" in normalized.columns:
        normalized["waterfront_normalized"] = _normalize_numeric_like(
            normalized["waterfront"], {"NO": 0, "YES": 1}, uppercase=True
        )
    if "view" in normalized.columns:
        normalized["view_normalized"] = _normalize_numeric_like(
            normalized["view"],
            {"NONE": 0, "FAIR": 1, "AVERAGE": 2, "GOOD": 3, "EXCELLENT": 4},
            uppercase=True,
        )
    if "condition" in normalized.columns:
        normalized["condition_normalized"] = _normalize_numeric_like(
            normalized["condition"],
            {"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5},
            uppercase=False,
        )
    if "grade" in normalized.columns:
        grade_string = normalized["grade"].astype("string")
        numeric_grade = pd.to_numeric(grade_string, errors="coerce")
        extracted_grade = pd.to_numeric(grade_string.str.extract(r"(\d+)", expand=False), errors="coerce")
        normalized["grade_normalized"] = numeric_grade.fillna(extracted_grade)
    if "sqft_basement" in normalized.columns:
        normalized["sqft_basement"] = normalized["sqft_basement"].replace({"?": pd.NA})

    numeric_candidates = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
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
    invalid_summary: dict[str, int] = {}
    for column in numeric_candidates:
        if column not in normalized.columns:
            continue
        series = pd.to_numeric(normalized[column], errors="coerce")
        if column in {"price", "sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"}:
            invalid_summary[f"{column}_non_positive"] = int((series <= 0).fillna(False).sum())
        if column in {"bedrooms", "bathrooms", "floors", "grade", "sqft_basement", "yr_built", "yr_renovated"}:
            invalid_summary[f"{column}_negative"] = int((series < 0).fillna(False).sum())

    if "waterfront_normalized" in normalized.columns:
        invalid_summary["waterfront_invalid"] = int(
            (normalized["waterfront"].notna() & normalized["waterfront_normalized"].isna()).sum()
        )
    if "view_normalized" in normalized.columns:
        invalid_summary["view_invalid"] = int(
            (normalized["view"].notna() & normalized["view_normalized"].isna()).sum()
        )
    if "condition_normalized" in normalized.columns:
        invalid_summary["condition_invalid"] = int(
            (normalized["condition"].notna() & normalized["condition_normalized"].isna()).sum()
        )
    if "grade_normalized" in normalized.columns:
        invalid_summary["grade_invalid"] = int((normalized["grade"].notna() & normalized["grade_normalized"].isna()).sum())

    type_summary = {column: str(dtype) for column, dtype in dataframe.dtypes.items()}
    return ValidationReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        missing_columns=missing_columns,
        duplicate_rows=duplicate_rows,
        missing_summary=missing_summary,
        invalid_summary=invalid_summary,
        type_summary=type_summary,
    )


def validation_report_frame(report: ValidationReport) -> pd.DataFrame:
    missing_items = [{"metric": column, "value": count, "section": "missing"} for column, count in report.missing_summary.items()]
    invalid_items = [{"metric": metric, "value": count, "section": "invalid"} for metric, count in report.invalid_summary.items()]
    headline_items = [
        {"metric": "row_count", "value": report.row_count, "section": "overview"},
        {"metric": "column_count", "value": report.column_count, "section": "overview"},
        {"metric": "duplicate_rows", "value": report.duplicate_rows, "section": "overview"},
        {"metric": "missing_required_columns", "value": len(report.missing_columns), "section": "overview"},
    ]
    return pd.DataFrame(headline_items + missing_items + invalid_items)


def summarize_missingness(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_pct", ascending=False)
    )
    summary["missing_pct"] = summary["missing_pct"].map(lambda value: format_float(value, digits=2))
    return summary
