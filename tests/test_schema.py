from dc_reif.config import REQUIRED_COLUMNS
from dc_reif.data_validation import validate_schema


def test_schema_validation_accepts_required_columns(sample_dataframe):
    report = validate_schema(sample_dataframe, REQUIRED_COLUMNS)
    assert report.missing_columns == []
    assert report.duplicate_rows == 0
    assert report.row_count == len(sample_dataframe)

