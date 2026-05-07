import pandas as pd

from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.feature_engineering import build_feature_matrix
from dc_reif.splitting import chronological_split


def test_chronological_split_preserves_time_order(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    features = build_feature_matrix(cleaned).dataframe
    split = chronological_split(features)

    assert split.train_df["date"].max() <= split.validation_df["date"].min()
    assert split.validation_df["date"].max() <= split.test_df["date"].min()
    assert len(split.train_df) + len(split.validation_df) + len(split.test_df) == len(features)


def test_chronological_split_uses_date_cutoffs_when_available(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    features = build_feature_matrix(cleaned).dataframe
    split = chronological_split(features, train_end_date="2014-06-15", validation_end_date="2014-07-15")

    assert split.train_df["date"].max() <= pd.Timestamp("2014-06-15")
    assert split.validation_df["date"].min() > pd.Timestamp("2014-06-15")
    assert split.validation_df["date"].max() <= pd.Timestamp("2014-07-15")
    assert split.test_df["date"].min() > pd.Timestamp("2014-07-15")
