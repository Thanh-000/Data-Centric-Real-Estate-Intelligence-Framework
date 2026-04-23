import pandas as pd

from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.feature_engineering import build_feature_matrix
from dc_reif.preprocessing import build_preprocessor


def test_preprocessor_handles_unseen_categories_after_train_fit(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    feature_set = build_feature_matrix(cleaned)
    feature_columns = feature_set.predictive_features

    train_df = feature_set.dataframe.iloc[:80].copy()
    validation_df = feature_set.dataframe.iloc[80:100].copy()
    validation_df["zipcode"] = "99999"

    preprocessing = build_preprocessor(train_df, feature_columns, scale_numeric=False)
    transformer = preprocessing.transformer.fit(train_df[feature_columns])
    transformed = transformer.transform(validation_df[feature_columns])

    assert transformed.shape[0] == len(validation_df)

