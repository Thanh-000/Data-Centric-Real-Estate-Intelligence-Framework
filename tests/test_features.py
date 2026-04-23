from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.feature_engineering import assert_no_target_leakage, build_feature_matrix


def test_feature_engineering_keeps_target_derived_features_out_of_predictive_branch(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    feature_set = build_feature_matrix(cleaned)

    assert "price_per_sqft" in feature_set.descriptive_features
    assert "price_per_sqft" not in feature_set.predictive_features
    assert_no_target_leakage(feature_set.predictive_features)

