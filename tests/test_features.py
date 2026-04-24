from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.feature_engineering import assert_no_target_leakage, build_feature_matrix


def test_feature_engineering_keeps_target_derived_features_out_of_predictive_branch(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    feature_set = build_feature_matrix(cleaned)

    assert "price_per_sqft" in feature_set.descriptive_features
    assert "price_per_sqft" not in feature_set.predictive_features
    assert_no_target_leakage(feature_set.predictive_features)


def test_enhanced_feature_branch_adds_safe_dataset_aligned_features(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    feature_set = build_feature_matrix(cleaned, include_enhanced_features=True)

    expected_features = {
        "total_sqft",
        "living_to_lot_ratio",
        "basement_share",
        "bathrooms_per_bedroom",
        "sqft_per_floor",
        "relative_living_area",
        "relative_lot_size",
        "sale_month_sin",
        "sale_month_cos",
    }
    assert expected_features.issubset(feature_set.dataframe.columns)
    assert expected_features.issubset(feature_set.predictive_features)
    assert "price_per_sqft" not in feature_set.predictive_features
    assert_no_target_leakage(feature_set.predictive_features)
