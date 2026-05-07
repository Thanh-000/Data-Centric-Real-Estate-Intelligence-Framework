from dc_reif.data_cleaning import clean_king_county_data
import numpy as np
import pandas as pd

from dc_reif.feature_engineering import add_safe_derived_features, assert_no_target_leakage, build_feature_matrix


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
        "prior_zipcode_median_price",
        "prior_neighbor_median_price",
    }
    assert expected_features.issubset(feature_set.dataframe.columns)
    assert expected_features.issubset(feature_set.predictive_features)
    assert "yr_renovated" not in feature_set.predictive_features
    assert "price_per_sqft" not in feature_set.predictive_features
    assert_no_target_leakage(feature_set.predictive_features)


def test_historical_price_features_use_strictly_prior_sale_dates_only():
    frame = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "date": pd.to_datetime(["2014-01-01", "2014-01-01", "2014-01-02", "2014-01-03"]),
            "price": [100_000.0, 900_000.0, 300_000.0, 400_000.0],
            "bedrooms": [3, 3, 3, 3],
            "bathrooms": [2.0, 2.0, 2.0, 2.0],
            "sqft_living": [1500, 1500, 1500, 1500],
            "sqft_lot": [5000, 5000, 5000, 5000],
            "floors": [1.0, 1.0, 1.0, 1.0],
            "waterfront": [0, 0, 0, 0],
            "view": [0, 0, 0, 0],
            "condition": [3, 3, 3, 3],
            "grade": [7, 7, 7, 7],
            "sqft_above": [1500, 1500, 1500, 1500],
            "sqft_basement": [0, 0, 0, 0],
            "yr_built": [1980, 1980, 1980, 1980],
            "yr_renovated": [0, 0, 0, 0],
            "zipcode": ["98001", "98001", "98001", "98001"],
            "lat": [47.50, 47.5001, 47.5002, 47.5003],
            "long": [-122.20, -122.2001, -122.2002, -122.2003],
            "sqft_living15": [1500, 1500, 1500, 1500],
            "sqft_lot15": [5000, 5000, 5000, 5000],
        }
    )

    features = add_safe_derived_features(frame)

    assert np.isnan(features.loc[1, "prior_zipcode_median_price"])
    assert np.isnan(features.loc[1, "prior_neighbor_median_price"])
    assert np.isnan(features.loc[2, "prior_zipcode_median_price"])
    assert np.isnan(features.loc[2, "prior_neighbor_median_price"])
    assert features.loc[3, "prior_zipcode_median_price"] == 300_000.0
    assert features.loc[3, "prior_neighbor_median_price"] == 300_000.0
