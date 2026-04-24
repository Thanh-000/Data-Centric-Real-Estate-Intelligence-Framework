from dc_reif.feature_store import descriptive_only_features, predictive_feature_policy


def test_feature_store_policy_preserves_descriptive_only_features():
    policy = predictive_feature_policy()
    assert policy["train_only_preprocessing_required"] is True
    assert "price_per_sqft" in descriptive_only_features()

