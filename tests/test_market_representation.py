from dc_reif.data_cleaning import clean_king_county_data
from dc_reif.feature_engineering import build_feature_matrix
from dc_reif.market_representation import assign_submarket_segments, fit_submarket_clustering


def test_market_representation_wrapper_fits_and_assigns_segments(sample_dataframe):
    cleaned = clean_king_county_data(sample_dataframe).dataframe
    feature_set = build_feature_matrix(cleaned, include_enhanced_features=True)
    modeling_df = feature_set.dataframe.sort_values("date").reset_index(drop=True)
    train_df = modeling_df.iloc[:80].copy()

    artifacts = fit_submarket_clustering(train_df, random_state=42, include_enhanced_features=True)
    assigned = assign_submarket_segments(modeling_df.iloc[80:].copy(), artifacts)

    assert artifacts.n_clusters >= 3
    assert "selection_score" in artifacts.selection_summary.columns
    assert assigned.str.startswith("segment_").all()
