from dc_reif.valuation_core import chronological_split, make_time_series_cv


def test_valuation_core_exports_split_helpers(sample_dataframe):
    split = chronological_split(sample_dataframe.assign(date=sample_dataframe["date"].astype("datetime64[ns]")))
    assert len(split.train_df) > 0
    assert make_time_series_cv(3).n_splits == 3

