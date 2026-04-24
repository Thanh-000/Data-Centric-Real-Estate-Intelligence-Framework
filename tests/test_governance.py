from dc_reif.governance import REQUIRED_SCHEMA_CONTRACT, build_schema_contract


def test_governance_contract_exposes_required_columns():
    contract = build_schema_contract()
    assert contract["required_columns"] == REQUIRED_SCHEMA_CONTRACT["required_columns"]
    assert contract["policy"]["target_derived_features_forbidden_in_predictive_path"] is True

