# Upgrade Recommendation

- Frozen baseline validation/test RMSE: `129084.55` / `147002.45`
- Candidate v1.6 validation/test RMSE: `110852.91` / `121629.66`
- Candidate selected model: `xgboost`
- Candidate interval coverage: `0.8831`

## Decision
- Adopt the candidate `xgboost` track only if you want the v1.6 branch to become the new operational champion.

## Rationale
- The candidate experiment was intentionally bounded to one challenger model and one enhanced geospatial feature branch.
- Recommendation favors robust and interpretable gains over marginal metric movement.
- Official frozen results remain the source of truth until an explicit adoption decision is made.

## Comparison Snapshot
track,feature_branch,comparison_role,model_name,validation_rmse,validation_mae,validation_mape,validation_r2,test_rmse,test_mae,test_mape,test_r2
frozen_baseline,stable,champion,random_forest,129084.55038257678,70975.28957140997,13.505059258326288,0.8593851075041401,147002.4512449295,79150.1726926637,13.246161507849132,0.8476432376371759
candidate_v1_6,enhanced_geospatial_v1_6,challenger,xgboost,110852.90650752607,65525.936800517135,12.933957223664525,0.8963004300020924,121629.65792329612,70271.43457844801,12.43762588653313,0.8956982602311996
candidate_v1_6,enhanced_geospatial_v1_6,champion,random_forest,123812.3065497971,70682.46856513753,13.625743020522751,0.8706368997755388,136676.47616177713,76955.18395268475,13.191050541075718,0.8682956472872185