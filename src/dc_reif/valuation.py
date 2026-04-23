from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from dc_reif.preprocessing import build_preprocessor
from dc_reif.splitting import make_time_series_cv


@dataclass
class ValuationArtifacts:
    model_name: str
    model_pipeline: Pipeline
    model_comparison: pd.DataFrame
    fair_value_hat_oof: pd.Series
    fair_value_hat_test: pd.Series
    validation_predictions: dict[str, pd.Series]
    test_predictions_by_model: dict[str, pd.Series]
    evaluation_summary: dict[str, dict[str, float]]


def _make_estimator(model_name: str, random_state: int) -> object:
    if model_name == "linear_regression":
        return LinearRegression()
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _scale_numeric(model_name: str) -> bool:
    return model_name == "linear_regression"


def _fit_pipeline(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    random_state: int,
) -> Pipeline:
    preprocessing = build_preprocessor(train_df, feature_columns, scale_numeric=_scale_numeric(model_name))
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.transformer),
            ("model", _make_estimator(model_name, random_state=random_state)),
        ]
    )
    pipeline.fit(train_df[feature_columns], train_df[target_column])
    return pipeline


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def generate_oof_predictions(
    train_validation_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    n_splits: int,
    random_state: int,
) -> pd.Series:
    ordered = train_validation_df.sort_values(["date", "id"]).copy()
    cv = make_time_series_cv(n_splits=max(2, min(n_splits, len(ordered) - 1)))
    oof_predictions = pd.Series(np.nan, index=ordered.index, dtype=float)

    for train_idx, validation_idx in cv.split(ordered):
        fold_train = ordered.iloc[train_idx]
        fold_validation = ordered.iloc[validation_idx]
        pipeline = _fit_pipeline(
            fold_train,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=model_name,
            random_state=random_state,
        )
        oof_predictions.loc[fold_validation.index] = pipeline.predict(fold_validation[feature_columns])
    return oof_predictions.sort_index()


def train_and_select_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_splits: int,
    random_state: int,
) -> ValuationArtifacts:
    model_names = ["linear_regression", "random_forest"]
    rows: list[dict[str, float | str]] = []
    validation_predictions: dict[str, pd.Series] = {}
    test_predictions_by_model: dict[str, pd.Series] = {}
    evaluation_summary: dict[str, dict[str, float]] = {}

    for model_name in model_names:
        validation_pipeline = _fit_pipeline(
            train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=model_name,
            random_state=random_state,
        )
        validation_pred = pd.Series(
            validation_pipeline.predict(validation_df[feature_columns]),
            index=validation_df.index,
            name=f"{model_name}_validation_prediction",
        )
        validation_predictions[model_name] = validation_pred
        validation_metrics = regression_metrics(validation_df[target_column], validation_pred)

        test_pipeline = _fit_pipeline(
            train_validation_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=model_name,
            random_state=random_state,
        )
        test_pred = pd.Series(
            test_pipeline.predict(test_df[feature_columns]),
            index=test_df.index,
            name=f"{model_name}_test_prediction",
        )
        test_predictions_by_model[model_name] = test_pred
        test_metrics = regression_metrics(test_df[target_column], test_pred)

        evaluation_summary[model_name] = {
            **{f"validation_{metric}": value for metric, value in validation_metrics.items()},
            **{f"test_{metric}": value for metric, value in test_metrics.items()},
        }
        rows.append(
            {
                "model_name": model_name,
                **{f"validation_{metric}": value for metric, value in validation_metrics.items()},
                **{f"test_{metric}": value for metric, value in test_metrics.items()},
            }
        )

    comparison = pd.DataFrame(rows).sort_values("validation_rmse").reset_index(drop=True)
    lr_rmse = float(comparison.loc[comparison["model_name"] == "linear_regression", "validation_rmse"].iloc[0])
    rf_rmse = float(comparison.loc[comparison["model_name"] == "random_forest", "validation_rmse"].iloc[0])
    chosen_model = "random_forest" if rf_rmse <= lr_rmse * 1.05 else "linear_regression"

    final_pipeline = _fit_pipeline(
        train_validation_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=chosen_model,
        random_state=random_state,
    )
    fair_value_hat_oof = generate_oof_predictions(
        train_validation_df=train_validation_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=chosen_model,
        n_splits=n_splits,
        random_state=random_state,
    )
    fair_value_hat_test = pd.Series(
        final_pipeline.predict(test_df[feature_columns]),
        index=test_df.index,
        name="fair_value_hat_test",
    )

    return ValuationArtifacts(
        model_name=chosen_model,
        model_pipeline=final_pipeline,
        model_comparison=comparison,
        fair_value_hat_oof=fair_value_hat_oof,
        fair_value_hat_test=fair_value_hat_test,
        validation_predictions=validation_predictions,
        test_predictions_by_model=test_predictions_by_model,
        evaluation_summary=evaluation_summary,
    )

