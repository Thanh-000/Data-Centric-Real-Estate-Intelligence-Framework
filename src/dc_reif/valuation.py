from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from dc_reif.preprocessing import build_preprocessor
from dc_reif.splitting import make_time_series_cv

try:  # pragma: no cover - dependency validated in runtime/tests
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


OFFICIAL_MODEL_NAME = "xgboost"


@dataclass(frozen=True)
class OfficialModelConfig:
    name: str
    estimator_params: dict[str, Any]
    target_strategy: str = "raw"
    high_price_weight: float = 1.0
    high_price_quantile: float = 0.8


@dataclass
class ValuationArtifacts:
    model_name: str
    model_pipeline: Pipeline
    valuation_metrics: pd.DataFrame
    fair_value_hat_oof: pd.Series
    fair_value_hat_test: pd.Series
    validation_predictions: dict[str, pd.Series]
    test_predictions_by_model: dict[str, pd.Series]
    evaluation_summary: dict[str, dict[str, float]]
    selected_parameters: dict[str, Any]
    target_strategy: str
    high_price_weight: float
    selection_summary: pd.DataFrame


@dataclass
class ModelSuiteArtifacts:
    valuation_metrics: pd.DataFrame
    validation_predictions: dict[str, pd.Series]
    test_predictions_by_model: dict[str, pd.Series]
    evaluation_summary: dict[str, dict[str, float]]


def _make_estimator(model_name: str, random_state: int, estimator_params: dict[str, Any] | None = None) -> object:
    estimator_params = estimator_params or {}
    if model_name == "linear_regression":
        return LinearRegression(**estimator_params)
    if model_name == "random_forest":
        params = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(estimator_params)
        return RandomForestRegressor(**params)
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it to run the official valuation pipeline.")
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(estimator_params)
        return XGBRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def official_model_available(model_name: str = OFFICIAL_MODEL_NAME) -> bool:
    return model_name != "xgboost" or XGBRegressor is not None


def _scale_numeric(model_name: str) -> bool:
    return model_name == "linear_regression"


def _transform_target(target: pd.Series, target_strategy: str) -> np.ndarray:
    values = target.astype(float).to_numpy()
    if target_strategy == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return values


def _inverse_transform_predictions(predictions: np.ndarray | pd.Series, target_strategy: str) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if target_strategy == "log1p":
        return np.expm1(values).clip(min=0.0)
    return values


def _high_price_weights(
    dataframe: pd.DataFrame,
    target_column: str,
    high_price_weight: float,
    high_price_quantile: float,
) -> np.ndarray | None:
    if high_price_weight <= 1.0:
        return None
    threshold = dataframe[target_column].quantile(high_price_quantile)
    weights = np.ones(len(dataframe), dtype=float)
    weights[dataframe[target_column].to_numpy() >= threshold] = float(high_price_weight)
    return weights


def _fit_pipeline(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    random_state: int,
    estimator_params: dict[str, Any] | None = None,
    target_strategy: str = "raw",
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    preprocessing = build_preprocessor(train_df, feature_columns, scale_numeric=_scale_numeric(model_name))
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing.transformer),
            ("model", _make_estimator(model_name, random_state=random_state, estimator_params=estimator_params)),
        ]
    )
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(
        train_df[feature_columns],
        _transform_target(train_df[target_column], target_strategy=target_strategy),
        **fit_kwargs,
    )
    return pipeline


def _predict_pipeline(pipeline: Pipeline, dataframe: pd.DataFrame, feature_columns: list[str], target_strategy: str) -> np.ndarray:
    predictions = pipeline.predict(dataframe[feature_columns])
    return _inverse_transform_predictions(predictions, target_strategy=target_strategy)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def _price_band_labels(series: pd.Series, n_bands: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="string")
    n_quantiles = min(n_bands, max(valid.nunique(), 1))
    labels = [f"Q{index}" for index in range(1, n_quantiles + 1)]
    ranked = valid.rank(method="first")
    bands = pd.qcut(ranked, q=n_quantiles, labels=labels)
    output = pd.Series(pd.NA, index=series.index, dtype="string")
    output.loc[valid.index] = bands.astype("string")
    return output


def _upper_tail_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    bands = _price_band_labels(actual)
    q5_mask = bands.eq("Q5")
    if not q5_mask.any():
        q5_mask = actual.notna()
    q5_actual = actual.loc[q5_mask]
    q5_predicted = predicted.loc[q5_mask]
    q5_signed_error = q5_actual - q5_predicted
    return {
        "validation_q5_count": float(len(q5_actual)),
        "validation_q5_rmse": float(np.sqrt(mean_squared_error(q5_actual, q5_predicted))),
        "validation_q5_mae": float(mean_absolute_error(q5_actual, q5_predicted)),
        "validation_q5_mean_signed_error": float(q5_signed_error.mean()),
    }


def _selection_score(metrics: dict[str, float], upper_tail: dict[str, float]) -> float:
    return (
        metrics["rmse"]
        + 0.20 * upper_tail["validation_q5_mae"]
        + 0.05 * abs(upper_tail["validation_q5_mean_signed_error"])
    )


def _official_xgboost_search_space() -> list[OfficialModelConfig]:
    return [
        OfficialModelConfig(
            name="xgb_raw_default",
            estimator_params={},
            target_strategy="raw",
            high_price_weight=1.0,
        ),
        OfficialModelConfig(
            name="xgb_log_default",
            estimator_params={},
            target_strategy="log1p",
            high_price_weight=1.0,
        ),
        OfficialModelConfig(
            name="xgb_log_tail_light",
            estimator_params={
                "n_estimators": 500,
                "learning_rate": 0.04,
                "max_depth": 5,
                "min_child_weight": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.5,
            },
            target_strategy="log1p",
            high_price_weight=1.2,
        ),
        OfficialModelConfig(
            name="xgb_log_tail_medium",
            estimator_params={
                "n_estimators": 550,
                "learning_rate": 0.04,
                "max_depth": 5,
                "min_child_weight": 2,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.5,
                "reg_alpha": 0.05,
            },
            target_strategy="log1p",
            high_price_weight=1.35,
        ),
        OfficialModelConfig(
            name="xgb_log_conservative",
            estimator_params={
                "n_estimators": 650,
                "learning_rate": 0.035,
                "max_depth": 4,
                "min_child_weight": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
            },
            target_strategy="log1p",
            high_price_weight=1.1,
        ),
        OfficialModelConfig(
            name="xgb_log_upper_tail_focus",
            estimator_params={
                "n_estimators": 500,
                "learning_rate": 0.04,
                "max_depth": 6,
                "min_child_weight": 2,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
            },
            target_strategy="log1p",
            high_price_weight=1.45,
        ),
    ]


def generate_oof_predictions(
    train_validation_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    n_splits: int,
    random_state: int,
    estimator_params: dict[str, Any] | None = None,
    target_strategy: str = "raw",
    high_price_weight: float = 1.0,
    high_price_quantile: float = 0.8,
) -> pd.Series:
    ordered = train_validation_df.sort_values(["date", "id"]).copy()
    cv = make_time_series_cv(n_splits=max(2, min(n_splits, len(ordered) - 1)))
    oof_predictions = pd.Series(np.nan, index=ordered.index, dtype=float)

    for train_idx, validation_idx in cv.split(ordered):
        fold_train = ordered.iloc[train_idx]
        fold_validation = ordered.iloc[validation_idx]
        fold_weights = _high_price_weights(
            fold_train,
            target_column=target_column,
            high_price_weight=high_price_weight,
            high_price_quantile=high_price_quantile,
        )
        pipeline = _fit_pipeline(
            fold_train,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=model_name,
            random_state=random_state,
            estimator_params=estimator_params,
            target_strategy=target_strategy,
            sample_weight=fold_weights,
        )
        oof_predictions.loc[fold_validation.index] = _predict_pipeline(
            pipeline,
            fold_validation,
            feature_columns=feature_columns,
            target_strategy=target_strategy,
        )
    return oof_predictions.sort_index()


def evaluate_model_suite(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str],
    random_state: int,
) -> ModelSuiteArtifacts:
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
            _predict_pipeline(validation_pipeline, validation_df, feature_columns=feature_columns, target_strategy="raw"),
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
            _predict_pipeline(test_pipeline, test_df, feature_columns=feature_columns, target_strategy="raw"),
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
    return ModelSuiteArtifacts(
        valuation_metrics=comparison,
        validation_predictions=validation_predictions,
        test_predictions_by_model=test_predictions_by_model,
        evaluation_summary=evaluation_summary,
    )


def fit_selected_model_artifacts(
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    n_splits: int,
    random_state: int,
    estimator_params: dict[str, Any] | None = None,
    target_strategy: str = "raw",
    high_price_weight: float = 1.0,
    high_price_quantile: float = 0.8,
) -> tuple[Pipeline, pd.Series, pd.Series]:
    train_weights = _high_price_weights(
        train_validation_df,
        target_column=target_column,
        high_price_weight=high_price_weight,
        high_price_quantile=high_price_quantile,
    )
    final_pipeline = _fit_pipeline(
        train_validation_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=model_name,
        random_state=random_state,
        estimator_params=estimator_params,
        target_strategy=target_strategy,
        sample_weight=train_weights,
    )
    fair_value_hat_oof = generate_oof_predictions(
        train_validation_df=train_validation_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=model_name,
        n_splits=n_splits,
        random_state=random_state,
        estimator_params=estimator_params,
        target_strategy=target_strategy,
        high_price_weight=high_price_weight,
        high_price_quantile=high_price_quantile,
    )
    fair_value_hat_test = pd.Series(
        _predict_pipeline(final_pipeline, test_df, feature_columns=feature_columns, target_strategy=target_strategy),
        index=test_df.index,
        name="fair_value_hat_test",
    )
    return final_pipeline, fair_value_hat_oof, fair_value_hat_test


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
    if not official_model_available():
        raise ImportError("xgboost is required for the official valuation workflow.")

    search_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, pd.Series] = {}
    search_space = _official_xgboost_search_space()

    for config_option in search_space:
        sample_weight = _high_price_weights(
            train_df,
            target_column=target_column,
            high_price_weight=config_option.high_price_weight,
            high_price_quantile=config_option.high_price_quantile,
        )
        validation_pipeline = _fit_pipeline(
            train_df=train_df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=OFFICIAL_MODEL_NAME,
            random_state=random_state,
            estimator_params=config_option.estimator_params,
            target_strategy=config_option.target_strategy,
            sample_weight=sample_weight,
        )
        validation_pred = pd.Series(
            _predict_pipeline(
                validation_pipeline,
                validation_df,
                feature_columns=feature_columns,
                target_strategy=config_option.target_strategy,
            ),
            index=validation_df.index,
            name=config_option.name,
        )
        validation_predictions[config_option.name] = validation_pred
        validation_metrics = regression_metrics(validation_df[target_column], validation_pred)
        upper_tail = _upper_tail_metrics(validation_df[target_column], validation_pred)
        selection_score = _selection_score(validation_metrics, upper_tail)
        search_rows.append(
            {
                "config_name": config_option.name,
                "model_name": OFFICIAL_MODEL_NAME,
                "target_strategy": config_option.target_strategy,
                "high_price_weight": config_option.high_price_weight,
                "high_price_quantile": config_option.high_price_quantile,
                "estimator_params_json": json.dumps(config_option.estimator_params, sort_keys=True),
                "selection_score": float(selection_score),
                **{f"validation_{metric}": value for metric, value in validation_metrics.items()},
                **upper_tail,
            }
        )

    selection_summary = pd.DataFrame(search_rows).sort_values(
        ["selection_score", "validation_rmse", "validation_q5_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    chosen_row = selection_summary.iloc[0]
    chosen_config = next(config_option for config_option in search_space if config_option.name == chosen_row["config_name"])

    final_pipeline, fair_value_hat_oof, fair_value_hat_test = fit_selected_model_artifacts(
        train_validation_df=train_validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=OFFICIAL_MODEL_NAME,
        n_splits=n_splits,
        random_state=random_state,
        estimator_params=chosen_config.estimator_params,
        target_strategy=chosen_config.target_strategy,
        high_price_weight=chosen_config.high_price_weight,
        high_price_quantile=chosen_config.high_price_quantile,
    )

    validation_pipeline = _fit_pipeline(
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=OFFICIAL_MODEL_NAME,
        random_state=random_state,
        estimator_params=chosen_config.estimator_params,
        target_strategy=chosen_config.target_strategy,
        sample_weight=_high_price_weights(
            train_df,
            target_column=target_column,
            high_price_weight=chosen_config.high_price_weight,
            high_price_quantile=chosen_config.high_price_quantile,
        ),
    )
    final_validation_pred = pd.Series(
        _predict_pipeline(
            validation_pipeline,
            validation_df,
            feature_columns=feature_columns,
            target_strategy=chosen_config.target_strategy,
        ),
        index=validation_df.index,
        name=f"{OFFICIAL_MODEL_NAME}_validation_prediction",
    )
    validation_metrics = regression_metrics(validation_df[target_column], final_validation_pred)
    test_metrics = regression_metrics(test_df[target_column], fair_value_hat_test)
    valuation_metrics = pd.DataFrame(
        [
            {
                "model_name": OFFICIAL_MODEL_NAME,
                **{f"validation_{metric}": value for metric, value in validation_metrics.items()},
                **{f"test_{metric}": value for metric, value in test_metrics.items()},
            }
        ]
    )
    evaluation_summary = {
        OFFICIAL_MODEL_NAME: {
            **{f"validation_{metric}": value for metric, value in validation_metrics.items()},
            **{f"test_{metric}": value for metric, value in test_metrics.items()},
        }
    }

    return ValuationArtifacts(
        model_name=OFFICIAL_MODEL_NAME,
        model_pipeline=final_pipeline,
        valuation_metrics=valuation_metrics,
        fair_value_hat_oof=fair_value_hat_oof,
        fair_value_hat_test=fair_value_hat_test,
        validation_predictions={OFFICIAL_MODEL_NAME: final_validation_pred},
        test_predictions_by_model={OFFICIAL_MODEL_NAME: fair_value_hat_test},
        evaluation_summary=evaluation_summary,
        selected_parameters=chosen_config.estimator_params,
        target_strategy=chosen_config.target_strategy,
        high_price_weight=chosen_config.high_price_weight,
        selection_summary=selection_summary,
    )
