from __future__ import annotations

from pathlib import Path
from typing import Any

from dc_reif.config import ProjectConfig
from dc_reif.utils import get_logger


LOGGER = get_logger(__name__)


def log_pipeline_run(
    config: ProjectConfig,
    outputs: dict[str, str],
    metrics: dict[str, Any],
    params: dict[str, Any],
) -> str | None:
    if not config.enable_mlflow:
        return None

    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOGGER.warning("MLflow logging requested but unavailable: %s", exc)
        return None

    tracking_uri = config.mlflow_tracking_uri
    if not tracking_uri:
        tracking_uri = (config.paths.outputs_dir / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("dc-reif")

    with mlflow.start_run(run_name="dc-reif-pipeline") as run:
        for key, value in params.items():
            if value is not None:
                mlflow.log_param(key, value)
        for key, value in metrics.items():
            if isinstance(value, bool):
                mlflow.log_metric(key, float(value))
            elif isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        for key, value in outputs.items():
            path = Path(value)
            if path.exists() and path.is_file():
                try:
                    mlflow.log_artifact(str(path), artifact_path=key)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Could not log artifact %s: %s", path, exc)

        LOGGER.info("MLflow run logged: %s", run.info.run_id)
        return str(run.info.run_id)
