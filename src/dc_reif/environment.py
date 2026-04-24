from __future__ import annotations

import os
from pathlib import Path

from dc_reif.utils import ensure_directory, get_logger

LOGGER = get_logger(__name__)


def is_running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # pragma: no cover

        return True
    except ImportError:
        return False


def maybe_mount_drive(force: bool = False) -> Path | None:
    if not is_running_in_colab():
        return None

    drive_root = Path("/content/drive")
    if drive_root.exists() and not force:
        return drive_root

    try:
        from google.colab import drive  # type: ignore  # pragma: no cover

        drive.mount(str(drive_root))
        return drive_root
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Google Drive mount skipped: %s", exc)
        return None


def resolve_data_path(filename: str, data_path: str | None = None) -> Path:
    env_path = data_path or os.getenv("DATA_PATH")
    if env_path:
        resolved_path = Path(env_path).expanduser()
        return resolved_path if resolved_path.suffix else resolved_path / filename

    drive_path = Path("/content/drive/MyDrive/dc_reif/data/raw")
    if is_running_in_colab() and drive_path.exists():
        return drive_path / filename

    local_path = Path.cwd() / "data" / "raw"
    ensure_directory(local_path)
    return local_path / filename


def resolve_output_root() -> Path:
    env_output_root = os.getenv("OUTPUT_DIR")
    if env_output_root:
        return ensure_directory(Path(env_output_root).expanduser())

    if is_running_in_colab():
        drive_reports = Path("/content/drive/MyDrive/dc_reif/outputs")
        if drive_reports.exists():
            return ensure_directory(drive_reports)
        return ensure_directory(Path("/content/outputs"))

    return ensure_directory(Path.cwd() / "outputs")
