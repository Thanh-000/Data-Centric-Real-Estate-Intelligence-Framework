from __future__ import annotations

from pathlib import Path

import pandas as pd

from dc_reif.utils import get_logger, sha256_file, utc_timestamp, write_json

LOGGER = get_logger(__name__)


def load_raw_data(csv_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, str | int]]:
    dataframe = pd.read_csv(csv_path)
    manifest = {
        "filename": csv_path.name,
        "path": str(csv_path.resolve()),
        "file_size_bytes": csv_path.stat().st_size,
        "timestamp_utc": utc_timestamp(),
        "checksum_sha256": sha256_file(csv_path),
    }
    write_json(manifest_path, manifest)
    LOGGER.info("Raw data manifest written to %s", manifest_path)
    return dataframe, manifest

