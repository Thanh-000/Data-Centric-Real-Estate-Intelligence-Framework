from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from dc_reif.environment import resolve_output_root
from dc_reif.paths import ProjectPaths, project_root_from_file


DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/randellmwania/"
    "Kings-County-Housing-Project/master/data/kc_house_data.csv"
)
DEFAULT_DATA_CHECKSUM = "970a5ee2b0294257cdb18952813df2dd05974f923a9a07ae17fa1af39da71dce"

REQUIRED_COLUMNS = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

DESCRIPTIVE_ONLY_FEATURES = [
    "price_per_sqft",
]

TARGET_DERIVED_FEATURES = {
    "price_per_sqft",
    "fair_value_hat",
    "fair_value_hat_oof",
    "valuation_gap",
    "anomaly_score",
}


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ProjectConfig:
    project_root: Path
    paths: ProjectPaths
    data_url: str = DEFAULT_DATA_URL
    data_filename: str = "kc_house_data.csv"
    data_checksum: str | None = None
    use_aria2: bool = True
    force_download: bool = False
    random_state: int = 42
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    n_splits: int = 5
    alpha: float = 0.1
    target_column: str = "price"
    date_column: str = "date"
    id_column: str = "id"
    required_columns: list[str] = field(default_factory=lambda: REQUIRED_COLUMNS.copy())
    descriptive_only_features: list[str] = field(default_factory=lambda: DESCRIPTIVE_ONLY_FEATURES.copy())

    @property
    def data_dir(self) -> Path:
        return self.paths.raw_dir

    @property
    def manifest_path(self) -> Path:
        return self.paths.artifacts_dir / "raw_data_manifest.json"

    @property
    def validation_report_path(self) -> Path:
        return self.paths.reports_dir / "data_quality_report.json"

    @property
    def cleaned_dataset_path(self) -> Path:
        return self.paths.processed_dir / "kc_house_data_clean.csv"

    @property
    def feature_dataset_path(self) -> Path:
        return self.paths.processed_dir / "kc_house_features.csv"

    @classmethod
    def default(cls) -> "ProjectConfig":
        project_root = project_root_from_file(Path(__file__))
        output_root = resolve_output_root()
        paths = ProjectPaths.from_root(project_root, outputs_root=output_root).ensure()
        return cls(
            project_root=project_root,
            paths=paths,
            data_url=os.getenv("DATA_URL", DEFAULT_DATA_URL),
            data_filename=os.getenv("DATA_FILENAME", "kc_house_data.csv"),
            data_checksum=(
                os.getenv("DATA_CHECKSUM")
                or (DEFAULT_DATA_CHECKSUM if os.getenv("DATA_URL", DEFAULT_DATA_URL) == DEFAULT_DATA_URL else None)
            ),
            use_aria2=parse_bool(os.getenv("USE_ARIA2"), default=True),
            force_download=parse_bool(os.getenv("FORCE_DOWNLOAD"), default=False),
            random_state=int(os.getenv("RANDOM_STATE", 42)),
            alpha=float(os.getenv("PREDICTION_ALPHA", 0.1)),
        )

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ProjectConfig":
        parser = argparse.ArgumentParser(description="DC-REIF configuration")
        parser.add_argument("--data-url", default=os.getenv("DATA_URL"))
        parser.add_argument("--data-filename", default=os.getenv("DATA_FILENAME"))
        parser.add_argument("--data-dir", default=os.getenv("DATA_DIR"))
        parser.add_argument("--data-checksum", default=os.getenv("DATA_CHECKSUM"))
        parser.add_argument("--use-aria2", default=os.getenv("USE_ARIA2"))
        parser.add_argument("--force-download", default=os.getenv("FORCE_DOWNLOAD"))
        parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR"))
        parser.add_argument("--alpha", type=float, default=os.getenv("PREDICTION_ALPHA"))
        args = parser.parse_args(argv)

        config = cls.default()
        project_root = config.project_root

        data_root = Path(args.data_dir).expanduser() if args.data_dir else None
        output_root = Path(args.output_dir).expanduser() if args.output_dir else config.paths.outputs_dir
        config.paths = ProjectPaths.from_root(project_root, data_root=data_root, outputs_root=output_root).ensure()

        if args.data_url:
            config.data_url = args.data_url
            if not args.data_checksum:
                config.data_checksum = DEFAULT_DATA_CHECKSUM if args.data_url == DEFAULT_DATA_URL else None
        if args.data_filename:
            config.data_filename = args.data_filename
        if args.data_checksum:
            config.data_checksum = args.data_checksum
        if args.use_aria2 is not None:
            config.use_aria2 = parse_bool(args.use_aria2, default=config.use_aria2)
        if args.force_download is not None:
            config.force_download = parse_bool(args.force_download, default=config.force_download)
        if args.alpha is not None:
            config.alpha = float(args.alpha)
        return config

