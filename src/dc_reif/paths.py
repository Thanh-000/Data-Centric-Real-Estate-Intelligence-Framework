from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dc_reif.utils import ensure_directory


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    notebooks_dir: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path, data_root: Path | None = None, outputs_root: Path | None = None) -> "ProjectPaths":
        resolved_root = root.resolve()
        resolved_data_root = (data_root or resolved_root / "data").resolve()
        resolved_outputs_root = (outputs_root or resolved_root / "outputs").resolve()
        return cls(
            root=resolved_root,
            data_root=resolved_data_root,
            raw_dir=resolved_data_root / "raw",
            interim_dir=resolved_data_root / "interim",
            processed_dir=resolved_data_root / "processed",
            artifacts_dir=resolved_data_root / "artifacts",
            notebooks_dir=resolved_root / "notebooks",
            outputs_dir=resolved_outputs_root,
            figures_dir=resolved_outputs_root / "figures",
            tables_dir=resolved_outputs_root / "tables",
            reports_dir=resolved_outputs_root / "reports",
        )

    def ensure(self) -> "ProjectPaths":
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.figures_dir,
            self.tables_dir,
            self.reports_dir,
        ):
            ensure_directory(directory)
        return self


def project_root_from_file(path: Path) -> Path:
    resolved = path.resolve()
    for root_option in [resolved] + list(resolved.parents):
        if (root_option / "pyproject.toml").exists() or (root_option / "README.md").exists():
            return root_option
    return resolved.parents[2]
