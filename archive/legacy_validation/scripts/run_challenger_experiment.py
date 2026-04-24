from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.diagnostics import (
    collect_slice_diagnostics,
    geospatial_feature_notes,
    slice_interpretation_notes,
    upgrade_recommendation,
)
from dc_reif.pipeline import run_candidate_v16_experiment
from dc_reif.report_results import build_final_results_summary, build_report_results_pack
from dc_reif.utils import ensure_directory


def _write_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def _flatten_summary(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else key, nested)
            return
        rows.append({"metric": prefix, "value": value})

    visit("", summary)
    return pd.DataFrame(rows)


def main() -> None:
    config = ProjectConfig.from_cli()
    build_report_results_pack(config)
    baseline_summary = build_final_results_summary(config)
    baseline_diagnostics = collect_slice_diagnostics(config)

    experiment_outputs = run_candidate_v16_experiment(config)
    candidate_summary = json.loads(Path(experiment_outputs["candidate_results_json"]).read_text(encoding="utf-8"))
    candidate_feature_dataset = Path(experiment_outputs["candidate_feature_dataset"])
    candidate_property_table = Path(experiment_outputs["candidate_property_intelligence"])
    candidate_diagnostics = collect_slice_diagnostics(
        config,
        feature_dataset_path=candidate_feature_dataset,
        property_table_path=candidate_property_table,
    )

    baseline_comparison = pd.read_csv(config.paths.tables_dir / "model_comparison.csv")
    baseline_rf = baseline_comparison.loc[baseline_comparison["model_name"] == "random_forest"].copy()
    baseline_rf.insert(0, "track", "frozen_baseline")
    baseline_rf.insert(1, "feature_branch", "stable")
    baseline_rf.insert(2, "comparison_role", "champion")

    candidate_comparison = pd.read_csv(experiment_outputs["candidate_model_comparison"]).copy()
    candidate_comparison.insert(0, "track", "candidate_v1_6")

    comparison = pd.concat([baseline_rf, candidate_comparison], ignore_index=True)
    comparison_path = config.paths.outputs_dir / "final_report_pack" / "15_champion_challenger_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    candidate_summary_csv = config.paths.reports_dir / "candidate_v1_6_results.csv"
    _flatten_summary(candidate_summary).to_csv(candidate_summary_csv, index=False)

    notes_path = _write_text(
        config.paths.outputs_dir / "final_report_pack" / "16_geospatial_feature_notes.md",
        geospatial_feature_notes(),
    )
    slice_notes_path = _write_text(
        config.paths.outputs_dir / "final_report_pack" / "17_slice_interpretation_notes.md",
        slice_interpretation_notes(
            baseline_summary,
            baseline_diagnostics,
            candidate_summary=candidate_summary,
            candidate_diagnostics=candidate_diagnostics,
        ),
    )
    recommendation_path = _write_text(
        config.paths.outputs_dir / "final_report_pack" / "18_upgrade_recommendation.md",
        upgrade_recommendation(baseline_summary, candidate_summary, comparison),
    )

    print("DC-REIF v1.6 challenger experiment complete.")
    print(f"candidate_results_json: {experiment_outputs['candidate_results_json']}")
    print(f"candidate_results_csv: {candidate_summary_csv}")
    print(f"comparison: {comparison_path}")
    print(f"geospatial_feature_notes: {notes_path}")
    print(f"slice_interpretation_notes: {slice_notes_path}")
    print(f"upgrade_recommendation: {recommendation_path}")


if __name__ == "__main__":
    main()
