from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.diagnostics import build_diagnostics_artifacts
from dc_reif.report_results import build_final_results_summary


def main() -> None:
    config = ProjectConfig.from_cli()
    summary = build_final_results_summary(config)
    artifacts = build_diagnostics_artifacts(config, summary=summary)
    print("Diagnostics artifacts:")
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
