from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

from dc_reif.config import ProjectConfig


def main() -> None:
    config = ProjectConfig.from_cli()
    uncertainty_path = config.paths.reports_dir / "uncertainty_metrics.json"
    if not uncertainty_path.exists():
        raise FileNotFoundError(
            f"Missing {uncertainty_path}. Run `python scripts/run_pipeline.py` to generate uncertainty outputs."
        )
    payload = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    print("DC-REIF uncertainty calibration summary")
    print(payload)


if __name__ == "__main__":
    main()
