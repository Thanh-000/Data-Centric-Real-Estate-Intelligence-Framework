from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.market_representation import market_representation_status


def main() -> None:
    config = ProjectConfig.from_cli()
    cluster_summary = config.paths.reports_dir / "cluster_summary.json"
    if not cluster_summary.exists():
        raise FileNotFoundError(
            f"Missing {cluster_summary}. Run `python scripts/run_pipeline.py` to generate the implemented market representation outputs."
        )
    print("DC-REIF market representation summary")
    print(f"Implemented status: {market_representation_status()}")
    print(f"Existing clustering summary: {cluster_summary}")


if __name__ == "__main__":
    main()
