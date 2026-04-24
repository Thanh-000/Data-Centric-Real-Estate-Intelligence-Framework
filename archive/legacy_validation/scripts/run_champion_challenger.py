from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from dc_reif.config import ProjectConfig


def main() -> None:
    config = ProjectConfig.from_cli()
    model_comparison = config.paths.tables_dir / "model_comparison.csv"
    if not model_comparison.exists():
        raise FileNotFoundError(
            f"Missing {model_comparison}. Run `python scripts/run_pipeline.py` to generate champion-challenger results."
        )
    comparison = pd.read_csv(model_comparison)
    champion = comparison.sort_values("validation_rmse").iloc[0]
    print("DC-REIF valuation comparison summary")
    print(comparison.to_string(index=False))
    print(f"Champion: {champion['model_name']}")


if __name__ == "__main__":
    main()
