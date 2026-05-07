from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from dc_reif.config import ProjectConfig
from dc_reif.product_analytics import slice_metrics, write_analysis_tables


def main() -> None:
    config = ProjectConfig.from_cli()
    input_path = config.paths.tables_dir / "property_intelligence_table.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run `python scripts/run_pipeline.py` before product analytics."
        )

    dataframe = pd.read_csv(input_path)
    group_columns = [
        "zipcode",
        "segment_label",
        "predicted_price_band",
        "observed_price_band",
        "grade_band",
        "house_age_band",
        "evidence_strength",
        "slice_risk_level",
    ]
    tables = slice_metrics(dataframe, group_columns)
    paths = write_analysis_tables(tables, config.paths.tables_dir, "slice_metrics")

    print("Slice evaluation")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
