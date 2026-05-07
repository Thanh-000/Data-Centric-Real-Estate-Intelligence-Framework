from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.pipeline import run_full_pipeline


def _print_model_comparison(path: str | None) -> None:
    if not path:
        return
    import pandas as pd

    comparison_path = Path(path)
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    columns = ["model_name", "test_rmse", "test_mae", "test_mape", "test_r2"]
    if not set(columns).issubset(comparison.columns):
        return
    display = comparison[columns].sort_values("test_rmse").copy()
    display["test_rmse"] = display["test_rmse"].map(lambda value: f"{value:,.0f}")
    display["test_mae"] = display["test_mae"].map(lambda value: f"{value:,.0f}")
    display["test_mape"] = display["test_mape"].map(lambda value: f"{value:.2f}%")
    display["test_r2"] = display["test_r2"].map(lambda value: f"{value:.3f}")
    print("\nModel comparison:")
    print(display.to_string(index=False))


def main() -> None:
    config = ProjectConfig.from_cli()
    outputs = run_full_pipeline(config)
    print("Pipeline completed.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    _print_model_comparison(outputs.get("model_baseline_comparison"))


if __name__ == "__main__":
    main()
