from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.pipeline import run_full_pipeline


def main() -> None:
    config = ProjectConfig.from_cli()
    outputs = run_full_pipeline(config)
    print("Pipeline completed.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
