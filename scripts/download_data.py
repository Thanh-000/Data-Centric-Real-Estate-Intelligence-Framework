from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dc_reif.config import ProjectConfig
from dc_reif.data_download import download_dataset


def main() -> None:
    config = ProjectConfig.from_cli()
    path = download_dataset(config)
    print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
