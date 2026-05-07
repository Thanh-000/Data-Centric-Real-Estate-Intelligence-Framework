from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run(label: str, command: list[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run DC-REIF end-to-end on the public King County dataset. "
            "The dataset is downloaded automatically before the pipeline runs."
        )
    )
    parser.add_argument("--install", action="store_true", help="Install requirements before running.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the dataset even if a valid local copy exists.")
    parser.add_argument("--no-aria2", action="store_true", help="Disable aria2 and use Python download fallbacks.")
    parser.add_argument("--with-tests", action="store_true", help="Run pytest after building outputs.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"), help="Output directory.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optional Optuna trials for XGBoost tuning.")
    parser.add_argument("--enable-mlflow", action="store_true", help="Log the pipeline run to local MLflow tracking.")
    args = parser.parse_args()

    python = sys.executable
    output_dir = Path(args.output_dir)

    if args.install:
        _run("Install dependencies", [python, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    use_aria2 = not args.no_aria2
    if use_aria2 and shutil.which("aria2c") is None:
        print("aria2c was not found in PATH; the downloader will fall back to requests/wget/urllib.", flush=True)

    common_args = ["--output-dir", str(output_dir), "--use-aria2", str(use_aria2).lower()]
    if args.force_download:
        common_args.extend(["--force-download", "true"])
    if args.optuna_trials:
        common_args.extend(["--optuna-trials", str(args.optuna_trials)])
    if args.enable_mlflow:
        common_args.extend(["--enable-mlflow", "true"])

    _run("Download official dataset", [python, "scripts/download_data.py", *common_args])
    _run("Run pipeline", [python, "scripts/run_pipeline.py", *common_args])
    _run("Analyze abstention", [python, "scripts/analyze_abstention.py", *common_args])
    _run("Evaluate slices", [python, "scripts/evaluate_slices.py", *common_args])
    if args.with_tests:
        _run("Run tests", [python, "-m", "pytest", "-q"])

    print(f"\nQuickstart complete. Outputs are in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
