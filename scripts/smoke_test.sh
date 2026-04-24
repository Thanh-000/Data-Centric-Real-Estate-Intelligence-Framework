#!/usr/bin/env bash
set -euo pipefail

python scripts/build_report_results.py
python -m pytest -q

