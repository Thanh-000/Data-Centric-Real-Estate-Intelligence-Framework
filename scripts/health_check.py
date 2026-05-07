from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _status(ok: bool, label: str, detail: str, warn: bool = False) -> tuple[str, str]:
    if ok:
        return "PASS", f"PASS {label}: {detail}"
    if warn:
        return "WARN", f"WARN {label}: {detail}"
    return "FAIL", f"FAIL {label}: {detail}"


def main() -> int:
    reports = ROOT / "outputs" / "reports"
    tables = ROOT / "outputs" / "tables"
    figures = ROOT / "outputs" / "figures"

    checks: list[tuple[str, str]] = []

    valuation_path = tables / "valuation_metrics.csv"
    uncertainty_path = reports / "local_conformal_calibration_summary.json"
    coverage_path = tables / "test_interval_coverage_by_price_band.csv"
    property_path = tables / "property_intelligence_table.csv"
    flagged_path = tables / "model_flagged_cases.csv"
    shap_path = figures / "shap_summary.png"
    trust_path = reports / "trust_summary.md"

    required_paths = [valuation_path, uncertainty_path, coverage_path, property_path, shap_path, trust_path]
    missing = [path for path in required_paths if not path.exists()]
    checks.append(_status(not missing, "required outputs", f"{len(required_paths) - len(missing)}/{len(required_paths)} present"))
    if missing:
        for path in missing:
            print(f"FAIL missing output: {path}")
        return 1

    valuation = pd.read_csv(valuation_path)
    metrics = valuation.iloc[0].to_dict()
    coverage = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    coverage_by_band = pd.read_csv(coverage_path)
    property_table = pd.read_csv(property_path)

    test_mape = float(metrics.get("test_mape", float("nan")))
    checks.append(_status(test_mape <= 15.0, "test MAPE", f"{test_mape:.2f}% <= 15%"))

    global_coverage = float(coverage.get("global_empirical_coverage", float("nan")))
    checks.append(_status(global_coverage >= 0.90, "global coverage", f"{global_coverage:.1%} >= 90%"))

    q5_row = coverage_by_band.loc[coverage_by_band["price_band"].eq("Q5")]
    q5_coverage = float(q5_row["empirical_coverage"].iloc[0]) if not q5_row.empty else float("nan")
    q5_width = float(q5_row["average_interval_width"].iloc[0]) if not q5_row.empty else float("nan")
    checks.append(_status(q5_coverage >= 0.90, "Q5 coverage", f"{q5_coverage:.1%} >= 90%"))
    checks.append(_status(q5_width <= 900_000, "Q5 interval width", f"${q5_width:,.0f} <= $900,000", warn=True))

    model_flagged = int(property_table["anomaly_flag"].isin(["potentially_over_valued", "potentially_under_valued"]).sum())
    checks.append(_status(model_flagged > 0, "model-flagged cases", f"{model_flagged:,} cases"))
    checks.append(_status(flagged_path.exists(), "model_flagged_cases.csv", str(flagged_path)))

    failed = False
    for level, message in checks:
        print(message)
        failed = failed or level == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
