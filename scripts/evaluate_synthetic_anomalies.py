from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "outputs" / "tables" / "property_intelligence_table.csv"
OUTPUT_TABLE = ROOT / "outputs" / "tables" / "synthetic_anomaly_recall.csv"
OUTPUT_REPORT = ROOT / "outputs" / "reports" / "synthetic_anomaly_recall.md"


def synthetic_recall(
    property_ledger: pd.DataFrame,
    *,
    n_per_direction: int = 100,
    shocks: tuple[float, ...] = (0.30, 0.40, 0.50),
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject price shocks into within-range rows and test whether interval labels catch them."""
    candidates = property_ledger.loc[
        property_ledger["anomaly_flag"].eq("within_expected_range")
        & property_ledger["observed_price"].notna()
        & property_ledger["lower_bound"].notna()
        & property_ledger["upper_bound"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=["scenario", "shock", "sample_size", "detected", "recall", "median_abs_gap_after_shock"]
        )

    sample_size = min(n_per_direction, len(candidates))
    rows: list[dict[str, object]] = []
    for shock in shocks:
        for scenario, multiplier, expected_flag in [
            ("synthetic_over_value", 1.0 + shock, "potentially_over_valued"),
            ("synthetic_under_value", 1.0 - shock, "potentially_under_valued"),
        ]:
            sample = candidates.sample(n=sample_size, random_state=random_state + len(rows)).copy()
            sample["synthetic_observed_price"] = sample["observed_price"] * multiplier
            sample["synthetic_flag"] = np.select(
                [
                    sample["synthetic_observed_price"] < sample["lower_bound"],
                    sample["synthetic_observed_price"] > sample["upper_bound"],
                ],
                ["potentially_under_valued", "potentially_over_valued"],
                default="within_expected_range",
            )
            detected = int(sample["synthetic_flag"].eq(expected_flag).sum())
            rows.append(
                {
                    "scenario": scenario,
                    "shock": shock,
                    "sample_size": sample_size,
                    "detected": detected,
                    "recall": detected / sample_size if sample_size else np.nan,
                    "median_abs_gap_after_shock": float(
                        (sample["synthetic_observed_price"] - sample["fair_value_hat"]).abs().median()
                    ),
                }
            )

    overall = pd.DataFrame(rows)
    for shock, group in overall.groupby("shock", sort=True):
        total_sample = int(group["sample_size"].sum())
        total_detected = int(group["detected"].sum())
        overall.loc[len(overall)] = {
            "scenario": "overall",
            "shock": float(shock),
            "sample_size": total_sample,
            "detected": total_detected,
            "recall": total_detected / total_sample if total_sample else np.nan,
            "median_abs_gap_after_shock": float("nan"),
        }
    return overall


def write_report(results: pd.DataFrame, path: Path) -> Path:
    lines = [
        "# Synthetic Anomaly Recall Check",
        "",
        "This is a sensitivity sanity check, not ground-truth precision/recall.",
        "It samples within-range transactions, injects +/-30%, +/-40%, and +/-50% sale-price shocks, and checks whether the existing interval decision rule catches the manipulated cases.",
        "",
        "| Scenario | Shock | Sample size | Detected | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.shock:.0%} | {int(row.sample_size)} | {int(row.detected)} | {row.recall:.1%} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: high synthetic recall means the interval rule reacts to large artificial price distortions. It does not estimate production precision because no human-reviewed anomaly labels are available.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    if not TABLE_PATH.exists():
        raise SystemExit(f"Missing property ledger: {TABLE_PATH}. Run scripts/run_pipeline.py first.")
    property_ledger = pd.read_csv(TABLE_PATH)
    results = synthetic_recall(property_ledger)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE, index=False)
    write_report(results, OUTPUT_REPORT)
    print(f"saved table: {OUTPUT_TABLE}")
    print(f"saved report: {OUTPUT_REPORT}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
