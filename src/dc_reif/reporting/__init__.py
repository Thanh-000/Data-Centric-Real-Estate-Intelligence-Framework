"""Reporting and results presentation helpers for the current DC-REIF workflow."""

from dc_reif.diagnostics import DiagnosticsError, build_diagnostics_artifacts
from dc_reif.reporting.artifacts import (
    create_eda_figures,
    save_dataframe,
    save_json,
    write_summary_report,
)
from dc_reif.report_results import (
    ReportResultsError,
    build_final_results_summary,
    build_report_results_pack,
    latex_core_metrics_table,
    markdown_summary_block,
    terminal_results_block,
)

__all__ = [
    "DiagnosticsError",
    "ReportResultsError",
    "build_final_results_summary",
    "build_diagnostics_artifacts",
    "build_report_results_pack",
    "create_eda_figures",
    "latex_core_metrics_table",
    "markdown_summary_block",
    "save_dataframe",
    "save_json",
    "terminal_results_block",
    "write_summary_report",
]
