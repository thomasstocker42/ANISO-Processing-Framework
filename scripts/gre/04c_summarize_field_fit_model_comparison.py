#!/usr/bin/env python3
"""
APF GRE04c summary: cohort decision table for four-scenario field-fit comparison.

Scientific purpose
------------------
Summarize the finalized GRE04c four-scenario comparison into compact cohort
tables suitable for APF documentation and thesis Results.

Inputs
------
derivatives/summary/GRE/
  gre04c_field_fit_model_comparison_qc.csv

Outputs
-------
derivatives/summary/GRE/
  gre04c_field_fit_model_decision_table.md
  gre04c_field_fit_model_decision_summary.csv
  gre04c_field_fit_model_decision_summary.json

The script intentionally avoids pandas.to_markdown()/tabulate so it has no
optional tabulate dependency.

Run
---
./apf scripts/gre/04c_summarize_field_fit_model_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from _field_fit_common import now_iso, resolve_root


SCRIPT_VERSION = "0.1.0"
MODULE_NAME = "gre04c"
METHOD_NAME = "four_scenario_field_fit_decision_summary"

DEFAULT_INPUT = (
    Path("derivatives")
    / "summary"
    / "GRE"
    / "gre04c_field_fit_model_comparison_qc.csv"
)

DEFAULT_MARKDOWN = (
    Path("derivatives")
    / "summary"
    / "GRE"
    / "gre04c_field_fit_model_decision_table.md"
)

DEFAULT_SUMMARY_CSV = (
    Path("derivatives")
    / "summary"
    / "GRE"
    / "gre04c_field_fit_model_decision_summary.csv"
)

DEFAULT_SUMMARY_JSON = (
    Path("derivatives")
    / "summary"
    / "GRE"
    / "gre04c_field_fit_model_decision_summary.json"
)


PER_CASE_COLUMNS = [
    ("case_id", "Case", "s"),
    ("weighting_legacy_rms_hz", "Weighting no parity RMS Hz", ".4f"),
    ("parity_ols_rms_hz", "Parity OLS RMS Hz", ".4f"),
    ("parity_wls_rms_hz", "Parity WLS RMS Hz", ".4f"),
    ("weighting_parity_rms_hz", "Weighting + parity RMS Hz", ".4f"),
    ("interaction_rms_hz", "Interaction RMS Hz", ".4f"),
    ("median_ols_parity_rmse_rad", "OLS parity RMSE rad", ".4f"),
    ("median_wls_parity_rmse_rad", "WLS parity RMSE rad", ".4f"),
    ("median_ols_parity_se_hz", "OLS parity SE Hz", ".4f"),
    ("median_wls_parity_se_hz", "WLS parity SE Hz", ".4f"),
]

SUMMARY_COLUMNS = [
    "weighting_legacy_rms_hz",
    "parity_ols_rms_hz",
    "parity_wls_rms_hz",
    "weighting_parity_rms_hz",
    "interaction_rms_hz",
    "median_ols_parity_rmse_rad",
    "median_wls_parity_rmse_rad",
    "median_ols_parity_r2",
    "median_wls_parity_r2",
    "median_ols_parity_se_hz",
    "median_wls_parity_se_hz",
    "fraction_parity_lower_rmse_ols",
    "fraction_parity_lower_rmse_wls",
    "fraction_wls_lower_rmse_parity",
    "median_wls_legacy_weighted_rmse_rad",
    "median_wls_parity_weighted_rmse_rad",
    "median_wls_legacy_weighted_r2",
    "median_wls_parity_weighted_r2",
    "fraction_parity_lower_weighted_rmse_wls",
    "beta_ols_wls_pearson_r",
    "beta_wls_minus_ols_rms_rad",
]


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid or missing numeric value for column {key!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in column {key!r}: {value}")
    return value


def read_ok_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if row.get("status") == "OK"]

    if not rows:
        raise ValueError("No status=OK rows found in GRE04c cohort CSV")

    required = {
        "case_id",
        "status",
        *(key for key, _, _ in PER_CASE_COLUMNS),
        *SUMMARY_COLUMNS,
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(
            "Missing expected GRE04c columns:\n  " + "\n  ".join(missing)
        )

    return rows, fieldnames


def cohort_summary(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for key in SUMMARY_COLUMNS:
        values = np.asarray(
            [parse_float(row, key) for row in rows],
            dtype=np.float64,
        )
        result[key] = {
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return result


def markdown_table(
    headers: list[str],
    body: list[list[str]],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_value(raw: str, fmt: str) -> str:
    if fmt == "s":
        return raw
    return format(float(raw), fmt)


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    summary: dict[str, dict[str, float]],
) -> None:
    per_case_headers = [label for _, label, _ in PER_CASE_COLUMNS]
    per_case_body = [
        [
            format_value(row[key], fmt)
            for key, _, fmt in PER_CASE_COLUMNS
        ]
        for row in rows
    ]

    summary_headers = ["Metric", "Median", "Minimum", "Maximum"]
    summary_body = [
        [
            key,
            f"{summary[key]['median']:.6f}",
            f"{summary[key]['minimum']:.6f}",
            f"{summary[key]['maximum']:.6f}",
        ]
        for key in SUMMARY_COLUMNS
    ]

    text = (
        "# GRE04c field-fit decision table\n\n"
        "Generated by "
        "`scripts/gre/04c_summarize_field_fit_model_comparison.py`.\n\n"
        "## Per-case compact table\n\n"
        + markdown_table(per_case_headers, per_case_body)
        + "\n\n"
        "## Cohort summary\n\n"
        + markdown_table(summary_headers, summary_body)
        + "\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_summary_csv(
    path: Path,
    summary: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "median", "minimum", "maximum"],
        )
        writer.writeheader()
        for key in SUMMARY_COLUMNS:
            writer.writerow(
                {
                    "metric": key,
                    "median": summary[key]["median"],
                    "minimum": summary[key]["minimum"],
                    "maximum": summary[key]["maximum"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "APF GRE04c: summarize the four-scenario field-fit comparison "
            "into documentation-ready cohort tables."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="APF repository root; defaults to APF_ROOT or cwd",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Override GRE04c cohort QC CSV",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Override Markdown decision-table output",
    )
    parser.add_argument(
        "--summary-csv-output",
        type=Path,
        help="Override compact cohort-summary CSV output",
    )
    parser.add_argument(
        "--summary-json-output",
        type=Path,
        help="Override compact cohort-summary JSON output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    input_csv = (
        args.input_csv.resolve()
        if args.input_csv
        else root / DEFAULT_INPUT
    )
    markdown_output = (
        args.markdown_output.resolve()
        if args.markdown_output
        else root / DEFAULT_MARKDOWN
    )
    summary_csv_output = (
        args.summary_csv_output.resolve()
        if args.summary_csv_output
        else root / DEFAULT_SUMMARY_CSV
    )
    summary_json_output = (
        args.summary_json_output.resolve()
        if args.summary_json_output
        else root / DEFAULT_SUMMARY_JSON
    )

    input_csv = require_file(input_csv, "GRE04c cohort QC CSV")
    rows, _ = read_ok_rows(input_csv)
    summary = cohort_summary(rows)

    write_markdown(markdown_output, rows, summary)
    write_summary_csv(summary_csv_output, summary)

    payload = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "created_utc": now_iso(),
        "input_csv": str(input_csv),
        "n_ok_cases": len(rows),
        "summary": summary,
        "outputs": {
            "markdown": str(markdown_output),
            "summary_csv": str(summary_csv_output),
            "summary_json": str(summary_json_output),
        },
    }
    summary_json_output.parent.mkdir(parents=True, exist_ok=True)
    summary_json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"APF GRE04c summary v{SCRIPT_VERSION}")
    print(f"Root:       {root}")
    print(f"Input:      {input_csv}")
    print(f"OK cases:   {len(rows)}")
    print()
    print("COHORT MEDIANS")
    print("=" * 72)

    for key in (
        "weighting_legacy_rms_hz",
        "parity_ols_rms_hz",
        "parity_wls_rms_hz",
        "weighting_parity_rms_hz",
        "interaction_rms_hz",
        "median_ols_parity_rmse_rad",
        "median_wls_parity_rmse_rad",
        "median_ols_parity_se_hz",
        "median_wls_parity_se_hz",
        "median_wls_legacy_weighted_rmse_rad",
        "median_wls_parity_weighted_rmse_rad",
        "fraction_parity_lower_weighted_rmse_wls",
    ):
        print(f"{key:44s}: {summary[key]['median']:.6f}")

    print()
    print(f"Markdown: {markdown_output}")
    print(f"CSV:      {summary_csv_output}")
    print(f"JSON:     {summary_json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
