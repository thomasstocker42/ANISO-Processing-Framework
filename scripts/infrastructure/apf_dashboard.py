#!/usr/bin/env python3
"""
APF infrastructure tool: apf_dashboard.py

Create project-level dashboard summaries from APF manifests.

Outputs
-------
derivatives/summary/apf_dashboard_cases.csv
derivatives/summary/apf_dashboard_steps.csv
derivatives/summary/apf_dashboard_summary.json
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from apf_manifest import load_manifest, validate_manifest_schema
from apf_paths import get_paths


def step_status(step: dict) -> str:
    if step.get("completed") and step.get("valid"):
        return "valid"
    if step.get("completed") and not step.get("valid"):
        return "invalid"
    if step.get("date_started") and not step.get("date_finished"):
        return "started"
    return "not_started"


def main() -> int:
    paths = get_paths()
    summary_dir = paths.derivatives_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_files = sorted(paths.derivatives_dir.glob("*/processing/manifest.json"))

    case_rows = []
    step_rows = []

    total_cases = 0
    total_steps = 0
    valid_steps = 0
    invalid_steps = 0
    started_steps = 0
    not_started_steps = 0
    schema_invalid_cases = 0

    print("=" * 70)
    print("APF dashboard")
    print("=" * 70)

    for manifest_file in manifest_files:
        case_id = manifest_file.parents[1].name
        total_cases += 1

        try:
            manifest = load_manifest(case_id, paths=paths)
            schema_valid, schema_errors = validate_manifest_schema(manifest)
        except Exception as exc:
            schema_valid = False
            schema_errors = [str(exc)]
            manifest = None

        if not schema_valid:
            schema_invalid_cases += 1

        case_valid_steps = 0
        case_total_steps = 0

        if manifest is not None and schema_valid:
            for module_name, steps in manifest["modules"].items():
                for step_name, step in steps.items():
                    status = step_status(step)

                    total_steps += 1
                    case_total_steps += 1

                    if status == "valid":
                        valid_steps += 1
                        case_valid_steps += 1
                    elif status == "invalid":
                        invalid_steps += 1
                    elif status == "started":
                        started_steps += 1
                    else:
                        not_started_steps += 1

                    step_rows.append(
                        {
                            "case": case_id,
                            "case_type": manifest.get("case_type", ""),
                            "module": module_name,
                            "step": step_name,
                            "status": status,
                            "completed": step.get("completed"),
                            "valid": step.get("valid"),
                            "script": step.get("script", ""),
                            "script_version": step.get("script_version", ""),
                            "algorithm_version": step.get("algorithm_version", ""),
                            "date_started": step.get("date_started", ""),
                            "date_finished": step.get("date_finished", ""),
                            "duration_seconds": step.get("duration_seconds"),
                            "n_outputs": len(step.get("outputs", [])),
                            "n_warnings": len(step.get("warnings", [])),
                            "n_errors": len(step.get("errors", [])),
                            "notes": step.get("notes", ""),
                        }
                    )

        completion_fraction = (
            case_valid_steps / case_total_steps if case_total_steps else 0.0
        )

        case_rows.append(
            {
                "case": case_id,
                "case_type": manifest.get("case_type", "") if manifest else "",
                "schema_valid": schema_valid,
                "schema_errors": "; ".join(schema_errors),
                "valid_steps": case_valid_steps,
                "total_steps": case_total_steps,
                "completion_fraction": completion_fraction,
            }
        )

        print(
            f"{case_id}: "
            f"{case_valid_steps}/{case_total_steps} valid steps "
            f"schema={'valid' if schema_valid else 'INVALID'}"
        )

    cases_csv = summary_dir / "apf_dashboard_cases.csv"
    steps_csv = summary_dir / "apf_dashboard_steps.csv"
    summary_json = summary_dir / "apf_dashboard_summary.json"

    with cases_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    with steps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)

    summary = {
        "total_cases": total_cases,
        "schema_invalid_cases": schema_invalid_cases,
        "total_steps": total_steps,
        "valid_steps": valid_steps,
        "invalid_steps": invalid_steps,
        "started_steps": started_steps,
        "not_started_steps": not_started_steps,
        "completion_fraction": valid_steps / total_steps if total_steps else 0.0,
        "outputs": {
            "cases_csv": str(cases_csv),
            "steps_csv": str(steps_csv),
            "summary_json": str(summary_json),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Dashboard summary")
    print("=" * 70)
    print(f"Cases:       {total_cases}")
    print(f"Steps:       {total_steps}")
    print(f"Valid:       {valid_steps}")
    print(f"Invalid:     {invalid_steps}")
    print(f"Started:     {started_steps}")
    print(f"Not started: {not_started_steps}")
    print(f"Cases CSV:   {cases_csv}")
    print(f"Steps CSV:   {steps_csv}")
    print(f"Summary:     {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())