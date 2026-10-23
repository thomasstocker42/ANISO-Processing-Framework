from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def add_case_selection_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("case", nargs="?", help="Single case ID")
    group.add_argument("--cases", nargs="+", help="Explicit list of case IDs")
    group.add_argument("--all", action="store_true", help="All active cases")
    group.add_argument(
        "--all-dwi-ready",
        action="store_true",
        help="All active DWI-ready cases",
    )
    group.add_argument(
        "--all-gre-ready",
        action="store_true",
        help="All active GRE-ready cases",
    )
    group.add_argument(
        "--all-mp2rage-ready",
        action="store_true",
        help="All active MP2RAGE-ready cases",
    )


def read_registry(registry_csv: Path) -> list[dict[str, str]]:
    if not registry_csv.exists():
        raise FileNotFoundError(f"Registry not found: {registry_csv}")

    with registry_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Registry contains no cases: {registry_csv}")

    if "case" not in rows[0]:
        raise ValueError(
            f"Registry lacks required 'case' column: {registry_csv}"
        )

    return rows


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def _validate_cases(
    cases: list[str],
    rows: list[dict[str, str]],
) -> None:
    known = {
        row["case"].strip()
        for row in rows
        if row.get("case", "").strip()
    }

    unknown = [case for case in cases if case not in known]

    if unknown:
        msg = [
            "Unknown case(s):",
            *[f"  - {case}" for case in unknown],
            "",
            "Known cases:",
            *[f"  - {case}" for case in sorted(known)],
        ]
        raise ValueError("\n".join(msg))


def _active_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Return active registry rows.

    Supported registry schemas:
    - status column, e.g. status=active
    - boolean active column
    - boolean include column

    The current APF registry uses the status column.
    """

    active_statuses = {
        "active",
        "new",
        "ready",
        "included",
    }

    active_rows: list[dict[str, str]] = []

    for row in rows:
        status = str(row.get("status", "")).strip().lower()

        if (
            status in active_statuses
            or truthy(row.get("active"))
            or truthy(row.get("include"))
        ):
            active_rows.append(row)

    return active_rows


def _ready_rows(
    rows: list[dict[str, str]],
    readiness_column: str,
) -> list[dict[str, str]]:
    """
    Return active rows marked ready for a specific modality.

    Examples of readiness columns:
    - dwi_ready
    - gre_ready
    - mp2rage_ready
    """

    return [
        row
        for row in _active_rows(rows)
        if truthy(row.get(readiness_column))
    ]


def _require_nonempty(
    cases: list[str],
    selection_name: str,
) -> list[str]:
    if not cases:
        raise ValueError(
            f"Case selection '{selection_name}' resolved to zero cases. "
            "Check the registry columns and values."
        )

    return cases


def resolve_cases(
    args: argparse.Namespace,
    registry_csv: Path,
) -> list[str]:
    rows = read_registry(registry_csv)

    if args.case:
        cases = [args.case]
        _validate_cases(cases, rows)
        return cases

    if args.cases:
        cases = unique_preserve_order(args.cases)
        _validate_cases(cases, rows)
        return cases

    if args.all:
        cases = [
            row["case"].strip()
            for row in _active_rows(rows)
            if row.get("case", "").strip()
        ]
        return _require_nonempty(cases, "--all")

    readiness_map = {
        "all_dwi_ready": "dwi_ready",
        "all_gre_ready": "gre_ready",
        "all_mp2rage_ready": "mp2rage_ready",
    }

    for arg_name, column in readiness_map.items():
        if getattr(args, arg_name, False):
            cases = [
                row["case"].strip()
                for row in _ready_rows(rows, column)
                if row.get("case", "").strip()
            ]
            return _require_nonempty(
                cases,
                f"--{arg_name.replace('_', '-')}",
            )

    raise ValueError("No case selection mode resolved.")


def print_selected_cases(cases: list[str]) -> None:
    print("=" * 72)
    print(f"Selected cases ({len(cases)})")
    print("=" * 72)

    for case in cases:
        print(f"  ✓ {case}")

    print("=" * 72)