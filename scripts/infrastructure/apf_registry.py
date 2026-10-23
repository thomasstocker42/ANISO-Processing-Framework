#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
apf_registry.py

Purpose
-------
Provide the only approved interface for reading APF case registries.

Scientific scripts must not read config/InS_cases.csv or config/ExS_cases.csv
directly. They should call functions from this module.

Version
-------
APF core version: 0.1
Created: 2026-06-17
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import csv

from apf_paths import get_paths, APFPaths


@dataclass(frozen=True)
class APFCase:
    """Single APF case registry entry."""

    case: str
    status: str
    type: str
    date_added: str
    raw_root: str
    notes: str = ""


VALID_DEFAULT_STATUSES = ("active", "new")


def _read_registry_csv(path: Path) -> list[APFCase]:
    """Read a case registry CSV into APFCase objects."""
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")

    cases: list[APFCase] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"case", "status", "type", "date_added", "raw_root"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Registry {path} is missing required columns: {sorted(missing)}")

        for row in reader:
            case_id = (row.get("case") or "").strip()
            if not case_id:
                continue

            cases.append(
                APFCase(
                    case=case_id,
                    status=(row.get("status") or "").strip(),
                    type=(row.get("type") or "").strip(),
                    date_added=(row.get("date_added") or "").strip(),
                    raw_root=(row.get("raw_root") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            )

    return cases


def get_cases(
    *,
    case_type: str = "InS",
    status: Optional[Iterable[str]] = VALID_DEFAULT_STATUSES,
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> list[APFCase]:
    """
    Return registered cases.

    Parameters
    ----------
    case_type:
        "InS" or "ExS".
    status:
        Iterable of allowed status values. Default: active and new.
        Use status=None to return all statuses.
    root:
        Optional project root override.
    paths:
        Optional APFPaths object. If supplied, root is ignored.

    Returns
    -------
    list[APFCase]
    """
    paths = paths or get_paths(root)

    if case_type == "InS":
        registry_path = paths.ins_registry
    elif case_type == "ExS":
        registry_path = paths.exs_registry
    else:
        raise ValueError(f"Unsupported case_type: {case_type!r}")

    cases = _read_registry_csv(registry_path)

    if status is not None:
        allowed = {s.lower() for s in status}
        cases = [c for c in cases if c.status.lower() in allowed]

    return cases


def get_case(
    case_id: str,
    *,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> APFCase:
    """Return one case by case ID."""
    cases = get_cases(case_type=case_type, status=None, root=root, paths=paths)
    matches = [c for c in cases if c.case == case_id]

    if not matches:
        raise KeyError(f"Case not found in {case_type} registry: {case_id}")

    if len(matches) > 1:
        raise ValueError(f"Duplicate case entries found for {case_id}")

    return matches[0]


def select_cases_from_cli(
    *,
    case: Optional[str],
    cases: Optional[str],
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> list[APFCase]:
    """
    Resolve common APF CLI case selection.

    Exactly one of `case` or `cases` must be supplied.

    Examples
    --------
    select_cases_from_cli(case="ANISO_00_000_InS", cases=None)
    select_cases_from_cli(case=None, cases="all")
    """
    if bool(case) == bool(cases):
        raise ValueError("Specify exactly one of --case CASE_ID or --cases all.")

    if case:
        return [get_case(case, case_type=case_type, root=root, paths=paths)]

    if cases != "all":
        raise ValueError("Only --cases all is currently supported.")

    return get_cases(case_type=case_type, status=VALID_DEFAULT_STATUSES, root=root, paths=paths)
