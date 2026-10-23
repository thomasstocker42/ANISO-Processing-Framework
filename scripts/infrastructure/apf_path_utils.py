#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
apf_path_utils.py

Purpose
-------
Utilities for APF path serialization and reconstruction.

APF scripts may use absolute paths internally during execution, but persisted
metadata should store project-relative paths whenever possible.

Version
-------
APF core version: 0.3
Created: 2026-06-30
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from apf_paths import APFPaths


PROJECT_MARKERS: tuple[str, ...] = (
    "raw_dicom",
    "nifti",
    "derivatives",
    "scripts",
    "config",
    "docs",
    "results",
    "figures",
)


def _as_path(path: str | Path) -> Path:
    """Return path as a pathlib.Path."""
    return path if isinstance(path, Path) else Path(path)


def _strip_known_project_prefix(path: Path, markers: Iterable[str] = PROJECT_MARKERS) -> Path | None:
    """
    Convert a path containing a known APF project marker into a project-relative path.

    Example
    -------
    /path/to/APF/nifti/example_case/file.nii.gz
    -> nifti/case/file.nii.gz
    """
    parts = path.parts

    for marker in markers:
        if marker in parts:
            idx = parts.index(marker)
            return Path(*parts[idx:])

    return None


def to_project_path(path: str | Path, paths: APFPaths) -> str:
    """
    Serialize a path for APF metadata.

    Preferred output is project-relative, e.g.

        nifti/<case>/DWI/file.nii.gz
        derivatives/<case>/DWI/eddy/file.json

    If the path cannot be related to the APF project, the original path string is
    returned as a safe fallback.
    """
    p = _as_path(path)

    if str(p) == "":
        return ""

    if not p.is_absolute():
        return str(p)

    try:
        return str(p.resolve().relative_to(paths.root.resolve()))
    except ValueError:
        stripped = _strip_known_project_prefix(p)
        if stripped is not None:
            return str(stripped)

    return str(p)


def from_project_path(path: str | Path, paths: APFPaths) -> Path:
    """
    Reconstruct an executable absolute Path from APF metadata.

    Handles:
    - project-relative APF paths
    - absolute paths already under the current APF root
    - old absolute paths from previous roots, such as /mnt/c/.../ANISO_processing/...
    """
    p = _as_path(path)

    if str(p) == "":
        return p

    if not p.is_absolute():
        return paths.root / p

    try:
        p.resolve().relative_to(paths.root.resolve())
        return p
    except ValueError:
        stripped = _strip_known_project_prefix(p)
        if stripped is not None:
            return paths.root / stripped

    return p


def is_project_relative(path: str | Path) -> bool:
    """
    Return True if path is relative and begins with a known APF project marker.
    """
    p = _as_path(path)

    if p.is_absolute() or str(p) == "":
        return False

    return len(p.parts) > 0 and p.parts[0] in PROJECT_MARKERS