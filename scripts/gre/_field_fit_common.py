#!/usr/bin/env python3
"""
Shared utilities for APF GRE04 multi-echo field estimation.

This module contains:
- APF root and case discovery
- GRE03/GRE02 input discovery
- echo-time parsing
- geometry validation
- NIfTI writing
- summary/provenance helpers
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np


GYROMAGNETIC_RATIO_HZ_PER_T = 42.57747892e6


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        root = cli_root.expanduser().resolve()
    elif os.environ.get("APF_ROOT"):
        root = Path(os.environ["APF_ROOT"]).expanduser().resolve()
    else:
        root = Path.cwd().resolve()

    if not root.exists():
        raise FileNotFoundError(f"APF root does not exist: {root}")
    return root


def unique_match(paths: Iterable[Path], description: str) -> Path:
    unique = sorted({p.resolve() for p in paths if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"No {description} found")
    if len(unique) > 1:
        joined = "\n  ".join(str(p) for p in unique)
        raise RuntimeError(
            f"Ambiguous {description}; found {len(unique)} files:\n  {joined}"
        )
    return unique[0]


def case_search_roots(root: Path, case_id: str) -> list[Path]:
    candidates = [
        root / "derivatives" / case_id,
        root / "nifti" / case_id,
        root / "processing" / case_id,
    ]
    return [p for p in candidates if p.exists()]


def glob_many(roots: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for base in roots:
        for pattern in patterns:
            found.extend(base.rglob(pattern))
    return found


def discover_unwrapped_phase(roots: Sequence[Path]) -> Path:
    preferred = [
        p
        for p in glob_many(
            roots,
            [
                "*GRE_phase_unwrapped_romeo_rad_4d.nii.gz",
                "*GRE_phase_unwrapped_romeo_rad_4d.nii",
            ],
        )
        if "/GRE/03_romeo_unwrap/" in str(p)
    ]
    if preferred:
        return unique_match(preferred, "GRE03 ROMEO unwrapped 4D phase")

    fallback = [
        p
        for p in glob_many(
            roots,
            [
                "*phase_unwrapped_romeo*4d*.nii.gz",
                "*phase_unwrapped_romeo*4d*.nii",
            ],
        )
        if "/04a_field_ols/" not in str(p)
        and "/04b_field_wls/" not in str(p)
    ]
    return unique_match(fallback, "GRE03 ROMEO unwrapped 4D phase")


def discover_romeo_mask(roots: Sequence[Path]) -> Path:
    preferred = [
        p
        for p in glob_many(
            roots,
            [
                "*GRE_romeo_mask.nii.gz",
                "*GRE_romeo_mask.nii",
            ],
        )
        if "/GRE/03_romeo_unwrap/" in str(p)
    ]
    if preferred:
        return unique_match(preferred, "GRE03 effective ROMEO mask")

    fallback = glob_many(
        roots,
        [
            "*romeo_mask.nii.gz",
            "*romeo_mask.nii",
        ],
    )
    return unique_match(fallback, "GRE03 effective ROMEO mask")


def discover_echo_times(roots: Sequence[Path]) -> Path:
    all_files = glob_many(
        roots,
        [
            "*echo_times_ms.txt",
            "*echo_times_s.txt",
            "*echo_times*.tsv",
            "*echo_times*.csv",
            "*echo_times*.json",
        ],
    )

    candidates = [
        p
        for p in all_files
        if "relaxometry" not in str(p).lower()
        and "rlx" not in p.name.lower()
    ]

    gre02_ms = [
        p
        for p in candidates
        if "phase_preparation_gre02" in str(p).lower()
        and "echo_times_ms" in p.name.lower()
    ]
    if len(gre02_ms) == 1:
        return gre02_ms[0].resolve()
    if len(gre02_ms) > 1:
        return unique_match(
            gre02_ms,
            "canonical GRE02 echo-time file in milliseconds",
        )

    gre02_s = [
        p
        for p in candidates
        if "phase_preparation_gre02" in str(p).lower()
        and "echo_times_s" in p.name.lower()
    ]
    if len(gre02_s) == 1:
        return gre02_s[0].resolve()
    if len(gre02_s) > 1:
        return unique_match(
            gre02_s,
            "canonical GRE02 echo-time file in seconds",
        )

    return unique_match(candidates, "GRE02 echo-time file")


def parse_echo_times_ms(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        candidates = None
        if isinstance(obj, list):
            candidates = obj
        elif isinstance(obj, dict):
            for key in (
                "echo_times_ms",
                "EchoTimesMs",
                "echo_times",
                "EchoTimes",
                "TE_ms",
                "TEs",
            ):
                if key in obj:
                    candidates = obj[key]
                    break
        if candidates is None:
            raise ValueError(f"Could not locate echo-time array in JSON: {path}")
        values = [float(x) for x in candidates]
    else:
        tokens = re.findall(
            r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            text,
        )
        values = [float(x) for x in tokens]

    if not values:
        raise ValueError(f"No echo times parsed from {path}")
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"Echo times must be finite and positive: {values}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"Echo times must be strictly increasing: {values}")

    if max(values) < 1.0:
        values = [1000.0 * v for v in values]

    return values


def validate_geometry(
    phase_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    tes_ms: Sequence[float],
) -> tuple[int, int, int, int]:
    phase_shape = tuple(int(x) for x in phase_img.shape)
    mask_shape = tuple(int(x) for x in mask_img.shape)

    if len(phase_shape) != 4:
        raise ValueError(f"Unwrapped phase must be 4D; got {phase_shape}")
    if mask_shape != phase_shape[:3]:
        raise ValueError(
            f"Mask shape {mask_shape} does not match phase shape {phase_shape[:3]}"
        )
    if phase_shape[3] != len(tes_ms):
        raise ValueError(
            f"Echo-count mismatch: phase has {phase_shape[3]} echoes, "
            f"TE file has {len(tes_ms)}"
        )
    if not np.allclose(
        phase_img.affine,
        mask_img.affine,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("Phase and mask affines differ")

    return phase_shape


def make_output_header(
    reference_img: nib.Nifti1Image,
    dtype: np.dtype | type,
) -> nib.Nifti1Header:
    header = reference_img.header.copy()
    header.set_data_dtype(dtype)
    return header


def save_nifti(
    data: np.ndarray,
    reference_img: nib.Nifti1Image,
    path: Path,
    dtype: np.dtype | type,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=dtype)
    header = make_output_header(reference_img, dtype)
    nib.save(
        nib.Nifti1Image(array, reference_img.affine, header),
        str(path),
    )


def write_case_summary_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_cohort_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    dict_rows = [asdict(row) for row in rows]
    fieldnames = list(dict_rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def list_cases(root: Path) -> list[str]:
    case_ids: set[str] = set()
    for parent_name in ("derivatives", "nifti", "processing"):
        parent = root / parent_name
        if not parent.exists():
            continue
        for p in parent.iterdir():
            if p.is_dir() and re.fullmatch(r"ANISO_\d{2}_\d{3}_InS", p.name):
                case_ids.add(p.name)
    return sorted(case_ids)
