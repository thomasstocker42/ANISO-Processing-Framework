#!/usr/bin/env python3
"""
APF GRE03: multi-echo phase unwrapping with ROMEO.

Purpose
-------
Run ROMEO on GRE02-prepared 4D phase data in radians, using the corresponding
4D magnitude data, echo times, and preferably the standardized eroded GRE mask.

Scientific boundary
-------------------
This module performs phase unwrapping only. It does not estimate the final
total-field map, remove background fields, perform dipole inversion, or produce
a susceptibility map.

Expected GRE02 inputs
---------------------
Per case, the script auto-discovers:
  - *phase_rad_4d*.nii[.gz]
  - *magnitude_4d*.nii[.gz]
  - an echo-time text/CSV/JSON file
  - canonical GRE02b full HD-BET brain mask, with legacy s11b compatibility

ROMEO is called with:
  - 4D phase
  - 4D magnitude
  - echo times in milliseconds
  - full GRE02b HD-BET mask by default; legacy s11b eroded mask as compatibility fallback
  - --mask-unwrapped
  - --no-phase-rescale, because GRE02 already wrote phase in radians

Outputs
-------
derivatives/<CASE>/GRE/03_romeo_unwrap/
  <CASE>_GRE_phase_unwrapped_romeo_rad_4d.nii.gz
  <CASE>_GRE_romeo_mask.nii.gz
  <CASE>_GRE_romeo_command.txt
  <CASE>_GRE_romeo_stdout.log
  <CASE>_GRE_romeo_stderr.log
  <CASE>_GRE_romeo_qc.json

Cohort outputs
--------------
derivatives/summary/GRE
  gre03_ROMEO_unwrap_qc.csv
  gre03_ROMEO_unwrap_run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "0.2.2"
MODULE_NAME = "gre03"
OUTPUT_SUBDIR = Path("GRE") / "03_romeo_unwrap"


@dataclass(frozen=True)
class Inputs:
    case_id: str
    phase_4d: Path
    magnitude_4d: Path
    echo_times_file: Path
    mask: Path | None
    mask_source: str


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None
    n_echoes: int | None = None
    te_min_ms: float | None = None
    te_max_ms: float | None = None
    finite_fraction_masked: float | None = None
    zero_fraction_masked: float | None = None
    output_min_rad: float | None = None
    output_max_rad: float | None = None
    output_median_rad: float | None = None
    output_path: str | None = None
    mask_path: str | None = None
    mask_source: str | None = None
    romeo_version: str | None = None


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


def find_executable(name_or_path: str) -> str:
    candidate = Path(name_or_path).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"ROMEO executable not found: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise PermissionError(f"ROMEO path is not executable: {candidate}")
        return str(candidate.resolve())

    found = shutil.which(name_or_path)
    if found is None:
        raise FileNotFoundError(
            f"Could not find '{name_or_path}' on PATH. "
            "Install ROMEO or pass --romeo-bin /absolute/path/to/romeo."
        )
    return found


def romeo_version(romeo_bin: str) -> str:
    try:
        cp = subprocess.run(
            [romeo_bin, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        text = (cp.stdout or cp.stderr).strip()
        return text if text else f"returncode={cp.returncode}"
    except Exception as exc:  # version reporting must not block processing
        return f"unavailable: {type(exc).__name__}: {exc}"


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def unique_match(paths: Iterable[Path], description: str) -> Path:
    unique = sorted({p.resolve() for p in paths if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"No {description} found")
    if len(unique) > 1:
        joined = "\n  ".join(str(p) for p in unique)
        raise RuntimeError(f"Ambiguous {description}; found {len(unique)} files:\n  {joined}")
    return unique[0]


def case_search_roots(root: Path, case_id: str) -> list[Path]:
    candidates = [
        root / "derivatives" / case_id,
        root / "processing" / case_id,
        root / "nifti" / case_id,
    ]
    return [p for p in candidates if p.exists()]


def glob_many(roots: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for base in roots:
        for pattern in patterns:
            found.extend(base.rglob(pattern))
    return found


def discover_phase(roots: Sequence[Path]) -> Path:
    """
    Discover the canonical GRE02 prepared phase volume.

    Preference order:
      1. Exact GRE02 phase-preparation output
      2. Guarded GRE02-named fallback

    Downstream GRE03/GRE04 products are explicitly excluded.
    """
    preferred = [
        p.resolve()
        for p in glob_many(
            roots,
            [
                "GRE/phase_preparation_gre02/gre02_*_phase_rad_4d.nii.gz",
                "GRE/phase_preparation_gre02/gre02_*_phase_rad_4d.nii",
                "gre/phase_preparation_gre02/gre02_*_phase_rad_4d.nii.gz",
                "gre/phase_preparation_gre02/gre02_*_phase_rad_4d.nii",
            ],
        )
        if p.is_file()
    ]
    if preferred:
        return unique_match(preferred, "canonical GRE02 4D phase-in-radians input")

    fallback = [
        p.resolve()
        for p in glob_many(
            roots,
            [
                "gre02_*_phase_rad_4d.nii.gz",
                "gre02_*_phase_rad_4d.nii",
            ],
        )
        if p.is_file()
        and "03_romeo_unwrap" not in str(p).lower()
        and "04a_field_ols" not in str(p).lower()
        and "04b_field_wls" not in str(p).lower()
        and "predicted" not in p.name.lower()
        and "residual" not in p.name.lower()
        and "unwrapped" not in p.name.lower()
    ]
    return unique_match(fallback, "GRE02 4D phase-in-radians input")


def discover_magnitude(roots: Sequence[Path]) -> Path:
    """
    Discover the canonical GRE02 prepared magnitude volume.

    Preference order:
      1. Exact GRE02 phase-preparation output
      2. Guarded GRE02-named fallback
    """
    preferred = [
        p.resolve()
        for p in glob_many(
            roots,
            [
                "GRE/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz",
                "GRE/phase_preparation_gre02/gre02_*_magnitude_4d.nii",
                "gre/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz",
                "gre/phase_preparation_gre02/gre02_*_magnitude_4d.nii",
            ],
        )
        if p.is_file()
    ]
    if preferred:
        return unique_match(preferred, "canonical GRE02 4D magnitude input")

    fallback = [
        p.resolve()
        for p in glob_many(
            roots,
            [
                "gre02_*_magnitude_4d.nii.gz",
                "gre02_*_magnitude_4d.nii",
            ],
        )
        if p.is_file()
        and "02b_brain_mask" not in str(p).lower()
        and "03_romeo_unwrap" not in str(p).lower()
        and "04a_field_ols" not in str(p).lower()
        and "04b_field_wls" not in str(p).lower()
    ]
    return unique_match(fallback, "GRE02 4D magnitude input")


def discover_echo_times(roots: Sequence[Path]) -> Path:
    """
    Discover the canonical GRE02 echo-time file.

    Preference order:
      1. GRE02 echo times in milliseconds
      2. GRE02 echo times in seconds
      3. Generic GRE echo-time files as a guarded fallback

    Relaxometry echo-time files are explicitly excluded.
    """
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

    gre_ms = [
        p
        for p in candidates
        if "gre" in str(p).lower()
        and "echo_times_ms" in p.name.lower()
    ]
    if len(gre_ms) == 1:
        return gre_ms[0].resolve()

    return unique_match(candidates, "GRE02 echo-time file")

def discover_gre_analysis_mask(
    roots: Sequence[Path],
) -> tuple[Path | None, str]:
    """
    Discover the canonical GRE analysis mask.

    Strict preference order:
      1. APF GRE02b full HD-BET brain mask
      2. Legacy s11b one-voxel-eroded HD-BET mask
      3. No mask; ROMEO robustmask remains available only through the
         explicit --allow-romeo-robustmask option

    Generic masks, non-eroded masks, relaxometry masks, and previous GRE03
    output masks are intentionally not accepted. This prevents silent changes
    in the anatomical analysis domain.
    """
    apf_masks: list[Path] = []
    legacy_masks: list[Path] = []

    for case_root in roots:
        # APF-native canonical mask.
        apf_masks.extend(
            case_root.glob(
                "GRE/02b_brain_mask/*_GRE02b_brain_mask.nii.gz"
            )
        )
        apf_masks.extend(
            case_root.glob(
                "GRE/02b_brain_mask/*_GRE02b_brain_mask.nii"
            )
        )
        apf_masks.extend(
            case_root.glob(
                "gre/02b_brain_mask/*_GRE02b_brain_mask.nii.gz"
            )
        )
        apf_masks.extend(
            case_root.glob(
                "gre/02b_brain_mask/*_GRE02b_brain_mask.nii"
            )
        )

        # Backward-compatible legacy mask generated by InS scripts 11/11b.
        legacy_masks.extend(
            case_root.glob(
                "brain_masks/s11b_GRE_e1_hdbet_cpu_notta_mask_ero1.nii.gz"
            )
        )
        legacy_masks.extend(
            case_root.glob(
                "brain_masks/s11b_GRE_e1_hdbet_cpu_notta_mask_ero1.nii"
            )
        )

    apf_unique = sorted({p.resolve() for p in apf_masks if p.is_file()})
    if len(apf_unique) == 1:
        return apf_unique[0], "gre02b_apf_full"
    if len(apf_unique) > 1:
        return (
            unique_match(apf_unique, "APF GRE02b full brain mask"),
            "gre02b_apf_full",
        )

    legacy_unique = sorted({p.resolve() for p in legacy_masks if p.is_file()})
    if len(legacy_unique) == 1:
        return legacy_unique[0], "legacy_s11b_ero1"
    if len(legacy_unique) > 1:
        return (
            unique_match(legacy_unique, "legacy s11b eroded GRE mask"),
            "legacy_s11b_ero1",
        )

    return None, "none"

def parse_echo_times(path: Path) -> list[float]:
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
        # Accept whitespace, comma, semicolon, or TSV separated values and ignore labels.
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

    # GRE02 is expected to write milliseconds. A seconds-valued vector would usually
    # have a maximum below 1; convert explicitly and record the effective values.
    if max(values) < 1.0:
        values = [1000.0 * v for v in values]

    return values


def validate_inputs(inputs: Inputs, tes_ms: Sequence[float]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    phase_img = nib.load(str(inputs.phase_4d))
    mag_img = nib.load(str(inputs.magnitude_4d))
    phase_shape = tuple(int(x) for x in phase_img.shape)
    mag_shape = tuple(int(x) for x in mag_img.shape)

    if len(phase_shape) != 4:
        raise ValueError(f"Phase must be 4D; got {phase_shape}: {inputs.phase_4d}")
    if phase_shape != mag_shape:
        raise ValueError(f"Phase/magnitude shape mismatch: {phase_shape} vs {mag_shape}")
    if phase_shape[3] != len(tes_ms):
        raise ValueError(
            f"Echo count mismatch: phase has {phase_shape[3]} echoes, "
            f"TE file has {len(tes_ms)} values"
        )
    if not np.allclose(phase_img.affine, mag_img.affine, rtol=1e-5, atol=1e-5):
        raise ValueError("Phase and magnitude affines differ")

    if inputs.mask is not None:
        mask_img = nib.load(str(inputs.mask))
        mask_shape = tuple(int(x) for x in mask_img.shape)
        if mask_shape != phase_shape[:3]:
            raise ValueError(
                f"Mask shape {mask_shape} does not match phase spatial shape {phase_shape[:3]}"
            )
        if not np.allclose(phase_img.affine, mask_img.affine, rtol=1e-5, atol=1e-5):
            raise ValueError("Phase and mask affines differ")

    # Avoid loading the entire 4D volume solely to inspect range.
    sample = np.asanyarray(phase_img.dataobj[..., 0], dtype=np.float32)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError("First phase echo contains no finite values")
    if float(np.nanmax(np.abs(finite))) > (math.pi + 0.25):
        raise ValueError(
            "GRE02 phase does not appear confined to approximately [-pi, pi]. "
            "Refusing to pass --no-phase-rescale."
        )

    return phase_shape, mag_shape


def format_te_argument(tes_ms: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in tes_ms) + "]"


def build_command(
    romeo_bin: str,
    inputs: Inputs,
    output_nii: Path,
    tes_ms: Sequence[float],
    *,
    allow_romeo_robustmask: bool,
    no_mmap: bool,
    verbose: bool,
) -> list[str]:
    cmd = [
        romeo_bin,
        "-p",
        str(inputs.phase_4d),
        "-m",
        str(inputs.magnitude_4d),
        "-t",
        format_te_argument(tes_ms),
        "-o",
        str(output_nii),
        "--no-phase-rescale",
    ]

    if inputs.mask is not None:
        cmd += ["-k", str(inputs.mask), "--mask-unwrapped"]
    elif allow_romeo_robustmask:
        cmd += ["-k", "robustmask", "--mask-unwrapped"]
    else:
        raise RuntimeError(
            "No canonical GRE analysis mask was found. Run GRE02b, supply --mask, "
            "or explicitly permit ROMEO's generated robust mask with "
            "--allow-romeo-robustmask."
        )

    if no_mmap:
        cmd.append("--no-mmap")
    if verbose:
        cmd.append("--verbose")
    return cmd


def write_mask_copy(mask_source: Path | None, output_mask: Path, output_nii: Path) -> None:
    if mask_source is not None:
        img = nib.load(str(mask_source))
        data = (np.asanyarray(img.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(output_mask))
        return

    # When ROMEO generated its own robust mask and did not expose a separate mask,
    # derive the effective support from voxels that are nonzero in any output echo.
    out_img = nib.load(str(output_nii))
    data = np.asanyarray(out_img.dataobj)
    mask = np.any(np.isfinite(data) & (data != 0), axis=3).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, out_img.affine, out_img.header), str(output_mask))


def qc_output(output_nii: Path, mask_path: Path, expected_shape: tuple[int, ...]) -> dict:
    out_img = nib.load(str(output_nii))
    out_shape = tuple(int(x) for x in out_img.shape)
    if out_shape != expected_shape:
        raise ValueError(f"ROMEO output shape mismatch: expected {expected_shape}, got {out_shape}")

    data = np.asanyarray(out_img.dataobj, dtype=np.float32)
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
    if mask.shape != expected_shape[:3]:
        raise ValueError(f"QC mask shape mismatch: {mask.shape} vs {expected_shape[:3]}")
    if not np.any(mask):
        raise ValueError("QC mask is empty")

    masked = data[mask, :]
    finite = np.isfinite(masked)
    finite_fraction = float(finite.mean())
    finite_values = masked[finite]
    if finite_values.size == 0:
        raise ValueError("ROMEO output has no finite voxels inside mask")

    return {
        "shape": list(out_shape),
        "dtype": str(data.dtype),
        "finite_fraction_masked": finite_fraction,
        "zero_fraction_masked": float(np.mean(masked == 0)),
        "min_rad": float(np.min(finite_values)),
        "max_rad": float(np.max(finite_values)),
        "median_rad": float(np.median(finite_values)),
        "p01_rad": float(np.percentile(finite_values, 1)),
        "p99_rad": float(np.percentile(finite_values, 99)),
    }


def discover_inputs(
    root: Path,
    case_id: str,
    phase: Path | None,
    magnitude: Path | None,
    echo_times: Path | None,
    mask: Path | None,
) -> Inputs:
    roots = case_search_roots(root, case_id)
    if not roots:
        raise FileNotFoundError(f"No case directories found for {case_id}")

    if mask is not None:
        resolved_mask = mask.expanduser().resolve()
        mask_source = "explicit_cli"
    else:
        resolved_mask, mask_source = discover_gre_analysis_mask(roots)

    return Inputs(
        case_id=case_id,
        phase_4d=phase.resolve() if phase else discover_phase(roots),
        magnitude_4d=magnitude.resolve() if magnitude else discover_magnitude(roots),
        echo_times_file=echo_times.resolve() if echo_times else discover_echo_times(roots),
        mask=resolved_mask,
        mask_source=mask_source,
    )


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


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case_id / OUTPUT_SUBDIR
    prefix = f"{case_id}_GRE"
    return {
        "dir": outdir,
        "unwrapped": outdir / f"{prefix}_phase_unwrapped_romeo_rad_4d.nii.gz",
        "mask": outdir / f"{prefix}_romeo_mask.nii.gz",
        "command": outdir / f"{prefix}_romeo_command.txt",
        "stdout": outdir / f"{prefix}_romeo_stdout.log",
        "stderr": outdir / f"{prefix}_romeo_stderr.log",
        "qc": outdir / f"{prefix}_romeo_qc.json",
    }


def process_case(
    root: Path,
    case_id: str,
    romeo_bin: str,
    romeo_ver: str,
    args: argparse.Namespace,
) -> CaseResult:
    started = time.perf_counter()
    paths = output_paths(root, case_id)

    try:
        explicit_allowed = len(args.case) == 1
        inputs = discover_inputs(
            root,
            case_id,
            args.phase if explicit_allowed else None,
            args.magnitude if explicit_allowed else None,
            args.echo_times if explicit_allowed else None,
            args.mask if explicit_allowed else None,
        )
        tes_ms = parse_echo_times(inputs.echo_times_file)
        phase_shape, _ = validate_inputs(inputs, tes_ms)
        effective_mask_source = (
            inputs.mask_source
            if inputs.mask is not None
            else "romeo_robustmask"
            if args.allow_romeo_robustmask
            else "none"
        )

        if paths["unwrapped"].exists() and not args.force:
            return CaseResult(
                case_id=case_id,
                status="SKIPPED_EXISTS",
                message="Output exists; use --force to overwrite",
                output_path=str(paths["unwrapped"]),
                mask_path=str(paths["mask"]) if paths["mask"].exists() else None,
                mask_source=effective_mask_source,
                romeo_version=romeo_ver,
            )

        cmd = build_command(
            romeo_bin,
            inputs,
            paths["unwrapped"],
            tes_ms,
            allow_romeo_robustmask=args.allow_romeo_robustmask,
            no_mmap=args.no_mmap,
            verbose=args.romeo_verbose,
        )

        if args.dry_run:
            print(f"[DRY-RUN] {case_id}: mask_source={effective_mask_source}")
            print(f"[DRY-RUN] {case_id}: {shlex.join(cmd)}")
            return CaseResult(
                case_id=case_id,
                status="DRY_RUN",
                message=shlex.join(cmd),
                n_echoes=len(tes_ms),
                te_min_ms=min(tes_ms),
                te_max_ms=max(tes_ms),
                mask_path=str(inputs.mask) if inputs.mask else None,
                mask_source=effective_mask_source,
                romeo_version=romeo_ver,
            )

        paths["dir"].mkdir(parents=True, exist_ok=True)
        if args.force:
            for key in ("unwrapped", "mask", "command", "stdout", "stderr", "qc"):
                paths[key].unlink(missing_ok=True)

        paths["command"].write_text(shlex.join(cmd) + "\n", encoding="utf-8")

        cp = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )
        paths["stdout"].write_text(cp.stdout or "", encoding="utf-8")
        paths["stderr"].write_text(cp.stderr or "", encoding="utf-8")

        if cp.returncode != 0:
            raise RuntimeError(
                f"ROMEO failed with return code {cp.returncode}; "
                f"see {paths['stderr']} and {paths['stdout']}"
            )
        if not paths["unwrapped"].exists():
            raise FileNotFoundError(
                f"ROMEO returned success but expected output is missing: {paths['unwrapped']}"
            )

        write_mask_copy(inputs.mask, paths["mask"], paths["unwrapped"])
        qc = qc_output(paths["unwrapped"], paths["mask"], phase_shape)
        runtime_s = time.perf_counter() - started

        provenance = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": runtime_s,
            "inputs": {
                "phase_4d": str(inputs.phase_4d),
                "magnitude_4d": str(inputs.magnitude_4d),
                "echo_times_file": str(inputs.echo_times_file),
                "mask": str(inputs.mask) if inputs.mask else None,
                "mask_source": effective_mask_source,
            },
            "echo_times_ms": tes_ms,
            "romeo": {
                "executable": romeo_bin,
                "version": romeo_ver,
                "command": cmd,
                "phase_rescaling_disabled": True,
                "magnitude_supplied": True,
                "explicit_mask_supplied": inputs.mask is not None,
                "mask_source": effective_mask_source,
                "mask_discovery_policy": [
                    "gre02b_apf_full",
                    "legacy_s11b_ero1",
                    "romeo_robustmask_only_if_explicitly_allowed",
                ],
                "mask_unwrapped_output": True,
                "individual_unwrapping": False,
                "compute_B0": False,
                "phase_offset_correction": False,
            },
            "outputs": {
                "unwrapped_phase_rad_4d": str(paths["unwrapped"]),
                "effective_mask": str(paths["mask"]),
                "stdout": str(paths["stdout"]),
                "stderr": str(paths["stderr"]),
            },
            "qc": qc,
        }
        paths["qc"].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

        return CaseResult(
            case_id=case_id,
            status="OK",
            message="ROMEO unwrapping completed",
            runtime_s=runtime_s,
            n_echoes=len(tes_ms),
            te_min_ms=min(tes_ms),
            te_max_ms=max(tes_ms),
            finite_fraction_masked=qc["finite_fraction_masked"],
            zero_fraction_masked=qc["zero_fraction_masked"],
            output_min_rad=qc["min_rad"],
            output_max_rad=qc["max_rad"],
            output_median_rad=qc["median_rad"],
            output_path=str(paths["unwrapped"]),
            mask_path=str(paths["mask"]),
            mask_source=effective_mask_source,
            romeo_version=romeo_ver,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - started,
            romeo_version=romeo_ver,
        )


def write_summary(root: Path, results: Sequence[CaseResult], run_metadata: dict) -> None:
    summary_dir = root / "derivatives" / "summary"/ "GRE"
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "gre03_ROMEO_unwrap_qc.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else list(CaseResult.__annotations__)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    run_path = summary_dir / "gre03_ROMEO_unwrap_run.json"
    run_metadata["results"] = [asdict(r) for r in results]
    run_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="APF GRE03: unwrap GRE02 multi-echo phase with ROMEO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, help="APF repository root; defaults to APF_ROOT or cwd")
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID; repeat for multiple cases",
    )
    p.add_argument(
        "--all-gre-ready",
        action="store_true",
        help="Attempt all discovered ANISO_*_InS cases; input validation determines readiness",
    )
    p.add_argument("--romeo-bin", default="romeo", help="ROMEO executable or absolute path")
    p.add_argument("--force", action="store_true", help="Overwrite existing GRE03 outputs")
    p.add_argument("--dry-run", action="store_true", help="Validate and print commands only")
    p.add_argument(
        "--allow-romeo-robustmask",
        action="store_true",
        help="Permit ROMEO robustmask only when no explicit GRE mask is found",
    )
    p.add_argument(
        "--no-mmap",
        action="store_true",
        help="Pass --no-mmap to ROMEO, useful on network-mounted storage",
    )
    p.add_argument("--romeo-verbose", action="store_true", help="Pass --verbose to ROMEO")

    # Explicit single-case overrides.
    p.add_argument("--phase", type=Path, help="Explicit 4D phase input; single case only")
    p.add_argument("--magnitude", type=Path, help="Explicit 4D magnitude input; single case only")
    p.add_argument("--echo-times", type=Path, help="Explicit TE file; single case only")
    p.add_argument("--mask", type=Path, help="Explicit 3D mask; single case only")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit("Select at least one --case CASE_ID or use --all-gre-ready")

    explicit_inputs = [args.phase, args.magnitude, args.echo_times, args.mask]
    if any(x is not None for x in explicit_inputs) and len(cases) != 1:
        raise SystemExit("Explicit --phase/--magnitude/--echo-times/--mask require one case")

    romeo_bin = find_executable(args.romeo_bin)
    version = romeo_version(romeo_bin)

    run_metadata = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "started_utc": now_iso(),
        "root": str(root),
        "romeo_executable": romeo_bin,
        "romeo_version": version,
        "cases_requested": cases,
        "arguments": vars(args) | {
            "root": str(args.root) if args.root else None,
            "phase": str(args.phase) if args.phase else None,
            "magnitude": str(args.magnitude) if args.magnitude else None,
            "echo_times": str(args.echo_times) if args.echo_times else None,
            "mask": str(args.mask) if args.mask else None,
        },
    }

    print(f"APF GRE03 v{SCRIPT_VERSION}")
    print(f"Root:  {root}")
    print(f"ROMEO: {romeo_bin}")
    print(f"Version: {version}")
    print(f"Cases: {len(cases)}")

    results: list[CaseResult] = []
    for case_id in cases:
        result = process_case(root, case_id, romeo_bin, version, args)
        results.append(result)
        print(f"[{result.status}] {case_id}: {result.message}")

    run_metadata["finished_utc"] = now_iso()
    write_summary(root, results, run_metadata)

    n_failed = sum(r.status == "FAILED" for r in results)
    n_ok = sum(r.status == "OK" for r in results)
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
