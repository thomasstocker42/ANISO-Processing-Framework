#!/usr/bin/env python3
"""
APF GRE 02: Prepare standardized multi-echo GRE phase and magnitude inputs.

Scientific scope
----------------
- discover and validate paired native GRE magnitude/phase echoes
- read echo times from JSON sidecars
- classify Siemens phase scaling from observed value ranges
- convert wrapped phase to radians in [-pi, pi)
- build matched 4D magnitude and phase stacks
- derive the B0 direction in NIfTI voxel coordinates
- write per-case provenance/QC and a persistent cohort summary

This stage deliberately does NOT perform:
- spatial phase unwrapping
- multi-echo field fitting
- background-field removal
- QSM inversion

Those operations belong to gre03 and later stages.

Examples
--------
  ./apf scripts/gre/02_prepare_phase.py CASE
  ./apf scripts/gre/02_prepare_phase.py --cases CASE1 CASE2
  ./apf scripts/gre/02_prepare_phase.py --all-gre-ready
  ./apf scripts/gre/02_prepare_phase.py CASE --force

Phase scaling modes
-------------------
  auto
      Detect one of the supported encodings from the observed data range.

  siemens_signed_4096
      phase_rad = raw * pi / 4096
      Expected raw range approximately [-4096, 4095].

  siemens_unsigned_4096
      phase_rad = raw * 2*pi / 4096 - pi
      Expected raw range approximately [0, 4095].

  radians
      Input is already in radians, approximately [-pi, pi].
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np

from apf_case_selection import add_case_selection_args, print_selected_cases
from apf_paths import APFPaths, get_paths
from apf_registry import VALID_DEFAULT_STATUSES, get_case, get_cases


SCRIPT = "02_prepare_phase.py"
VERSION = "0.1.1"
TAG = "gre02"
LEGACY_TAGS = ("s22_phase_preparation",)

EXPECTED_ECHOES = tuple(range(1, 13))
ECHO_PATTERN = re.compile(
    r"_e(?P<echo>\d+)(?:_ph)?\.nii(?:\.gz)?$",
    re.IGNORECASE,
)
GEOMETRY_ATOL = 1e-5
TE_ATOL_S = 1e-9

SCALING_MODES = (
    "auto",
    "siemens_signed_4096",
    "siemens_unsigned_4096",
    "radians",
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def project_relative(path: Path, paths: APFPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.root.resolve()))
    except ValueError:
        return str(path)


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def strip_nii_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def json_sidecar(nii: Path) -> Path:
    return nii.parent / f"{strip_nii_suffix(nii)}.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing JSON sidecar: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON sidecar: {path}: {error}") from error


def echo_number_from_name(path: Path) -> Optional[int]:
    match = ECHO_PATTERN.search(path.name)
    return int(match.group("echo")) if match else None


def nifti_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))


def gre_readiness(paths: APFPaths, case: str) -> tuple[bool, str]:
    gre_dir = paths.case_nifti_dir(case) / "GRE"
    mag_files = nifti_files(gre_dir / "mag")
    phase_files = nifti_files(gre_dir / "phase")

    if not gre_dir.is_dir():
        return False, "GRE directory missing"
    if not mag_files:
        return False, "GRE magnitude files missing"
    if not phase_files:
        return False, "GRE phase files missing"

    return True, f"{len(mag_files)} magnitude and {len(phase_files)} phase file(s)"


def resolve_cases(args: argparse.Namespace, paths: APFPaths) -> list[str]:
    if args.case:
        get_case(args.case, case_type="InS", paths=paths)
        return [args.case]

    if args.cases:
        cases = unique_preserve_order(args.cases)
        for case in cases:
            get_case(case, case_type="InS", paths=paths)
        return cases

    registry_cases = get_cases(
        case_type="InS",
        status=VALID_DEFAULT_STATUSES,
        paths=paths,
    )

    if args.all:
        return [entry.case for entry in registry_cases]

    if args.all_gre_ready:
        ready_cases: list[str] = []
        print("\nGRE readiness")
        for entry in registry_cases:
            ready, reason = gre_readiness(paths, entry.case)
            print(f"  {'READY' if ready else 'SKIP ':5}: {entry.case} — {reason}")
            if ready:
                ready_cases.append(entry.case)

        if not ready_cases:
            raise RuntimeError("No registered cases passed GRE readiness checks.")
        return ready_cases

    if args.all_dwi_ready or args.all_mp2rage_ready:
        raise ValueError(
            "This is a GRE script. Use CASE, --cases, --all, or --all-gre-ready."
        )

    raise ValueError("No GRE case-selection mode resolved.")


def discover_echoes(folder: Path, expected_part: str) -> dict[int, Path]:
    files = nifti_files(folder)
    by_echo: dict[int, Path] = {}

    for path in files:
        echo = echo_number_from_name(path)
        if echo is None:
            continue
        if echo in by_echo:
            raise RuntimeError(
                f"Duplicate {expected_part} echo {echo}: "
                f"{by_echo[echo].name} and {path.name}"
            )
        by_echo[echo] = path

    missing = sorted(set(EXPECTED_ECHOES).difference(by_echo))
    unexpected = sorted(set(by_echo).difference(EXPECTED_ECHOES))

    if missing or unexpected:
        raise RuntimeError(
            f"{expected_part} echo set invalid: "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )

    return by_echo


def metadata_echo_number(meta: dict[str, Any], fallback: int) -> int:
    value = meta.get("EchoNumber", fallback)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid EchoNumber value: {value!r}") from error


def metadata_echo_time(meta: dict[str, Any], path: Path) -> float:
    value = meta.get("EchoTime")
    try:
        te = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing/invalid EchoTime in {path}") from error

    if not np.isfinite(te) or te <= 0:
        raise ValueError(f"Non-positive or non-finite EchoTime in {path}: {te}")
    return te


def load_echo_metadata(
    mag_by_echo: dict[int, Path],
    phase_by_echo: dict[int, Path],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    echo_times_s: list[float] = []
    mag_meta_all: list[dict[str, Any]] = []
    phase_meta_all: list[dict[str, Any]] = []

    previous_te = -math.inf

    for echo in EXPECTED_ECHOES:
        mag_json = json_sidecar(mag_by_echo[echo])
        phase_json = json_sidecar(phase_by_echo[echo])

        mag_meta = read_json(mag_json)
        phase_meta = read_json(phase_json)

        mag_echo = metadata_echo_number(mag_meta, echo)
        phase_echo = metadata_echo_number(phase_meta, echo)
        if mag_echo != echo or phase_echo != echo:
            raise RuntimeError(
                f"EchoNumber mismatch at echo {echo}: "
                f"magnitude={mag_echo}, phase={phase_echo}"
            )

        mag_te = metadata_echo_time(mag_meta, mag_json)
        phase_te = metadata_echo_time(phase_meta, phase_json)

        if abs(mag_te - phase_te) > TE_ATOL_S:
            raise RuntimeError(
                f"Magnitude/phase TE mismatch at echo {echo}: "
                f"{mag_te:.12g} s vs {phase_te:.12g} s"
            )

        if phase_te <= previous_te:
            raise RuntimeError(
                f"Echo times are not strictly increasing at echo {echo}: "
                f"{phase_te:.12g} s <= {previous_te:.12g} s"
            )

        previous_te = phase_te
        echo_times_s.append(phase_te)
        mag_meta_all.append(mag_meta)
        phase_meta_all.append(phase_meta)

    return (
        np.asarray(echo_times_s, dtype=np.float64),
        mag_meta_all,
        phase_meta_all,
    )


def validate_geometry(
    mag_by_echo: dict[int, Path],
    phase_by_echo: dict[int, Path],
) -> tuple[tuple[int, int, int], np.ndarray, nib.Nifti1Header]:
    reference_path = mag_by_echo[1]
    reference = nib.load(str(reference_path))

    if len(reference.shape) != 3:
        raise RuntimeError(
            f"Expected a 3D native echo, got {reference.shape}: {reference_path}"
        )

    reference_shape = tuple(int(v) for v in reference.shape)
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    reference_header = reference.header.copy()

    for echo in EXPECTED_ECHOES:
        for part, path in (
            ("magnitude", mag_by_echo[echo]),
            ("phase", phase_by_echo[echo]),
        ):
            image = nib.load(str(path))
            shape = tuple(int(v) for v in image.shape)
            affine = np.asarray(image.affine, dtype=np.float64)

            if shape != reference_shape:
                raise RuntimeError(
                    f"Geometry mismatch at {part} echo {echo}: "
                    f"shape={shape}, reference={reference_shape}"
                )

            if not np.allclose(
                affine,
                reference_affine,
                rtol=0.0,
                atol=GEOMETRY_ATOL,
            ):
                max_difference = float(np.max(np.abs(affine - reference_affine)))
                raise RuntimeError(
                    f"Affine mismatch at {part} echo {echo}; "
                    f"max absolute difference={max_difference:.6g}"
                )

    return reference_shape, reference_affine, reference_header


def finite_range(data: np.ndarray, path: Path) -> tuple[float, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        raise RuntimeError(f"No finite voxels in {path}")
    values = data[finite]
    return float(values.min()), float(values.max())


def inspect_phase_ranges(
    phase_by_echo: dict[int, Path],
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    global_min = math.inf
    global_max = -math.inf

    for echo in EXPECTED_ECHOES:
        path = phase_by_echo[echo]
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
        raw_min, raw_max = finite_range(data, path)

        global_min = min(global_min, raw_min)
        global_max = max(global_max, raw_max)

        rows.append(
            {
                "echo": echo,
                "file": path.name,
                "dtype": str(image.get_data_dtype()),
                "raw_min": raw_min,
                "raw_max": raw_max,
            }
        )

    return rows, float(global_min), float(global_max)


def classify_phase_scaling(
    requested_mode: str,
    raw_min: float,
    raw_max: float,
) -> tuple[str, str]:
    if requested_mode != "auto":
        return requested_mode, "explicit_user_selection"

    tolerance_rad = 0.10
    if raw_min >= -math.pi - tolerance_rad and raw_max <= math.pi + tolerance_rad:
        return "radians", "auto_detected_range_within_approximately_minus_pi_to_pi"

    if raw_min >= -4096.5 and raw_max <= 4095.5 and raw_min < -10:
        return (
            "siemens_signed_4096",
            "auto_detected_signed_Siemens_integer_range",
        )

    if raw_min >= -0.5 and raw_max <= 4095.5 and raw_max > 10:
        return (
            "siemens_unsigned_4096",
            "auto_detected_unsigned_Siemens_integer_range",
        )

    raise RuntimeError(
        "Phase scaling could not be determined safely from the observed range "
        f"[{raw_min:.6g}, {raw_max:.6g}]. "
        "Inspect the source data and rerun with --phase-scaling explicitly."
    )


def scale_phase_to_radians(data: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(data, dtype=np.float32)

    if mode == "radians":
        radians = raw
    elif mode == "siemens_signed_4096":
        radians = raw * np.float32(math.pi / 4096.0)
    elif mode == "siemens_unsigned_4096":
        radians = raw * np.float32(2.0 * math.pi / 4096.0) - np.float32(math.pi)
    else:
        raise ValueError(f"Unsupported phase scaling mode: {mode}")

    # Standardize wrapped phase to [-pi, pi). This leaves already valid values
    # unchanged except for the mathematically equivalent +pi -> -pi endpoint.
    radians = (radians + np.float32(math.pi)) % np.float32(
        2.0 * math.pi
    ) - np.float32(math.pi)

    return np.asarray(radians, dtype=np.float32)


def b0_direction_voxel(affine: np.ndarray) -> np.ndarray:
    """
    Return scanner/world +Z expressed in NIfTI voxel-axis coordinates.

    The affine maps voxel coordinates to NIfTI world coordinates. Therefore,
    applying the inverse 3x3 affine to world +Z expresses B0 in voxel axes.
    The result is normalized because voxel sizes are embedded in the affine.
    """
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    if abs(np.linalg.det(linear)) < 1e-12:
        raise RuntimeError("Cannot derive B0 direction from singular affine.")

    direction = np.linalg.solve(linear, np.asarray([0.0, 0.0, 1.0]))
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid B0 direction derived from affine.")

    return direction / norm


def write_txt_vector(path: Path, values: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values), fmt=fmt)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def merge_summary_by_case(
    existing_path: Path,
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Persist cohort state without deleting rows from cases not selected now.
    """
    merged: dict[str, dict[str, Any]] = {}

    if existing_path.exists():
        with existing_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                case = str(row.get("case", "")).strip()
                if case:
                    merged[case] = dict(row)

    for row in new_rows:
        case = str(row.get("case", "")).strip()
        if not case:
            raise ValueError("Cannot merge summary row without case.")
        merged[case] = row

    return [merged[case] for case in sorted(merged)]


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    reference_header: nib.Nifti1Header,
    path: Path,
    dtype: np.dtype,
) -> None:
    header = reference_header.copy()
    header.set_data_dtype(dtype)
    header.set_data_shape(data.shape)

    image = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        affine,
        header=header,
    )
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def prepare_case(
    paths: APFPaths,
    case: str,
    requested_scaling: str,
    force: bool,
) -> dict[str, Any]:
    gre_dir = paths.case_nifti_dir(case) / "GRE"
    mag_by_echo = discover_echoes(gre_dir / "mag", "magnitude")
    phase_by_echo = discover_echoes(gre_dir / "phase", "phase")

    echo_times_s, mag_meta_all, phase_meta_all = load_echo_metadata(
        mag_by_echo,
        phase_by_echo,
    )
    shape, affine, reference_header = validate_geometry(
        mag_by_echo,
        phase_by_echo,
    )

    phase_range_rows, raw_min, raw_max = inspect_phase_ranges(phase_by_echo)
    scaling_mode, scaling_reason = classify_phase_scaling(
        requested_scaling,
        raw_min,
        raw_max,
    )

    out_dir = (
        paths.case_derivatives_dir(case)
        / "GRE"
        / "phase_preparation_gre02"
    )

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    magnitude_path = out_dir / f"{TAG}_{case}_magnitude_4d.nii.gz"
    phase_path = out_dir / f"{TAG}_{case}_phase_rad_4d.nii.gz"
    te_s_path = out_dir / f"{TAG}_{case}_echo_times_s.txt"
    te_ms_path = out_dir / f"{TAG}_{case}_echo_times_ms.txt"
    b0_path = out_dir / f"{TAG}_{case}_b0_direction.json"
    ranges_path = out_dir / f"{TAG}_{case}_phase_ranges.csv"
    summary_csv_path = out_dir / f"{TAG}_{case}_summary.csv"
    summary_json_path = out_dir / f"{TAG}_{case}_summary.json"

    expected_outputs = (
        magnitude_path,
        phase_path,
        te_s_path,
        te_ms_path,
        b0_path,
        ranges_path,
        summary_csv_path,
        summary_json_path,
    )

    existing = [path for path in expected_outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "gre02 outputs already exist. Use --force to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    n_echoes = len(EXPECTED_ECHOES)
    magnitude_4d = np.empty(shape + (n_echoes,), dtype=np.float32)
    phase_4d = np.empty(shape + (n_echoes,), dtype=np.float32)

    converted_range_rows: list[dict[str, Any]] = []

    for index, echo in enumerate(EXPECTED_ECHOES):
        mag_path = mag_by_echo[echo]
        ph_path = phase_by_echo[echo]

        mag_image = nib.load(str(mag_path))
        ph_image = nib.load(str(ph_path))

        magnitude = np.asarray(
            mag_image.get_fdata(dtype=np.float32),
            dtype=np.float32,
        )
        raw_phase = np.asarray(ph_image.dataobj)
        phase_rad = scale_phase_to_radians(raw_phase, scaling_mode)

        if not np.all(np.isfinite(magnitude)):
            n_bad = int(np.size(magnitude) - np.count_nonzero(np.isfinite(magnitude)))
            raise RuntimeError(
                f"Non-finite magnitude voxels at echo {echo}: {n_bad}"
            )

        if not np.all(np.isfinite(phase_rad)):
            n_bad = int(np.size(phase_rad) - np.count_nonzero(np.isfinite(phase_rad)))
            raise RuntimeError(
                f"Non-finite converted phase voxels at echo {echo}: {n_bad}"
            )

        magnitude_4d[..., index] = magnitude
        phase_4d[..., index] = phase_rad

        raw_min_echo, raw_max_echo = finite_range(raw_phase, ph_path)
        rad_min_echo, rad_max_echo = finite_range(phase_rad, ph_path)

        # Echo-wise QC is evaluated within non-zero magnitude support to avoid
        # zero-padded background dominating the descriptive statistics.
        signal_mask = magnitude > 0
        signal_voxels = int(np.count_nonzero(signal_mask))
        signal_fraction = float(signal_voxels / magnitude.size)

        if signal_voxels == 0:
            raise RuntimeError(
                f"No non-zero magnitude voxels available for QC at echo {echo}."
            )

        magnitude_signal = magnitude[signal_mask]
        phase_signal = phase_rad[signal_mask]

        magnitude_median = float(np.median(magnitude_signal))
        magnitude_mad = float(
            np.median(np.abs(magnitude_signal - magnitude_median))
        )
        phase_median = float(np.median(phase_signal))
        phase_mad = float(np.median(np.abs(phase_signal - phase_median)))

        converted_range_rows.append(
            {
                "script_tag": TAG,
                "case": case,
                "echo": echo,
                "echo_time_s": float(echo_times_s[index]),
                "echo_time_ms": float(echo_times_s[index] * 1000.0),
                "phase_file": ph_path.name,
                "raw_dtype": str(ph_image.get_data_dtype()),
                "raw_min": raw_min_echo,
                "raw_max": raw_max_echo,
                "phase_scaling": scaling_mode,
                "rad_min": rad_min_echo,
                "rad_max": rad_max_echo,
                "signal_voxels_magnitude_gt_0": signal_voxels,
                "signal_fraction_magnitude_gt_0": signal_fraction,
                "magnitude_mean_signal": float(np.mean(magnitude_signal)),
                "magnitude_std_signal": float(np.std(magnitude_signal)),
                "magnitude_median_signal": magnitude_median,
                "magnitude_mad_signal": magnitude_mad,
                "phase_mean_rad_signal": float(np.mean(phase_signal)),
                "phase_std_rad_signal": float(np.std(phase_signal)),
                "phase_median_rad_signal": phase_median,
                "phase_mad_rad_signal": phase_mad,
            }
        )

    save_nifti(
        magnitude_4d,
        affine,
        reference_header,
        magnitude_path,
        np.dtype(np.float32),
    )
    save_nifti(
        phase_4d,
        affine,
        reference_header,
        phase_path,
        np.dtype(np.float32),
    )

    write_txt_vector(te_s_path, echo_times_s, "%.9f")
    write_txt_vector(te_ms_path, echo_times_s * 1000.0, "%.6f")
    write_csv(converted_range_rows, ranges_path)

    b0_voxel = b0_direction_voxel(affine)
    b0_payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "created": now_iso(),
        "case": case,
        "definition": (
            "Scanner/world +Z (assumed B0 direction) expressed in NIfTI "
            "voxel-axis coordinates and normalized."
        ),
        "world_b0_direction": [0.0, 0.0, 1.0],
        "voxel_b0_direction": [float(value) for value in b0_voxel],
        "affine": affine.tolist(),
        "image_orientation_patient_dicom": phase_meta_all[0].get(
            "ImageOrientationPatientDICOM",
            "",
        ),
    }
    b0_path.write_text(json.dumps(b0_payload, indent=2), encoding="utf-8")

    converted_min = float(np.min(phase_4d))
    converted_max = float(np.max(phase_4d))
    magnitude_min = float(np.min(magnitude_4d))
    magnitude_max = float(np.max(magnitude_4d))

    summary = {
        "script_tag": TAG,
        "case": case,
        "status": "READY_FOR_GRE03",
        "n_echoes": n_echoes,
        "shape_4d": "x".join(str(v) for v in magnitude_4d.shape),
        "voxel_sizes_mm": "x".join(
            f"{float(v):g}" for v in reference_header.get_zooms()[:3]
        ),
        "echo_times_ms": ";".join(
            f"{float(value):.6g}" for value in echo_times_s * 1000.0
        ),
        "first_echo_time_ms": float(echo_times_s[0] * 1000.0),
        "last_echo_time_ms": float(echo_times_s[-1] * 1000.0),
        "requested_phase_scaling": requested_scaling,
        "resolved_phase_scaling": scaling_mode,
        "phase_scaling_reason": scaling_reason,
        "raw_phase_global_min": raw_min,
        "raw_phase_global_max": raw_max,
        "converted_phase_global_min_rad": converted_min,
        "converted_phase_global_max_rad": converted_max,
        "magnitude_global_min": magnitude_min,
        "magnitude_global_max": magnitude_max,
        "b0_voxel_x": float(b0_voxel[0]),
        "b0_voxel_y": float(b0_voxel[1]),
        "b0_voxel_z": float(b0_voxel[2]),
        "magnitude_4d": project_relative(magnitude_path, paths),
        "phase_rad_4d": project_relative(phase_path, paths),
        "echo_times_s_file": project_relative(te_s_path, paths),
        "echo_times_ms_file": project_relative(te_ms_path, paths),
        "b0_direction_file": project_relative(b0_path, paths),
        "phase_ranges_file": project_relative(ranges_path, paths),
        "qc_notes": "OK",
    }

    write_csv([summary], summary_csv_path)

    payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "legacy_tags": list(LEGACY_TAGS),
        "created": now_iso(),
        "case": case,
        "root": str(paths.root),
        "command": " ".join(sys.argv),
        "execution": {
            "host": platform.node(),
            "python": platform.python_version(),
            "nibabel": nib.__version__,
            "numpy": np.__version__,
        },
        "input": {
            "magnitude_echoes": {
                str(echo): project_relative(path, paths)
                for echo, path in mag_by_echo.items()
            },
            "phase_echoes": {
                str(echo): project_relative(path, paths)
                for echo, path in phase_by_echo.items()
            },
            "magnitude_json": [
                project_relative(json_sidecar(mag_by_echo[echo]), paths)
                for echo in EXPECTED_ECHOES
            ],
            "phase_json": [
                project_relative(json_sidecar(phase_by_echo[echo]), paths)
                for echo in EXPECTED_ECHOES
            ],
        },
        "phase_scaling": {
            "requested": requested_scaling,
            "resolved": scaling_mode,
            "reason": scaling_reason,
            "raw_global_min": raw_min,
            "raw_global_max": raw_max,
            "formula": {
                "radians": "phase_rad = raw",
                "siemens_signed_4096": "phase_rad = raw * pi / 4096",
                "siemens_unsigned_4096": (
                    "phase_rad = raw * 2*pi / 4096 - pi"
                ),
            }[scaling_mode],
            "standardized_wrapped_interval": "[-pi, pi)",
        },
        "geometry": {
            "shape_3d": list(shape),
            "shape_4d": list(magnitude_4d.shape),
            "affine": affine.tolist(),
            "voxel_sizes_mm": [
                float(v) for v in reference_header.get_zooms()[:3]
            ],
        },
        "echo_times_s": [float(v) for v in echo_times_s],
        "echo_times_ms": [float(v * 1000.0) for v in echo_times_s],
        "b0": b0_payload,
        "phase_ranges": converted_range_rows,
        "summary": summary,
    }
    summary_json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    # Release large arrays before the next case.
    del magnitude_4d
    del phase_4d

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF GRE 02: standardize native multi-echo magnitude and phase "
            "for ROMEO/total-field estimation"
        )
    )
    add_case_selection_args(parser)
    parser.add_argument(
        "--phase-scaling",
        choices=SCALING_MODES,
        default="auto",
        help=(
            "Phase scaling mode. Default: auto. "
            "Auto fails rather than guessing when the observed range is ambiguous."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing gre02 outputs for selected cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths)

    start = now_iso()
    t0 = time.monotonic()

    print("=" * 76)
    print("APF GRE 02: Prepare multi-echo phase and magnitude")
    print("=" * 76)
    print(f"Script        : {SCRIPT}")
    print(f"Version       : {VERSION}")
    print(f"Tag           : {TAG}")
    print(f"Root          : {paths.root}")
    print(f"Phase scaling : {args.phase_scaling}")
    print(f"Force         : {args.force}")
    print_selected_cases(cases)

    summaries: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    for case in cases:
        case_t0 = time.monotonic()
        print(f"\n{'=' * 76}\n{TAG}: {case}\n{'=' * 76}")

        try:
            summary = prepare_case(
                paths=paths,
                case=case,
                requested_scaling=args.phase_scaling,
                force=args.force,
            )
            summaries.append(summary)
            statuses.append(
                {
                    "case": case,
                    "status": "OK",
                    "message": "",
                    "runtime_seconds": round(time.monotonic() - case_t0, 3),
                }
            )

            print(f"  status              : {summary['status']}")
            print(
                "  phase scaling       : "
                f"{summary['resolved_phase_scaling']} "
                f"({summary['phase_scaling_reason']})"
            )
            print(
                "  raw phase range     : "
                f"{summary['raw_phase_global_min']:.6g} to "
                f"{summary['raw_phase_global_max']:.6g}"
            )
            print(
                "  converted range     : "
                f"{summary['converted_phase_global_min_rad']:.6g} to "
                f"{summary['converted_phase_global_max_rad']:.6g} rad"
            )
            print(
                "  B0 voxel direction  : "
                f"[{summary['b0_voxel_x']:.6f}, "
                f"{summary['b0_voxel_y']:.6f}, "
                f"{summary['b0_voxel_z']:.6f}]"
            )
            print(f"  output phase        : {summary['phase_rad_4d']}")
            print(f"  output magnitude    : {summary['magnitude_4d']}")

        except Exception as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            statuses.append(
                {
                    "case": case,
                    "status": "FAILED",
                    "message": str(error),
                    "runtime_seconds": round(time.monotonic() - case_t0, 3),
                }
            )

    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    cohort_csv = paths.summary_dir / f"{TAG}_GRE_phase_preparation_qc.csv"
    run_json = paths.summary_dir / f"{TAG}_GRE_phase_preparation_run.json"

    merged_summaries = merge_summary_by_case(cohort_csv, summaries)
    write_csv(merged_summaries, cohort_csv)

    runtime = round(time.monotonic() - t0, 3)
    run_payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "created": now_iso(),
        "start": start,
        "root": str(paths.root),
        "command": " ".join(sys.argv),
        "selected_cases": cases,
        "n_selected_cases": len(cases),
        "n_successful": sum(item["status"] == "OK" for item in statuses),
        "n_failed": sum(item["status"] == "FAILED" for item in statuses),
        "runtime_seconds": runtime,
        "case_status": statuses,
        "updated_case_summaries": summaries,
        "cohort_summary_csv": project_relative(cohort_csv, paths),
    }
    run_json.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    successful = sum(item["status"] == "OK" for item in statuses)

    print("\n" + "=" * 76)
    print(f"{TAG} complete: {successful}/{len(cases)} successful")
    print(f"Cohort summary: {cohort_csv}")
    print(f"Run record    : {run_json}")
    print(f"Runtime       : {runtime:.3f} s")
    print("=" * 76)

    return 0 if successful == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
