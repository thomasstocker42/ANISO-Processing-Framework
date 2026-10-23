#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
Relaxometry 01 — T2 acquisition inventory

Purpose
-------
Discover and validate candidate quantitative T2 acquisitions for one or more
registered in-situ cases.

Supported acquisition classes
-----------------------------
- dual_tse:
    Exactly two TSE echoes with distinct EchoTime values and matching geometry.
- multi_echo_t2:
    Four or more spin-echo/TSE echoes with distinct EchoTime values and
    matching geometry.

This script is read-only with respect to source NIfTI/JSON data. It writes only
inventory and selection metadata.

Version
-------
0.4.0
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from apf_case_selection import (
    add_case_selection_args,
    print_selected_cases,
    resolve_cases,
)
from apf_paths import APFPaths, get_paths


SCRIPT = "01_inventory.py"
TAG = "rlx01"
VERSION = "0.4.0"
MODULE = "relaxometry"
STAGE = "rlx01_inventory"
SUPPORTED_SOURCE_DIRS = ("T2", "SE")


@dataclass(frozen=True)
class EchoRecord:
    image: str
    json_sidecar: str
    echo_number: int | None
    echo_time_s: float
    series_number: int | None
    series_description: str
    protocol_name: str
    pulse_sequence_name: str
    pulse_sequence_details: str
    scanning_sequence: str
    echo_train_length: int | None
    flip_angle_deg: float | None


@dataclass
class AcquisitionRecord:
    acquisition_id: str
    source_directory: str
    classification: str
    ready: bool
    selected: bool
    readiness_reason: str
    warnings: list[str]
    n_echoes: int
    echo_numbers: list[int | None]
    echo_times_ms: list[float]
    matrix: list[int]
    voxel_size_mm: list[float]
    orientation: str
    affine_match: bool
    geometry_match: bool
    json_complete: bool
    series_number: int | None
    series_description: str
    protocol_name: str
    echoes: list[EchoRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_selection_args(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rlx01 case outputs.",
    )
    return parser.parse_args()


def project_relative(path: Path, root: Path) -> str:
    """Return a project-relative POSIX path when path is inside APF_ROOT."""
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def nifti_for_json(json_path: Path) -> Path | None:
    """Return the matching .nii.gz or .nii file, if present."""
    stem = json_path.name.removesuffix(".json")
    for suffix in (".nii.gz", ".nii"):
        candidate = json_path.parent / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def strip_echo_suffix(stem: str) -> str:
    """Strip only a terminal dcm2niix-style _eN or -eN echo suffix."""
    lower = stem.lower()
    for marker in ("_e", "-e"):
        position = lower.rfind(marker)
        suffix = lower[position + len(marker):] if position >= 0 else ""
        if position >= 0 and suffix.isdigit():
            return stem[:position]
    return stem


def acquisition_group_key(
    json_path: Path,
    metadata: dict[str, Any],
) -> tuple[str, str, str]:
    """Group echoes by source directory, SeriesNumber and shared stem."""
    source_directory = json_path.parent.name
    series_number = str(metadata.get("SeriesNumber", "unknown"))
    series_stem = strip_echo_suffix(json_path.name.removesuffix(".json"))
    return source_directory, series_number, series_stem


def sequence_text(metadata: dict[str, Any]) -> str:
    fields = (
        "SeriesDescription",
        "ProtocolName",
        "PulseSequenceName",
        "PulseSequenceDetails",
        "ScanningSequence",
    )
    return " ".join(str(metadata.get(field, "")) for field in fields).lower()


def classify_acquisition(
    n_echoes: int,
    representative_metadata: dict[str, Any],
) -> str:
    text = sequence_text(representative_metadata)
    scanning_sequence = str(
        representative_metadata.get("ScanningSequence", "")
    ).strip().upper()

    is_tse = "tse" in text or "turbo spin echo" in text
    is_spin_echo_family = is_tse or scanning_sequence == "SE"

    if n_echoes == 2 and is_tse:
        return "dual_tse"
    if n_echoes >= 4 and is_spin_echo_family:
        return "multi_echo_t2"
    return "unsupported_or_ambiguous"


def geometry_summary(
    image_paths: list[Path],
) -> tuple[list[int], list[float], str, bool, bool]:
    if not image_paths:
        raise ValueError("Cannot inspect geometry without NIfTI images.")

    images = [nib.load(str(path)) for path in image_paths]
    reference_shape = tuple(int(value) for value in images[0].shape)
    reference_zooms = tuple(
        float(value) for value in images[0].header.get_zooms()[:3]
    )
    reference_affine = images[0].affine
    orientation = "".join(nib.aff2axcodes(reference_affine))

    geometry_match = all(
        tuple(int(value) for value in image.shape) == reference_shape
        and np.allclose(
            image.header.get_zooms()[:3],
            reference_zooms,
            atol=1e-5,
            rtol=0,
        )
        for image in images[1:]
    )
    affine_match = all(
        np.allclose(
            image.affine,
            reference_affine,
            atol=1e-4,
            rtol=0,
        )
        for image in images[1:]
    )

    return (
        list(reference_shape),
        list(reference_zooms),
        orientation,
        affine_match,
        geometry_match,
    )


def discover_orphan_niftis(directory: Path) -> list[Path]:
    """Find NIfTI files without a same-stem JSON sidecar."""
    niftis = list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz"))
    orphaned: list[Path] = []

    for nifti_path in sorted(niftis):
        name = nifti_path.name
        stem = (
            name.removesuffix(".nii.gz")
            if name.endswith(".nii.gz")
            else name.removesuffix(".nii")
        )
        if not (directory / f"{stem}.json").is_file():
            orphaned.append(nifti_path)

    return orphaned


def discover_case_acquisitions(
    paths: APFPaths,
    case: str,
) -> tuple[list[AcquisitionRecord], list[str]]:
    case_nifti = paths.case_nifti_dir(case)
    candidate_dirs = [
        case_nifti / source_name
        for source_name in SUPPORTED_SOURCE_DIRS
    ]

    case_warnings: list[str] = []
    grouped: dict[
        tuple[str, str, str],
        list[tuple[Path, Path | None, dict[str, Any]]],
    ] = {}

    existing_dirs = [
        directory
        for directory in candidate_dirs
        if directory.is_dir()
    ]
    if not existing_dirs:
        case_warnings.append(
            "Neither T2 nor SE source directory exists for this case."
        )

    for directory in existing_dirs:
        for orphan in discover_orphan_niftis(directory):
            case_warnings.append(
                "NIfTI without JSON sidecar: "
                f"{project_relative(orphan, paths.root)}"
            )

        for json_path in sorted(directory.glob("*.json")):
            try:
                metadata = load_json(json_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                case_warnings.append(
                    "Unreadable JSON: "
                    f"{project_relative(json_path, paths.root)} ({exc})"
                )
                continue

            if metadata.get("EchoTime") is None:
                continue

            image_path = nifti_for_json(json_path)
            key = acquisition_group_key(json_path, metadata)
            grouped.setdefault(key, []).append(
                (json_path, image_path, metadata)
            )

    provisional: list[AcquisitionRecord] = []

    for (source_name, series_key, series_stem), items in sorted(grouped.items()):
        warnings: list[str] = []
        echoes: list[EchoRecord] = []
        image_paths: list[Path] = []
        json_complete = True
        representative_metadata = items[0][2]

        for json_path, image_path, metadata in items:
            if image_path is None:
                json_complete = False
                warnings.append(
                    "Missing NIfTI for JSON: "
                    f"{project_relative(json_path, paths.root)}"
                )
                continue

            try:
                echo_time_s = float(metadata["EchoTime"])
            except (TypeError, ValueError, KeyError):
                json_complete = False
                warnings.append(
                    "Invalid EchoTime: "
                    f"{project_relative(json_path, paths.root)}"
                )
                continue

            echo_number_raw = metadata.get("EchoNumber")
            try:
                echo_number = (
                    int(echo_number_raw)
                    if echo_number_raw is not None
                    else None
                )
            except (TypeError, ValueError):
                echo_number = None
                warnings.append(
                    "Invalid EchoNumber: "
                    f"{project_relative(json_path, paths.root)}"
                )

            try:
                series_number = (
                    int(metadata["SeriesNumber"])
                    if metadata.get("SeriesNumber") is not None
                    else None
                )
            except (TypeError, ValueError):
                series_number = None
                warnings.append(
                    "Invalid SeriesNumber: "
                    f"{project_relative(json_path, paths.root)}"
                )

            try:
                echo_train_length = (
                    int(metadata["EchoTrainLength"])
                    if metadata.get("EchoTrainLength") is not None
                    else None
                )
            except (TypeError, ValueError):
                echo_train_length = None
                warnings.append(
                    "Invalid EchoTrainLength: "
                    f"{project_relative(json_path, paths.root)}"
                )

            try:
                flip_angle_deg = (
                    float(metadata["FlipAngle"])
                    if metadata.get("FlipAngle") is not None
                    else None
                )
            except (TypeError, ValueError):
                flip_angle_deg = None
                warnings.append(
                    "Invalid FlipAngle: "
                    f"{project_relative(json_path, paths.root)}"
                )

            image_paths.append(image_path)
            echoes.append(
                EchoRecord(
                    image=project_relative(image_path, paths.root),
                    json_sidecar=project_relative(json_path, paths.root),
                    echo_number=echo_number,
                    echo_time_s=echo_time_s,
                    series_number=series_number,
                    series_description=str(
                        metadata.get("SeriesDescription", "")
                    ),
                    protocol_name=str(metadata.get("ProtocolName", "")),
                    pulse_sequence_name=str(
                        metadata.get("PulseSequenceName", "")
                    ),
                    pulse_sequence_details=str(
                        metadata.get("PulseSequenceDetails", "")
                    ),
                    scanning_sequence=str(
                        metadata.get("ScanningSequence", "")
                    ),
                    echo_train_length=echo_train_length,
                    flip_angle_deg=flip_angle_deg,
                )
            )

        order = sorted(
            range(len(echoes)),
            key=lambda index: (
                echoes[index].echo_time_s,
                echoes[index].echo_number
                if echoes[index].echo_number is not None
                else 0,
            ),
        )
        echoes = [echoes[index] for index in order]
        image_paths = [image_paths[index] for index in order]

        if not echoes:
            case_warnings.extend(warnings)
            continue

        classification = classify_acquisition(
            len(echoes),
            representative_metadata,
        )

        echo_times = [echo.echo_time_s for echo in echoes]
        unique_echo_times = (
            len({round(value, 9) for value in echo_times})
            == len(echo_times)
        )
        increasing_echo_times = all(
            later > earlier
            for earlier, later in zip(echo_times, echo_times[1:])
        )

        matrix: list[int] = []
        voxel_size_mm: list[float] = []
        orientation = ""
        affine_match = False
        geometry_match = False

        try:
            (
                matrix,
                voxel_size_mm,
                orientation,
                affine_match,
                geometry_match,
            ) = geometry_summary(image_paths)
        except Exception as exc:
            warnings.append(f"Geometry read failed: {exc}")

        if not unique_echo_times:
            warnings.append("Duplicate EchoTime values.")
        if not increasing_echo_times:
            warnings.append("EchoTime values are not strictly increasing.")
        if not geometry_match:
            warnings.append("Echo matrix or voxel-size mismatch.")
        if not affine_match:
            warnings.append("Echo affine mismatch.")
        if classification == "unsupported_or_ambiguous":
            warnings.append(
                "Acquisition does not meet dual-TSE or multi-echo T2 criteria."
            )

        ready = (
            classification in {"dual_tse", "multi_echo_t2"}
            and json_complete
            and unique_echo_times
            and increasing_echo_times
            and geometry_match
            and affine_match
        )

        provisional.append(
            AcquisitionRecord(
                acquisition_id=(
                    f"{source_name}_series-{series_key}_{series_stem}"
                ),
                source_directory=project_relative(
                    paths.case_nifti_dir(case) / source_name,
                    paths.root,
                ),
                classification=classification,
                ready=ready,
                selected=False,
                readiness_reason=(
                    "ready" if ready else "; ".join(warnings) or "not ready"
                ),
                warnings=warnings,
                n_echoes=len(echoes),
                echo_numbers=[echo.echo_number for echo in echoes],
                echo_times_ms=[
                    round(1000.0 * echo.echo_time_s, 6)
                    for echo in echoes
                ],
                matrix=matrix,
                voxel_size_mm=voxel_size_mm,
                orientation=orientation,
                affine_match=affine_match,
                geometry_match=geometry_match,
                json_complete=json_complete,
                series_number=echoes[0].series_number,
                series_description=echoes[0].series_description,
                protocol_name=echoes[0].protocol_name,
                echoes=echoes,
            )
        )

    # Select a class only when exactly one valid candidate exists.
    for classification in ("dual_tse", "multi_echo_t2"):
        candidates = [
            acquisition
            for acquisition in provisional
            if acquisition.classification == classification
            and acquisition.ready
        ]

        if len(candidates) == 1:
            candidates[0].selected = True
        elif len(candidates) > 1:
            reason = (
                f"Ambiguous: {len(candidates)} ready {classification} "
                "candidates; manual selection required."
            )
            for candidate in candidates:
                candidate.ready = False
                candidate.selected = False
                candidate.readiness_reason = reason
                candidate.warnings.append(reason)

    return provisional, case_warnings


def inventory_output_dir(paths: APFPaths, case: str) -> Path:
    return paths.case_stage_dir(
        case=case,
        module=MODULE,
        stage=STAGE,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_acquisition_csv(
    path: Path,
    case: str,
    acquisitions: list[AcquisitionRecord],
) -> None:
    fieldnames = [
        "case",
        "acquisition_id",
        "classification",
        "series_number",
        "series_description",
        "protocol_name",
        "source_directory",
        "n_echoes",
        "echo_numbers",
        "echo_times_ms",
        "matrix",
        "voxel_size_mm",
        "orientation",
        "affine_match",
        "geometry_match",
        "json_complete",
        "ready",
        "selected",
        "readiness_reason",
    ]

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for acquisition in acquisitions:
            row = asdict(acquisition)
            row["case"] = case
            row["echo_numbers"] = ";".join(
                "" if value is None else str(value)
                for value in acquisition.echo_numbers
            )
            row["echo_times_ms"] = ";".join(
                f"{value:g}" for value in acquisition.echo_times_ms
            )
            row["matrix"] = "x".join(
                str(value) for value in acquisition.matrix
            )
            row["voxel_size_mm"] = "x".join(
                f"{value:g}" for value in acquisition.voxel_size_mm
            )

            writer.writerow(
                {field: row.get(field, "") for field in fieldnames}
            )

    temporary.replace(path)


def process_case(
    paths: APFPaths,
    case: str,
    overwrite: bool,
    command: list[str],
) -> dict[str, Any]:
    started = time.time()
    output_dir = inventory_output_dir(paths, case)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_json = output_dir / f"{TAG}_{case}_inventory.json"
    acquisitions_csv = output_dir / f"{TAG}_{case}_acquisitions.csv"
    selected_json = output_dir / f"{TAG}_{case}_selected_inputs.json"

    outputs = [inventory_json, acquisitions_csv, selected_json]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{case}: inventory outputs already exist; use --overwrite."
        )

    acquisitions, case_warnings = discover_case_acquisitions(paths, case)
    selected = [
        acquisition
        for acquisition in acquisitions
        if acquisition.selected
    ]

    dual_selected = next(
        (
            acquisition
            for acquisition in selected
            if acquisition.classification == "dual_tse"
        ),
        None,
    )
    multi_selected = next(
        (
            acquisition
            for acquisition in selected
            if acquisition.classification == "multi_echo_t2"
        ),
        None,
    )

    finished = time.time()
    status = "ready" if selected else "not_ready"

    payload = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "command": command,
        },
        "timing": {
            "start_epoch": started,
            "end_epoch": finished,
            "runtime_seconds": finished - started,
        },
        "paths": {
            "project_root_source": "APFPaths",
            "case_nifti_dir": project_relative(
                paths.case_nifti_dir(case),
                paths.root,
            ),
            "output_dir": project_relative(output_dir, paths.root),
        },
        "summary": {
            "n_acquisitions": len(acquisitions),
            "n_ready": sum(acquisition.ready for acquisition in acquisitions),
            "n_selected": len(selected),
            "dual_tse_selected": dual_selected is not None,
            "multi_echo_t2_selected": multi_selected is not None,
        },
        "warnings": case_warnings,
        "acquisitions": [
            asdict(acquisition)
            for acquisition in acquisitions
        ],
    }

    selected_payload = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "dual_tse": (
            asdict(dual_selected)
            if dual_selected is not None
            else None
        ),
        "multi_echo_t2": (
            asdict(multi_selected)
            if multi_selected is not None
            else None
        ),
    }

    write_json(inventory_json, payload)
    write_json(selected_json, selected_payload)
    write_acquisition_csv(acquisitions_csv, case, acquisitions)

    def protocol_row(
        acquisition: AcquisitionRecord | None,
    ) -> dict[str, Any] | None:
        if acquisition is None:
            return None

        echo_times = acquisition.echo_times_ms
        return {
            "case": case,
            "status": "ready",
            "acquisition_id": acquisition.acquisition_id,
            "classification": acquisition.classification,
            "n_echoes": acquisition.n_echoes,
            "first_te_ms": echo_times[0],
            "last_te_ms": echo_times[-1],
            "delta_te_ms": (
                round(echo_times[1] - echo_times[0], 6)
                if acquisition.classification == "dual_tse"
                and len(echo_times) == 2
                else ""
            ),
            "echo_times_ms": ";".join(f"{value:g}" for value in echo_times),
            "matrix": "x".join(str(value) for value in acquisition.matrix),
            "voxel_size_mm": "x".join(
                f"{value:g}" for value in acquisition.voxel_size_mm
            ),
            "orientation": acquisition.orientation,
            "series_number": acquisition.series_number,
            "series_description": acquisition.series_description,
            "protocol_name": acquisition.protocol_name,
        }

    return {
        "case": case,
        "status": status,
        "n_acquisitions": len(acquisitions),
        "n_ready": payload["summary"]["n_ready"],
        "dual_tse_selected": int(dual_selected is not None),
        "multi_echo_t2_selected": int(multi_selected is not None),
        "runtime_seconds": round(finished - started, 3),
        "_dual_protocol": protocol_row(dual_selected),
        "_multi_protocol": protocol_row(multi_selected),
    }


def write_cohort_summary(
    paths: APFPaths,
    rows: list[dict[str, Any]],
) -> Path:
    summary_dir = paths.summary_module_dir(MODULE)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"{TAG}_t2_inventory_summary.csv"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")

    fieldnames = [
        "case",
        "status",
        "n_acquisitions",
        "n_ready",
        "dual_tse_selected",
        "multi_echo_t2_selected",
        "runtime_seconds",
    ]

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(summary_path)
    return summary_path



def write_protocol_summary(
    paths: APFPaths,
    classification: str,
    rows: list[dict[str, Any]],
) -> Path:
    summary_dir = paths.summary_module_dir(MODULE)
    summary_dir.mkdir(parents=True, exist_ok=True)

    name = "dual_tse" if classification == "dual_tse" else "multi_echo_t2"
    summary_path = summary_dir / f"{TAG}_{name}_summary.csv"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")

    fieldnames = [
        "case",
        "status",
        "acquisition_id",
        "classification",
        "n_echoes",
        "first_te_ms",
        "last_te_ms",
        "delta_te_ms",
        "echo_times_ms",
        "matrix",
        "voxel_size_mm",
        "orientation",
        "series_number",
        "series_description",
        "protocol_name",
    ]

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(summary_path)
    return summary_path


def variant_counts(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[tuple[tuple[str, ...], int]]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def print_variant_group(
    title: str,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    formatter: Any,
) -> int:
    variants = variant_counts(rows, fields)
    print(f"\n{title}:")
    for values, count in variants:
        print(f"  {formatter(values)} ({count})")
    return len(variants)


def print_protocol_consistency(
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    if not rows:
        print("No selected acquisitions available.")
        return

    print(f"Selected cases: {len(rows)}")
    variant_numbers: list[int] = []

    if rows[0]["classification"] == "dual_tse":
        variant_numbers.append(
            print_variant_group(
                "Echo times and spacing",
                rows,
                ("first_te_ms", "last_te_ms", "delta_te_ms"),
                lambda values: (
                    f"TE1={values[0]} ms, TE2={values[1]} ms, "
                    f"deltaTE={values[2]} ms"
                ),
            )
        )
    else:
        variant_numbers.append(
            print_variant_group(
                "Echo schedule",
                rows,
                ("n_echoes", "first_te_ms", "last_te_ms", "echo_times_ms"),
                lambda values: (
                    f"n={values[0]}, TE={values[1]}-{values[2]} ms"
                ),
            )
        )

    variant_numbers.append(
        print_variant_group(
            "Voxel size",
            rows,
            ("voxel_size_mm",),
            lambda values: f"{values[0]} mm",
        )
    )
    variant_numbers.append(
        print_variant_group(
            "Matrix",
            rows,
            ("matrix",),
            lambda values: values[0],
        )
    )
    variant_numbers.append(
        print_variant_group(
            "Orientation",
            rows,
            ("orientation",),
            lambda values: values[0] or "unknown",
        )
    )

    if all(number == 1 for number in variant_numbers):
        print("\nResult: cohort acquisition protocol is consistent.")
    else:
        print("\nResult: acquisition variants detected; use case-specific metadata.")


def main() -> None:
    args = parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths.ins_registry)

    print_selected_cases(cases)
    print(f"{TAG} | {SCRIPT} | version {VERSION}")
    print(f"APF root: {paths.root}")

    results: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        print("\n" + "=" * 72)
        print(f"{TAG}: {case}")
        print("=" * 72)

        try:
            result = process_case(
                paths=paths,
                case=case,
                overwrite=args.overwrite,
                command=list(sys.argv),
            )
            results.append(result)

            print(
                f"status={result['status']} | "
                f"acquisitions={result['n_acquisitions']} | "
                f"ready={result['n_ready']} | "
                f"dual={result['dual_tse_selected']} | "
                f"multi={result['multi_echo_t2_selected']}"
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR in {case}: {exc}", file=sys.stderr)

    if results:
        summary_path = write_cohort_summary(paths, results)
        print(f"\nInventory summary: {summary_path}")

        dual_rows = [
            row["_dual_protocol"]
            for row in results
            if row.get("_dual_protocol") is not None
        ]
        multi_rows = [
            row["_multi_protocol"]
            for row in results
            if row.get("_multi_protocol") is not None
        ]

        dual_path = write_protocol_summary(paths, "dual_tse", dual_rows)
        multi_path = write_protocol_summary(
            paths,
            "multi_echo_t2",
            multi_rows,
        )
        print(f"Dual-TSE summary: {dual_path}")
        print(f"Multi-echo summary: {multi_path}")

        print_protocol_consistency(
            "Dual-TSE cohort protocol summary",
            dual_rows,
        )
        print_protocol_consistency(
            "Multi-echo T2 cohort protocol summary",
            multi_rows,
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
