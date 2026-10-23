#!/usr/bin/env python3
"""
APF GRE 01: Native multi-echo GRE inventory and QC.

Legacy predecessors:
  s18 / 18_inventory_GRE_R2star_inputs_all_cases.py
  s21 / 21_inventory_phase_QSM_chisep_inputs_all_cases.py

Case selection:
  ./apf scripts/gre/01_inventory.py CASE
  ./apf scripts/gre/01_inventory.py --cases CASE1 CASE2
  ./apf scripts/gre/01_inventory.py --all
  ./apf scripts/gre/01_inventory.py --all-gre-ready

Scope:
  - inventory native GRE magnitude and phase echoes
  - validate echo numbering, JSON sidecars, geometry, and TE pairing
  - report companion acquisition availability
  - do not inventory downstream GRE derivatives
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import re
import struct
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from apf_case_selection import add_case_selection_args, print_selected_cases
from apf_paths import APFPaths, get_paths
from apf_registry import VALID_DEFAULT_STATUSES, get_case, get_cases

SCRIPT = "01_inventory.py"
VERSION = "0.1"
TAG = "gre01"
LEGACY_TAGS = ("s18", "s21")

EXPECTED_ECHOES = tuple(range(1, 13))
TE_TOLERANCE_S = 1e-9
GEOMETRY_TOLERANCE = 1e-6

ECHO_PATTERN = re.compile(r"_e(?P<echo>\d+)(?:_ph)?\.nii(?:\.gz)?$", re.IGNORECASE)


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
    if not mag_files and not phase_files:
        return False, "no GRE magnitude or phase NIfTI files"
    if not mag_files:
        return False, f"magnitude missing; {len(phase_files)} phase file(s)"
    if not phase_files:
        return False, f"phase missing; {len(mag_files)} magnitude file(s)"

    return True, f"{len(mag_files)} magnitude and {len(phase_files)} phase file(s)"


def resolve_cases(args: argparse.Namespace, paths: APFPaths) -> list[str]:
    """
    Resolve registered InS cases.

    GRE readiness is intentionally filesystem-derived. The central registry
    supplies candidate cases; this script applies its own modality-specific
    readiness predicate so incomplete GRE acquisitions can still be inventoried.
    """
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


def read_json(path: Optional[Path]) -> tuple[dict[str, Any], Optional[str]]:
    if path is None:
        return {}, "missing_json"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as error:
        return {}, f"unreadable_json:{error}"


def strip_nii_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def json_sidecar(nii: Path) -> Optional[Path]:
    candidate = nii.parent / f"{strip_nii_suffix(nii)}.json"
    return candidate if candidate.exists() else None


def echo_number_from_name(path: Path) -> Optional[int]:
    match = ECHO_PATTERN.search(path.name)
    return int(match.group("echo")) if match else None


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def nifti_header_info(
    path: Path,
) -> tuple[
    Optional[tuple[int, ...]],
    Optional[tuple[float, ...]],
    Optional[str],
    Optional[str],
]:
    try:
        import nibabel as nib

        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
        dtype = str(image.get_data_dtype())
        return shape, zooms, dtype, None
    except Exception as nib_error:
        try:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rb") as file:
                header = file.read(348)

            little = struct.unpack("<i", header[:4])[0]
            big = struct.unpack(">i", header[:4])[0]
            if little == 348:
                endian = "<"
            elif big == 348:
                endian = ">"
            else:
                raise ValueError("invalid NIfTI sizeof_hdr")

            dims = struct.unpack(endian + "8h", header[40:56])
            pixdim = struct.unpack(endian + "8f", header[76:108])
            datatype_code = struct.unpack(endian + "h", header[70:72])[0]
            ndim = int(dims[0])
            shape = tuple(int(dims[index]) for index in range(1, ndim + 1))
            zooms = tuple(float(pixdim[index]) for index in range(1, min(ndim, 3) + 1))
            return (
                shape,
                zooms,
                f"nifti_datatype_code_{datatype_code}",
                f"nibabel_failed_used_header_fallback:{nib_error}",
            )
        except Exception as header_error:
            return None, None, None, f"could_not_read_nifti_header:{header_error}"


def fmt_tuple(values: Optional[tuple[Any, ...]]) -> str:
    if values is None:
        return ""
    return "x".join(f"{value:g}" if isinstance(value, float) else str(value) for value in values)


def fmt_list(values: Sequence[Any], separator: str = ";") -> str:
    return separator.join(str(value) for value in values)


def float_lists_close(
    first: Optional[tuple[float, ...]],
    second: Optional[tuple[float, ...]],
    tolerance: float = GEOMETRY_TOLERANCE,
) -> bool:
    if first is None or second is None or len(first) != len(second):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(first, second))


def image_type_part(meta: dict[str, Any], fallback_part: str) -> str:
    image_type = meta.get("ImageType", [])
    if isinstance(image_type, list):
        values = {str(value).upper() for value in image_type}
        if "PHASE" in values or "P" in values:
            return "phase"
        if "M" in values or "MAGNITUDE" in values:
            return "magnitude"
    return fallback_part


def echo_row(paths: APFPaths, case: str, part: str, nii: Path) -> dict[str, Any]:
    sidecar = json_sidecar(nii)
    meta, json_error = read_json(sidecar)
    shape, zooms, dtype, header_note = nifti_header_info(nii)

    name_echo = echo_number_from_name(nii)
    json_echo = as_int(meta.get("EchoNumber"))
    resolved_echo = json_echo if json_echo is not None else name_echo

    notes: list[str] = []
    if name_echo is None:
        notes.append("echo_number_not_parseable_from_filename")
    if json_echo is None:
        notes.append("missing_or_invalid_EchoNumber")
    if name_echo is not None and json_echo is not None and name_echo != json_echo:
        notes.append(f"filename_json_echo_mismatch_{name_echo}_vs_{json_echo}")
    if json_error:
        notes.append(json_error)
    if header_note:
        notes.append(header_note)

    metadata_part = image_type_part(meta, part)
    if metadata_part != part:
        notes.append(f"folder_ImageType_part_mismatch_{part}_vs_{metadata_part}")

    echo_time_s = as_float(meta.get("EchoTime"))
    if echo_time_s is None:
        notes.append("missing_or_invalid_EchoTime")

    return {
        "script_tag": TAG,
        "case": case,
        "part": part,
        "echo": resolved_echo if resolved_echo is not None else "",
        "filename_echo": name_echo if name_echo is not None else "",
        "json_echo": json_echo if json_echo is not None else "",
        "relative_path": project_relative(nii, paths),
        "file": nii.name,
        "has_json": sidecar is not None,
        "json_path": project_relative(sidecar, paths) if sidecar else "",
        "echo_time_s": echo_time_s if echo_time_s is not None else "",
        "echo_time_ms": echo_time_s * 1000.0 if echo_time_s is not None else "",
        "series_number": meta.get("SeriesNumber", ""),
        "series_description": meta.get("SeriesDescription", ""),
        "protocol_name": meta.get("ProtocolName", ""),
        "sequence_name": meta.get("SequenceName", ""),
        "scanning_sequence": meta.get("ScanningSequence", ""),
        "image_type": json.dumps(meta.get("ImageType", "")),
        "repetition_time_s": meta.get("RepetitionTime", ""),
        "flip_angle_deg": meta.get("FlipAngle", ""),
        "manufacturer": meta.get("Manufacturer", ""),
        "model": meta.get("ManufacturersModelName", ""),
        "magnetic_field_strength_t": meta.get("MagneticFieldStrength", ""),
        "slice_thickness_mm": meta.get("SliceThickness", ""),
        "shape": fmt_tuple(shape),
        "zooms_mm": fmt_tuple(zooms),
        "dtype": dtype or "",
        "image_orientation_patient_dicom": json.dumps(
            meta.get("ImageOrientationPatientDICOM", "")
        ),
        "qc_notes": ";".join(notes) if notes else "OK",
    }


def collect_echo_rows(
    paths: APFPaths,
    case: str,
    part: str,
    folder: Path,
) -> list[dict[str, Any]]:
    return [echo_row(paths, case, part, nii) for nii in nifti_files(folder)]


def rows_by_echo(rows: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        echo = row.get("echo")
        if isinstance(echo, int):
            grouped.setdefault(echo, []).append(row)
    return grouped


def count_niftis(folder: Path, recursive: bool = False) -> int:
    if not folder.is_dir():
        return 0
    iterator = folder.rglob if recursive else folder.glob
    return len(list(iterator("*.nii"))) + len(list(iterator("*.nii.gz")))


def count_matching_niftis(folder: Path, patterns: Sequence[str]) -> int:
    if not folder.is_dir():
        return 0
    found: set[Path] = set()
    for pattern in patterns:
        found.update(folder.glob(f"{pattern}.nii"))
        found.update(folder.glob(f"{pattern}.nii.gz"))
    return len(found)


def companion_availability(paths: APFPaths, case: str) -> dict[str, Any]:
    case_dir = paths.case_nifti_dir(case)
    t1_dir = case_dir / "T1"

    return {
        "mp2rage_inv1_count": count_matching_niftis(t1_dir, ("*MP2RAGE*INV1*",)),
        "mp2rage_inv2_count": count_matching_niftis(t1_dir, ("*MP2RAGE*INV2*",)),
        "mp2rage_uni_count": count_matching_niftis(t1_dir, ("*MP2RAGE*UNI*",)),
        "flair_count": count_niftis(case_dir / "FLAIR", recursive=True),
        "se_count": count_niftis(case_dir / "SE", recursive=True),
        "t2_count": count_niftis(case_dir / "T2", recursive=True),
        "dwi_count": count_niftis(case_dir / "DWI", recursive=True),
        "t2starfit_count": count_niftis(case_dir / "T2STARFIT", recursive=True),
    }


def geometry_signature(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("shape", "")), str(row.get("zooms_mm", ""))


def summarize_case(
    paths: APFPaths,
    case: str,
    mag_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mag_grouped = rows_by_echo(mag_rows)
    phase_grouped = rows_by_echo(phase_rows)

    mag_echoes = sorted(mag_grouped)
    phase_echoes = sorted(phase_grouped)
    expected = set(EXPECTED_ECHOES)

    missing_mag = sorted(expected.difference(mag_echoes))
    missing_phase = sorted(expected.difference(phase_echoes))
    duplicate_mag = sorted(echo for echo, rows in mag_grouped.items() if len(rows) > 1)
    duplicate_phase = sorted(echo for echo, rows in phase_grouped.items() if len(rows) > 1)
    paired_echoes = sorted(set(mag_echoes).intersection(phase_echoes))
    unpaired_mag = sorted(set(mag_echoes).difference(phase_echoes))
    unpaired_phase = sorted(set(phase_echoes).difference(mag_echoes))

    te_mismatch_echoes: list[int] = []
    missing_te_echoes: list[int] = []
    geometry_mismatch_echoes: list[int] = []

    for echo in paired_echoes:
        mag_row = mag_grouped[echo][0]
        phase_row = phase_grouped[echo][0]
        mag_te = as_float(mag_row.get("echo_time_s"))
        phase_te = as_float(phase_row.get("echo_time_s"))

        if mag_te is None or phase_te is None:
            missing_te_echoes.append(echo)
        elif abs(mag_te - phase_te) > TE_TOLERANCE_S:
            te_mismatch_echoes.append(echo)

        if geometry_signature(mag_row) != geometry_signature(phase_row):
            geometry_mismatch_echoes.append(echo)

    mag_geometry = Counter(geometry_signature(row) for row in mag_rows)
    phase_geometry = Counter(geometry_signature(row) for row in phase_rows)
    mag_json_missing = sum(not bool(row["has_json"]) for row in mag_rows)
    phase_json_missing = sum(not bool(row["has_json"]) for row in phase_rows)

    mag_te_ms = [
        as_float(mag_grouped[echo][0].get("echo_time_ms"))
        for echo in mag_echoes
        if mag_grouped[echo]
    ]
    mag_te_ms = [value for value in mag_te_ms if value is not None]

    complete = (
        mag_echoes == list(EXPECTED_ECHOES)
        and phase_echoes == list(EXPECTED_ECHOES)
        and not duplicate_mag
        and not duplicate_phase
        and not te_mismatch_echoes
        and not missing_te_echoes
        and not geometry_mismatch_echoes
        and mag_json_missing == 0
        and phase_json_missing == 0
        and len(mag_geometry) <= 1
        and len(phase_geometry) <= 1
    )

    ready, readiness_reason = gre_readiness(paths, case)
    if complete:
        inventory_status = "READY_COMPLETE"
    elif ready:
        inventory_status = "READY_INCOMPLETE"
    else:
        inventory_status = "NOT_READY"

    notes: list[str] = []
    if missing_mag:
        notes.append(f"missing_magnitude_echoes_{fmt_list(missing_mag, ',')}")
    if missing_phase:
        notes.append(f"missing_phase_echoes_{fmt_list(missing_phase, ',')}")
    if duplicate_mag:
        notes.append(f"duplicate_magnitude_echoes_{fmt_list(duplicate_mag, ',')}")
    if duplicate_phase:
        notes.append(f"duplicate_phase_echoes_{fmt_list(duplicate_phase, ',')}")
    if unpaired_mag:
        notes.append(f"unpaired_magnitude_echoes_{fmt_list(unpaired_mag, ',')}")
    if unpaired_phase:
        notes.append(f"unpaired_phase_echoes_{fmt_list(unpaired_phase, ',')}")
    if te_mismatch_echoes:
        notes.append(f"magnitude_phase_TE_mismatch_{fmt_list(te_mismatch_echoes, ',')}")
    if missing_te_echoes:
        notes.append(f"missing_TE_{fmt_list(missing_te_echoes, ',')}")
    if geometry_mismatch_echoes:
        notes.append(f"magnitude_phase_geometry_mismatch_{fmt_list(geometry_mismatch_echoes, ',')}")
    if len(mag_geometry) > 1:
        notes.append("inconsistent_magnitude_geometry")
    if len(phase_geometry) > 1:
        notes.append("inconsistent_phase_geometry")
    if mag_json_missing:
        notes.append(f"missing_magnitude_json_{mag_json_missing}")
    if phase_json_missing:
        notes.append(f"missing_phase_json_{phase_json_missing}")
    if not ready:
        notes.append(readiness_reason.replace(" ", "_"))

    representative = mag_rows[0] if mag_rows else (phase_rows[0] if phase_rows else {})
    companion = companion_availability(paths, case)

    return {
        "script_tag": TAG,
        "case": case,
        "inventory_status": inventory_status,
        "gre_ready": ready,
        "readiness_reason": readiness_reason,
        "gre_dir_present": (paths.case_nifti_dir(case) / "GRE").is_dir(),
        "mag_dir_present": (paths.case_nifti_dir(case) / "GRE" / "mag").is_dir(),
        "phase_dir_present": (paths.case_nifti_dir(case) / "GRE" / "phase").is_dir(),
        "n_magnitude_files": len(mag_rows),
        "n_phase_files": len(phase_rows),
        "n_unique_magnitude_echoes": len(mag_echoes),
        "n_unique_phase_echoes": len(phase_echoes),
        "magnitude_echoes": fmt_list(mag_echoes),
        "phase_echoes": fmt_list(phase_echoes),
        "missing_magnitude_echoes": fmt_list(missing_mag),
        "missing_phase_echoes": fmt_list(missing_phase),
        "duplicate_magnitude_echoes": fmt_list(duplicate_mag),
        "duplicate_phase_echoes": fmt_list(duplicate_phase),
        "paired_echoes": fmt_list(paired_echoes),
        "unpaired_magnitude_echoes": fmt_list(unpaired_mag),
        "unpaired_phase_echoes": fmt_list(unpaired_phase),
        "te_mismatch_echoes": fmt_list(te_mismatch_echoes),
        "missing_te_echoes": fmt_list(missing_te_echoes),
        "geometry_mismatch_echoes": fmt_list(geometry_mismatch_echoes),
        "echo_times_ms": fmt_list([f"{value:.5g}" for value in mag_te_ms]),
        "first_echo_time_ms": min(mag_te_ms) if mag_te_ms else "",
        "last_echo_time_ms": max(mag_te_ms) if mag_te_ms else "",
        "shape": representative.get("shape", ""),
        "zooms_mm": representative.get("zooms_mm", ""),
        "dtype": representative.get("dtype", ""),
        "series_description": representative.get("series_description", ""),
        "protocol_name": representative.get("protocol_name", ""),
        "repetition_time_s": representative.get("repetition_time_s", ""),
        "flip_angle_deg": representative.get("flip_angle_deg", ""),
        "manufacturer": representative.get("manufacturer", ""),
        "model": representative.get("model", ""),
        "magnetic_field_strength_t": representative.get("magnetic_field_strength_t", ""),
        "image_orientation_patient_dicom": representative.get(
            "image_orientation_patient_dicom", ""
        ),
        **companion,
        "qc_notes": ";".join(notes) if notes else "OK",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def inventory_case(
    paths: APFPaths,
    case: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gre_dir = paths.case_nifti_dir(case) / "GRE"
    mag_rows = collect_echo_rows(paths, case, "magnitude", gre_dir / "mag")
    phase_rows = collect_echo_rows(paths, case, "phase", gre_dir / "phase")
    echo_rows = sorted(
        mag_rows + phase_rows,
        key=lambda row: (
            0 if row["part"] == "magnitude" else 1,
            row["echo"] if isinstance(row["echo"], int) else 10**9,
            row["file"],
        ),
    )
    summary = summarize_case(paths, case, mag_rows, phase_rows)

    print(
        f"  {summary['inventory_status']}: "
        f"mag={summary['n_unique_magnitude_echoes']}/12 "
        f"phase={summary['n_unique_phase_echoes']}/12 "
        f"paired={len(summary['paired_echoes'].split(';')) if summary['paired_echoes'] else 0}/12"
    )
    print(
        f"    TE={summary['first_echo_time_ms'] or 'NA'}–"
        f"{summary['last_echo_time_ms'] or 'NA'} ms "
        f"shape={summary['shape'] or 'NA'} "
        f"zooms={summary['zooms_mm'] or 'NA'}"
    )
    print(f"    notes={summary['qc_notes']}")

    out_dir = paths.case_derivatives_dir(case) / "GRE" / "inventory_gre01"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / f"{TAG}_{case}_summary.csv"
    echoes_csv = out_dir / f"{TAG}_{case}_echoes.csv"
    json_path = out_dir / f"{TAG}_{case}_inventory.json"

    write_csv([summary], summary_csv)
    write_csv(echo_rows, echoes_csv)

    payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "legacy_tags": list(LEGACY_TAGS),
        "created": now_iso(),
        "case": case,
        "root": str(paths.root),
        "summary": summary,
        "echoes": echo_rows,
        "outputs": {
            "summary_csv": project_relative(summary_csv, paths),
            "echoes_csv": project_relative(echoes_csv, paths),
            "json": project_relative(json_path, paths),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary, echo_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF GRE 01: native multi-echo GRE inventory and QC"
    )
    add_case_selection_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths)
    start = now_iso()
    t0 = time.monotonic()

    print("=" * 72)
    print("APF GRE 01: Native multi-echo GRE inventory and QC")
    print("=" * 72)
    print(f"Script : {SCRIPT}")
    print(f"Version: {VERSION}")
    print(f"Tag    : {TAG}")
    print(f"Legacy : {', '.join(LEGACY_TAGS)}")
    print(f"Root   : {paths.root}")
    print_selected_cases(cases)

    summaries: list[dict[str, Any]] = []
    all_echo_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    for case in cases:
        print(f"\n{'=' * 72}\n{TAG}: {case}\n{'=' * 72}")
        try:
            summary, echo_rows = inventory_case(paths, case)
            summaries.append(summary)
            all_echo_rows.extend(echo_rows)
            statuses.append(
                {
                    "case": case,
                    "status": "OK",
                    "inventory_status": summary["inventory_status"],
                    "n_echo_rows": len(echo_rows),
                    "message": "",
                }
            )
        except Exception as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            statuses.append(
                {
                    "case": case,
                    "status": "FAILED",
                    "inventory_status": "",
                    "n_echo_rows": 0,
                    "message": str(error),
                }
            )

    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = paths.summary_dir / f"{TAG}_GRE_inventory_qc.csv"
    echoes_csv = paths.summary_dir / f"{TAG}_GRE_echo_inventory.csv"
    summary_json = paths.summary_dir / f"{TAG}_GRE_inventory_qc.json"

    write_csv(summaries, summary_csv)
    write_csv(all_echo_rows, echoes_csv)

    end = now_iso()
    runtime = round(time.monotonic() - t0, 3)
    payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "legacy_tags": list(LEGACY_TAGS),
        "created": end,
        "root": str(paths.root),
        "selected_cases": cases,
        "n_cases": len(cases),
        "n_case_summaries": len(summaries),
        "n_echo_rows": len(all_echo_rows),
        "timing": {
            "start": start,
            "end": end,
            "runtime_seconds": runtime,
        },
        "execution": {
            "host": platform.node(),
            "threads": 1,
            "tool": "Python",
            "tool_version": platform.python_version(),
            "command": " ".join(sys.argv),
        },
        "case_status": statuses,
        "case_summaries": summaries,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    successful = sum(item["status"] == "OK" for item in statuses)
    complete = sum(
        item.get("inventory_status") == "READY_COMPLETE" for item in statuses
    )

    print("\n" + "=" * 72)
    print(f"{TAG} complete: {successful}/{len(cases)} inventoried successfully")
    print(f"GRE complete: {complete}/{len(cases)} cases")
    print(f"Summary CSV: {summary_csv}")
    print(f"Echo CSV   : {echoes_csv}")
    print(f"JSON       : {summary_json}")
    print(f"Runtime    : {runtime:.3f} s")
    print("=" * 72)

    return 0 if successful == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
