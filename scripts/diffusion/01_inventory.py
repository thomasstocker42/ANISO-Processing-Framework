#!/usr/bin/env python3
"""
APF DWI 01: Diffusion inventory and QC
Legacy predecessor: s27 / 27_diffusion_inventory_qc_all_cases.py

Case selection:
  ./apf scripts/diffusion/01_inventory.py CASE
  ./apf scripts/diffusion/01_inventory.py --cases CASE1 CASE2
  ./apf scripts/diffusion/01_inventory.py --all
  ./apf scripts/diffusion/01_inventory.py --all-dwi-ready
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import struct
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from apf_paths import APFPaths, get_paths
from apf_registry import VALID_DEFAULT_STATUSES, get_case, get_cases

SCRIPT = "01_inventory.py"
VERSION = "0.1"
TAG = "dwi01"
LEGACY_TAG = "s27"

DIFF_KEYWORDS = ("diff", "dwi", "dti", "b0", "b1000", "b2000", "n64", "n_64", "tracew", "ep2d_diff")
EXCLUDE_KEYWORDS = ("adc", "fa", "colfa", "tensor", "derived", "trace", "exp", "mprage", "mp2rage",
                    "t1", "t2star", "t2starfit", "t2s", "gre", "flair", "localizer", "survey")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_case_selection_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("case", nargs="?", help="Single registered case ID")
    group.add_argument("--cases", nargs="+", help="Explicit registered case IDs")
    group.add_argument("--all", action="store_true", help="All active/new registered InS cases")
    group.add_argument("--all-dwi-ready", action="store_true",
                       help="All active/new registered InS cases containing DWI NIfTI data")


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def dwi_readiness(paths: APFPaths, case: str) -> tuple[bool, str]:
    dwi_dir = paths.case_nifti_dir(case) / "DWI"
    if not dwi_dir.is_dir():
        return False, "DWI directory missing"
    niftis = sorted(dwi_dir.glob("*.nii")) + sorted(dwi_dir.glob("*.nii.gz"))
    if not niftis:
        return False, "no DWI NIfTI files"
    return True, f"{len(niftis)} DWI NIfTI file(s)"


def resolve_cases(args: argparse.Namespace, paths: APFPaths) -> list[str]:
    if args.case:
        get_case(args.case, case_type="InS", paths=paths)
        return [args.case]
    if args.cases:
        cases = unique_preserve_order(args.cases)
        for case in cases:
            get_case(case, case_type="InS", paths=paths)
        return cases

    registry_cases = get_cases(case_type="InS", status=VALID_DEFAULT_STATUSES, paths=paths)
    if args.all:
        return [entry.case for entry in registry_cases]

    ready_cases: list[str] = []
    print("\nDWI readiness")
    for entry in registry_cases:
        ready, reason = dwi_readiness(paths, entry.case)
        print(f"  {'READY' if ready else 'SKIP ':5}: {entry.case} — {reason}")
        if ready:
            ready_cases.append(entry.case)
    if not ready_cases:
        raise RuntimeError("No registered cases passed DWI readiness checks.")
    return ready_cases


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def strip_nii_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def sidecar_paths(nii: Path) -> dict[str, Optional[Path]]:
    base = nii.parent / strip_nii_suffix(nii)
    candidates = {"json": base.with_suffix(".json"), "bval": base.with_suffix(".bval"),
                  "bvec": base.with_suffix(".bvec")}
    return {key: value if value.exists() else None for key, value in candidates.items()}


def normalized_text(*parts: Any) -> str:
    return " ".join(str(part).lower() for part in parts if part is not None)


def is_probably_diffusion(nii: Path, meta: dict[str, Any]) -> bool:
    sidecars = sidecar_paths(nii)
    text = normalized_text(nii.name, nii.parent.name, nii.parent.parent.name,
                           meta.get("SeriesDescription"), meta.get("ProtocolName"),
                           meta.get("SequenceName"), meta.get("ScanningSequence"),
                           meta.get("ImageType"))
    has_diff_key = any(key in text for key in DIFF_KEYWORDS)
    has_gradients = sidecars["bval"] is not None or sidecars["bvec"] is not None
    excluded = any(key in text for key in EXCLUDE_KEYWORDS) and not has_gradients
    return (has_diff_key or has_gradients) and not excluded


def read_bvals(path: Optional[Path]) -> list[float]:
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8").strip().replace("\n", " ")
        return [float(value) for value in text.split() if value.strip()]
    except Exception:
        return []


def read_bvec_count(path: Optional[Path]) -> tuple[int, str]:
    if path is None:
        return 0, "missing"
    try:
        rows = [[float(value) for value in line.split()]
                for line in path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    except Exception:
        return 0, "unreadable"
    if len(rows) == 3:
        return max(len(row) for row in rows), "3_rows"
    if rows and len(rows[0]) == 3:
        return len(rows), "N_rows"
    return 0, f"unexpected_{len(rows)}rows"


def rounded_shells(bvals: Sequence[float]) -> list[int]:
    return [0 if b < 50 else int(round(b / 100.0) * 100) for b in bvals]


def nifti_shape_basic(path: Path) -> tuple[Optional[tuple[int, ...]], Optional[tuple[float, ...]], Optional[str]]:
    try:
        import nibabel as nib
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:len(shape)])
        return shape, zooms, None
    except Exception as nib_error:
        try:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rb") as file:
                header = file.read(348)
            sizeof_hdr = struct.unpack("<i", header[:4])[0]
            endian = "<" if sizeof_hdr == 348 else ">"
            dims = struct.unpack(endian + "8h", header[40:56])
            pixdim = struct.unpack(endian + "8f", header[76:108])
            ndim = int(dims[0])
            shape = tuple(int(dims[i]) for i in range(1, ndim + 1) if dims[i] > 0)
            zooms = tuple(float(pixdim[i]) for i in range(1, ndim + 1) if dims[i] > 0)
            return shape, zooms, f"nibabel_failed_used_header_fallback: {nib_error}"
        except Exception as header_error:
            return None, None, f"could_not_read_nifti_header: {header_error}"


def fmt_tuple(values: Optional[tuple[Any, ...]]) -> str:
    return "" if values is None else "x".join(str(value) for value in values)


def infer_pe(meta: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(meta.get("PhaseEncodingDirection", "")),
            str(meta.get("InPlanePhaseEncodingDirectionDICOM", "")),
            str(meta.get("TotalReadoutTime", "")),
            str(meta.get("EffectiveEchoSpacing", "")))


def classify_candidate(meta: dict[str, Any], bvals: Sequence[float],
                       shape: Optional[tuple[int, ...]]) -> str:
    shells = sorted(set(rounded_shells(bvals)))
    nonzero = [shell for shell in shells if shell > 0]
    nvols = shape[3] if shape is not None and len(shape) >= 4 else len(bvals)
    text = normalized_text(meta.get("SeriesDescription"), meta.get("ProtocolName"))
    if "n64" in text or "n_64" in text:
        return "N64_candidate"
    if len(nonzero) >= 2:
        return "multi_shell_candidate"
    if nonzero and nvols >= 30:
        return "single_shell_DTI_candidate"
    if not bvals:
        return "diffusion_like_missing_bvals"
    return "limited_or_unclear_diffusion"


def qc_notes(meta: dict[str, Any], bvals: Sequence[float], bvec_count: int,
             shape: Optional[tuple[int, ...]]) -> list[str]:
    notes: list[str] = []
    n_bvals = len(bvals)
    nvols = shape[3] if shape is not None and len(shape) >= 4 else None
    if n_bvals == 0:
        notes.append("missing_bval")
    if bvec_count == 0:
        notes.append("missing_bvec")
    if nvols is not None and n_bvals and nvols != n_bvals:
        notes.append(f"nvols_bval_mismatch_{nvols}_vs_{n_bvals}")
    if n_bvals and bvec_count and n_bvals != bvec_count:
        notes.append(f"bval_bvec_mismatch_{n_bvals}_vs_{bvec_count}")
    if 0 not in sorted(set(rounded_shells(bvals))) and n_bvals:
        notes.append("no_b0_shell_detected")
    pe_dir, _, total_ro, ees = infer_pe(meta)
    if not pe_dir:
        notes.append("missing_PhaseEncodingDirection")
    if not total_ro and not ees:
        notes.append("missing_TotalReadoutTime_or_EffectiveEchoSpacing")
    return notes


def project_relative(path: Path, paths: APFPaths) -> str:
    try:
        return str(path.resolve().relative_to(paths.root.resolve()))
    except ValueError:
        return str(path)


def candidate_row(paths: APFPaths, case: str, nii: Path) -> dict[str, Any]:
    sidecars = sidecar_paths(nii)
    meta = read_json(sidecars["json"]) if sidecars["json"] else {}
    bvals = read_bvals(sidecars["bval"])
    bvec_count, bvec_format = read_bvec_count(sidecars["bvec"])
    shell_counter = Counter(rounded_shells(bvals))
    shape, zooms, header_note = nifti_shape_basic(nii)
    pe_dir, pe_dicom, total_ro, ees = infer_pe(meta)
    notes = qc_notes(meta, bvals, bvec_count, shape)
    if header_note:
        notes.append(header_note)
    return {
        "script_tag": TAG, "legacy_tag": LEGACY_TAG, "case": case,
        "relative_path": project_relative(nii, paths),
        "folder": project_relative(nii.parent, paths), "file": nii.name,
        "candidate_class": classify_candidate(meta, bvals, shape),
        "series_description": meta.get("SeriesDescription", ""),
        "protocol_name": meta.get("ProtocolName", ""),
        "sequence_name": meta.get("SequenceName", ""),
        "scanning_sequence": meta.get("ScanningSequence", ""),
        "manufacturer": meta.get("Manufacturer", ""),
        "model": meta.get("ManufacturersModelName", ""),
        "magnetic_field_strength": meta.get("MagneticFieldStrength", ""),
        "shape": fmt_tuple(shape), "zooms": fmt_tuple(zooms),
        "n_volumes": shape[3] if shape is not None and len(shape) >= 4 else "",
        "has_json": sidecars["json"] is not None, "has_bval": sidecars["bval"] is not None,
        "has_bvec": sidecars["bvec"] is not None, "n_bvals": len(bvals),
        "n_bvecs": bvec_count, "bvec_format": bvec_format,
        "shells_rounded": ";".join(str(shell) for shell in sorted(shell_counter)),
        "shell_counts": ";".join(f"{shell}:{shell_counter[shell]}" for shell in sorted(shell_counter)),
        "n_b0": shell_counter.get(0, 0), "max_bval": max(bvals) if bvals else "",
        "phase_encoding_direction": pe_dir, "phase_encoding_direction_dicom": pe_dicom,
        "total_readout_time": total_ro, "effective_echo_spacing": ees,
        "image_orientation_patient_dicom": json.dumps(meta.get("ImageOrientationPatientDICOM", "")),
        "qc_notes": ";".join(notes) if notes else "OK",
    }


def no_candidate_row(case: str) -> dict[str, Any]:
    row = {key: "" for key in (
        "relative_path", "folder", "file", "series_description", "protocol_name",
        "sequence_name", "scanning_sequence", "manufacturer", "model",
        "magnetic_field_strength", "shape", "zooms", "n_volumes", "shells_rounded",
        "shell_counts", "max_bval", "phase_encoding_direction",
        "phase_encoding_direction_dicom", "total_readout_time",
        "effective_echo_spacing", "image_orientation_patient_dicom")}
    row.update({"script_tag": TAG, "legacy_tag": LEGACY_TAG, "case": case,
                "candidate_class": "no_candidate_found", "has_json": False,
                "has_bval": False, "has_bvec": False, "n_bvals": 0, "n_bvecs": 0,
                "bvec_format": "missing", "n_b0": 0,
                "qc_notes": "no_diffusion_like_nifti_found_under_case_DWI_folder"})
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def inventory_case(paths: APFPaths, case: str) -> list[dict[str, Any]]:
    dwi_dir = paths.case_nifti_dir(case) / "DWI"
    rows: list[dict[str, Any]] = []
    if dwi_dir.is_dir():
        niftis = sorted(dwi_dir.rglob("*.nii")) + sorted(dwi_dir.rglob("*.nii.gz"))
        for nii in niftis:
            sidecars = sidecar_paths(nii)
            meta = read_json(sidecars["json"]) if sidecars["json"] else {}
            if not is_probably_diffusion(nii, meta):
                continue
            row = candidate_row(paths, case, nii)
            rows.append(row)
            print(f"  + {row['candidate_class']}: {row['relative_path']}")
            print(f"    shape={row['shape']} shells={row['shell_counts'] or 'NA'} "
                  f"PE={row['phase_encoding_direction'] or 'NA'} notes={row['qc_notes']}")
    if not rows:
        print("  ! no diffusion-like NIfTI candidates found")
        rows = [no_candidate_row(case)]

    out_dir = paths.case_derivatives_dir(case) / "DWI" / "inventory_dwi01"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{TAG}_{case}_inventory.csv"
    json_path = out_dir / f"{TAG}_{case}_inventory.json"
    write_csv(rows, csv_path)
    payload = {"script": SCRIPT, "version": VERSION, "tag": TAG, "legacy_tag": LEGACY_TAG,
               "created": now_iso(), "case": case, "root": str(paths.root),
               "n_rows": len(rows), "outputs": {
                   "csv": project_relative(csv_path, paths),
                   "json": project_relative(json_path, paths)}, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APF DWI 01: diffusion inventory and QC")
    add_case_selection_args(parser)
    return parser.parse_args()


def summary_scope(args: argparse.Namespace, cases: list[str]) -> str:
    """Return a stable filename scope for the current case selection mode."""
    if getattr(args, "all_dwi_ready", False):
        return "dwi_ready_cohort"

    if getattr(args, "all", False):
        return "all_registered_cases"

    explicit_cases = getattr(args, "cases", None)
    positional_case = getattr(args, "case", None)

    if explicit_cases:
        if len(cases) == 1:
            return cases[0]
        return "selected_cases"

    if positional_case:
        return positional_case

    if len(cases) == 1:
        return cases[0]

    return "selected_cases"


def main() -> int:
    args = parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths)
    start = now_iso()
    t0 = time.monotonic()

    print("=" * 72)
    print("APF DWI 01: Diffusion inventory and QC")
    print("=" * 72)
    print(f"Script : {SCRIPT}\nVersion: {VERSION}\nTag    : {TAG}\nLegacy : {LEGACY_TAG}")
    print(f"Root   : {paths.root}\nCases  : {len(cases)}")
    print("=" * 72)
    for case in cases:
        print(f"  {case}")
    print("=" * 72)

    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for case in cases:
        print(f"\n{case}")
        try:
            rows = inventory_case(paths, case)
            all_rows.extend(rows)
            statuses.append({"case": case, "status": "OK", "n_inventory_rows": len(rows), "message": ""})
        except Exception as error:
            print(f"  ERROR: {error}")
            statuses.append({"case": case, "status": "FAILED", "n_inventory_rows": 0, "message": str(error)})

    summary_dir = (
        paths.summary_dir
        / "diffusion"
        / "dwi01_inventory"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    scope = summary_scope(args, cases)

    csv_path = (
        summary_dir
        / f"{TAG}_{scope}_inventory_qc.csv"
    )
    json_path = (
        summary_dir
        / f"{TAG}_{scope}_inventory_qc.json"
    )

    write_csv(all_rows, csv_path)

    end = now_iso()
    runtime = round(time.monotonic() - t0, 3)

    payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "legacy_tag": LEGACY_TAG,
        "created": end,
        "root": str(paths.root),
        "selection_scope": scope,
        "selected_cases": cases,
        "n_cases": len(cases),
        "n_rows": len(all_rows),
        "outputs": {
            "csv": project_relative(csv_path, paths),
            "json": project_relative(json_path, paths),
        },
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
        "rows": all_rows,
    }

    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    successful = sum(item["status"] == "OK" for item in statuses)
    print("\n" + "=" * 72)
    print(f"{TAG} complete: {successful}/{len(cases)} successful")
    print(f"CSV : {csv_path}\nJSON: {json_path}\nRuntime: {runtime:.3f} s")
    print("=" * 72)
    return 0 if successful == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
