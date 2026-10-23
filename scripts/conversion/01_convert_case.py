#!/usr/bin/env python3
"""
APF Conversion: 01_convert_case.py v0.1

Purpose
-------
Convert MRI DICOM series for one APF case into standardized NIfTI folders.

This script performs:
- DICOM -> NIfTI conversion with dcm2niix
- basic GRE magnitude/phase organization

It does not perform BET, masking, segmentation, registration, or scientific
image processing.

CAVE: Wenn nifti bereits vorhanden, konvertiert erneut mit append -a.ni... tst

Existing NIfTI outputs must be handled deliberately before rerunning
conversion; this script does not currently implement a general idempotent
skip-existing policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from apf_paths import get_paths


SCRIPT = "01_convert_case.py"
VERSION = "0.1"
TAG = "conv01"

SEQUENCES = [
    ("*gre_12ecos_1x1x1*", "GRE", "GRE_12echo"),
    (
        "*gre_12ecos_t2starfit*",
        "T2STARFIT",
        "GRE_T2starfit",
    ),
    (
        "*se_12ecos_t2fit*",
        "SE",
        "SE_T2fit",
    ),
    (
        "*diff_tra_multi_b_N30_GRAPPA*",
        "DWI",
        "DWI_multi_b_N30",
    ),
    (
        "*diff_tra_N64_b2000*",
        "DWI",
        "DWI_N64_b2000",
    ),
    (
        "*diff_tra_b0_rev_phase_encoding*",
        "DWI",
        "DWI_b0_revPE",
    ),
    (
        "*MP2RAGE_sag_1mm_iso_INV1*",
        "T1",
        "MP2RAGE_INV1",
    ),
    (
        "*MP2RAGE_sag_1mm_iso_INV2*",
        "T1",
        "MP2RAGE_INV2",
    ),
    (
        "*MP2RAGE_sag_1mm_iso_UNI_Images*",
        "T1",
        "MP2RAGE_UNI",
    ),
    (
        "*t2_flair_TIadapt*",
        "FLAIR",
        "FLAIR_TIadapt",
    ),
    (
        "*t2_flair_T1adapt*",
        "FLAIR",
        "FLAIR_T1adapt",
    ),
    (
        "*t2_flair_TI800*",
        "FLAIR",
        "FLAIR_TI0800",
    ),
    (
        "*t2_flair_TI1000*",
        "FLAIR",
        "FLAIR_TI1000",
    ),
    (
        "*t2_flair_TI1150*",
        "FLAIR",
        "FLAIR_TI1150",
    ),
    (
        "*t2_flair_TI1300*",
        "FLAIR",
        "FLAIR_TI1300",
    ),
    (
        "*t2_flair_TI1550*",
        "FLAIR",
        "FLAIR_TI1550",
    ),
    (
        "*t2_flair_TI1800*",
        "FLAIR",
        "FLAIR_TI1800",
    ),
    (
        "*t2_flair_TI2800*",
        "FLAIR",
        "FLAIR_TI2800",
    ),
    (
        "*t2_tse_1x1x2*",
        "T2",
        "T2_TSE_1x1x2",
    ),
    (
        "*t2_tse_1x1x1*",
        "T2",
        "T2_TSE_1x1x1",
    ),
    (
        "*t2_space_sag_TE13*",
        "T2",
        "T2_SPACE_TE13",
    ),
    (
        "*t2_space_sag_TE122*",
        "T2",
        "T2_SPACE_TE122",
    ),
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_mri_root(raw_case_dir: Path) -> Path:
    mri_dir = raw_case_dir / "MRI"
    if not mri_dir.is_dir():
        raise FileNotFoundError(f"MRI directory not found: {mri_dir}")

    markers = (
        "gre_12ecos",
        "diff_tra",
        "MP2RAGE",
        "t2_flair",
        "t2_tse",
        "se_12ecos",
    )

    candidates = []
    for p in mri_dir.rglob("*"):
        if not p.is_dir():
            continue
        child_names = [c.name for c in p.iterdir() if c.is_dir()]
        score = sum(any(name.startswith(marker) for name in child_names) for marker in markers)
        if score:
            candidates.append((score, p))

    if not candidates:
        return mri_dir

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def convert_sequence(paths, case_id: str, mri_root: Path, pattern: str, outsub: str, label: str) -> list[dict]:
    rows = []
    outdir = paths.case_nifti_dir(case_id) / outsub
    outdir.mkdir(parents=True, exist_ok=True)

    seqdirs = sorted(p for p in mri_root.rglob("*") if p.is_dir() and p.match(pattern))

    if not seqdirs:
        print(f"  WARNING: {label} not found")
        return [{
            "case": case_id,
            "pattern": pattern,
            "label": label,
            "outsub": outsub,
            "status": "missing",
            "input_dir": "",
            "output_dir": str(outdir.relative_to(paths.root)),
            "n_new_files": 0,
            "new_files": "",
        }]

    for seqdir in seqdirs:
        before = set(outdir.iterdir())

        print("-" * 72)
        print(f"Converting {case_id} | {label}")
        print(f"Input : {seqdir.relative_to(paths.root)}")
        print(f"Output: {outdir.relative_to(paths.root)}")

        cmd = [
            "dcm2niix",
            "-z", "y",
            "-ba", "y",
            "-f", f"{case_id}_{label}_%s",
            "-o", str(outdir),
            str(seqdir),
        ]
        run(cmd)

        after = set(outdir.iterdir())
        new_files = sorted(after - before)

        rows.append({
            "case": case_id,
            "pattern": pattern,
            "label": label,
            "outsub": outsub,
            "status": "converted",
            "input_dir": str(seqdir.relative_to(paths.root)),
            "output_dir": str(outdir.relative_to(paths.root)),
            "n_new_files": len(new_files),
            "new_files": ";".join(str(p.relative_to(paths.root)) for p in new_files),
        })

    return rows


def organize_gre(paths, case_id: str) -> dict:
    gre_dir = paths.case_nifti_dir(case_id) / "GRE"
    mag_dir = gre_dir / "mag"
    phase_dir = gre_dir / "phase"

    mag_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)

    moved_mag = 0
    moved_phase = 0
    skipped_existing = 0

    for p in sorted(gre_dir.iterdir()):
        if not p.is_file():
            continue

        if "_ph" in p.name:
            dest = phase_dir / p.name
            if dest.exists():
                skipped_existing += 1
            else:
                shutil.move(str(p), str(dest))
                moved_phase += 1

        elif p.name.endswith(".nii.gz") or p.suffix == ".json" or p.suffix == ".nii":
            dest = mag_dir / p.name
            if dest.exists():
                skipped_existing += 1
            else:
                shutil.move(str(p), str(dest))
                moved_mag += 1

    return {
        "case": case_id,
        "gre_dir": str(gre_dir.relative_to(paths.root)),
        "mag_dir": str(mag_dir.relative_to(paths.root)),
        "phase_dir": str(phase_dir.relative_to(paths.root)),
        "moved_mag": moved_mag,
        "moved_phase": moved_phase,
        "skipped_existing": skipped_existing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APF DICOM to NIfTI conversion v0.1")
    parser.add_argument("case_id", help="Case ID, e.g. ANISO_00_000_InS")
    args = parser.parse_args()

    case_id = args.case_id
    paths = get_paths()

    print("=" * 72)
    print("APF DICOM -> NIfTI conversion")
    print("=" * 72)
    print(f"Script : {SCRIPT}")
    print(f"Version: {VERSION}")
    print(f"Tag    : {TAG}")
    print(f"Case   : {case_id}")
    print(f"ROOT   : {paths.root}")
    print("=" * 72)

    if shutil.which("dcm2niix") is None:
        raise RuntimeError("dcm2niix not found on PATH")

    raw_case_dir = paths.find_case_raw_dir(case_id)
    if raw_case_dir is None:
        raise FileNotFoundError(f"Raw DICOM case directory not found for {case_id}")

    mri_root = find_mri_root(raw_case_dir)

    print(f"Raw case dir: {raw_case_dir.relative_to(paths.root)}")
    print(f"MRI root    : {mri_root.relative_to(paths.root)}")

    for sub in ["GRE", "DWI", "T1", "T2", "FLAIR", "T2STARFIT", "SE", "metadata"]:
        (paths.case_nifti_dir(case_id) / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for pattern, outsub, label in SEQUENCES:
        rows.extend(convert_sequence(paths, case_id, mri_root, pattern, outsub, label))

    gre_org = organize_gre(paths, case_id)

    summary_dir = (
        paths.summary_dir
        / "conversion"
        / "conv01"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        summary_dir
        / f"{TAG}_{case_id}_conversion_summary.csv"
    )

    json_path = (
        summary_dir
        / f"{TAG}_{case_id}_conversion_summary.json"
    )
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    end = now_iso()

    payload = {
        "script": SCRIPT,
        "version": VERSION,
        "tag": TAG,
        "created": end,
        "case": case_id,
        "root": str(paths.root),
        "raw_case_dir": str(raw_case_dir),
        "mri_root": str(mri_root),
        "outputs": {
            "csv": str(csv_path.relative_to(paths.root)),
            "json": str(json_path.relative_to(paths.root)),
        },
        "gre_organization": gre_org,
        "rows": rows,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nGRE organization")
    print(f"  moved mag      : {gre_org['moved_mag']}")
    print(f"  moved phase    : {gre_org['moved_phase']}")
    print(f"  skipped existing: {gre_org['skipped_existing']}")

    print("\nSummary")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
