#!/usr/bin/env python3
"""
Script diffusion/02: Prepare TOPUP inputs for in-situ DWI cases , version 2

Tag: dwi02

Usage
-----
Single case:
    ./apf scripts/diffusion/02_prepare_topup_inputs.py ANISO_00_000_InS

All active in-situ cases:
    ./apf scripts/diffusion/02_prepare_topup_inputs.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apf_paths import get_paths
from apf_path_utils import to_project_path
from apf_registry import get_cases


SCRIPT = "02_prepare_topup_inputs.py"
TAG = "dwi02"

PATHS = get_paths()
ROOT = PATHS.root
NIFTI_DIR = PATHS.nifti_dir
DERIV_DIR = PATHS.derivatives_dir
SUMMARY_DIR = DERIV_DIR / "summary"

MAIN_DWI_KEY = "multi_b_N30"
REVPE_KEY = "b0_revPE"

ACQPARAM_MAIN_J_MINUS = "0 -1 0"
ACQPARAM_REV_J_PLUS = "0 1 0"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TOPUP inputs for in-situ DWI cases."
    )
    parser.add_argument(
        "case_id",
        nargs="?",
        help="Case ID, e.g. ANISO_00_000_InS",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all active in-situ cases from the APF registry.",
    )
    return parser.parse_args()


def resolve_cases(args: argparse.Namespace) -> List[str]:
    if args.all and args.case_id:
        raise SystemExit("Use either a single case ID or --all, not both.")

    if args.all:
        cases = get_cases(case_type="InS", paths=PATHS)
        return [case.case for case in cases]

    if args.case_id:
        return [args.case_id]

    raise SystemExit("Provide a case ID or use --all.")


def run(cmd: List[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fsl_dim4(path: Path) -> int:
    result = subprocess.run(
        ["fslval", str(path), "dim4"],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def sidecar(path: Path, suffix: str) -> Path:
    return path.parent / f"{strip_nii_suffix(path)}.{suffix}"


def read_bvals(path: Path) -> List[float]:
    txt = path.read_text(encoding="utf-8").strip().replace("\n", " ")
    return [float(x) for x in txt.split() if x.strip()]


def find_one(case_dir: Path, key: str) -> Optional[Path]:
    matches = sorted(case_dir.rglob(f"*{key}*.nii.gz"))
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    case_ids = resolve_cases(args)

    print("=" * 72)
    print(f"Script: {SCRIPT}")
    print(f"Tag:    {TAG}")
    print("Task:   Prepare TOPUP inputs for in-situ DWI")
    print(f"ROOT:   {ROOT}")
    print(f"Cases:  {len(case_ids)}")
    print("=" * 72)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for case in case_ids:
        case_dir = NIFTI_DIR / case

        print("\n" + "=" * 72)
        print(case)
        print("=" * 72)

        if not case_dir.exists():
            row = {
                "script_tag": TAG,
                "case": case,
                "status": "failed",
                "notes": "missing_nifti_case_directory",
            }
            rows.append(row)
            print(f"  ! missing NIfTI case directory: {case_dir}")
            continue

        main_dwi = find_one(case_dir, MAIN_DWI_KEY)
        revpe = find_one(case_dir, REVPE_KEY)

        out_dir = DERIV_DIR / case / "DWI" / "topup"
        out_dir.mkdir(parents=True, exist_ok=True)

        row: Dict[str, Any] = {
            "script_tag": TAG,
            "case": case,
            #"main_dwi": str(main_dwi.relative_to(ROOT)) if main_dwi else "",
            #"revpe_b0": str(revpe.relative_to(ROOT)) if revpe else "",
            "main_dwi": to_project_path(main_dwi, PATHS) if main_dwi else "",
            "revpe_b0": to_project_path(revpe, PATHS) if revpe else "",
            "status": "not_started",
            "notes": "",
        }

        if main_dwi is None:
            row["status"] = "failed"
            row["notes"] = "missing_multi_b_N30_main_DWI"
            rows.append(row)
            print("  ! missing main multi_b_N30 DWI")
            continue

        if revpe is None:
            row["status"] = "failed"
            row["notes"] = "missing_b0_revPE"
            rows.append(row)
            print("  ! missing b0_revPE")
            continue

        main_json = sidecar(main_dwi, "json")
        main_bval = sidecar(main_dwi, "bval")
        main_bvec = sidecar(main_dwi, "bvec")
        rev_json = sidecar(revpe, "json")

        missing = [
            p.name
            for p in [main_json, main_bval, main_bvec, rev_json]
            if not p.exists()
        ]
        if missing:
            row["status"] = "failed"
            row["notes"] = "missing_required_sidecars: " + ";".join(missing)
            rows.append(row)
            print(f"  ! missing required sidecars: {missing}")
            continue

        main_meta = read_json(main_json)
        rev_meta = read_json(rev_json)

        main_pe = str(main_meta.get("PhaseEncodingDirection", ""))
        rev_pe = str(rev_meta.get("PhaseEncodingDirection", ""))
        total_ro = str(main_meta.get("TotalReadoutTime", ""))

        if main_pe != "j-" or rev_pe != "j":
            row["status"] = "failed"
            row["notes"] = f"unexpected_phase_encoding_main_{main_pe}_rev_{rev_pe}"
            rows.append(row)
            print(f"  ! unexpected PE directions: main={main_pe}, rev={rev_pe}")
            continue

        if not total_ro:
            row["status"] = "failed"
            row["notes"] = "missing_TotalReadoutTime"
            rows.append(row)
            print("  ! missing TotalReadoutTime")
            continue

        bvals = read_bvals(main_bval)
        b0_indices = [i for i, b in enumerate(bvals) if b < 50]

        if not b0_indices:
            row["status"] = "failed"
            row["notes"] = "no_b0_volumes_in_main_DWI"
            rows.append(row)
            print("  ! no b0 volumes in main DWI")
            continue

        n_revpe_volumes = fsl_dim4(revpe)

        first_main_b0 = out_dir / f"{TAG}_{case}_main_b0s.nii.gz"
        topup_pair = out_dir / f"{TAG}_{case}_b0_pair_for_topup.nii.gz"
        acqparams = out_dir / f"{TAG}_{case}_acqparams.txt"
        index_file = out_dir / f"{TAG}_{case}_index.txt"
        selected_json = out_dir / f"{TAG}_{case}_selected_inputs.json"

        temp_b0s = []
        for n, idx in enumerate(b0_indices):
            tmp = out_dir / f"{TAG}_{case}_tmp_main_b0_{n:02d}.nii.gz"
            run(["fslroi", str(main_dwi), str(tmp), str(idx), "1"])
            temp_b0s.append(tmp)

        run(["fslmerge", "-t", str(first_main_b0)] + [str(p) for p in temp_b0s])
        run(["fslmerge", "-t", str(topup_pair), str(first_main_b0), str(revpe)])

        acq_lines = (
            [f"{ACQPARAM_MAIN_J_MINUS} {total_ro}"] * len(b0_indices)
            + [f"{ACQPARAM_REV_J_PLUS} {total_ro}"] * n_revpe_volumes
        )
        acqparams.write_text("\n".join(acq_lines) + "\n", encoding="utf-8")

        index_file.write_text(" ".join(["1"] * len(bvals)) + "\n", encoding="utf-8")

        for tmp in temp_b0s:
            tmp.unlink(missing_ok=True)

        payload = {
            "script": SCRIPT,
            "tag": TAG,
            "created": now_iso(),
            "case": case,
            #"main_dwi": str(main_dwi),
            #"main_json": str(main_json),
            #"main_bval": str(main_bval),
            #"main_bvec": str(main_bvec),
            #"revpe_b0": str(revpe),
            #"revpe_json": str(rev_json),
            "main_dwi": to_project_path(main_dwi, PATHS),
            "main_json": to_project_path(main_json, PATHS),
            "main_bval": to_project_path(main_bval, PATHS),
            "main_bvec": to_project_path(main_bvec, PATHS),
            "revpe_b0": to_project_path(revpe, PATHS),
            "revpe_json": to_project_path(rev_json, PATHS),
            "main_phase_encoding_direction": main_pe,
            "revpe_phase_encoding_direction": rev_pe,
            "total_readout_time": total_ro,
            "main_b0_indices_0based": b0_indices,
            "n_main_volumes": len(bvals),
            "n_main_b0": len(b0_indices),
            "n_revpe_volumes": n_revpe_volumes,
            "n_topup_volumes": len(b0_indices) + n_revpe_volumes,
            "outputs": {
                #"main_b0s": str(first_main_b0),
                #"topup_pair": str(topup_pair),
                #"acqparams": str(acqparams),
                #"index": str(index_file),
                "main_b0s": to_project_path(first_main_b0, PATHS),
                "topup_pair": to_project_path(topup_pair, PATHS),
                "acqparams": to_project_path(acqparams, PATHS),
                "index": to_project_path(index_file, PATHS),
            },
        }
        selected_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        row.update(
            {
                "status": "OK",
                "notes": "prepared_topup_inputs",
                "main_pe": main_pe,
                "revpe_pe": rev_pe,
                "total_readout_time": total_ro,
                "n_main_volumes": len(bvals),
                "n_main_b0": len(b0_indices),
                "n_revpe_volumes": n_revpe_volumes,
                "n_topup_volumes": len(b0_indices) + n_revpe_volumes,
                "b0_indices_0based": ";".join(str(i) for i in b0_indices),
                #"topup_pair": str(topup_pair.relative_to(ROOT)),
                #"acqparams": str(acqparams.relative_to(ROOT)),
                #"index": str(index_file.relative_to(ROOT)),
                #"selected_inputs_json": str(selected_json.relative_to(ROOT)),
                "topup_pair": to_project_path(topup_pair, PATHS),
                "acqparams": to_project_path(acqparams, PATHS),
                "index": to_project_path(index_file, PATHS),
                "selected_inputs_json": to_project_path(selected_json, PATHS),
            }
        )
        rows.append(row)

        print(f"  OK: {topup_pair}")
        print(f"  main b0 indices: {b0_indices}")
        print(f"  reverse PE volumes: {n_revpe_volumes}")
        print(f"  TOPUP volumes: {len(b0_indices) + n_revpe_volumes}")

    csv_path = SUMMARY_DIR / f"{TAG}_prepare_topup_inputs.csv"
    json_path = SUMMARY_DIR / f"{TAG}_prepare_topup_inputs.json"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "script": SCRIPT,
        "tag": TAG,
        "created": now_iso(),
        #"root": str(ROOT),
        "root": ".",
        "case_mode": "all" if args.all else "single",
        "n_cases": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "OK"),
        "n_failed": sum(1 for r in rows if r.get("status") != "OK"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Done.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
