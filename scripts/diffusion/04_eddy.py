#!/usr/bin/env python3
"""
Script diffusion/04: Run FSL EDDY for in-situ DWI cases

Tag: dwi04
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apf_paths import get_paths

#path standardisation#
from apf_path_utils import to_project_path, from_project_path

from apf_registry import get_cases


MODULE = "diffusion"
STEP = 4
NAME = "eddy"

SCRIPT = f"{STEP:02d}_{NAME}.py"
TAG = f"dwi{STEP:02d}"
PREP_TAG = "dwi02"
TOPUP_TAG = "dwi03"

PATHS = get_paths()
ROOT = PATHS.root
NIFTI_DIR = PATHS.nifti_dir
DERIV_DIR = PATHS.derivatives_dir
SUMMARY_DIR = DERIV_DIR / "summary"
QC_DIR = SUMMARY_DIR / "QC"

EDDY_EXECUTABLE = "eddy"
DEFAULT_THREADS = os.cpu_count() or 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FSL EDDY for in-situ DWI cases.")
    parser.add_argument("case_id", nargs="?", help="Case ID, e.g. ANISO_00_000_InS")
    parser.add_argument("--all", action="store_true", help="Run all active/new InS cases.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dwi04 outputs.")
    #parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Threads for eddy --nthr.")
    parser.add_argument(
    "--threads",
    type=int,
    default=DEFAULT_THREADS,
    help="Threads for eddy (--nthr).",
    )

    parser.add_argument(
        "--repol",
        action="store_true",
        help=(
            "Enable FSL EDDY outlier replacement (--repol). "
            "Disabled by default because postmortem data generally have no subject motion "
            "and this option substantially increases runtime."
        ),
    )
    return parser.parse_args()


def resolve_cases(args: argparse.Namespace) -> List[str]:
    if args.all and args.case_id:
        raise SystemExit("Use either a single case ID or --all, not both.")
    if args.all:
        return [case.case for case in get_cases(case_type="InS", paths=PATHS)]
    if args.case_id:
        return [args.case_id]
    raise SystemExit("Provide a case ID or use --all.")


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


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


def run_capture(cmd: List[str], log_path: Path) -> None:
    print("  $ " + " ".join(cmd))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with return code {proc.returncode}")


def command_version(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return proc.stdout.splitlines()[0] if proc.stdout else ""
    except Exception:
        return ""


def fsl_dim4(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(["fslval", str(path), "dim4"], check=True, text=True, capture_output=True)
        return int(result.stdout.strip())
    except Exception:
        return None


def fslstats_voxels(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(["fslstats", str(path), "-V"], check=True, text=True, capture_output=True)
        return int(float(result.stdout.strip().split()[0]))
    except Exception:
        return None


def read_motion_stats(path: Path) -> Dict[str, Any]:
    if not file_exists(path):
        return {
            "motion_rms_mean": "",
            "motion_rms_max": "",
            "motion_rms_n_values": "",
        }

    values: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for item in line.split():
            try:
                values.append(float(item))
            except ValueError:
                pass

    if not values:
        return {
            "motion_rms_mean": "",
            "motion_rms_max": "",
            "motion_rms_n_values": 0,
        }

    return {
        "motion_rms_mean": sum(values) / len(values),
        "motion_rms_max": max(values),
        "motion_rms_n_values": len(values),
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def find_hdbet_mask(eddy_dir: Path, case: str) -> Path: #implicit way of mask find#
    patterns = [
        f"{TAG}_{case}_mean_b0_hdbet_bet.nii.gz",
        f"{TAG}_{case}_mean_b0_hdbet_mask.nii.gz",
        f"{TAG}_{case}_mean_b0_hdbet*mask*.nii.gz",
        f"{TAG}_{case}_mean_b0_hdbet*bet*.nii.gz",
    ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(eddy_dir.glob(pattern)))

    unique = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)

    binary = []
    for path in unique:
        try:
            result = subprocess.run(
                ["fslstats", str(path), "-R"],
                check=True,
                text=True,
                capture_output=True,
            )
            vmin, vmax = [float(x) for x in result.stdout.strip().split()]
            if vmin == 0.0 and vmax == 1.0:
                binary.append(path)
        except Exception:
            pass

    if len(binary) == 1:
        return binary[0]

    if len(binary) > 1:
        raise RuntimeError(
            "Multiple binary HD-BET mask candidates found: "
            + "; ".join(str(p) for p in binary)
        )

    raise RuntimeError("No valid binary HD-BET mask found.")


def main() -> None:
    args = parse_args()
    case_ids = resolve_cases(args)
    case_source = "APF registry" if args.all else "Command-line argument"

    print("=" * 72)
    print(f"Script:      {SCRIPT}")
    print(f"Tag:         {TAG}")
    print("Task:        Eddy-current and motion correction with FSL EDDY")
    print(f"ROOT:        {ROOT}")
    print(f"Case source: {case_source}")
    if args.all:
        print("Registry:    config/InS_cases.csv (active/new)")
        print(f"Cases:       {len(case_ids)}")
    else:
        print(f"Case:        {case_ids[0]}")
    print(f"Threads:     {args.threads}")
    print(f"REPOL:       {'enabled' if args.repol else 'disabled'}")
    print("=" * 72)

    if shutil.which(EDDY_EXECUTABLE) is None:
        raise SystemExit(f"Required executable not found: {EDDY_EXECUTABLE}")

    if shutil.which("hd-bet") is None:
        raise SystemExit("Required executable not found: hd-bet")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    qc_runtime_rows: List[Dict[str, Any]] = []
    qc_output_rows: List[Dict[str, Any]] = []
    qc_volume_rows: List[Dict[str, Any]] = []
    qc_motion_rows: List[Dict[str, Any]] = []

    eddy_version = command_version([EDDY_EXECUTABLE, "--help"])
    hdbet_version = command_version(["hd-bet", "--version"])

    for case in case_ids:
        start_iso = now_iso()
        start_time = time.time()

        print("\n" + "=" * 72)
        print(case)
        print("=" * 72)

        topup_dir = DERIV_DIR / case / "DWI" / "topup"
        eddy_dir = DERIV_DIR / case / "DWI" / "eddy"
        eddy_dir.mkdir(parents=True, exist_ok=True)

        selected_inputs_json = topup_dir / f"{PREP_TAG}_{case}_selected_inputs.json"
        acqparams = topup_dir / f"{PREP_TAG}_{case}_acqparams.txt"
        index_file = topup_dir / f"{PREP_TAG}_{case}_index.txt"
        corrected_b0s = topup_dir / f"{TOPUP_TAG}_{case}_corrected_b0s.nii.gz"
        topup_base = topup_dir / f"{TOPUP_TAG}_{case}_topup"
        topup_fieldcoef = topup_dir / f"{TOPUP_TAG}_{case}_topup_fieldcoef.nii.gz"
        topup_movpar = topup_dir / f"{TOPUP_TAG}_{case}_topup_movpar.txt"

        mean_b0 = eddy_dir / f"{TAG}_{case}_mean_b0.nii.gz"
        hdbet_out = eddy_dir / f"{TAG}_{case}_mean_b0_hdbet.nii.gz"

        #hdbet_mask = eddy_dir / f"{TAG}_{case}_mean_b0_hdbet_mask.nii.gz" #mask explicit, must know name convention of hd-bet 25.6. tst#
        hdbet_mask_expected = eddy_dir / f"{TAG}_{case}_mean_b0_hdbet_mask.nii.gz"
        hdbet_mask = hdbet_mask_expected

        eddy_mask = eddy_dir / f"{TAG}_{case}_eddy_mask.nii.gz"


        out_base = eddy_dir / f"{TAG}_{case}_eddy_corrected"
        eddy_corrected = eddy_dir / f"{TAG}_{case}_eddy_corrected.nii.gz"
        rotated_bvecs = eddy_dir / f"{TAG}_{case}_eddy_corrected.eddy_rotated_bvecs"
        eddy_parameters = eddy_dir / f"{TAG}_{case}_eddy_corrected.eddy_parameters"
        movement_rms = eddy_dir / f"{TAG}_{case}_eddy_corrected.eddy_movement_rms"

        log_path = eddy_dir / f"{TAG}_{case}_eddy.log"
        metadata_json = eddy_dir / f"{TAG}_{case}_metadata.json"

        row: Dict[str, Any] = {
            "script_tag": TAG,
            "case": case,
            "status": "not_started",
            "notes": "",
            #"eddy_dir": str(eddy_dir.relative_to(ROOT)),
            "eddy_dir": to_project_path(eddy_dir, PATHS),
        }

        selected_payload = read_json(selected_inputs_json)
        #main_dwi = Path(selected_payload.get("main_dwi", "")) if selected_payload.get("main_dwi") else None
        main_dwi = (
            from_project_path(selected_payload["main_dwi"], PATHS)
            if selected_payload.get("main_dwi")
            else None
        )

        if main_dwi is None:
            main_matches = sorted((NIFTI_DIR / case).rglob("*multi_b_N30*.nii.gz"))
            main_dwi = main_matches[0] if main_matches else None

        main_bval = sidecar(main_dwi, "bval") if main_dwi else None
        main_bvec = sidecar(main_dwi, "bvec") if main_dwi else None

        required = [
            selected_inputs_json,
            acqparams,
            index_file,
            corrected_b0s,
            topup_fieldcoef,
            topup_movpar,
            main_dwi,
            main_bval,
            main_bvec,
        ]

        missing = [p for p in required if p is None or not file_exists(p)]

        if missing:
            status = "failed"
            notes = "missing_required_inputs"
            runtime_s = round(time.time() - start_time, 3)
            end_iso = now_iso()

            row.update(
                {
                    "status": status,
                    "notes": notes,
                    "start": start_iso,
                    "end": end_iso,
                    "runtime_seconds": runtime_s,
                }
            )
            rows.append(row)

            print("Input")
            for p in required:
                label = str(p) if p else "None"
                print(f"  {label}: {'OK' if p and file_exists(p) else 'MISSING'}")

        elif file_exists(eddy_corrected) and file_exists(rotated_bvecs) and not args.force:
            status = "skipped"
            notes = "existing_outputs_use_force_to_overwrite"
            runtime_s = round(time.time() - start_time, 3)
            end_iso = now_iso()

            row.update(
                {
                    "status": status,
                    "notes": notes,
                    "start": start_iso,
                    "end": end_iso,
                    "runtime_seconds": runtime_s,
                    #"main_dwi": str(main_dwi.relative_to(ROOT)),
                    #"eddy_corrected": str(eddy_corrected.relative_to(ROOT)),
                    #"rotated_bvecs": str(rotated_bvecs.relative_to(ROOT)),
                    #"metadata_json": str(metadata_json.relative_to(ROOT)) if file_exists(metadata_json) else "",
                    #"log": str(log_path.relative_to(ROOT)) if file_exists(log_path) else "",
                    "main_dwi": to_project_path(main_dwi, PATHS),
                    "eddy_corrected": to_project_path(eddy_corrected, PATHS),
                    "rotated_bvecs": to_project_path(rotated_bvecs, PATHS),
                    "metadata_json": to_project_path(metadata_json, PATHS) if file_exists(metadata_json) else "",
                    "log": to_project_path(log_path, PATHS) if file_exists(log_path) else "",
                }
            )
            rows.append(row)
            print("  SKIP: existing dwi04 outputs found. Use --force to overwrite.")

        else:
            log_path.write_text(f"{SCRIPT} {case}\nStarted: {start_iso}\n", encoding="utf-8")

            print("Input")
            print(f"  main DWI:       OK  {main_dwi.name}")
            print(f"  bval:           OK  {main_bval.name}")
            print(f"  bvec:           OK  {main_bvec.name}")
            print(f"  acqparams:      OK  {acqparams.name}")
            print(f"  index:          OK  {index_file.name}")
            print(f"  TOPUP base:     OK  {topup_base.name}")
            print(f"  corrected b0s:  OK  {corrected_b0s.name}")

            status = "OK"
            notes = "eddy_completed"

            try:
                run_capture(["fslmaths", str(corrected_b0s), "-Tmean", str(mean_b0)], log_path)

                run_capture(
                    [
                        "hd-bet",
                        "-i", str(mean_b0),
                        "-o", str(hdbet_out),
                        "-device", "cpu",
                        "--disable_tta",
                        "--save_bet_mask",
                    ],
                    log_path,
                )

                hdbet_mask = find_hdbet_mask(eddy_dir, case) # find mask implicit #
                run_capture(["fslmaths", str(hdbet_mask), "-bin", str(eddy_mask)], log_path)

                eddy_cmd = [
                    EDDY_EXECUTABLE,
                    f"--imain={main_dwi}",
                    f"--mask={eddy_mask}",
                    f"--acqp={acqparams}",
                    f"--index={index_file}",
                    f"--bvecs={main_bvec}",
                    f"--bvals={main_bval}",
                    f"--topup={topup_base}",
                    f"--out={out_base}",
                    #"--repol",
                    f"--nthr={args.threads}",
                ]

                if args.repol:
                    eddy_cmd.append("--repol")

                run_capture(eddy_cmd, log_path)

            except Exception as exc:
                status = "failed"
                notes = f"processing_failed: {exc}"
                print(f"  ! EDDY failed: {exc}")

            runtime_s = round(time.time() - start_time, 3)
            end_iso = now_iso()

            n_input_volumes = fsl_dim4(main_dwi)
            n_corrected_volumes = fsl_dim4(eddy_corrected) if file_exists(eddy_corrected) else None
            mask_voxels = fslstats_voxels(eddy_mask)
            motion_stats = read_motion_stats(movement_rms)

            output_checks = {
                "mean_b0": file_exists(mean_b0),
                "hdbet_mask": file_exists(hdbet_mask),
                "eddy_mask": file_exists(eddy_mask),
                "eddy_corrected": file_exists(eddy_corrected),
                "rotated_bvecs": file_exists(rotated_bvecs),
                "eddy_parameters": file_exists(eddy_parameters),
                "movement_rms": file_exists(movement_rms),
                "log": file_exists(log_path),
            }

            if status == "OK" and not output_checks["eddy_corrected"]:
                status = "failed"
                notes = "missing_eddy_corrected_output"

            eddy_cmd_str = " ".join(eddy_cmd)
            #eddy_cmd_str = " ".join(
            #    [
            #        EDDY_EXECUTABLE,
            #        f"--imain={main_dwi}",
            #        f"--mask={eddy_mask}",
            #        f"--acqp={acqparams}",
            #        f"--index={index_file}",
            #        f"--bvecs={main_bvec}",
            #        f"--bvals={main_bval}",
            #        f"--topup={topup_base}",
            #        f"--out={out_base}",
            #        "--repol",
            #        f"--nthr={args.threads}",
            #    ]
            #)

            metadata = {
                "script": SCRIPT,
                "tag": TAG,
                "module": MODULE,
                "step": STEP,
                "created": now_iso(),
                "case": case,
                "timing": {
                    "start": start_iso,
                    "end": end_iso,
                    "runtime_seconds": runtime_s,
                },
                "execution": {
                    "host": socket.gethostname(),
                    "threads": args.threads,
                    "repol": args.repol,
                    "tool": EDDY_EXECUTABLE,
                    "tool_version": eddy_version,
                    "command": eddy_cmd_str,
                    "tool_versions": {
                        "eddy": eddy_version,
                        "hd-bet": hdbet_version,
                    },
                },
                "status": status,
                "notes": notes,
                "inputs": {
                    #"main_dwi": str(main_dwi),
                    #"main_bval": str(main_bval),
                    #"main_bvec": str(main_bvec),
                    #"acqparams": str(acqparams),
                    #"index": str(index_file),
                    #"corrected_b0s": str(corrected_b0s),
                    #"topup_base": str(topup_base),
                    #"selected_inputs_json": str(selected_inputs_json),
                    "main_dwi": to_project_path(main_dwi, PATHS),
                    "main_bval": to_project_path(main_bval, PATHS),
                    "main_bvec": to_project_path(main_bvec, PATHS),
                    "acqparams": to_project_path(acqparams, PATHS),
                    "index": to_project_path(index_file, PATHS),
                    "corrected_b0s": to_project_path(corrected_b0s, PATHS),
                    "topup_base": to_project_path(topup_base, PATHS),
                    "selected_inputs_json": to_project_path(selected_inputs_json, PATHS),
                    "dwi02_metadata": selected_payload,
                },
                "outputs": {
                    #"mean_b0": str(mean_b0),
                    #"hdbet_out": str(hdbet_out),
                    #"hdbet_mask": str(hdbet_mask),
                    #"eddy_mask": str(eddy_mask),
                    #"eddy_corrected": str(eddy_corrected),
                    #"rotated_bvecs": str(rotated_bvecs),
                    #"eddy_parameters": str(eddy_parameters),
                    #"movement_rms": str(movement_rms),
                    #"log": str(log_path),
                    "mean_b0": to_project_path(mean_b0, PATHS),
                    "hdbet_out": to_project_path(hdbet_out, PATHS),
                    "hdbet_mask": to_project_path(hdbet_mask, PATHS),
                    "eddy_mask": to_project_path(eddy_mask, PATHS),
                    "eddy_corrected": to_project_path(eddy_corrected, PATHS),
                    "rotated_bvecs": to_project_path(rotated_bvecs, PATHS),
                    "eddy_parameters": to_project_path(eddy_parameters, PATHS),
                    "movement_rms": to_project_path(movement_rms, PATHS),
                    "log": to_project_path(log_path, PATHS),

                },
                "qc": {
                    "n_input_volumes": n_input_volumes,
                    "n_corrected_volumes": n_corrected_volumes,
                    "mask_voxels": mask_voxels,
                    "output_checks": output_checks,
                    "motion_statistics": motion_stats,
                },
            }

            metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            row.update(
                {
                    "status": status,
                    "notes": notes,
                    "start": start_iso,
                    "end": end_iso,
                    "runtime_seconds": runtime_s,
                    #"main_dwi": str(main_dwi.relative_to(ROOT)),
                    "main_dwi": to_project_path(main_dwi, PATHS),
                    "n_input_volumes": n_input_volumes if n_input_volumes is not None else "",
                    "n_corrected_volumes": n_corrected_volumes if n_corrected_volumes is not None else "",
                    "mask_voxels": mask_voxels if mask_voxels is not None else "",
                    #"eddy_corrected": str(eddy_corrected.relative_to(ROOT)) if file_exists(eddy_corrected) else "",
                    #"rotated_bvecs": str(rotated_bvecs.relative_to(ROOT)) if file_exists(rotated_bvecs) else "",
                    #"eddy_parameters": str(eddy_parameters.relative_to(ROOT)) if file_exists(eddy_parameters) else "",
                    #"movement_rms": str(movement_rms.relative_to(ROOT)) if file_exists(movement_rms) else "",
                    #"metadata_json": str(metadata_json.relative_to(ROOT)),
                    #"log": str(log_path.relative_to(ROOT)) if file_exists(log_path) else "",
                    "eddy_corrected": to_project_path(eddy_corrected, PATHS) if file_exists(eddy_corrected) else "",
                    "rotated_bvecs": to_project_path(rotated_bvecs, PATHS) if file_exists(rotated_bvecs) else "",
                    "eddy_parameters": to_project_path(eddy_parameters, PATHS) if file_exists(eddy_parameters) else "",
                    "movement_rms": to_project_path(movement_rms, PATHS) if file_exists(movement_rms) else "",
                    "metadata_json": to_project_path(metadata_json, PATHS),
                    "log": to_project_path(log_path, PATHS) if file_exists(log_path) else "",
                    **motion_stats,
                }
            )
            rows.append(row)

            print("Output")
            print(f"  mean b0:          {'OK' if output_checks['mean_b0'] else 'MISSING'}")
            print(f"  HD-BET mask:      {'OK' if output_checks['hdbet_mask'] else 'MISSING'}")
            print(f"  EDDY mask:        {'OK' if output_checks['eddy_mask'] else 'MISSING'}")
            print(f"  corrected DWI:    {'OK' if output_checks['eddy_corrected'] else 'MISSING'}")
            print(f"  rotated bvecs:    {'OK' if output_checks['rotated_bvecs'] else 'MISSING'}")
            print("Status")
            print(f"  {status}: {notes}")
            print(f"  Runtime: {runtime_s:.1f} s")

        qc_runtime_rows.append(
            {
                "case": case,
                "status": row.get("status", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "notes": row.get("notes", ""),
            }
        )

        qc_output_rows.append(
            {
                "case": case,
                "status": row.get("status", ""),
                "mean_b0": "OK" if file_exists(mean_b0) else "MISSING",
                "hdbet_mask": "OK" if file_exists(hdbet_mask) else "MISSING",
                "eddy_mask": "OK" if file_exists(eddy_mask) else "MISSING",
                "eddy_corrected": "OK" if file_exists(eddy_corrected) else "MISSING",
                "rotated_bvecs": "OK" if file_exists(rotated_bvecs) else "MISSING",
                "eddy_parameters": "OK" if file_exists(eddy_parameters) else "MISSING",
                "movement_rms": "OK" if file_exists(movement_rms) else "MISSING",
                "metadata_json": "OK" if file_exists(metadata_json) else "MISSING",
                "log": "OK" if file_exists(log_path) else "MISSING",
            }
        )

        qc_volume_rows.append(
            {
                "case": case,
                "status": row.get("status", ""),
                "n_input_volumes": fsl_dim4(main_dwi) if main_dwi and file_exists(main_dwi) else "",
                "n_corrected_volumes": fsl_dim4(eddy_corrected) if file_exists(eddy_corrected) else "",
                "mask_voxels": fslstats_voxels(eddy_mask) if file_exists(eddy_mask) else "",
            }
        )

        qc_motion_rows.append(
            {
                "case": case,
                "status": row.get("status", ""),
                **read_motion_stats(movement_rms),
            }
        )

    csv_path = SUMMARY_DIR / f"{TAG}_eddy.csv"
    json_path = SUMMARY_DIR / f"{TAG}_eddy.json"

    write_csv(csv_path, rows)

    summary = {
        "script": SCRIPT,
        "tag": TAG,
        "created": now_iso(),
        #"root": str(ROOT),
        "root": ".",
        "case_mode": "all" if args.all else "single",
        "case_source": case_source,
        "registry": "config/InS_cases.csv (active/new)" if args.all else "",
        "threads": args.threads,
        "n_cases": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "OK"),
        "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "rows": rows,
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_csv(QC_DIR / f"{TAG}_case_runtime.csv", qc_runtime_rows)
    write_csv(QC_DIR / f"{TAG}_output_check.csv", qc_output_rows)
    write_csv(QC_DIR / f"{TAG}_volume_statistics.csv", qc_volume_rows)
    write_csv(QC_DIR / f"{TAG}_motion_statistics.csv", qc_motion_rows)

    print("\n" + "=" * 72)
    print("Done.")
    print(f"OK:      {summary['n_ok']}")
    print(f"Failed:  {summary['n_failed']}")
    print(f"Skipped: {summary['n_skipped']}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print(f"QC:      {QC_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()