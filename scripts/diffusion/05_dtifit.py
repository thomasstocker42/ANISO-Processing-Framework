#!/usr/bin/env python3
"""
Script diffusion/05: Run FSL DTIFIT for in-situ DWI cases

Tag: dwi05
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apf_paths import get_paths
from apf_path_utils import to_project_path
from apf_registry import get_cases


MODULE = "diffusion"
STEP = 5
NAME = "dtifit"

SCRIPT = f"{STEP:02d}_{NAME}.py"
TAG = f"dwi{STEP:02d}"
EDDY_TAG = "dwi04"

PATHS = get_paths()
ROOT = PATHS.root
NIFTI_DIR = PATHS.nifti_dir
DERIV_DIR = PATHS.derivatives_dir
SUMMARY_DIR = DERIV_DIR / "summary"
QC_DIR = SUMMARY_DIR / "QC"

DEFAULT_DATASET = "multi_b_N30"
DEFAULT_SHELL = "2000"
DEFAULT_FIT = "wls"

SHELLS = ["1000", "2000", "3000"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FSL DTIFIT for in-situ DWI cases.")
    parser.add_argument("case_id", nargs="?", help="Case ID, e.g. ANISO_00_000_InS")
    parser.add_argument("--all", action="store_true", help="Run all active/new InS cases.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dwi05 outputs.")

    parser.add_argument(
        "--dataset",
        choices=["multi_b_N30", "N64_b2000"],
        default=DEFAULT_DATASET,
        help=f"DWI dataset to use (default: {DEFAULT_DATASET}).",
    )

    parser.add_argument(
        "--shell",
        choices=["1000", "2000", "3000", "all"],
        default=DEFAULT_SHELL,
        help=f"Shell to fit for multi_b_N30. 'all' runs separate b1000/b2000/b3000 fits. Default: {DEFAULT_SHELL}.",
    )

    parser.add_argument(
        "--fit",
        choices=["wls", "ols"],
        default=DEFAULT_FIT,
        help=f"Tensor fitting method. WLS adds dtifit --wls. Default: {DEFAULT_FIT}.",
    )

    parser.add_argument(
        "--legacy-leonard",
        action="store_true",
        help="Leonard-compatible mode: dataset=N64_b2000, fit=ols.",
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


def file_exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists() and path.stat().st_size > 0


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
            tail = "\n".join(proc.stdout.splitlines()[-20:])
            raise RuntimeError(
                f"Command failed with return code {proc.returncode}\n"
                f"Last output lines:\n{tail}"
            )


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


def fslstats_median(path: Path, mask: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            ["fslstats", str(path), "-k", str(mask), "-P", "50"],
            check=True,
            text=True,
            capture_output=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_bvals(path: Path) -> List[float]:
    text = path.read_text(encoding="utf-8").strip()
    return [float(x) for x in text.split()]


def read_bvecs(path: Path) -> List[List[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        rows.append([float(x) for x in line.split()])
    if len(rows) != 3:
        raise RuntimeError(f"Expected 3-row bvec file, got {len(rows)} rows: {path}")
    return rows


def write_bvals(path: Path, values: List[float]) -> None:
    path.write_text(" ".join(f"{v:g}" for v in values) + "\n", encoding="utf-8")


def write_bvecs(path: Path, rows: List[List[float]]) -> None:
    path.write_text(
        "\n".join(" ".join(f"{v:.10g}" for v in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def selected_indices_for_shell(bvals: List[float], shell: str, tolerance: float = 100.0) -> List[int]:
    target = float(shell)
    return [i for i, b in enumerate(bvals) if b < 50 or abs(b - target) <= tolerance]


def select_bvec_columns(bvecs: List[List[float]], indices: List[int]) -> List[List[float]]:
    return [[row[i] for i in indices] for row in bvecs]


def find_primary_dwi_bval(case: str, dataset: str) -> Optional[Path]:
    dwi_dir = NIFTI_DIR / case / "DWI"
    if dataset == "multi_b_N30":
        matches = sorted(dwi_dir.glob("*multi_b_N30*.bval"))
    else:
        matches = sorted(dwi_dir.glob("*N64*b2000*.bval")) + sorted(dwi_dir.glob("*N64*B2000*.bval"))
    return matches[0] if matches else None


def build_shell_inputs(
    case: str,
    dataset: str,
    shell: str,
    eddy_corrected: Path,
    rotated_bvecs: Path,
    original_bval: Path,
    shell_dir: Path,
    log_path: Path,
) -> tuple[Path, Path, Path, List[int]]:
    if dataset == "N64_b2000":
        return eddy_corrected, original_bval, rotated_bvecs, list(range(read_bvals(original_bval).__len__()))

    bvals = read_bvals(original_bval)
    bvecs = read_bvecs(rotated_bvecs)

    indices = selected_indices_for_shell(bvals, shell)
    if not indices:
        raise RuntimeError(f"No volumes selected for shell b{shell}")

    selected_dwi = shell_dir / f"{TAG}_{case}_b{shell}_eddy_corrected.nii.gz"
    selected_bval = shell_dir / f"{TAG}_{case}_b{shell}.bval"
    selected_bvec = shell_dir / f"{TAG}_{case}_b{shell}.bvec"

    idx_spec = ",".join(str(i) for i in indices)
    run_capture(
        ["fslselectvols", "-i", str(eddy_corrected), "-o", str(selected_dwi), "--vols=" + idx_spec],
        log_path,
    )

    write_bvals(selected_bval, [bvals[i] for i in indices])
    write_bvecs(selected_bvec, select_bvec_columns(bvecs, indices))

    return selected_dwi, selected_bval, selected_bvec, indices


def expected_outputs(out_base: Path) -> Dict[str, Path]:
    return {
        "FA": out_base.parent / f"{out_base.name}_FA.nii.gz",
        "MD": out_base.parent / f"{out_base.name}_MD.nii.gz",
        "L1": out_base.parent / f"{out_base.name}_L1.nii.gz",
        "L2": out_base.parent / f"{out_base.name}_L2.nii.gz",
        "L3": out_base.parent / f"{out_base.name}_L3.nii.gz",
        "V1": out_base.parent / f"{out_base.name}_V1.nii.gz",
        "V2": out_base.parent / f"{out_base.name}_V2.nii.gz",
        "V3": out_base.parent / f"{out_base.name}_V3.nii.gz",
        "MO": out_base.parent / f"{out_base.name}_MO.nii.gz",
        "S0": out_base.parent / f"{out_base.name}_S0.nii.gz",
        "tensor": out_base.parent / f"{out_base.name}_tensor.nii.gz",
    }


def main() -> None:
    args = parse_args()

    if args.legacy_leonard:
        args.dataset = "N64_b2000"
        args.fit = "ols"
        args.shell = "2000"

    case_ids = resolve_cases(args)
    shells = SHELLS if args.shell == "all" else [args.shell]

    print("=" * 72)
    print(f"Script:      {SCRIPT}")
    print(f"Tag:         {TAG}")
    print("Task:        Diffusion tensor fitting with FSL DTIFIT")
    print(f"ROOT:        {ROOT}")
    print(f"Cases:       {len(case_ids)}")
    print(f"Dataset:     {args.dataset}")
    print(f"Shell:       {args.shell}")
    print(f"Fit:         {args.fit.upper()}")
    print("Tensor:      saved")
    print("=" * 72)

    if shutil.which("dtifit") is None:
        raise SystemExit("Required executable not found: dtifit")
    if shutil.which("fslselectvols") is None:
        raise SystemExit("Required executable not found: fslselectvols")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    dtifit_version = command_version(["dtifit", "--help"])

    rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []

    for case in case_ids:
        print("\n" + "=" * 72)
        print(case)
        print("=" * 72)

        eddy_dir = DERIV_DIR / case / "DWI" / "eddy"
        dtifit_dir = DERIV_DIR / case / "DWI" / "dtifit"
        dtifit_dir.mkdir(parents=True, exist_ok=True)

        log_path = dtifit_dir / f"{TAG}_{case}_dtifit.log"
        log_path.write_text(f"{SCRIPT} {case}\nStarted: {now_iso()}\n", encoding="utf-8")

        eddy_corrected = eddy_dir / f"{EDDY_TAG}_{case}_eddy_corrected.nii.gz"
        rotated_bvecs = eddy_dir / f"{EDDY_TAG}_{case}_eddy_corrected.eddy_rotated_bvecs"
        eddy_mask = eddy_dir / f"{EDDY_TAG}_{case}_eddy_mask.nii.gz"
        original_bval = find_primary_dwi_bval(case, args.dataset)

        required = [eddy_corrected, rotated_bvecs, eddy_mask, original_bval]
        missing = [p for p in required if not file_exists(p)]

        if missing:
            print("Input")
            for p in required:
                print(f"  {p if p else 'None'}: {'OK' if file_exists(p) else 'MISSING'}")
            for shell in shells:
                rows.append({
                    "case": case,
                    "shell": shell,
                    "dataset": args.dataset,
                    "fit": args.fit,
                    "status": "failed",
                    "notes": "missing_required_inputs",
                })
            continue

        print("Input")
        print(f"  corrected DWI:  OK  {eddy_corrected.name}")
        print(f"  rotated bvecs:   OK  {rotated_bvecs.name}")
        print(f"  bval:            OK  {original_bval.name}")
        print(f"  mask:            OK  {eddy_mask.name}")

        case_records: List[Dict[str, Any]] = []

        print("\nDTI Fits")
        print("-" * 56)

        for shell in shells:
            start_iso = now_iso()
            start_time = time.time()

            shell_label = f"b{shell}"
            shell_dir = dtifit_dir / shell_label
            shell_dir.mkdir(parents=True, exist_ok=True)

            out_base = shell_dir / f"{TAG}_{case}_{args.dataset}_{shell_label}_{args.fit}"
            outs = expected_outputs(out_base)
            metadata_json = shell_dir / f"{TAG}_{case}_{args.dataset}_{shell_label}_{args.fit}_metadata.json"

            if file_exists(outs["FA"]) and file_exists(outs["V1"]) and not args.force:
                status = "skipped"
                notes = "existing_outputs_use_force_to_overwrite"
                runtime_s = 0.0
                fa_median = fslstats_median(outs["FA"], eddy_mask)
                md_median = fslstats_median(outs["MD"], eddy_mask) if file_exists(outs["MD"]) else None
                print(f"✓ {shell_label} SKIPPED existing outputs")
            else:
                status = "OK"
                notes = "dtifit_completed"
                try:
                    selected_dwi, selected_bval, selected_bvec, indices = build_shell_inputs(
                        case=case,
                        dataset=args.dataset,
                        shell=shell,
                        eddy_corrected=eddy_corrected,
                        rotated_bvecs=rotated_bvecs,
                        original_bval=original_bval,
                        shell_dir=shell_dir,
                        log_path=log_path,
                    )

                    cmd = [
                        "dtifit",
                        "-k", str(selected_dwi),
                        "-o", str(out_base),
                        "-m", str(eddy_mask),
                        "-r", str(selected_bvec),
                        "-b", str(selected_bval),
                        "--save_tensor",
                    ]

                    if args.fit == "wls":
                        cmd.append("--wls")

                    run_capture(cmd, log_path)

                except Exception as exc:
                    status = "failed"
                    notes = f"processing_failed: {exc}"
                    indices = []
                    print(f"! {shell_label} failed: {exc}")

                runtime_s = round(time.time() - start_time, 3)
                fa_median = fslstats_median(outs["FA"], eddy_mask) if file_exists(outs["FA"]) else None
                md_median = fslstats_median(outs["MD"], eddy_mask) if file_exists(outs["MD"]) else None

            end_iso = now_iso()
            output_checks = {k: file_exists(v) for k, v in outs.items()}
            n_volumes = fsl_dim4(shell_dir / f"{TAG}_{case}_b{shell}_eddy_corrected.nii.gz") if args.dataset == "multi_b_N30" else fsl_dim4(eddy_corrected)
            mask_voxels = fslstats_voxels(eddy_mask)

            record = {
                "script": SCRIPT,
                "tag": TAG,
                "module": MODULE,
                "step": STEP,
                "case": case,
                "dataset": args.dataset,
                "shell": shell,
                "fit": args.fit,
                "status": status,
                "notes": notes,
                "start": start_iso,
                "end": end_iso,
                "runtime_seconds": runtime_s,
                "n_volumes": n_volumes if n_volumes is not None else "",
                "mask_voxels": mask_voxels if mask_voxels is not None else "",
                "fa_median": fa_median if fa_median is not None else "",
                "md_median": md_median if md_median is not None else "",
                "output_dir": to_project_path(shell_dir, PATHS),
                "FA": to_project_path(outs["FA"], PATHS) if file_exists(outs["FA"]) else "",
                "MD": to_project_path(outs["MD"], PATHS) if file_exists(outs["MD"]) else "",
                "V1": to_project_path(outs["V1"], PATHS) if file_exists(outs["V1"]) else "",
                "tensor": to_project_path(outs["tensor"], PATHS) if file_exists(outs["tensor"]) else "",
                "metadata_json": to_project_path(metadata_json, PATHS),
                "log": to_project_path(log_path, PATHS),
            }

            metadata = {
                **record,
                "created": now_iso(),
                "execution": {
                    "host": socket.gethostname(),
                    "tool": "dtifit",
                    "tool_version": dtifit_version,
                    "fit": args.fit,
                    "tensor_saved": True,
                },
                "inputs": {
                    "eddy_corrected": to_project_path(eddy_corrected, PATHS),
                    "rotated_bvecs": to_project_path(rotated_bvecs, PATHS),
                    "bval": to_project_path(original_bval, PATHS),
                    "mask": to_project_path(eddy_mask, PATHS),
                },
                "outputs": {k: to_project_path(v, PATHS) for k, v in outs.items()},
                "qc": {
                    "output_checks": output_checks,
                    "fa_median": fa_median,
                    "md_median": md_median,
                    "mask_voxels": mask_voxels,
                },
            }

            metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            rows.append(record)
            qc_rows.append({
                "case": case,
                "dataset": args.dataset,
                "shell": shell,
                "fit": args.fit,
                "status": status,
                "n_volumes": n_volumes if n_volumes is not None else "",
                "mask_voxels": mask_voxels if mask_voxels is not None else "",
                "fa_median": fa_median if fa_median is not None else "",
                "md_median": md_median if md_median is not None else "",
                **{f"has_{k}": "OK" if ok else "MISSING" for k, ok in output_checks.items()},
            })

            status_icon = "✓" if status in {"OK", "skipped"} else "!"
            print(f"{status_icon} {shell_label}")
            print(f"    Runtime:      {runtime_s:.1f} s")
            print(f"    Brain voxels: {mask_voxels if mask_voxels is not None else 'NA'}")
            print(f"    FA median:    {fa_median if fa_median is not None else 'NA'}")
            print(f"    MD median:    {md_median if md_median is not None else 'NA'}")
            print(f"    Status:       {status}")

            case_records.append(record)

        case_summary = dtifit_dir / f"{TAG}_{case}_summary.json"
        case_summary.write_text(json.dumps(case_records, indent=2), encoding="utf-8")

    csv_path = SUMMARY_DIR / f"{TAG}_dtifit.csv"
    json_path = SUMMARY_DIR / f"{TAG}_dtifit.json"

    write_csv(csv_path, rows)
    write_csv(QC_DIR / f"{TAG}_dtifit_qc.csv", qc_rows)

    summary = {
        "script": SCRIPT,
        "tag": TAG,
        "created": now_iso(),
        "root": ".",
        "dataset": args.dataset,
        "shell": args.shell,
        "fit": args.fit,
        "tensor_saved": True,
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "OK"),
        "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Done.")
    print(f"OK:      {summary['n_ok']}")
    print(f"Failed:  {summary['n_failed']}")
    print(f"Skipped: {summary['n_skipped']}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print(f"QC:      {QC_DIR / f'{TAG}_dtifit_qc.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()