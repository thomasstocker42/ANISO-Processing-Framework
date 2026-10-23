#!/usr/bin/env python3
"""
Script diffusion/03: Run FSL TOPUP for in-situ DWI cases

Tag: dwi03
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apf_paths import get_paths
from apf_path_utils import to_project_path
from apf_registry import get_cases


SCRIPT = "03_topup.py"
TAG = "dwi03"
PREV_TAG = "dwi02"

PATHS = get_paths()
ROOT = PATHS.root
DERIV_DIR = PATHS.derivatives_dir
SUMMARY_DIR = DERIV_DIR / "summary"
QC_DIR = SUMMARY_DIR / "QC"

TOPUP_CONFIG = "b02b0.cnf"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FSL TOPUP for in-situ DWI cases.")
    parser.add_argument("case_id", nargs="?", help="Case ID, e.g. ANISO_00_000_InS")
    parser.add_argument("--all", action="store_true", help="Run all active/new InS cases.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dwi03 outputs.")
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


def run_capture(cmd: List[str], log_path: Path) -> None:
    print("  $ " + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with return code {proc.returncode}")


def fsl_dim4(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            ["fslval", str(path), "dim4"],
            check=True,
            text=True,
            capture_output=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def fslstats_field(path: Path) -> Dict[str, Any]:
    if not file_exists(path):
        return {
            "field_min_Hz": "",
            "field_max_Hz": "",
            "field_mean_Hz": "",
            "field_std_Hz": "",
            "field_p01_Hz": "",
            "field_p99_Hz": "",
        }

    result = subprocess.run(
        ["fslstats", str(path), "-R", "-M", "-S", "-P", "1", "-P", "99"],
        check=True,
        text=True,
        capture_output=True,
    )
    values = [float(x) for x in result.stdout.strip().split()]
    return {
        "field_min_Hz": values[0],
        "field_max_Hz": values[1],
        "field_mean_Hz": values[2],
        "field_std_Hz": values[3],
        "field_p01_Hz": values[4],
        "field_p99_Hz": values[5],
    }


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({k for row in records for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    case_ids = resolve_cases(args)
    case_source = "APF registry" if args.all else "Command-line argument"

    print("=" * 72)
    print(f"Script:      {SCRIPT}")
    print(f"Tag:         {TAG}")
    print("Task:        Estimate susceptibility field with FSL TOPUP")
    print(f"ROOT:        {ROOT}")
    print(f"Case source: {case_source}")
    if args.all:
        print("Registry:    config/InS_cases.csv (active/new)")
        print(f"Cases:       {len(case_ids)}")
    else:
        print(f"Case:        {case_ids[0]}")
    print("=" * 72)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    qc_runtime_rows: List[Dict[str, Any]] = []
    qc_output_rows: List[Dict[str, Any]] = []
    qc_field_rows: List[Dict[str, Any]] = []

    for case in case_ids:
        start_iso = now_iso()
        start_time = time.time()

        print("\n" + "=" * 72)
        print(case)
        print("=" * 72)

        topup_dir = DERIV_DIR / case / "DWI" / "topup"
        topup_dir.mkdir(parents=True, exist_ok=True)

        imain = topup_dir / f"{PREV_TAG}_{case}_b0_pair_for_topup.nii.gz"
        acqparams = topup_dir / f"{PREV_TAG}_{case}_acqparams.txt"
        selected_inputs_json = topup_dir / f"{PREV_TAG}_{case}_selected_inputs.json"

        out_base = topup_dir / f"{TAG}_{case}_topup"
        field_hz = topup_dir / f"{TAG}_{case}_field_Hz.nii.gz"
        corrected_b0s = topup_dir / f"{TAG}_{case}_corrected_b0s.nii.gz"
        log_path = topup_dir / f"{TAG}_{case}_topup.log"
        metadata_json = topup_dir / f"{TAG}_{case}_metadata.json"

        fieldcoef = topup_dir / f"{TAG}_{case}_topup_fieldcoef.nii.gz"
        movpar = topup_dir / f"{TAG}_{case}_topup_movpar.txt"

        row: Dict[str, Any] = {
            "script_tag": TAG,
            "case": case,
            "status": "not_started",
            "notes": "",
            #"topup_dir": str(topup_dir.relative_to(ROOT)),
            #"imain": str(imain.relative_to(ROOT)),
            #"acqparams": str(acqparams.relative_to(ROOT)),
            "topup_dir": to_project_path(topup_dir, PATHS),
            "imain": to_project_path(imain, PATHS),
            "acqparams": to_project_path(acqparams, PATHS),
        }

        missing = [p for p in [imain, acqparams, selected_inputs_json] if not file_exists(p)]

        if missing:
            status = "failed"
            notes = "missing_dwi02_outputs: " + ";".join(p.name for p in missing)
            runtime_s = round(time.time() - start_time, 3)
            end_iso = now_iso()

            row.update({"status": status, "notes": notes, "runtime_seconds": runtime_s})
            rows.append(row)

            print("Input")
            for p in [imain, acqparams, selected_inputs_json]:
                print(f"  {p.name}: {'OK' if file_exists(p) else 'MISSING'}")

        else:
            outputs = [fieldcoef, movpar, field_hz, corrected_b0s, metadata_json, log_path]
            print("\nSkip check") #skip check, shows outputs already generated tst
            for p in outputs:
                print(f"  {p.name:<45} {file_exists(p)}")
            existing_outputs = any(file_exists(p) for p in outputs)

            if existing_outputs and not args.force:
                status = "skipped"
                notes = "existing_outputs_use_force_to_overwrite"
                runtime_s = round(time.time() - start_time, 3)
                end_iso = now_iso()

                n_input_volumes = fsl_dim4(imain)
                n_corrected_volumes = fsl_dim4(corrected_b0s) if file_exists(corrected_b0s) else None

                row.update(
                    {
                        "status": status,
                        "notes": notes,
                        "runtime_seconds": runtime_s,
                        "n_input_volumes": n_input_volumes if n_input_volumes is not None else "",
                        "n_corrected_volumes": n_corrected_volumes if n_corrected_volumes is not None else "",
                        #"fieldcoef": str(fieldcoef.relative_to(ROOT)) if file_exists(fieldcoef) else "",
                        #"movpar": str(movpar.relative_to(ROOT)) if file_exists(movpar) else "",
                        #"field_Hz": str(field_hz.relative_to(ROOT)) if file_exists(field_hz) else "",
                        #"corrected_b0s": str(corrected_b0s.relative_to(ROOT)) if file_exists(corrected_b0s) else "",
                        #"metadata_json": str(metadata_json.relative_to(ROOT)) if file_exists(metadata_json) else "",
                        #"log": str(log_path.relative_to(ROOT)) if file_exists(log_path) else "",
                        "fieldcoef": to_project_path(fieldcoef, PATHS) if file_exists(fieldcoef) else "",
                        "movpar": to_project_path(movpar, PATHS) if file_exists(movpar) else "",
                        "field_Hz": to_project_path(field_hz, PATHS) if file_exists(field_hz) else "",
                        "corrected_b0s": to_project_path(corrected_b0s, PATHS) if file_exists(corrected_b0s) else "",
                        "metadata_json": to_project_path(metadata_json, PATHS) if file_exists(metadata_json) else "",
                        "log": to_project_path(log_path, PATHS) if file_exists(log_path) else "",
                    }
                )
                rows.append(row)
                print("  SKIP: existing dwi03 outputs found. Use --force to overwrite.")

            else:
                print("Input")
                print(f"  b0 pair:              OK  {imain.name}")
                print(f"  acqparams:            OK  {acqparams.name}")
                print(f"  selected inputs JSON: OK  {selected_inputs_json.name}")

                cmd = [
                    "topup",
                    f"--imain={imain}",
                    f"--datain={acqparams}",
                    f"--config={TOPUP_CONFIG}",
                    f"--out={out_base}",
                    f"--fout={field_hz}",
                    f"--iout={corrected_b0s}",
                ]

                status = "OK"
                notes = "topup_completed"

                try:
                    run_capture(cmd, log_path)
                except Exception as exc:
                    status = "failed"
                    notes = f"topup_failed: {exc}"
                    print(f"  ! TOPUP failed: {exc}")

                runtime_s = round(time.time() - start_time, 3)
                end_iso = now_iso()

                n_input_volumes = fsl_dim4(imain)
                n_corrected_volumes = fsl_dim4(corrected_b0s) if file_exists(corrected_b0s) else None

                output_checks = {
                    "fieldcoef": file_exists(fieldcoef),
                    "movpar": file_exists(movpar),
                    "field_Hz": file_exists(field_hz),
                    "corrected_b0s": file_exists(corrected_b0s),
                    "log": file_exists(log_path),
                }

                if status == "OK" and not all(output_checks.values()):
                    status = "failed"
                    missing_outputs = [k for k, v in output_checks.items() if not v]
                    notes = "missing_expected_outputs: " + ";".join(missing_outputs)

                selected_payload = read_json(selected_inputs_json)
                field_stats = fslstats_field(field_hz)

                metadata = {
                    "script": SCRIPT,
                    "tag": TAG,
                    "created": now_iso(),
                    "case": case,
                    "timing": {
                        "start": start_iso,
                        "end": end_iso,
                        "runtime_seconds": runtime_s,
                    },
                    "status": status,
                    "notes": notes,
                    "software": {
                        "tool": "FSL TOPUP",
                        "config": TOPUP_CONFIG,
                    },
                    "inputs": {
                        #"imain": str(imain),
                        #"acqparams": str(acqparams),
                        #"selected_inputs_json": str(selected_inputs_json),
                        "imain": to_project_path(imain, PATHS),
                        "acqparams": to_project_path(acqparams, PATHS),
                        "selected_inputs_json": to_project_path(selected_inputs_json, PATHS),
                        "dwi02_metadata": selected_payload,
                    },
                    "outputs": {
                        #"out_base": str(out_base),
                        #"fieldcoef": str(fieldcoef),
                        #"movpar": str(movpar),
                        #"field_Hz": str(field_hz),
                        #"corrected_b0s": str(corrected_b0s),
                        #"log": str(log_path),
                        "out_base": to_project_path(out_base, PATHS),
                        "fieldcoef": to_project_path(fieldcoef, PATHS),
                        "movpar": to_project_path(movpar, PATHS),
                        "field_Hz": to_project_path(field_hz, PATHS),
                        "corrected_b0s": to_project_path(corrected_b0s, PATHS),
                        "log": to_project_path(log_path, PATHS),
                    },
                    "qc": {
                        "n_input_volumes": n_input_volumes,
                        "n_corrected_volumes": n_corrected_volumes,
                        "output_checks": output_checks,
                        "field_statistics": field_stats,
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
                        "n_input_volumes": n_input_volumes if n_input_volumes is not None else "",
                        "n_corrected_volumes": n_corrected_volumes if n_corrected_volumes is not None else "",
                        #"fieldcoef": str(fieldcoef.relative_to(ROOT)) if file_exists(fieldcoef) else "",
                        #"movpar": str(movpar.relative_to(ROOT)) if file_exists(movpar) else "",
                        #"field_Hz": str(field_hz.relative_to(ROOT)) if file_exists(field_hz) else "",
                        #"corrected_b0s": str(corrected_b0s.relative_to(ROOT)) if file_exists(corrected_b0s) else "",
                        #"metadata_json": str(metadata_json.relative_to(ROOT)),
                        #"log": str(log_path.relative_to(ROOT)) if file_exists(log_path) else "",
                        "fieldcoef": to_project_path(fieldcoef, PATHS) if file_exists(fieldcoef) else "",
                        "movpar": to_project_path(movpar, PATHS) if file_exists(movpar) else "",
                        "field_Hz": to_project_path(field_hz, PATHS) if file_exists(field_hz) else "",
                        "corrected_b0s": to_project_path(corrected_b0s, PATHS) if file_exists(corrected_b0s) else "",
                        "metadata_json": to_project_path(metadata_json, PATHS),
                        "log": to_project_path(log_path, PATHS) if file_exists(log_path) else "",
                        **field_stats,
                    }
                )
                rows.append(row)

                print("Output")
                print(f"  fieldcoef:     {'OK' if output_checks['fieldcoef'] else 'MISSING'}")
                print(f"  movpar:        {'OK' if output_checks['movpar'] else 'MISSING'}")
                print(f"  field_Hz:      {'OK' if output_checks['field_Hz'] else 'MISSING'}")
                print(f"  corrected b0s: {'OK' if output_checks['corrected_b0s'] else 'MISSING'}")
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
                "fieldcoef": "OK" if file_exists(fieldcoef) else "MISSING",
                "movpar": "OK" if file_exists(movpar) else "MISSING",
                "field_Hz": "OK" if file_exists(field_hz) else "MISSING",
                "corrected_b0s": "OK" if file_exists(corrected_b0s) else "MISSING",
                "metadata_json": "OK" if file_exists(metadata_json) else "MISSING",
                "log": "OK" if file_exists(log_path) else "MISSING",
            }
        )

        qc_field_rows.append(
            {
                "case": case,
                "status": row.get("status", ""),
                **fslstats_field(field_hz),
            }
        )

    csv_path = SUMMARY_DIR / f"{TAG}_topup.csv"
    json_path = SUMMARY_DIR / f"{TAG}_topup.json"

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
        "n_cases": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "OK"),
        "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_csv(QC_DIR / f"{TAG}_case_runtime.csv", qc_runtime_rows)
    write_csv(QC_DIR / f"{TAG}_output_check.csv", qc_output_rows)
    write_csv(QC_DIR / f"{TAG}_field_statistics.csv", qc_field_rows)

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



#!/usr/bin/env python3
"""
Script diffusion/03: Run FSL TOPUP for in-situ DWI cases, v1

Tag: dwi03

Purpose
-------
For each selected in-situ ANISO case:
- read TOPUP inputs prepared by diffusion/02
- run FSL TOPUP
- estimate susceptibility-induced off-resonance field
- write corrected b0 images for QC
- write per-case metadata JSON
- write project-level summary CSV/JSON

Usage
-----
Single case:
    ./apf scripts/diffusion/03_topup.py ANISO_00_000_InS

All active/new in-situ cases:
    ./apf scripts/diffusion/03_topup.py --all
"""

"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apf_paths import get_paths
from apf_registry import get_cases


SCRIPT = "03_topup.py"
TAG = "dwi03"
PREV_TAG = "dwi02"

PATHS = get_paths()
ROOT = PATHS.root
DERIV_DIR = PATHS.derivatives_dir
SUMMARY_DIR = DERIV_DIR / "summary"

TOPUP_CONFIG = "b02b0.cnf"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FSL TOPUP for in-situ DWI cases."
    )
    parser.add_argument(
        "case_id",
        nargs="?",
        help="Case ID, e.g. ANISO_00_000_InS",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all active/new in-situ cases from the APF registry.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dwi03 outputs.",
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


def run_capture(cmd: List[str], log_path: Path) -> None:
    print("  $ " + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with return code {proc.returncode}")


def fsl_dim4(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            ["fslval", str(path), "dim4"],
            check=True,
            text=True,
            capture_output=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    args = parse_args()
    case_ids = resolve_cases(args)

    case_source = "APF registry" if args.all else "Command-line argument"

    print("=" * 72)
    print(f"Script:      {SCRIPT}")
    print(f"Tag:         {TAG}")
    print("Task:        Estimate susceptibility field with FSL TOPUP")
    print(f"ROOT:        {ROOT}")
    print(f"Case source: {case_source}")
    if args.all:
        print("Registry:    config/InS_cases.csv (active/new)")
        print(f"Cases:       {len(case_ids)}")
    else:
        print(f"Case:        {case_ids[0]}")
    print("=" * 72)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for case in case_ids:
        start_time = time.time()

        print("\n" + "=" * 72)
        print(case)
        print("=" * 72)

        topup_dir = DERIV_DIR / case / "DWI" / "topup"
        topup_dir.mkdir(parents=True, exist_ok=True)

        imain = topup_dir / f"{PREV_TAG}_{case}_b0_pair_for_topup.nii.gz"
        acqparams = topup_dir / f"{PREV_TAG}_{case}_acqparams.txt"
        selected_inputs_json = topup_dir / f"{PREV_TAG}_{case}_selected_inputs.json"

        out_base = topup_dir / f"{TAG}_{case}_topup"
        field_hz = topup_dir / f"{TAG}_{case}_field_Hz.nii.gz"
        corrected_b0s = topup_dir / f"{TAG}_{case}_corrected_b0s.nii.gz"
        log_path = topup_dir / f"{TAG}_{case}_topup.log"
        metadata_json = topup_dir / f"{TAG}_{case}_metadata.json"

        fieldcoef = topup_dir / f"{TAG}_{case}_topup_fieldcoef.nii.gz"
        movpar = topup_dir / f"{TAG}_{case}_topup_movpar.txt"

        row: Dict[str, Any] = {
            "script_tag": TAG,
            "case": case,
            "status": "not_started",
            "notes": "",
            "topup_dir": str(topup_dir.relative_to(ROOT)),
            "imain": str(imain.relative_to(ROOT)),
            "acqparams": str(acqparams.relative_to(ROOT)),
        }

        missing = [p for p in [imain, acqparams, selected_inputs_json] if not file_exists(p)]
        if missing:
            row["status"] = "failed"
            row["notes"] = "missing_dwi02_outputs: " + ";".join(p.name for p in missing)
            rows.append(row)
            print("Input")
            for p in [imain, acqparams, selected_inputs_json]:
                print(f"  {p.name}: {'OK' if file_exists(p) else 'MISSING'}")
            continue

        outputs = [fieldcoef, movpar, field_hz, corrected_b0s, metadata_json, log_path]
        if any(file_exists(p) for p in outputs) and not args.force:
            row["status"] = "skipped"
            row["notes"] = "existing_outputs_use_force_to_overwrite"
            row["fieldcoef"] = str(fieldcoef.relative_to(ROOT)) if file_exists(fieldcoef) else ""
            row["movpar"] = str(movpar.relative_to(ROOT)) if file_exists(movpar) else ""
            row["field_Hz"] = str(field_hz.relative_to(ROOT)) if file_exists(field_hz) else ""
            row["corrected_b0s"] = str(corrected_b0s.relative_to(ROOT)) if file_exists(corrected_b0s) else ""
            rows.append(row)
            print("  SKIP: existing dwi03 outputs found. Use --force to overwrite.")
            continue

        print("Input")
        print(f"  b0 pair:              OK  {imain.name}")
        print(f"  acqparams:            OK  {acqparams.name}")
        print(f"  selected inputs JSON: OK  {selected_inputs_json.name}")

        cmd = [
            "topup",
            f"--imain={imain}",
            f"--datain={acqparams}",
            f"--config={TOPUP_CONFIG}",
            f"--out={out_base}",
            f"--fout={field_hz}",
            f"--iout={corrected_b0s}",
        ]

        status = "OK"
        notes = "topup_completed"

        try:
            run_capture(cmd, log_path)
        except Exception as exc:
            status = "failed"
            notes = f"topup_failed: {exc}"
            print(f"  ! TOPUP failed: {exc}")

        runtime_s = round(time.time() - start_time, 3)

        n_input_volumes = fsl_dim4(imain)
        n_corrected_volumes = fsl_dim4(corrected_b0s) if file_exists(corrected_b0s) else None

        output_checks = {
            "fieldcoef": file_exists(fieldcoef),
            "movpar": file_exists(movpar),
            "field_Hz": file_exists(field_hz),
            "corrected_b0s": file_exists(corrected_b0s),
            "log": file_exists(log_path),
        }

        if status == "OK" and not all(output_checks.values()):
            status = "failed"
            missing_outputs = [k for k, v in output_checks.items() if not v]
            notes = "missing_expected_outputs: " + ";".join(missing_outputs)

        selected_payload: Dict[str, Any] = {}
        try:
            selected_payload = json.loads(selected_inputs_json.read_text(encoding="utf-8"))
        except Exception:
            selected_payload = {}

        metadata = {
            "script": SCRIPT,
            "tag": TAG,
            "created": now_iso(),
            "case": case,
            "runtime_seconds": runtime_s,
            "status": status,
            "notes": notes,
            "software": {
                "tool": "FSL TOPUP",
                "config": TOPUP_CONFIG,
            },
            "inputs": {
                "imain": str(imain),
                "acqparams": str(acqparams),
                "selected_inputs_json": str(selected_inputs_json),
                "dwi02_metadata": selected_payload,
            },
            "outputs": {
                "out_base": str(out_base),
                "fieldcoef": str(fieldcoef),
                "movpar": str(movpar),
                "field_Hz": str(field_hz),
                "corrected_b0s": str(corrected_b0s),
                "log": str(log_path),
            },
            "qc": {
                "n_input_volumes": n_input_volumes,
                "n_corrected_volumes": n_corrected_volumes,
                "output_checks": output_checks,
            },
        }
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        row.update(
            {
                "status": status,
                "notes": notes,
                "runtime_seconds": runtime_s,
                "n_input_volumes": n_input_volumes if n_input_volumes is not None else "",
                "n_corrected_volumes": n_corrected_volumes if n_corrected_volumes is not None else "",
                "fieldcoef": str(fieldcoef.relative_to(ROOT)) if file_exists(fieldcoef) else "",
                "movpar": str(movpar.relative_to(ROOT)) if file_exists(movpar) else "",
                "field_Hz": str(field_hz.relative_to(ROOT)) if file_exists(field_hz) else "",
                "corrected_b0s": str(corrected_b0s.relative_to(ROOT)) if file_exists(corrected_b0s) else "",
                "metadata_json": str(metadata_json.relative_to(ROOT)),
                "log": str(log_path.relative_to(ROOT)) if file_exists(log_path) else "",
            }
        )
        rows.append(row)

        print("Output")
        print(f"  fieldcoef:     {'OK' if output_checks['fieldcoef'] else 'MISSING'}")
        print(f"  movpar:        {'OK' if output_checks['movpar'] else 'MISSING'}")
        print(f"  field_Hz:      {'OK' if output_checks['field_Hz'] else 'MISSING'}")
        print(f"  corrected b0s: {'OK' if output_checks['corrected_b0s'] else 'MISSING'}")
        print(f"Status")
        print(f"  {status}: {notes}")
        print(f"  Runtime: {runtime_s:.1f} s")

    csv_path = SUMMARY_DIR / f"{TAG}_topup.csv"
    json_path = SUMMARY_DIR / f"{TAG}_topup.json"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "script": SCRIPT,
        "tag": TAG,
        "created": now_iso(),
        "root": str(ROOT),
        "case_mode": "all" if args.all else "single",
        "case_source": case_source,
        "registry": "config/InS_cases.csv (active/new)" if args.all else "",
        "n_cases": len(rows),
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
    print("=" * 72)


if __name__ == "__main__":
    main()

"""