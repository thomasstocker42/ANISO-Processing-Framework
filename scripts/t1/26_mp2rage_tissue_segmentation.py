#!/usr/bin/env python3
"""
Script 26: MP2RAGE UNI tissue segmentation (APF production)
Tag: s26

Scientific processing is locked to the legacy Bash implementation.

Profiles
--------
legacy (default): HD-BET CPU, TTA disabled; legacy output directory.
workstation:       HD-BET CUDA, TTA enabled; separate output directory.
z.b.
./apf scripts/t1/26_mp2rage_tissue_segmentation.py \
  ANISO_00_000_InS \
  --profile workstation

FAST is identical for both profiles:
    fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0

FAST class mapping:
    pve_0 -> CSF, pve_1 -> GM, pve_2 -> WM

Binary masks:
    fslmaths <probability> -thr 0.8 -bin <mask>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np


SCRIPT_ID = "s26"
SCRIPT_NAME = Path(__file__).name
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MASK_THRESHOLD = 0.8

PROFILE_SETTINGS = {
    "legacy": {
        "device": "cpu",
        "tta": False,
        "output_dir": "MP2RAGE_s26",
    },
    "workstation": {
        "device": "cuda",
        "tta": True,
        "output_dir": "MP2RAGE_s26_workstation",
    },
}

FAST_SETTINGS = {
    "image_type": 1,
    "classes": 3,
    "mrf_beta": 0.1,
    "bias_iterations": 4,
    "bias_smoothness": 20.0,
}

TISSUE_MAPPING = {"CSF": 0, "GM": 1, "WM": 2}


class ProcessingError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shell_join(command: Sequence[str]) -> str:
    import shlex

    return shlex.join([str(x) for x in command])


def run_command(command: Sequence[str], log_path: Path) -> None:
    print(f"$ {shell_join(command)}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shell_join(command)}\n")
        result = subprocess.run(
            [str(x) for x in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise ProcessingError(
            f"Command failed ({result.returncode}): {shell_join(command)}. "
            f"See {log_path}"
        )


def require_tools() -> None:
    missing = [name for name in ("hd-bet", "fast", "fslmaths") if not shutil.which(name)]
    if missing:
        raise ProcessingError(f"Missing executable(s) on PATH: {', '.join(missing)}")


def tool_version(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            [str(x) for x in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip()
    return text.splitlines()[0] if text else None


def get_registry_cases(root: Path) -> list[str]:
    """Use APF registry; retain a simple CSV fallback for portability."""
    try:
        from apf_registry import get_cases  # type: ignore

        records = get_cases()
        cases: list[str] = []
        for record in records:
            if isinstance(record, str):
                case = record
            elif isinstance(record, dict):
                case = record.get("case") or record.get("case_id") or ""
            else:
                case = getattr(record, "case", "") or getattr(record, "case_id", "")
            if case:
                cases.append(str(case))
        if cases:
            return sorted(set(cases))
    except (ImportError, TypeError):
        pass

    csv_path = root / "config" / "InS_cases.csv"
    if not csv_path.is_file():
        raise ProcessingError(f"APF registry unavailable and CSV not found: {csv_path}")

    cases = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lowered = {str(k).lower(): str(v).strip() for k, v in row.items()}
            case = lowered.get("case") or lowered.get("case_id") or ""
            active = lowered.get("active", "true").lower()
            if case and active not in {"0", "false", "no", "inactive", "excluded"}:
                cases.append(case)
    return sorted(set(cases))


def find_uni_files(root: Path, case: str) -> list[Path]:
    return sorted((root / "nifti" / case / "T1").glob(f"{case}_MP2RAGE_UNI_*.nii.gz"))


def select_cases(args: argparse.Namespace, root: Path) -> list[str]:
    cases = get_registry_cases(root) if args.all else sorted(set(args.cases))
    if not cases:
        raise ProcessingError("Provide case ID(s), or use --all.")
    if args.all_ready:
        cases = [case for case in cases if find_uni_files(root, case)]
    return cases


def make_paths(root: Path, case: str, profile: str) -> dict[str, Path]:
    out_dir = (
        root
        / "derivatives"
        / case
        / "tissue_segmentation"
        / PROFILE_SETTINGS[profile]["output_dir"]
    )
    prefix = f"{SCRIPT_ID}_{case}"
    bet_base = out_dir / f"{prefix}_MP2RAGE_UNI_hdbet"
    fast_base = out_dir / f"{prefix}_MP2RAGE_UNI_FAST"

    return {
        "out_dir": out_dir,
        "uni": out_dir / f"{prefix}_MP2RAGE_UNI_input.nii.gz",
        "bet": Path(f"{bet_base}.nii.gz"),
        "bet_mask": Path(f"{bet_base}_bet.nii.gz"),
        "fast_base": fast_base,
        "pve0": Path(f"{fast_base}_pve_0.nii.gz"),
        "pve1": Path(f"{fast_base}_pve_1.nii.gz"),
        "pve2": Path(f"{fast_base}_pve_2.nii.gz"),
        "csf_prob": out_dir / f"{prefix}_CSF_prob.nii.gz",
        "gm_prob": out_dir / f"{prefix}_GM_prob.nii.gz",
        "wm_prob": out_dir / f"{prefix}_WM_prob.nii.gz",
        "csf_mask": out_dir / f"{prefix}_CSF_mask_thr0p8.nii.gz",
        "gm_mask": out_dir / f"{prefix}_GM_mask_thr0p8.nii.gz",
        "wm_mask": out_dir / f"{prefix}_WM_mask_thr0p8.nii.gz",
        "qc": out_dir / f"{prefix}_qc_summary.csv",
        "metadata": out_dir / f"{prefix}_metadata.json",
        "log": out_dir / f"{prefix}_processing.log",
    }


def clear_outputs(paths: dict[str, Path]) -> None:
    out_dir = paths["out_dir"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def resolve_bet_output(paths: dict[str, Path]) -> Path:
    """Standardize HD-BET output name if the installed version appends _bet."""
    if paths["bet"].is_file():
        return paths["bet"]

    alternate = paths["bet"].with_name(
        paths["bet"].name.removesuffix(".nii.gz") + "_bet.nii.gz"
    )
    if alternate.is_file():
        shutil.copy2(alternate, paths["bet"])
        return paths["bet"]

    raise ProcessingError("HD-BET brain image was not created.")


def assert_same_geometry(paths: Sequence[Path]) -> None:
    reference = nib.load(str(paths[0]))
    for path in paths[1:]:
        image = nib.load(str(path))
        if image.shape != reference.shape:
            raise ProcessingError(f"Shape mismatch: {path.name} != {paths[0].name}")
        if not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise ProcessingError(f"Affine mismatch: {path.name} != {paths[0].name}")


def calculate_qc(paths: dict[str, Path]) -> dict[str, Any]:
    brain = np.asarray(nib.load(str(paths["bet_mask"])).dataobj) > 0
    brain_voxels = int(brain.sum())
    if brain_voxels == 0:
        raise ProcessingError("HD-BET mask is empty.")

    qc: dict[str, Any] = {"brain_voxels": brain_voxels}
    for tissue in ("csf", "gm", "wm"):
        prob = np.asarray(nib.load(str(paths[f"{tissue}_prob"])).dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(str(paths[f"{tissue}_mask"])).dataobj) > 0
        if not np.isfinite(prob).all():
            raise ProcessingError(f"{tissue.upper()} probability map contains non-finite values.")
        voxels = int(mask.sum())
        qc[f"{tissue}_prob_min"] = float(prob.min())
        qc[f"{tissue}_prob_max"] = float(prob.max())
        qc[f"{tissue}_prob_mean_in_brain"] = float(prob[brain].mean())
        qc[f"{tissue}_mask_voxels"] = voxels
        qc[f"{tissue}_mask_fraction_of_brain"] = voxels / brain_voxels
    return qc


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def process_case(
    root: Path,
    case: str,
    profile: str,
    overwrite: bool,
    versions: dict[str, str | None],
) -> dict[str, Any]:
    started = time.time()
    started_iso = now_iso()
    cfg = PROFILE_SETTINGS[profile]
    paths = make_paths(root, case, profile)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    if overwrite:
        clear_outputs(paths)

    uni_files = find_uni_files(root, case)
    if not uni_files:
        raise ProcessingError(f"No MP2RAGE UNI file found for {case}.")
    uni = uni_files[0]
    message = "OK" if len(uni_files) == 1 else "Multiple UNI files; used first sorted file"

    print(f"UNI: {uni}")
    if len(uni_files) > 1:
        print(f"WARNING: {message}")
    shutil.copy2(uni, paths["uni"])

    hd_bet_cmd = [
        "hd-bet", "-i", str(paths["uni"]), "-o", str(paths["bet"]),
        "-device", cfg["device"],
    ]
    if not cfg["tta"]:
        hd_bet_cmd.append("--disable_tta")
    hd_bet_cmd.append("--save_bet_mask")

    hd_bet_ran = False
    if not paths["bet"].is_file() or not paths["bet_mask"].is_file():
        print(f"Running HD-BET: profile={profile}, device={cfg['device']}, TTA={cfg['tta']}")
        run_command(hd_bet_cmd, paths["log"])
        hd_bet_ran = True
    else:
        print("HD-BET outputs exist; skipping.")

    bet_image = resolve_bet_output(paths)
    if not paths["bet_mask"].is_file():
        raise ProcessingError(f"HD-BET mask missing: {paths['bet_mask']}")

    fast_cmd = [
        "fast",
        "-t", "1",
        "-n", "3",
        "-H", "0.1",
        "-I", "4",
        "-l", "20.0",
        "-o", str(paths["fast_base"]),
        str(bet_image),
    ]

    pves = [paths["pve0"], paths["pve1"], paths["pve2"]]
    fast_ran = False
    if not all(path.is_file() for path in pves):
        print("Running FAST with locked legacy parameters.")
        run_command(fast_cmd, paths["log"])
        fast_ran = True
    else:
        print("FAST outputs exist; skipping.")

    if not all(path.is_file() for path in pves):
        raise ProcessingError("One or more FAST PVE outputs are missing.")

    for tissue, index in TISSUE_MAPPING.items():
        shutil.copy2(paths[f"pve{index}"], paths[f"{tissue.lower()}_prob"])

    threshold_commands = []
    for tissue in ("csf", "gm", "wm"):
        command = [
            "fslmaths", str(paths[f"{tissue}_prob"]),
            "-thr", str(MASK_THRESHOLD), "-bin", str(paths[f"{tissue}_mask"]),
        ]
        run_command(command, paths["log"])
        threshold_commands.append(command)

    geometry_files = [
        paths["uni"], paths["bet"], paths["bet_mask"],
        paths["csf_prob"], paths["gm_prob"], paths["wm_prob"],
        paths["csf_mask"], paths["gm_mask"], paths["wm_mask"],
    ]
    assert_same_geometry(geometry_files)
    qc = calculate_qc(paths)
    write_single_row_csv(paths["qc"], {"case": case, "profile": profile, **qc})

    finished = time.time()
    metadata = {
        "script": SCRIPT_NAME,
        "script_id": SCRIPT_ID,
        "case": case,
        "status": "OK",
        "profile": profile,
        "profile_settings": cfg,
        "input": {
            "selected_uni": str(uni),
            "all_matching_uni_files": [str(path) for path in uni_files],
            "selection_message": message,
        },
        "method": {
            "hdbet": {"device": cfg["device"], "tta": cfg["tta"], "save_mask": True},
            "fast": FAST_SETTINGS,
            "tissue_mapping": TISSUE_MAPPING,
            "mask_threshold": MASK_THRESHOLD,
        },
        "outputs": {key: str(value) for key, value in paths.items() if key != "fast_base"},
        "qc": qc,
        "timing": {
            "start": started_iso,
            "end": now_iso(),
            "runtime_seconds": finished - started,
        },
        "execution": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "threads": int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)),
            "tool_versions": versions,
            "executed_command": shell_join(sys.argv),
            "commands": {
                "hd_bet": shell_join(hd_bet_cmd),
                "fast": shell_join(fast_cmd),
                "fslmaths": [shell_join(cmd) for cmd in threshold_commands],
            },
            "stages_executed": {"hd_bet": hd_bet_ran, "fast": fast_ran},
        },
    }
    with paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return {
        "case": case,
        "status": "OK",
        "profile": profile,
        "uni_input": str(uni),
        "out_dir": str(paths["out_dir"]),
        "message": message,
        "runtime_seconds": finished - started,
    }


def write_summary(root: Path, profile: str, rows: list[dict[str, Any]]) -> Path:
    path = root / "derivatives" / "summary" / f"{SCRIPT_ID}_MP2RAGE_FAST_{profile}_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "status", "profile", "uni_input", "out_dir", "message", "runtime_seconds"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF MP2RAGE UNI HD-BET + FAST tissue segmentation."
    )
    parser.add_argument("cases", nargs="*", help="Case IDs to process.")
    parser.add_argument("--all", action="store_true", help="Process all active registry cases.")
    parser.add_argument(
        "--all-ready",
        action="store_true",
        help="With --all, process only cases with an MP2RAGE UNI input.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SETTINGS),
        default="legacy",
        help="Processing profile (default: legacy).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    if args.all and args.cases:
        parser.error("Use explicit case IDs or --all, not both.")
    if args.all_ready and not args.all:
        parser.error("--all-ready requires --all.")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    print("=" * 72)
    print("s26: MP2RAGE UNI HD-BET + FAST tissue segmentation")
    print(f"Root: {root}")
    print(f"Profile: {args.profile}")
    print("=" * 72)

    try:
        require_tools()
        cases = select_cases(args, root)
    except ProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    versions = {
        "hd_bet": tool_version(["hd-bet", "--help"]),
        "fsl": tool_version(["fslversion"]),
    }

    rows = []
    failures = 0
    for case in cases:
        print(f"\n{'=' * 72}\ns26: {case}\n{'=' * 72}")
        try:
            rows.append(process_case(root, case, args.profile, args.overwrite, versions))
            print(f"Done: {case}")
        except Exception as exc:
            failures += 1
            print(f"ERROR in {case}: {exc}", file=sys.stderr)
            traceback.print_exc()
            rows.append({
                "case": case,
                "status": "FAILED",
                "profile": args.profile,
                "uni_input": "",
                "out_dir": str(make_paths(root, case, args.profile)["out_dir"]),
                "message": str(exc),
                "runtime_seconds": "",
            })

    summary = write_summary(root, args.profile, rows)
    print(f"\n{'=' * 72}")
    print(f"s26 complete: {len(rows) - failures}/{len(rows)} successful")
    print(f"Summary: {summary}")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
