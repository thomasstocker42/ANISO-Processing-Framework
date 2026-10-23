#!/usr/bin/env python3
"""
APF dwi06: Register DTI scalar maps to GRE space.

dwi06 = spatial registration only.

Registration image:
    dwi04 mean b0 HD-BET image from DWI/eddy/

Transformed outputs:
    dwi05 shell-specific scalar maps from:
        DWI/dtifit/b1000/
        DWI/dtifit/b2000/
        DWI/dtifit/b3000/

Important:
    V1/V2/V3 are only resampled for visual inspection.
    They are NOT reoriented here.
    Quantitative vector use starts only after dwi07.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


SCRIPT = "06_register_dti_to_gre.py"
TAG = "dwi06"
VERSION = "0.2.0"

SCALARS = ["FA", "MD", "L1", "L2", "L3", "MO", "S0"]
VECTOR_FIELDS = ["V1", "V2", "V3"]


try:
    from apf_paths import APFPaths
    from apf_registry import get_cases
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run via ./apf."
    ) from exc


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd: List[str], dry_run: bool = False) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def first_existing(candidates: List[Path], label: str) -> Path:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def get_fsl_version() -> Optional[str]:
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        return None
    f = Path(fsl_dir) / "etc" / "fslversion"
    return f.read_text().strip() if f.exists() else None


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def matrix_metrics(mat_path: Path) -> Dict:
    mat = np.loadtxt(mat_path)
    linear = mat[:3, :3]
    det = float(np.linalg.det(linear))
    translation = [float(x) for x in mat[:3, 3]]

    u, _, vh = np.linalg.svd(linear)
    rot = u @ vh
    cosang = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))

    return {
        "matrix_determinant": det,
        "approx_rotation_degrees": angle_deg,
        "translation_mm": translation,
    }


def dwi06_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "registration" / "dwi06"


def find_registration_b0(paths: APFPaths, case: str) -> Path:
    base = paths.derivatives_dir / case / "DWI" / "eddy"
    candidates = [
        base / f"dwi04_{case}_mean_b0_hdbet.nii.gz",
        base / f"dwi04_{case}_mean_b0_hdbet_bet.nii.gz",
        base / f"dwi04_{case}_mean_b0.nii.gz",
    ]
    return first_existing(candidates, "DWI dwi04 mean b0 registration image")


def find_registration_mask(paths: APFPaths, case: str) -> Path:
    base = paths.derivatives_dir / case / "DWI" / "eddy"
    candidates = [
        base / f"dwi04_{case}_eddy_mask.nii.gz",
        base / f"dwi04_{case}_mean_b0_hdbet_mask.nii.gz",
    ]
    return first_existing(candidates, "DWI dwi04 eddy/brain mask")


def find_dtifit_shell_dirs(paths: APFPaths, case: str) -> List[Path]:
    base = paths.derivatives_dir / case / "DWI" / "dtifit"
    if not base.exists():
        raise FileNotFoundError(f"Missing dtifit directory: {base}")

    shell_dirs = sorted(
        p for p in base.iterdir()
        if p.is_dir() and p.name.startswith("b")
    )

    if not shell_dirs:
        raise FileNotFoundError(f"No dtifit shell directories found in {base}")

    return shell_dirs


def find_shell_image(shell_dir: Path, case: str, basename: str) -> Path:
    candidates = sorted(shell_dir.glob(f"dwi05_{case}_*_{basename}.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {basename} image in shell directory: {shell_dir}"
        )
    return candidates[0]


def find_gre_reference(paths: APFPaths, case: str) -> Path:
    deriv = paths.derivatives_dir / case

    candidates = [
        deriv / "GRE" / "brain_extraction" / "s11b" / f"{case}_GRE_e1_brain.nii.gz",
        deriv / "GRE" / "brain_extraction" / "s11b" / f"{case}_e1_brain.nii.gz",
        deriv / "GRE" / "brain_extraction" / "s11b" / "GRE_e1_brain.nii.gz",
        deriv / "GRE" / "brain_extraction" / "s11b" / "e1_brain.nii.gz",
        deriv / "GRE" / "masked" / "s12" / f"{case}_GRE_e7_masked.nii.gz",
        deriv / "GRE" / "masked" / "s12" / "GRE_e7_masked.nii.gz",
    ]

    candidates.extend(sorted(deriv.glob("**/*GRE*e1*brain*.nii.gz")))
    candidates.extend(sorted(deriv.glob("**/*e1*brain*.nii.gz")))
    candidates.extend(sorted(deriv.glob("**/*e7*masked*.nii.gz")))

    return first_existing(candidates, "GRE reference image")


def make_qc_png(
    gre_ref: Path,
    b0_in_gre: Path,
    fa_in_gre: Optional[Path],
    out_png: Path,
) -> None:
    try:
        import nibabel as nib
        import matplotlib.pyplot as plt
    except ImportError:
        print("  QC PNG skipped: nibabel/matplotlib unavailable.")
        return

    gre = np.asanyarray(nib.load(str(gre_ref)).dataobj).astype(float)
    b0 = np.asanyarray(nib.load(str(b0_in_gre)).dataobj).astype(float)

    z = gre.shape[2] // 2
    y = gre.shape[1] // 2
    x = gre.shape[0] // 2

    def norm(a: np.ndarray) -> np.ndarray:
        finite = np.isfinite(a)
        if not finite.any():
            return np.zeros_like(a)
        lo, hi = np.percentile(a[finite], [1, 99])
        if hi <= lo:
            return np.zeros_like(a)
        return np.clip((a - lo) / (hi - lo), 0, 1)

    planes = [
        ("axial", gre[:, :, z].T, b0[:, :, z].T),
        ("coronal", gre[:, y, :].T, b0[:, y, :].T),
        ("sagittal", gre[x, :, :].T, b0[x, :, :].T),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    for col, (name, g, d) in enumerate(planes):
        axes[0, col].imshow(norm(g), cmap="gray", origin="lower")
        axes[0, col].imshow(norm(d), cmap="hot", alpha=0.35, origin="lower")
        axes[0, col].set_title(f"GRE + b0, {name}")
        axes[0, col].axis("off")

    if fa_in_gre and fa_in_gre.exists():
        fa = np.asanyarray(nib.load(str(fa_in_gre)).dataobj).astype(float)
        fa_planes = [
            ("axial", gre[:, :, z].T, fa[:, :, z].T),
            ("coronal", gre[:, y, :].T, fa[:, y, :].T),
            ("sagittal", gre[x, :, :].T, fa[x, :, :].T),
        ]

        for col, (name, g, f) in enumerate(fa_planes):
            axes[1, col].imshow(norm(g), cmap="gray", origin="lower")
            axes[1, col].imshow(norm(f), cmap="viridis", alpha=0.40, origin="lower")
            axes[1, col].set_title(f"GRE + FA, {name}")
            axes[1, col].axis("off")
    else:
        for col in range(3):
            axes[1, col].axis("off")

    fig.suptitle("dwi06 registration QC")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_case(
    case: str,
    paths: APFPaths,
    dof: int,
    cost: str,
    mask_thr: float,
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    start = now_iso()

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    require_executable("flirt")
    require_executable("fslmaths")

    out_dir = dwi06_dir(paths, case)
    out_dir.mkdir(parents=True, exist_ok=True)

    b0 = find_registration_b0(paths, case)
    nodif_mask = find_registration_mask(paths, case)
    shell_dirs = find_dtifit_shell_dirs(paths, case)
    gre_ref = find_gre_reference(paths, case)

    mat = out_dir / f"dwi06_{case}_b0_to_GRE.mat"
    b0_out = out_dir / f"dwi06_{case}_mean_b0_in_GRE.nii.gz"

    scalar_outputs: Dict[str, Dict[str, str]] = {}
    vector_outputs: Dict[str, Dict[str, str]] = {}
    mask_outputs: Dict[str, str] = {}

    if overwrite or not (mat.exists() and b0_out.exists()):
        run_cmd(
            [
                "flirt",
                "-in", str(b0),
                "-ref", str(gre_ref),
                "-out", str(b0_out),
                "-omat", str(mat),
                "-dof", str(dof),
                "-cost", cost,
                "-interp", "trilinear",
            ],
            dry_run=dry_run,
        )
    else:
        print("  Registration exists; skipping.")

    mask_float = out_dir / f"dwi06_{case}_dwi_mask_in_GRE_float.nii.gz"
    mask_bin = out_dir / f"dwi06_{case}_dwi_mask_in_GRE.nii.gz"
    mask_outputs["dwi_mask"] = str(mask_bin)

    if overwrite or not mask_bin.exists():
        run_cmd(
            [
                "flirt",
                "-in", str(nodif_mask),
                "-ref", str(gre_ref),
                "-applyxfm",
                "-init", str(mat),
                "-out", str(mask_float),
                "-interp", "trilinear",
            ],
            dry_run=dry_run,
        )
        run_cmd(
            [
                "fslmaths",
                str(mask_float),
                "-thr", str(mask_thr),
                "-bin",
                str(mask_bin),
            ],
            dry_run=dry_run,
        )

    for shell_dir in shell_dirs:
        shell = shell_dir.name
        shell_out_dir = out_dir / shell
        shell_out_dir.mkdir(parents=True, exist_ok=True)

        scalar_outputs[shell] = {}
        vector_outputs[shell] = {}

        for scalar in SCALARS:
            try:
                inp = find_shell_image(shell_dir, case, scalar)
            except FileNotFoundError:
                print(f"  WARNING: {shell} scalar missing, skipping: {scalar}")
                continue

            out = shell_out_dir / f"dwi06_{case}_{shell}_{scalar}_in_GRE.nii.gz"
            scalar_outputs[shell][scalar] = str(out)

            if overwrite or not out.exists():
                run_cmd(
                    [
                        "flirt",
                        "-in", str(inp),
                        "-ref", str(gre_ref),
                        "-applyxfm",
                        "-init", str(mat),
                        "-out", str(out),
                        "-interp", "trilinear",
                    ],
                    dry_run=dry_run,
                )

        for vec in VECTOR_FIELDS:
            try:
                inp = find_shell_image(shell_dir, case, vec)
            except FileNotFoundError:
                print(f"  WARNING: {shell} vector missing, skipping: {vec}")
                continue

            out = shell_out_dir / (
                f"dwi06_{case}_{shell}_{vec}"
                "_in_GRE_VISUAL_ONLY_NOT_REORIENTED.nii.gz"
            )
            vector_outputs[shell][vec] = str(out)

            if overwrite or not out.exists():
                run_cmd(
                    [
                        "flirt",
                        "-in", str(inp),
                        "-ref", str(gre_ref),
                        "-applyxfm",
                        "-init", str(mat),
                        "-out", str(out),
                        "-interp", "trilinear",
                    ],
                    dry_run=dry_run,
                )

    metrics = {}
    if mat.exists() and not dry_run:
        metrics = matrix_metrics(mat)

    fa_for_qc = None
    for preferred_shell in ["b2000", "b1000", "b3000"]:
        candidate = (
            out_dir
            / preferred_shell
            / f"dwi06_{case}_{preferred_shell}_FA_in_GRE.nii.gz"
        )
        if candidate.exists():
            fa_for_qc = candidate
            break

    qc_png = out_dir / f"dwi06_{case}_registration_qc.png"
    if not dry_run:
        make_qc_png(gre_ref, b0_out, fa_for_qc, qc_png)

    end = now_iso()

    transform_matrix = None                             #

    if mat.exists() and not dry_run:                    #
        transform_matrix = np.loadtxt(mat).tolist()     #new json

    metadata = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "created": end,
        "execution": {
            "start": start,
            "end": end,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "fsl_version": get_fsl_version(),
            "command": " ".join(sys.argv),
            "dry_run": dry_run,
        },
        #"registration": {
        #    "tool": "FLIRT",
        #    "input_image": str(b0),
        #    "reference_image": str(gre_ref),
        #    "transform_matrix": str(mat),
        #    "dof": dof,
        #    "cost": cost,
        #    "interpolation_registration": "trilinear",
        #    "output_geometry_defined_by_reference": True,
        #},
        "registration": {
            "tool": "FLIRT",
            "input_image": str(b0),
            "reference_image": str(gre_ref),

            "dof": dof,
            "cost": cost,
            "interpolation_registration": "trilinear",
            "output_geometry_defined_by_reference": True,

            "transform_matrix": {
                "matrix_file": str(mat),
                "matrix": transform_matrix,
            },
        },
        "mask_resampling": {
            "input_mask": str(nodif_mask),
            "method": "trilinear_then_threshold_and_binarize",
            "threshold": mask_thr,
            "output_mask": str(mask_bin),
        },
        "dtifit_shells": [p.name for p in shell_dirs],
        "vector_fields": {
            "V1_V2_V3_resampled_for_visual_inspection_if_present": True,
            "reoriented": False,
            "quantitative_use_allowed": False,
            "handled_by": "dwi07",
        },
        "outputs": {
            "registered_b0": str(b0_out),
            "scalar_maps": scalar_outputs,
            "masks": mask_outputs,
            "visual_only_vectors": vector_outputs,
            "qc_png": str(qc_png),
        },
        "qc": {
            "success": True,
            **metrics,
        },
    }

    meta_path = out_dir / f"dwi06_{case}_metadata.json"
    qc_csv = out_dir / f"dwi06_{case}_qc_summary.csv"

    if not dry_run:
        write_json(meta_path, metadata)
        write_csv(
            qc_csv,
            [
                {
                    "case": case,
                    "success": True,
                    "dof": dof,
                    "cost": cost,
                    "shells": ";".join(p.name for p in shell_dirs),
                    "matrix_determinant": metrics.get("matrix_determinant"),
                    "approx_rotation_degrees": metrics.get("approx_rotation_degrees"),
                    "translation_x_mm": metrics.get("translation_mm", [None, None, None])[0],
                    "translation_y_mm": metrics.get("translation_mm", [None, None, None])[1],
                    "translation_z_mm": metrics.get("translation_mm", [None, None, None])[2],
                    "qc_png": str(qc_png),
                    "metadata": str(meta_path),
                }
            ],
        )

    print(f"  DONE: {case}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF dwi06: register DTI scalar maps to GRE space."
    )

    parser.add_argument(
        "case",
        nargs="?",
        help="Case ID, e.g. ANISO_00_000_InS. Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all active InS cases from APF registry.",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[6, 7, 9, 12],
        help="FLIRT degrees of freedom. Default: 6.",
    )
    parser.add_argument(
        "--cost",
        default="corratio",
        choices=["corratio", "mutualinfo", "normmi", "normcorr", "leastsq"],
        help="FLIRT cost function. Default: corratio.",
    )
    parser.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
        help="Threshold after linear mask resampling. Default: 0.5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = APFPaths(root=Path.cwd())

    if args.all:
        cases = get_cases(case_type="InS", active_only=True)
    elif args.case:
        cases = [args.case]
    else:
        raise SystemExit("Provide a case ID or use --all.")

    all_rows = []

    for case in cases:
        try:
            meta = process_case(
                case=case,
                paths=paths,
                dof=args.dof,
                cost=args.cost,
                mask_thr=args.mask_thr,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            all_rows.append(
                {
                    "case": case,
                    "success": True,
                    "dof": args.dof,
                    "cost": args.cost,
                    "metadata": str(
                        dwi06_dir(paths, case) / f"dwi06_{case}_metadata.json"
                    ),
                }
            )
        except Exception as exc:
            print(f"ERROR in {case}: {exc}", file=sys.stderr)
            all_rows.append(
                {
                    "case": case,
                    "success": False,
                    "dof": args.dof,
                    "cost": args.cost,
                    "error": str(exc),
                }
            )
            if not args.all:
                raise

    if args.all and not args.dry_run:
        summary_dir = paths.derivatives_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_csv(summary_dir / "dwi06_registration_summary.csv", all_rows)


if __name__ == "__main__":
    main()