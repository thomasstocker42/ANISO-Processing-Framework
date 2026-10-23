#!/usr/bin/env python3
"""
APF R2PRIME02: rigid dual-TSE -> GRE registration and quantitative
R2app resampling.

Purpose
-------
R2PRIME02 estimates a rigid transform from the authoritative RLX01
dual-TSE echo 1 to GRE02b magnitude echo 1, then applies that transform
to the RLX05 high-resolution R2app map.

Registration contract
---------------------
Moving anatomical image:
    RLX01 selected dual-TSE echo 1

Fixed anatomical image:
    GRE02b magnitude echo 1

Transform:
    rigid, 6 DOF

Cost:
    normalized mutual information (normmi)

Registration interpolation:
    trilinear

Quantitative resampling
-----------------------
RLX05 R2app contains NaN outside its valid fitting support. Direct
trilinear interpolation of such a map can propagate non-finite values
and bias boundary voxels.

R2PRIME02 therefore performs mask-normalized interpolation:

    R2app_GRE = T(M * R2app) / T(M)

where M is the RLX05 fit_valid_mask and T denotes the rigid transform
with trilinear interpolation.

The transformed continuous mask weight is retained. A configurable
threshold defines the binary valid support of the final GRE-space
R2app map. Voxels outside that support are written as NaN.

R2PRIME02 does not calculate R2-prime. That is deferred to R2PRIME03.

Examples
--------
    ./apf scripts/relaxometry/r2prime/02_register_r2_to_gre.py \
        ANISO_00_000_InS

    ./apf scripts/relaxometry/r2prime/02_register_r2_to_gre.py --all

    ./apf scripts/relaxometry/r2prime/02_register_r2_to_gre.py \
        ANISO_00_000_InS --overwrite
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
from typing import Any, Dict, List, Sequence

import nibabel as nib
import numpy as np

SCRIPT = "02_register_r2_to_gre.py"
TAG = "r2prime02"
VERSION = "0.1.0"

DEFAULT_DOF = 6
DEFAULT_COST = "normmi"
DEFAULT_SUPPORT_THR = 0.5
DEFAULT_AFFINE_ATOL = 1e-5


try:
    from apf_paths import APFPaths, get_paths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run through ./apf."
    ) from exc


# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}:\n  {path}"
        )
    return path


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable not found on PATH: {name}"
        )


def get_fsl_version() -> str | None:
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        return None

    path = Path(fsl_dir) / "etc" / "fslversion"
    if not path.exists():
        return None

    return path.read_text().strip()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def run_command(
    command: Sequence[str],
    dry_run: bool,
    command_log: List[List[str]],
) -> None:
    command = [str(x) for x in command]
    command_log.append(command)

    print("  $ " + " ".join(command))

    if dry_run:
        return

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.stdout:
        print(result.stdout.rstrip())

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}:\n"
            + " ".join(command)
        )


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------


def r2prime01_metadata_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2prime"
        / "r2prime01_inventory"
        / f"r2prime01_{case}_metadata.json"
    )


def r2prime02_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2prime"
        / "r2prime02_register"
    )


def r2prime02_summary_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "r2prime"
        / "r2prime02_register"
    )


# -------------------------------------------------------------------------
# NIfTI / geometry
# -------------------------------------------------------------------------


def geometry_equal(
    a: nib.spatialimages.SpatialImage,
    b: nib.spatialimages.SpatialImage,
    atol: float,
) -> bool:
    return (
        a.shape == b.shape
        and np.allclose(
            a.affine,
            b.affine,
            rtol=0,
            atol=atol,
        )
    )


def save_array(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
    dtype: np.dtype,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference.header.copy()
    header.set_data_dtype(dtype)

    img = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        header=header,
    )

    nib.save(
        img,
        str(output_path),
    )


# -------------------------------------------------------------------------
# Transform QC
# -------------------------------------------------------------------------


def matrix_metrics(
    matrix_path: Path,
) -> Dict[str, Any]:
    mat = np.loadtxt(matrix_path)

    if mat.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 FLIRT matrix, got {mat.shape}."
        )

    linear = mat[:3, :3]
    det = float(
        np.linalg.det(linear)
    )

    translation = [
        float(x)
        for x in mat[:3, 3]
    ]

    u, _, vh = np.linalg.svd(linear)
    rotation = u @ vh

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    cos_angle = float(
        np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    angle_deg = float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )

    return {
        "matrix": mat.tolist(),
        "matrix_determinant": det,
        "approx_rotation_degrees": angle_deg,
        "translation_mm": translation,
    }


# -------------------------------------------------------------------------
# Registration QC figure
# -------------------------------------------------------------------------


def normalize_for_display(
    data: np.ndarray,
) -> np.ndarray:
    data = np.asarray(
        data,
        dtype=float,
    )

    finite = np.isfinite(data)

    if not np.any(finite):
        return np.zeros_like(
            data,
            dtype=float,
        )

    lo, hi = np.percentile(
        data[finite],
        [1, 99],
    )

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(
            data,
            dtype=float,
        )

    out = (
        data - lo
    ) / (
        hi - lo
    )

    return np.clip(
        out,
        0,
        1,
    )


def make_qc_png(
    fixed_path: Path,
    registered_tse_path: Path,
    r2app_path: Path,
    valid_mask_path: Path,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  QC PNG skipped: matplotlib unavailable."
        )
        return

    fixed = np.asarray(
        nib.load(
            str(fixed_path)
        ).dataobj,
        dtype=float,
    )

    moving = np.asarray(
        nib.load(
            str(registered_tse_path)
        ).dataobj,
        dtype=float,
    )

    r2app = np.asarray(
        nib.load(
            str(r2app_path)
        ).dataobj,
        dtype=float,
    )

    support = np.asarray(
        nib.load(
            str(valid_mask_path)
        ).dataobj,
        dtype=float,
    ) > 0

    x = fixed.shape[0] // 2
    y = fixed.shape[1] // 2
    z = fixed.shape[2] // 2

    fixed_n = normalize_for_display(
        fixed
    )
    moving_n = normalize_for_display(
        moving
    )

    planes = [
        (
            "Axial",
            fixed_n[:, :, z].T,
            moving_n[:, :, z].T,
            r2app[:, :, z].T,
            support[:, :, z].T,
        ),
        (
            "Coronal",
            fixed_n[:, y, :].T,
            moving_n[:, y, :].T,
            r2app[:, y, :].T,
            support[:, y, :].T,
        ),
        (
            "Sagittal",
            fixed_n[x, :, :].T,
            moving_n[x, :, :].T,
            r2app[x, :, :].T,
            support[x, :, :].T,
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    for col, (
        title,
        fixed_plane,
        moving_plane,
        r2_plane,
        support_plane,
    ) in enumerate(planes):

        axes[0, col].imshow(
            fixed_plane,
            cmap="gray",
            origin="lower",
        )

        axes[0, col].imshow(
            moving_plane,
            cmap="hot",
            alpha=0.35,
            origin="lower",
        )

        axes[0, col].set_title(
            f"GRE e1 + registered TSE e1\n{title}"
        )
        axes[0, col].axis("off")

        r2_show = np.ma.masked_where(
            ~support_plane
            | ~np.isfinite(
                r2_plane
            ),
            r2_plane,
        )

        axes[1, col].imshow(
            fixed_plane,
            cmap="gray",
            origin="lower",
        )

        axes[1, col].imshow(
            r2_show,
            cmap="viridis",
            alpha=0.55,
            origin="lower",
        )

        axes[1, col].set_title(
            f"GRE e1 + R2app support\n{title}"
        )
        axes[1, col].axis("off")

    fig.suptitle(
        "R2PRIME02 rigid TSE → GRE registration QC"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# -------------------------------------------------------------------------
# Input contract
# -------------------------------------------------------------------------


def load_r2prime01_contract(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    meta_path = require_file(
        r2prime01_metadata_path(
            paths,
            case,
        ),
        "R2PRIME01 metadata",
    )

    meta = json.loads(
        meta_path.read_text()
    )

    if meta.get("status") != "READY":
        raise RuntimeError(
            f"R2PRIME01 is not READY for {case}: "
            f"{meta.get('status')}"
        )

    inputs = meta["inputs"]

    dual = inputs["rlx01_dual_tse"]

    echoes = sorted(
        dual["echoes"],
        key=lambda x: x["echo_number"],
    )

    if len(echoes) != 2:
        raise RuntimeError(
            f"Expected exactly 2 dual-TSE echoes for {case}."
        )

    moving_tse = Path(
        echoes[0]["image"]
    )

    r2app = Path(
        inputs["rlx05_r2app"]
    )

    fixed_gre = Path(
        inputs["gre02b_magnitude_e1"]
    )

    r2star = Path(
        inputs["r2star02_r2star"]
    )

    valid_mask = (
        r2app.parent
        / f"rlx05_{case}_fit_valid_mask.nii.gz"
    )

    return {
        "r2prime01_metadata":
            meta_path,

        "moving_tse_e1":
            require_file(
                moving_tse,
                "RLX01 dual-TSE echo 1",
            ),

        "r2app":
            require_file(
                r2app,
                "RLX05 R2app",
            ),

        "valid_mask":
            require_file(
                valid_mask,
                "RLX05 fit_valid_mask",
            ),

        "fixed_gre_e1":
            require_file(
                fixed_gre,
                "GRE02b magnitude echo 1",
            ),

        "r2star":
            require_file(
                r2star,
                "R2STAR02 R2star",
            ),
    }


# -------------------------------------------------------------------------
# Main processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    support_thr: float,
    affine_atol: float,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    require_executable(
        "flirt"
    )

    start = now_iso()
    command_log: List[List[str]] = []

    inputs = load_r2prime01_contract(
        paths,
        case,
    )

    moving_tse = inputs[
        "moving_tse_e1"
    ]

    r2app_path = inputs[
        "r2app"
    ]

    valid_mask_path = inputs[
        "valid_mask"
    ]

    fixed_gre = inputs[
        "fixed_gre_e1"
    ]

    r2star_path = inputs[
        "r2star"
    ]

    # ------------------------------------------------------------------
    # Native geometry checks
    # ------------------------------------------------------------------

    moving_img = nib.load(
        str(moving_tse)
    )

    r2app_img = nib.load(
        str(r2app_path)
    )

    valid_img = nib.load(
        str(valid_mask_path)
    )

    fixed_img = nib.load(
        str(fixed_gre)
    )

    r2star_img = nib.load(
        str(r2star_path)
    )

    if not geometry_equal(
        moving_img,
        r2app_img,
        affine_atol,
    ):
        raise RuntimeError(
            "Native dual-TSE echo 1 and RLX05 R2app "
            "geometry do not match."
        )

    if not geometry_equal(
        r2app_img,
        valid_img,
        affine_atol,
    ):
        raise RuntimeError(
            "RLX05 R2app and fit_valid_mask geometry "
            "do not match."
        )

    if not geometry_equal(
        fixed_img,
        r2star_img,
        affine_atol,
    ):
        raise RuntimeError(
            "GRE02b echo 1 and R2STAR02 R2star "
            "geometry do not match."
        )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    out_dir = r2prime02_dir(
        paths,
        case,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_path = (
        out_dir
        / f"r2prime02_{case}_TSEe1_to_GREe1.mat"
    )

    registered_tse = (
        out_dir
        / f"r2prime02_{case}_TSEe1_in_GRE.nii.gz"
    )

    numerator_native = (
        out_dir
        / f".r2prime02_{case}_R2app_numerator_native.nii.gz"
    )

    weight_native = (
        out_dir
        / f".r2prime02_{case}_valid_weight_native.nii.gz"
    )

    numerator_gre = (
        out_dir
        / f".r2prime02_{case}_R2app_numerator_in_GRE.nii.gz"
    )

    valid_weight_gre = (
        out_dir
        / f"r2prime02_{case}_valid_weight_in_GRE.nii.gz"
    )

    valid_mask_gre = (
        out_dir
        / f"r2prime02_{case}_valid_mask_in_GRE.nii.gz"
    )

    r2app_gre = (
        out_dir
        / f"r2prime02_{case}_R2app_in_GRE_s-1.nii.gz"
    )

    qc_png = (
        out_dir
        / f"r2prime02_{case}_registration_qc.png"
    )

    metadata_path = (
        out_dir
        / f"r2prime02_{case}_metadata.json"
    )

    summary_path = (
        out_dir
        / f"r2prime02_{case}_summary.csv"
    )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    registration_needed = (
        overwrite
        or not matrix_path.exists()
        or not registered_tse.exists()
    )

    registration_cmd = [
        "flirt",
        "-in",
        str(moving_tse),
        "-ref",
        str(fixed_gre),
        "-out",
        str(registered_tse),
        "-omat",
        str(matrix_path),
        "-dof",
        str(DEFAULT_DOF),
        "-cost",
        DEFAULT_COST,
        "-interp",
        "trilinear",
    ]

    if registration_needed:
        run_command(
            registration_cmd,
            dry_run,
            command_log,
        )
    else:
        print(
            "  Registration outputs exist; skipping."
        )

    # ------------------------------------------------------------------
    # Dry-run stops before array-dependent steps.
    # ------------------------------------------------------------------

    if dry_run:
        print(
            "  DRY RUN: quantitative temporary images "
            "and applyxfm steps not executed."
        )

        return {
            "case": case,
            "status": "DRY_RUN",
            "support_threshold": support_thr,
            "matrix": str(matrix_path),
            "registered_tse": str(registered_tse),
            "r2app_in_GRE": str(r2app_gre),
        }

    require_file(
        matrix_path,
        "FLIRT transform matrix",
    )

    require_file(
        registered_tse,
        "registered TSE echo 1",
    )

    # ------------------------------------------------------------------
    # Prepare masked quantitative numerator + interpolation weights
    # ------------------------------------------------------------------

    r2app = np.asarray(
        r2app_img.dataobj,
        dtype=np.float32,
    )

    valid_mask = (
        np.asarray(
            valid_img.dataobj
        )
        > 0
    )

    native_support = (
        valid_mask
        & np.isfinite(
            r2app
        )
    )

    if not np.any(
        native_support
    ):
        raise RuntimeError(
            "RLX05 contains no finite R2app values "
            "inside fit_valid_mask."
        )

    numerator = np.zeros(
        r2app.shape,
        dtype=np.float32,
    )

    numerator[
        native_support
    ] = r2app[
        native_support
    ]

    weight = native_support.astype(
        np.float32
    )

    save_array(
        numerator,
        r2app_img,
        numerator_native,
        np.float32,
    )

    save_array(
        weight,
        r2app_img,
        weight_native,
        np.float32,
    )

    # ------------------------------------------------------------------
    # Apply rigid transform to numerator and weight
    # ------------------------------------------------------------------

    if overwrite or not numerator_gre.exists():
        run_command(
            [
                "flirt",
                "-in",
                str(numerator_native),
                "-ref",
                str(fixed_gre),
                "-applyxfm",
                "-init",
                str(matrix_path),
                "-out",
                str(numerator_gre),
                "-interp",
                "trilinear",
            ],
            False,
            command_log,
        )

    if overwrite or not valid_weight_gre.exists():
        run_command(
            [
                "flirt",
                "-in",
                str(weight_native),
                "-ref",
                str(fixed_gre),
                "-applyxfm",
                "-init",
                str(matrix_path),
                "-out",
                str(valid_weight_gre),
                "-interp",
                "trilinear",
            ],
            False,
            command_log,
        )

    # ------------------------------------------------------------------
    # Mask-normalized interpolation
    # ------------------------------------------------------------------

    numerator_gre_img = nib.load(
        str(numerator_gre)
    )

    weight_gre_img = nib.load(
        str(valid_weight_gre)
    )

    if not geometry_equal(
        numerator_gre_img,
        fixed_img,
        affine_atol,
    ):
        raise RuntimeError(
            "Resampled numerator does not match GRE geometry."
        )

    if not geometry_equal(
        weight_gre_img,
        fixed_img,
        affine_atol,
    ):
        raise RuntimeError(
            "Resampled valid weight does not match GRE geometry."
        )

    numerator_resampled = np.asarray(
        numerator_gre_img.dataobj,
        dtype=np.float32,
    )

    weight_resampled = np.asarray(
        weight_gre_img.dataobj,
        dtype=np.float32,
    )

    # Guard against tiny numerical overshoots.
    weight_resampled = np.clip(
        weight_resampled,
        0.0,
        1.0,
    )

    support = (
        np.isfinite(
            numerator_resampled
        )
        & np.isfinite(
            weight_resampled
        )
        & (
            weight_resampled
            >= support_thr
        )
    )

    final_r2app = np.full(
        fixed_img.shape,
        np.nan,
        dtype=np.float32,
    )

    final_r2app[
        support
    ] = (
        numerator_resampled[
            support
        ]
        / weight_resampled[
            support
        ]
    )

    finite_final = (
        support
        & np.isfinite(
            final_r2app
        )
    )

    support &= finite_final

    final_r2app[
        ~support
    ] = np.nan

    save_array(
        weight_resampled,
        fixed_img,
        valid_weight_gre,
        np.float32,
    )

    save_array(
        support.astype(
            np.uint8
        ),
        fixed_img,
        valid_mask_gre,
        np.uint8,
    )

    save_array(
        final_r2app,
        fixed_img,
        r2app_gre,
        np.float32,
    )

    # ------------------------------------------------------------------
    # Final geometry validation
    # ------------------------------------------------------------------

    final_img = nib.load(
        str(r2app_gre)
    )

    final_mask_img = nib.load(
        str(valid_mask_gre)
    )

    final_geometry_ok = (
        geometry_equal(
            final_img,
            fixed_img,
            affine_atol,
        )
        and geometry_equal(
            final_mask_img,
            fixed_img,
            affine_atol,
        )
        and geometry_equal(
            final_img,
            r2star_img,
            affine_atol,
        )
    )

    if not final_geometry_ok:
        raise RuntimeError(
            "Final R2app outputs do not match GRE/R2star geometry."
        )

    # ------------------------------------------------------------------
    # QC metrics
    # ------------------------------------------------------------------

    transform = matrix_metrics(
        matrix_path
    )

    n_native_support = int(
        np.count_nonzero(
            native_support
        )
    )

    n_gre_support = int(
        np.count_nonzero(
            support
        )
    )

    n_gre_voxels = int(
        np.prod(
            fixed_img.shape
        )
    )

    valid_values = final_r2app[
        support
    ]

    r2_min = float(
        np.min(
            valid_values
        )
    )

    r2_max = float(
        np.max(
            valid_values
        )
    )

    r2_median = float(
        np.median(
            valid_values
        )
    )

    r2_mean = float(
        np.mean(
            valid_values
        )
    )

    support_fraction = float(
        n_gre_support
        / n_gre_voxels
    )

    make_qc_png(
        fixed_gre,
        registered_tse,
        r2app_gre,
        valid_mask_gre,
        qc_png,
    )

    end = now_iso()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "created": end,
        "command": " ".join(
            sys.argv
        ),
        "case": case,

        "status": "DONE",

        "purpose": (
            "Rigidly register the high-resolution dual-TSE acquisition "
            "to native GRE space and resample RLX05 R2app using "
            "mask-normalized trilinear interpolation."
        ),

        "execution": {
            "start": start,
            "end": end,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "fsl_version": get_fsl_version(),
            "dry_run": False,
        },

        "upstream": {
            "r2prime01_metadata":
                str(
                    inputs[
                        "r2prime01_metadata"
                    ]
                ),
        },

        "inputs": {
            "moving_registration_image":
                str(
                    moving_tse
                ),

            "fixed_registration_image":
                str(
                    fixed_gre
                ),

            "r2app":
                str(
                    r2app_path
                ),

            "r2app_valid_mask":
                str(
                    valid_mask_path
                ),

            "r2star_geometry_reference":
                str(
                    r2star_path
                ),
        },

        "registration": {
            "tool": "FSL FLIRT",
            "direction": "dual-TSE -> GRE",
            "dof": DEFAULT_DOF,
            "cost": DEFAULT_COST,
            "registration_interpolation": "trilinear",

            "transform_matrix": {
                "matrix_file":
                    str(
                        matrix_path
                    ),

                "matrix":
                    transform[
                        "matrix"
                    ],
            },

            "matrix_determinant":
                transform[
                    "matrix_determinant"
                ],

            "approx_rotation_degrees":
                transform[
                    "approx_rotation_degrees"
                ],

            "translation_mm":
                transform[
                    "translation_mm"
                ],
        },

        "quantitative_resampling": {
            "method":
                "mask-normalized trilinear interpolation",

            "equation":
                "R2app_GRE = T(M * R2app) / T(M)",

            "native_support":
                "RLX05 fit_valid_mask AND finite R2app",

            "continuous_weight":
                str(
                    valid_weight_gre
                ),

            "support_threshold":
                support_thr,

            "binary_support":
                str(
                    valid_mask_gre
                ),

            "outside_support":
                "NaN",

            "output":
                str(
                    r2app_gre
                ),

            "units":
                "s^-1",
        },

        "geometry": {
            "output_matches_GRE":
                final_geometry_ok,

            "output_matches_R2STAR02":
                final_geometry_ok,

            "output_shape":
                list(
                    final_img.shape
                ),

            "voxel_size_mm":
                [
                    float(x)
                    for x in final_img.header.get_zooms()[:3]
                ],
        },

        "qc": {
            "native_valid_voxels":
                n_native_support,

            "gre_valid_voxels":
                n_gre_support,

            "gre_valid_fraction":
                support_fraction,

            "R2app_s-1_min":
                r2_min,

            "R2app_s-1_median":
                r2_median,

            "R2app_s-1_mean":
                r2_mean,

            "R2app_s-1_max":
                r2_max,

            "registration_qc_png":
                str(
                    qc_png
                ),
        },

        "commands": command_log,
    }

    write_json(
        metadata_path,
        metadata,
    )

    summary_row = {
        "case":
            case,

        "status":
            "DONE",

        "dof":
            DEFAULT_DOF,

        "cost":
            DEFAULT_COST,

        "support_threshold":
            support_thr,

        "native_valid_voxels":
            n_native_support,

        "GRE_valid_voxels":
            n_gre_support,

        "GRE_valid_fraction":
            support_fraction,

        "matrix_determinant":
            transform[
                "matrix_determinant"
            ],

        "approx_rotation_degrees":
            transform[
                "approx_rotation_degrees"
            ],

        "translation_x_mm":
            transform[
                "translation_mm"
            ][0],

        "translation_y_mm":
            transform[
                "translation_mm"
            ][1],

        "translation_z_mm":
            transform[
                "translation_mm"
            ][2],

        "R2app_min_s-1":
            r2_min,

        "R2app_median_s-1":
            r2_median,

        "R2app_mean_s-1":
            r2_mean,

        "R2app_max_s-1":
            r2_max,

        "GRE_geometry":
            "OK",

        "matrix":
            str(
                matrix_path
            ),

        "R2app_in_GRE":
            str(
                r2app_gre
            ),

        "valid_mask_in_GRE":
            str(
                valid_mask_gre
            ),

        "qc_png":
            str(
                qc_png
            ),

        "metadata":
            str(
                metadata_path
            ),
    }

    write_csv(
        summary_path,
        [
            summary_row
        ],
    )

    # Temporary numerator products are implementation details.
    for temporary in [
        numerator_native,
        weight_native,
        numerator_gre,
    ]:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    print(
        f"  native valid voxels : {n_native_support:,}"
    )

    print(
        f"  GRE valid voxels    : {n_gre_support:,}"
    )

    print(
        f"  GRE valid fraction  : {support_fraction:.6f}"
    )

    print(
        f"  transform rotation  : "
        f"{transform['approx_rotation_degrees']:.3f} deg"
    )

    print(
        "  translation [mm]    : "
        + ", ".join(
            f"{x:.3f}"
            for x in transform[
                "translation_mm"
            ]
        )
    )

    print(
        f"  R2app median        : {r2_median:.3f} s^-1"
    )

    print(
        f"  R2app range         : "
        f"{r2_min:.3f} .. {r2_max:.3f} s^-1"
    )

    print(
        "  output geometry     : GRE / R2STAR02 OK"
    )

    print(
        f"  QC                   : {qc_png}"
    )

    print(
        f"  DONE: {case}"
    )

    return summary_row


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF R2PRIME02: rigid dual-TSE to GRE registration and "
            "mask-normalized R2app resampling."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--support-thr",
        type=float,
        default=DEFAULT_SUPPORT_THR,
        help=(
            "Minimum transformed RLX05 valid-mask weight required "
            "for final GRE-space R2app support. Default: 0.5."
        ),
    )

    parser.add_argument(
        "--affine-atol",
        type=float,
        default=DEFAULT_AFFINE_ATOL,
        help=(
            "Absolute affine tolerance for geometry comparisons. "
            "Default: 1e-5."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registration/resampling outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print registration command without executing it.",
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.support_thr
        <= 1.0
    ):
        parser.error(
            "--support-thr must satisfy 0 < threshold <= 1."
        )

    return args


def main() -> None:
    args = parse_args()

    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )

    print_selected_cases(
        cases
    )

    rows: List[
        Dict[str, Any]
    ] = []

    failures: List[
        Dict[str, str]
    ] = []

    for case in cases:
        try:
            row = process_case(
                case=case,
                paths=paths,
                support_thr=args.support_thr,
                affine_atol=args.affine_atol,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            rows.append(
                row
            )

        except Exception as exc:
            print(
                f"ERROR: {case}: {exc}",
                file=sys.stderr,
            )

            failures.append(
                {
                    "case":
                        case,

                    "error":
                        str(
                            exc
                        ),
                }
            )

    if not args.dry_run:
        summary_dir = r2prime02_summary_dir(
            paths
        )

        summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cohort_csv = (
            summary_dir
            / "r2prime02_cohort_summary.csv"
        )

        cohort_json = (
            summary_dir
            / "r2prime02_cohort_metadata.json"
        )

        write_csv(
            cohort_csv,
            rows,
        )

        write_json(
            cohort_json,
            {
                "script": SCRIPT,
                "tag": TAG,
                "version": VERSION,
                "created": now_iso(),
                "command": " ".join(
                    sys.argv
                ),
                "support_threshold":
                    args.support_thr,
                "n_cases_requested":
                    len(
                        cases
                    ),
                "n_completed":
                    len(
                        rows
                    ),
                "n_failures":
                    len(
                        failures
                    ),
                "rows":
                    rows,
                "failures":
                    failures,
            },
        )

        print()
        print("=" * 72)
        print("R2PRIME02 cohort summary")
        print("=" * 72)
        print(
            f"Cases requested : {len(cases)}"
        )
        print(
            f"Completed       : {len(rows)}"
        )
        print(
            f"Failures        : {len(failures)}"
        )
        print(
            f"Cohort CSV      : {cohort_csv}"
        )
        print(
            f"Cohort metadata : {cohort_json}"
        )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
