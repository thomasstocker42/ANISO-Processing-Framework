#!/usr/bin/env python3
"""
APF R2PRIME02b: validation of R2PRIME02 TSE-to-GRE registration.

Purpose
-------
R2PRIME02b validates the spatial-support behavior of the production
R2PRIME02 registration/resampling step.

It separates three potential causes of reduced GRE-space R2app coverage:

1. anatomical coverage mismatch between the native dual-TSE acquisition
   and the GRE-defined brain;
2. voxels rejected by the RLX05 dual-echo fit-valid criteria;
3. support changes introduced during TSE-to-GRE resampling.

The module therefore reports:

    anatomical coverage:
        transformed RLX05a brain mask ∩ GRE02b brain
        ------------------------------------------------
                       GRE02b brain

    native fit-valid coverage:
        RLX05 fit_valid_mask ∩ RLX05a brain
        ------------------------------------
                 RLX05a brain

    final R2app coverage:
        R2PRIME02 valid mask ∩ GRE02b brain
        ------------------------------------
                  GRE02b brain

The RLX05a brain mask is transformed with the exact R2PRIME02 FLIRT
matrix using trilinear interpolation. A threshold of 0.5 is used by
default to recover binary anatomical support.

R2PRIME02b performs validation only. It does not alter the production
R2PRIME02 R2app output.

Examples
--------
    ./apf scripts/relaxometry/r2prime/02b_validate_registration.py \
        ANISO_00_000_InS

    ./apf scripts/relaxometry/r2prime/02b_validate_registration.py --all
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
from typing import Any, Dict, List

import nibabel as nib
import numpy as np


SCRIPT = "02b_validate_registration.py"
TAG = "r2prime02b"
VERSION = "0.1.0"

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
# Utilities
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


def write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


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
    header = reference.header.copy()
    header.set_data_dtype(dtype)

    img = nib.Nifti1Image(
        np.asarray(
            data,
            dtype=dtype,
        ),
        reference.affine,
        header=header,
    )

    nib.save(
        img,
        str(output_path),
    )


def run_command(
    command: List[str],
) -> None:
    command = [
        str(x)
        for x in command
    ]

    print(
        "  $ "
        + " ".join(
            command
        )
    )

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.stdout:
        print(
            result.stdout.rstrip()
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}:\n"
            + " ".join(
                command
            )
        )


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------


def rlx05a_brain_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05a"
        / f"rlx05a_{case}_brain_mask.nii.gz"
    )


def rlx05_valid_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05"
        / f"rlx05_{case}_fit_valid_mask.nii.gz"
    )


def gre02b_brain_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "GRE"
        / "02b_brain_mask"
        / f"{case}_GRE02b_brain_mask.nii.gz"
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


def r2prime02_matrix_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(
            paths,
            case,
        )
        / f"r2prime02_{case}_TSEe1_to_GREe1.mat"
    )


def r2prime02_valid_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(
            paths,
            case,
        )
        / f"r2prime02_{case}_valid_mask_in_GRE.nii.gz"
    )


def r2prime02_metadata_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(
            paths,
            case,
        )
        / f"r2prime02_{case}_metadata.json"
    )


def r2prime02b_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2prime"
        / "r2prime02b_validation"
    )


def r2prime02b_summary_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "r2prime"
        / "r2prime02b_validation"
    )


# -------------------------------------------------------------------------
# FLIRT matrix metrics
# -------------------------------------------------------------------------


def matrix_metrics(
    matrix_path: Path,
) -> Dict[str, Any]:

    mat = np.loadtxt(
        matrix_path
    )

    if mat.shape != (
        4,
        4,
    ):
        raise ValueError(
            f"Expected 4x4 FLIRT matrix, got {mat.shape}."
        )

    linear = mat[
        :3,
        :3,
    ]

    determinant = float(
        np.linalg.det(
            linear
        )
    )

    translation = [
        float(x)
        for x in mat[
            :3,
            3,
        ]
    ]

    u, _, vh = np.linalg.svd(
        linear
    )

    rotation = (
        u
        @ vh
    )

    if np.linalg.det(
        rotation
    ) < 0:
        u[:, -1] *= -1
        rotation = (
            u
            @ vh
        )

    cos_angle = float(
        np.clip(
            (
                np.trace(
                    rotation
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    rotation_deg = float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )

    return {
        "matrix":
            mat.tolist(),

        "matrix_determinant":
            determinant,

        "approx_rotation_degrees":
            rotation_deg,

        "translation_mm":
            translation,
    }


# -------------------------------------------------------------------------
# Coverage QC figure
# -------------------------------------------------------------------------


def normalize_for_display(
    data: np.ndarray,
) -> np.ndarray:
    data = np.asarray(
        data,
        dtype=float,
    )

    finite = np.isfinite(
        data
    )

    if not np.any(
        finite
    ):
        return np.zeros_like(
            data,
            dtype=float,
        )

    lo, hi = np.percentile(
        data[finite],
        [1, 99],
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):
        return np.zeros_like(
            data,
            dtype=float,
        )

    return np.clip(
        (data - lo)
        / (hi - lo),
        0,
        1,
    )


def make_coverage_qc_png(
    native_tse_path: Path,
    native_brain_mask_path: Path,
    native_fit_mask_path: Path,
    gre_path: Path,
    transformed_tse_brain_mask_path: Path,
    final_r2app_mask_path: Path,
    anatomical_coverage_pct: float,
    native_fit_coverage_pct: float,
    final_r2app_coverage_pct: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  QC PNG skipped: matplotlib unavailable."
        )
        return

    native_tse = np.asarray(
        nib.load(
            str(native_tse_path)
        ).dataobj,
        dtype=float,
    )

    native_brain = (
        np.asarray(
            nib.load(
                str(native_brain_mask_path)
            ).dataobj
        )
        > 0
    )

    native_fit = (
        np.asarray(
            nib.load(
                str(native_fit_mask_path)
            ).dataobj
        )
        > 0
    )

    gre = np.asarray(
        nib.load(
            str(gre_path)
        ).dataobj,
        dtype=float,
    )

    transformed_tse_brain = (
        np.asarray(
            nib.load(
                str(transformed_tse_brain_mask_path)
            ).dataobj
        )
        > 0
    )

    final_r2app = (
        np.asarray(
            nib.load(
                str(final_r2app_mask_path)
            ).dataobj
        )
        > 0
    )

    native_y = (
        native_tse.shape[1]
        // 2
    )

    gre_y = (
        gre.shape[1]
        // 2
    )

    native_plane = normalize_for_display(
        native_tse[
            :,
            native_y,
            :
        ].T
    )

    native_brain_plane = (
        native_brain[
            :,
            native_y,
            :
        ].T
    )

    native_fit_plane = (
        native_fit[
            :,
            native_y,
            :
        ].T
    )

    gre_plane = normalize_for_display(
        gre[
            :,
            gre_y,
            :
        ].T
    )

    transformed_brain_plane = (
        transformed_tse_brain[
            :,
            gre_y,
            :
        ].T
    )

    final_r2app_plane = (
        final_r2app[
            :,
            gre_y,
            :
        ].T
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    # Native row
    axes[0, 0].imshow(
        native_plane,
        cmap="gray",
        origin="lower",
    )
    axes[0, 0].set_title(
        "Native TSE e1"
    )

    axes[0, 1].imshow(
        native_plane,
        cmap="gray",
        origin="lower",
    )
    axes[0, 1].imshow(
        np.ma.masked_where(
            ~native_brain_plane,
            native_brain_plane,
        ),
        alpha=0.45,
        origin="lower",
    )
    axes[0, 1].set_title(
        "Native TSE + RLX05a brain mask"
    )

    axes[0, 2].imshow(
        native_plane,
        cmap="gray",
        origin="lower",
    )
    axes[0, 2].imshow(
        np.ma.masked_where(
            ~native_fit_plane,
            native_fit_plane,
        ),
        alpha=0.45,
        origin="lower",
    )
    axes[0, 2].set_title(
        "Native TSE + RLX05 fit-valid mask"
    )

    # GRE row
    axes[1, 0].imshow(
        gre_plane,
        cmap="gray",
        origin="lower",
    )
    axes[1, 0].set_title(
        "GRE e1"
    )

    axes[1, 1].imshow(
        gre_plane,
        cmap="gray",
        origin="lower",
    )
    axes[1, 1].imshow(
        np.ma.masked_where(
            ~transformed_brain_plane,
            transformed_brain_plane,
        ),
        alpha=0.45,
        origin="lower",
    )
    axes[1, 1].set_title(
        "GRE + transformed TSE brain mask"
    )

    axes[1, 2].imshow(
        gre_plane,
        cmap="gray",
        origin="lower",
    )
    axes[1, 2].imshow(
        np.ma.masked_where(
            ~final_r2app_plane,
            final_r2app_plane,
        ),
        alpha=0.45,
        origin="lower",
    )
    axes[1, 2].set_title(
        "GRE + final R2app-valid mask"
    )

    for ax in axes.ravel():
        ax.axis(
            "off"
        )

    fig.suptitle(
        "R2PRIME02b coverage validation\n"
        f"TSE anatomical coverage = "
        f"{anatomical_coverage_pct:.2f}%   |   "
        f"Native fit-valid coverage = "
        f"{native_fit_coverage_pct:.2f}%   |   "
        f"Final R2app GRE coverage = "
        f"{final_r2app_coverage_pct:.2f}%"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# -------------------------------------------------------------------------
# Per-case validation
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    support_thr: float,
    affine_atol: float,
    overwrite: bool,
) -> Dict[str, Any]:

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    require_executable(
        "flirt"
    )

    start = now_iso()

    native_brain_path = require_file(
        rlx05a_brain_mask_path(
            paths,
            case,
        ),
        "RLX05a brain mask",
    )

    native_fit_path = require_file(
        rlx05_valid_mask_path(
            paths,
            case,
        ),
        "RLX05 fit-valid mask",
    )

    gre_brain_path = require_file(
        gre02b_brain_mask_path(
            paths,
            case,
        ),
        "GRE02b brain mask",
    )

    matrix_path = require_file(
        r2prime02_matrix_path(
            paths,
            case,
        ),
        "R2PRIME02 transform matrix",
    )

    final_r2_mask_path = require_file(
        r2prime02_valid_mask_path(
            paths,
            case,
        ),
        "R2PRIME02 valid R2app mask",
    )

    upstream_metadata = require_file(
        r2prime02_metadata_path(
            paths,
            case,
        ),
        "R2PRIME02 metadata",
    )

    upstream = json.loads(
        upstream_metadata.read_text()
    )

    native_tse_path = require_file(
        Path(
            upstream["inputs"][
                "moving_registration_image"
            ]
        ),
        "R2PRIME02 moving TSE image",
    )

    gre_reference_path = require_file(
        Path(
            upstream["inputs"][
                "fixed_registration_image"
            ]
        ),
        "R2PRIME02 GRE reference image",
    )

    out_dir = r2prime02b_dir(
        paths,
        case,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    brain_weight_gre_path = (
        out_dir
        / f"r2prime02b_{case}_TSE_brain_weight_in_GRE.nii.gz"
    )

    brain_mask_gre_path = (
        out_dir
        / f"r2prime02b_{case}_TSE_brain_mask_in_GRE.nii.gz"
    )

    metadata_path = (
        out_dir
        / f"r2prime02b_{case}_metadata.json"
    )

    summary_path = (
        out_dir
        / f"r2prime02b_{case}_summary.csv"
    )

    qc_png = (
        out_dir
        / f"r2prime02b_{case}_coverage_qc.png"
    )

    # ------------------------------------------------------------------
    # Load native/reference masks
    # ------------------------------------------------------------------

    native_brain_img = nib.load(
        str(
            native_brain_path
        )
    )

    native_fit_img = nib.load(
        str(
            native_fit_path
        )
    )

    gre_brain_img = nib.load(
        str(
            gre_brain_path
        )
    )

    final_r2_img = nib.load(
        str(
            final_r2_mask_path
        )
    )

    if not geometry_equal(
        native_brain_img,
        native_fit_img,
        affine_atol,
    ):
        raise RuntimeError(
            "RLX05a brain mask and RLX05 fit-valid mask "
            "do not share native geometry."
        )

    if not geometry_equal(
        gre_brain_img,
        final_r2_img,
        affine_atol,
    ):
        raise RuntimeError(
            "GRE02b brain mask and R2PRIME02 valid mask "
            "do not share GRE geometry."
        )

    # ------------------------------------------------------------------
    # Transform RLX05a anatomical brain support
    # ------------------------------------------------------------------

    if (
        overwrite
        or not brain_weight_gre_path.exists()
    ):
        run_command(
            [
                "flirt",
                "-in",
                str(
                    native_brain_path
                ),
                "-ref",
                str(
                    gre_brain_path
                ),
                "-applyxfm",
                "-init",
                str(
                    matrix_path
                ),
                "-out",
                str(
                    brain_weight_gre_path
                ),
                "-interp",
                "trilinear",
            ]
        )

    brain_weight_img = nib.load(
        str(
            brain_weight_gre_path
        )
    )

    if not geometry_equal(
        brain_weight_img,
        gre_brain_img,
        affine_atol,
    ):
        raise RuntimeError(
            "Transformed TSE brain weight does not match GRE geometry."
        )

    brain_weight = np.asarray(
        brain_weight_img.dataobj,
        dtype=np.float32,
    )

    brain_weight = np.clip(
        brain_weight,
        0.0,
        1.0,
    )

    transformed_tse_brain = (
        np.isfinite(
            brain_weight
        )
        & (
            brain_weight
            >= support_thr
        )
    )

    save_array(
        transformed_tse_brain.astype(
            np.uint8
        ),
        gre_brain_img,
        brain_mask_gre_path,
        np.uint8,
    )

    # ------------------------------------------------------------------
    # Coverage decomposition
    # ------------------------------------------------------------------

    native_brain = (
        np.asarray(
            native_brain_img.dataobj
        )
        > 0
    )

    native_fit = (
        np.asarray(
            native_fit_img.dataobj
        )
        > 0
    )

    gre_brain = (
        np.asarray(
            gre_brain_img.dataobj
        )
        > 0
    )

    final_r2 = (
        np.asarray(
            final_r2_img.dataobj
        )
        > 0
    )

    n_native_brain = int(
        np.count_nonzero(
            native_brain
        )
    )

    n_native_fit = int(
        np.count_nonzero(
            native_fit
            & native_brain
        )
    )

    n_gre_brain = int(
        np.count_nonzero(
            gre_brain
        )
    )

    n_transformed_tse_brain = int(
        np.count_nonzero(
            transformed_tse_brain
        )
    )

    n_anatomical_intersection = int(
        np.count_nonzero(
            transformed_tse_brain
            & gre_brain
        )
    )

    n_final_r2 = int(
        np.count_nonzero(
            final_r2
        )
    )

    n_final_r2_intersection = int(
        np.count_nonzero(
            final_r2
            & gre_brain
        )
    )

    n_final_r2_outside_gre = int(
        np.count_nonzero(
            final_r2
            & ~gre_brain
        )
    )

    anatomical_coverage_pct = (
        100.0
        * n_anatomical_intersection
        / n_gre_brain
    )

    native_fit_coverage_pct = (
        100.0
        * n_native_fit
        / n_native_brain
    )

    final_r2app_coverage_pct = (
        100.0
        * n_final_r2_intersection
        / n_gre_brain
    )

    support_change_pct = (
        100.0
        * (
            n_final_r2
            - n_native_fit
        )
        / n_native_fit
    )

    final_outside_fraction_pct = (
        100.0
        * n_final_r2_outside_gre
        / n_final_r2
    )

    transform = matrix_metrics(
        matrix_path
    )

    make_coverage_qc_png(
        native_tse_path=native_tse_path,
        native_brain_mask_path=native_brain_path,
        native_fit_mask_path=native_fit_path,
        gre_path=gre_reference_path,
        transformed_tse_brain_mask_path=brain_mask_gre_path,
        final_r2app_mask_path=final_r2_mask_path,
        anatomical_coverage_pct=anatomical_coverage_pct,
        native_fit_coverage_pct=native_fit_coverage_pct,
        final_r2app_coverage_pct=final_r2app_coverage_pct,
        output_path=qc_png,
    )

    end = now_iso()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    row = {
        "case":
            case,

        "status":
            "PASS",

        "support_threshold":
            support_thr,

        "TSE_brain_coverage_in_GRE_pct":
            anatomical_coverage_pct,

        "native_fit_valid_coverage_pct":
            native_fit_coverage_pct,

        "final_R2app_coverage_in_GRE_pct":
            final_r2app_coverage_pct,

        "R2app_support_change_pct":
            support_change_pct,

        "R2app_outside_GRE_brain_voxels":
            n_final_r2_outside_gre,

        "R2app_outside_GRE_brain_pct":
            final_outside_fraction_pct,

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

        "native_TSE_brain_voxels":
            n_native_brain,

        "native_fit_valid_voxels":
            n_native_fit,

        "transformed_TSE_brain_voxels":
            n_transformed_tse_brain,

        "GRE_brain_voxels":
            n_gre_brain,

        "TSEbrain_GRE_intersection_voxels":
            n_anatomical_intersection,

        "final_R2app_voxels":
            n_final_r2,

        "R2app_GRE_intersection_voxels":
            n_final_r2_intersection,

        "TSE_brain_mask_in_GRE":
            str(
                brain_mask_gre_path
            ),

        "coverage_qc_png":
            str(
                qc_png
            ),

        "metadata":
            str(
                metadata_path
            ),
    }

    metadata = {
        "script":
            SCRIPT,

        "tag":
            TAG,

        "version":
            VERSION,

        "created":
            end,

        "command":
            " ".join(
                sys.argv
            ),

        "case":
            case,

        "status":
            "PASS",

        "purpose": (
            "Validate R2PRIME02 registration and decompose reduced "
            "GRE-space R2app coverage into anatomical-coverage, "
            "native-fit-validity, and resampling components."
        ),

        "execution": {
            "start":
                start,

            "end":
                end,

            "host":
                platform.node(),

            "platform":
                platform.platform(),

            "python":
                sys.version.split()[0],

            "fsl_version":
                get_fsl_version(),
        },

        "upstream": {
            "r2prime02_metadata":
                str(
                    upstream_metadata
                ),

            "transform_matrix":
                str(
                    matrix_path
                ),
        },

        "inputs": {
            "native_RLX05a_brain_mask":
                str(
                    native_brain_path
                ),

            "native_RLX05_fit_valid_mask":
                str(
                    native_fit_path
                ),

            "GRE02b_brain_mask":
                str(
                    gre_brain_path
                ),

            "R2PRIME02_valid_mask_in_GRE":
                str(
                    final_r2_mask_path
                ),
        },

        "anatomical_support_resampling": {
            "tool":
                "FSL FLIRT -applyxfm",

            "matrix":
                str(
                    matrix_path
                ),

            "interpolation":
                "trilinear",

            "support_threshold":
                support_thr,

            "continuous_weight_output":
                str(
                    brain_weight_gre_path
                ),

            "binary_mask_output":
                str(
                    brain_mask_gre_path
                ),
        },

        "coverage_metrics": {
            "definitions": {
                "TSE_brain_coverage_in_GRE_pct": (
                    "100 * |transformed RLX05a brain ∩ GRE02b brain| "
                    "/ |GRE02b brain|"
                ),

                "native_fit_valid_coverage_pct": (
                    "100 * |RLX05 fit-valid ∩ RLX05a brain| "
                    "/ |RLX05a brain|"
                ),

                "final_R2app_coverage_in_GRE_pct": (
                    "100 * |R2PRIME02 valid R2app ∩ GRE02b brain| "
                    "/ |GRE02b brain|"
                ),

                "R2app_support_change_pct": (
                    "100 * (|final R2app support| - |native fit-valid|) "
                    "/ |native fit-valid|"
                ),
            },

            "values":
                row,
        },

        "transform_qc":
            transform,

        "qc_outputs": {
            "coverage_qc_png":
                str(
                    qc_png
                ),
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    write_csv(
        summary_path,
        [
            row
        ],
    )

    print(
        f"  TSE brain / GRE brain : "
        f"{anatomical_coverage_pct:.3f}%"
    )

    print(
        f"  native fit coverage   : "
        f"{native_fit_coverage_pct:.3f}%"
    )

    print(
        f"  final R2app / GRE     : "
        f"{final_r2app_coverage_pct:.3f}%"
    )

    print(
        f"  support change        : "
        f"{support_change_pct:+.3f}%"
    )

    print(
        f"  R2app outside GRE     : "
        f"{final_outside_fraction_pct:.3f}%"
    )

    print(
        f"  determinant           : "
        f"{transform['matrix_determinant']:.8f}"
    )

    print(
        f"  rotation              : "
        f"{transform['approx_rotation_degrees']:.3f} deg"
    )

    print(
        f"  coverage QC           : "
        f"{qc_png}"
    )

    print(
        f"  DONE: {case}"
    )

    return row


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF R2PRIME02b: validate R2PRIME02 registration "
            "and quantitative-support coverage."
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
            "Threshold for transformed RLX05a brain-mask weights. "
            "Default: 0.5."
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
        help="Recompute transformed anatomical support.",
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
            rows.append(
                process_case(
                    case=case,
                    paths=paths,
                    support_thr=args.support_thr,
                    affine_atol=args.affine_atol,
                    overwrite=args.overwrite,
                )
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

    summary_dir = r2prime02b_summary_dir(
        paths
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_csv = (
        summary_dir
        / "r2prime02b_cohort_summary.csv"
    )

    cohort_json = (
        summary_dir
        / "r2prime02b_cohort_metadata.json"
    )

    write_csv(
        cohort_csv,
        rows,
    )

    write_json(
        cohort_json,
        {
            "script":
                SCRIPT,

            "tag":
                TAG,

            "version":
                VERSION,

            "created":
                now_iso(),

            "command":
                " ".join(
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
    print("R2PRIME02b cohort summary")
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
