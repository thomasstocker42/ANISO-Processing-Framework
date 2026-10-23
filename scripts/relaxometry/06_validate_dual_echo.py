#!/usr/bin/env python3
"""
APF rlx06: Validate high-resolution dual-echo R2app against the
validated 12-echo drop1 R2 reference.

Scientific purpose
------------------
The 12-echo mono-exponential drop1 estimate is treated as the quantitative
reference:

    R2_ref = rlx03 drop1 R2

The high-resolution dual-echo estimate from rlx05 is treated as the candidate
surrogate:

    R2_dual = rlx05 R2app

The dual-echo acquisition has substantially higher spatial resolution
(typically 1 x 1 x 1 mm) than the 12-echo acquisition
(typically 1 x 1 x 4 mm). Therefore quantitative comparison is performed in
the native 12-echo space.

Workflow
--------
1. Resolve authoritative rlx02 multi-echo stack and echo times.
2. Extract the first 12-echo image as the registration reference.
3. Resolve the native dual-TSE echo 1 from rlx01.
4. Rigidly register dual echo 1 -> 12-echo echo 1 using FSL FLIRT (6 DOF).
5. Apply the transform to:
       - dual R2app
       - dual T2app
       - dual valid mask
6. Restrict analysis to voxels valid in:
       - 12-echo drop1
       - transformed dual-echo estimate
7. Compare the two estimates using:
       - voxelwise difference maps
       - Pearson correlation
       - Spearman correlation
       - linear regression
       - absolute and relative bias
       - Bland-Altman statistics

Important interpretation
------------------------
rlx06 tests whether dual-echo R2app is a useful high-resolution surrogate.
It does NOT assume that the dual-echo and multi-echo estimates have perfect
absolute agreement.

Primary comparison:
    R2app_dual_in_12echo_space vs R2_drop1

Outputs
-------
Registration:
    rlx06_<case>_dualEcho1_to_12echoEcho1.mat
    rlx06_<case>_dualEcho1_in_12echo.nii.gz

Resampled dual-echo maps:
    rlx06_<case>_R2app_in_12echo.nii.gz
    rlx06_<case>_T2app_in_12echo.nii.gz
    rlx06_<case>_dual_valid_in_12echo.nii.gz

Comparison:
    rlx06_<case>_common_valid_mask.nii.gz
    rlx06_<case>_deltaR2_s-1.nii.gz
    rlx06_<case>_deltaT2_ms.nii.gz
    rlx06_<case>_comparison.csv
    rlx06_<case>_qc.png
    rlx06_<case>_metadata.json

Conventions
-----------
    deltaR2 = R2_dual - R2_12echo_drop1
    deltaT2 = T2_dual - T2_12echo_drop1

Run
---
    ./apf scripts/relaxometry/06_validate_dual_echo.py \
        ANISO_00_000_InS
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT = "06_validate_dual_echo.py"
TAG = "rlx06"
VERSION = "0.1.0"

DEFAULT_AFFINE_ATOL = 1e-5
DEFAULT_SCATTER_MAX_POINTS = 100_000
DEFAULT_RANDOM_SEED = 42


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("rlx06 requires nibabel.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("rlx06 requires matplotlib.") from exc

try:
    from scipy.stats import pearsonr, spearmanr, linregress
except ImportError as exc:
    raise ImportError("rlx06 requires scipy.") from exc

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
                fieldnames.append(
                    key
                )
                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def robust_stats(
    values: np.ndarray,
) -> Dict[str, Any]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }

    return {
        "n": int(values.size),
        "mean": float(
            np.mean(values)
        ),
        "sd": (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if values.size > 1
            else 0.0
        ),
        "median": float(
            np.median(values)
        ),
        "p05": float(
            np.percentile(
                values,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
    }


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference.header.copy()
    header.set_data_dtype(
        dtype
    )

    image = nib.Nifti1Image(
        data.astype(
            dtype,
            copy=False,
        ),
        reference.affine,
        header,
    )

    nib.save(
        image,
        str(path),
    )


def require_file(
    path: Path,
    label: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} not found:\n"
            f"  {path}"
        )

    return path


def resolve_project_path(
    raw: str,
    root: Path,
) -> Path:
    path = Path(
        raw
    ).expanduser()

    if not path.is_absolute():
        path = root / path

    return require_file(
        path.resolve(),
        "input",
    )


# -------------------------------------------------------------------------
# APF paths
# -------------------------------------------------------------------------


def rlx01_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx01_inventory"
    )


def rlx02_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx02"
    )


def rlx03_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx03"
    )


def rlx05_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05"
    )


def rlx06_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx06"
    )


def cohort_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "rlx06"
    )


# -------------------------------------------------------------------------
# Input resolution
# -------------------------------------------------------------------------


def find_rlx02_metadata(
    paths: APFPaths,
    case: str,
) -> Path:
    candidate = (
        paths.case_stage_dir(
            case=case,
            module="relaxometry",
            stage="rlx02_prepare_multi_echo_t2",
        )
        / f"rlx02_{case}_preparation.json"
    )

    return require_file(
        candidate,
        "rlx02 metadata",
    )


def load_rlx02_reference(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    metadata_path = find_rlx02_metadata(
        paths,
        case,
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    outputs = metadata.get(
        "outputs",
    )

    input_meta = metadata.get(
        "input",
    )

    if not isinstance(
        outputs,
        dict,
    ):
        raise ValueError(
            "rlx02 metadata lacks outputs object."
        )

    if not isinstance(
        input_meta,
        dict,
    ):
        raise ValueError(
            "rlx02 metadata lacks input object."
        )

    raw_stack = outputs.get(
        "series_4d"
    )

    if not isinstance(
        raw_stack,
        str,
    ):
        raise ValueError(
            "rlx02 metadata lacks outputs.series_4d."
        )

    stack_path = resolve_project_path(
        raw_stack,
        paths.root,
    )

    echo_times = np.asarray(
        input_meta.get(
            "echo_times_s"
        ),
        dtype=np.float64,
    )

    if (
        echo_times.ndim != 1
        or echo_times.size < 2
    ):
        raise ValueError(
            "Invalid rlx02 echo-time vector."
        )

    if np.max(
        echo_times
    ) > 1.0:
        echo_times = (
            echo_times
            / 1000.0
        )

    return {
        "metadata_path":
            metadata_path,

        "stack_path":
            stack_path,

        "echo_times_s":
            echo_times,
    }


def load_dual_echo1(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    metadata_path = (
        rlx01_dir(
            paths,
            case,
        )
        / f"rlx01_{case}_selected_inputs.json"
    )

    require_file(
        metadata_path,
        "rlx01 selected-input metadata",
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    dual = metadata.get(
        "dual_tse"
    )

    if not isinstance(
        dual,
        dict,
    ):
        raise ValueError(
            "rlx01 metadata lacks dual_tse."
        )

    if not dual.get(
        "ready",
        False,
    ):
        raise RuntimeError(
            f"dual_tse is not ready for {case}."
        )

    echoes = dual.get(
        "echoes"
    )

    if (
        not isinstance(
            echoes,
            list,
        )
        or len(echoes) != 2
    ):
        raise ValueError(
            "Expected two dual-TSE echoes."
        )

    echoes = sorted(
        echoes,
        key=lambda x: int(
            x["echo_number"]
        ),
    )

    echo1 = echoes[0]

    image_path = resolve_project_path(
        echo1["image"],
        paths.root,
    )

    return {
        "metadata_path":
            metadata_path,

        "image_path":
            image_path,

        "echo_time_s":
            float(
                echo1["echo_time_s"]
            ),
    }


def load_rlx03_reference_paths(
    paths: APFPaths,
    case: str,
) -> Dict[str, Path]:
    stage = rlx03_dir(
        paths,
        case,
    )

    model = stage / "drop1"

    stem = (
        f"rlx03_{case}_drop1"
    )

    return {
        "R2":
            require_file(
                model
                / f"{stem}_R2_s-1.nii.gz",
                "rlx03 drop1 R2",
            ),

        "T2":
            require_file(
                model
                / f"{stem}_T2_ms.nii.gz",
                "rlx03 drop1 T2",
            ),

        "valid":
            require_file(
                model
                / f"{stem}_fit_valid_mask.nii.gz",
                "rlx03 drop1 valid mask",
            ),
    }


def load_rlx05_paths(
    paths: APFPaths,
    case: str,
) -> Dict[str, Path]:
    stage = rlx05_dir(
        paths,
        case,
    )

    stem = f"rlx05_{case}"

    return {
        "R2":
            require_file(
                stage
                / f"{stem}_R2app_s-1.nii.gz",
                "rlx05 R2app",
            ),

        "T2":
            require_file(
                stage
                / f"{stem}_T2app_ms.nii.gz",
                "rlx05 T2app",
            ),

        "valid":
            require_file(
                stage
                / f"{stem}_fit_valid_mask.nii.gz",
                "rlx05 valid mask",
            ),
    }


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------


def check_fsl() -> None:
    if shutil.which(
        "flirt"
    ) is None:
        raise RuntimeError(
            "FSL FLIRT not found in PATH."
        )


def run_command(
    command: Sequence[str],
) -> None:
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
            "Command failed with exit code "
            f"{result.returncode}:\n"
            + " ".join(
                command
            )
        )


def extract_reference_echo1(
    stack_path: Path,
    output_path: Path,
) -> nib.spatialimages.SpatialImage:
    stack_img = nib.load(
        str(
            stack_path
        )
    )

    if (
        len(
            stack_img.shape
        )
        != 4
        or stack_img.shape[3] < 1
    ):
        raise ValueError(
            f"Expected 4D rlx02 stack, got "
            f"{stack_img.shape}."
        )

    echo1 = np.asarray(
        stack_img.dataobj[
            ...,
            0,
        ],
        dtype=np.float32,
    )

    save_nifti(
        echo1,
        stack_img,
        output_path,
        np.float32,
    )

    return nib.load(
        str(
            output_path
        )
    )


def register_dual_to_reference(
    dual_echo1: Path,
    reference_echo1: Path,
    registered_echo1: Path,
    matrix_path: Path,
) -> None:
    run_command(
        [
            "flirt",
            "-in",
            str(
                dual_echo1
            ),
            "-ref",
            str(
                reference_echo1
            ),
            "-out",
            str(
                registered_echo1
            ),
            "-omat",
            str(
                matrix_path
            ),
            "-dof",
            "6",
            "-cost",
            "normmi",
            "-interp",
            "trilinear",
        ]
    )


def apply_transform(
    input_path: Path,
    reference_path: Path,
    matrix_path: Path,
    output_path: Path,
    interpolation: str,
) -> None:
    run_command(
        [
            "flirt",
            "-in",
            str(
                input_path
            ),
            "-ref",
            str(
                reference_path
            ),
            "-applyxfm",
            "-init",
            str(
                matrix_path
            ),
            "-out",
            str(
                output_path
            ),
            "-interp",
            interpolation,
        ]
    )


# -------------------------------------------------------------------------
# Quantitative comparison
# -------------------------------------------------------------------------


def safe_pearson(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    if (
        x.size < 3
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return (
            float("nan"),
            float("nan"),
        )

    result = pearsonr(
        x,
        y,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    if x.size < 3:
        return (
            float("nan"),
            float("nan"),
        )

    result = spearmanr(
        x,
        y,
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def compare_maps(
    r2_ref: np.ndarray,
    t2_ref: np.ndarray,
    valid_ref: np.ndarray,
    r2_dual: np.ndarray,
    t2_dual: np.ndarray,
    valid_dual: np.ndarray,
) -> Tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    common = (
        valid_ref
        & valid_dual
        & np.isfinite(
            r2_ref
        )
        & np.isfinite(
            r2_dual
        )
        & np.isfinite(
            t2_ref
        )
        & np.isfinite(
            t2_dual
        )
    )

    if not np.any(
        common
    ):
        raise RuntimeError(
            "No common valid voxels."
        )

    ref_r2 = r2_ref[
        common
    ].astype(
        np.float64
    )

    dual_r2 = r2_dual[
        common
    ].astype(
        np.float64
    )

    ref_t2 = t2_ref[
        common
    ].astype(
        np.float64
    )

    dual_t2 = t2_dual[
        common
    ].astype(
        np.float64
    )

    delta_r2_values = (
        dual_r2
        - ref_r2
    )

    delta_t2_values = (
        dual_t2
        - ref_t2
    )

    mean_r2_values = (
        dual_r2
        + ref_r2
    ) / 2.0

    pearson_r, pearson_p = safe_pearson(
        ref_r2,
        dual_r2,
    )

    spearman_rho, spearman_p = safe_spearman(
        ref_r2,
        dual_r2,
    )

    regression = linregress(
        ref_r2,
        dual_r2,
    )

    mean_bias = float(
        np.mean(
            delta_r2_values
        )
    )

    sd_bias = float(
        np.std(
            delta_r2_values,
            ddof=1,
        )
    )

    loa_lower = (
        mean_bias
        - 1.96
        * sd_bias
    )

    loa_upper = (
        mean_bias
        + 1.96
        * sd_bias
    )

    proportional_bias = linregress(
        mean_r2_values,
        delta_r2_values,
    )

    relative_r2 = (
        delta_r2_values
        / ref_r2
    )

    summary: Dict[str, Any] = {
        "common_valid_voxels":
            int(
                np.count_nonzero(
                    common
                )
            ),

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        "spearman_rho":
            spearman_rho,

        "spearman_p":
            spearman_p,

        "regression_slope":
            float(
                regression.slope
            ),

        "regression_intercept":
            float(
                regression.intercept
            ),

        "regression_r2":
            float(
                regression.rvalue ** 2
            ),

        "regression_p":
            float(
                regression.pvalue
            ),

        "bland_altman_bias_R2_s-1":
            mean_bias,

        "bland_altman_sd_R2_s-1":
            sd_bias,

        "bland_altman_lower95_R2_s-1":
            float(
                loa_lower
            ),

        "bland_altman_upper95_R2_s-1":
            float(
                loa_upper
            ),

        "proportional_bias_slope":
            float(
                proportional_bias.slope
            ),

        "proportional_bias_p":
            float(
                proportional_bias.pvalue
            ),
    }

    for prefix, values in [
        (
            "R2_reference_s-1",
            ref_r2,
        ),
        (
            "R2_dual_s-1",
            dual_r2,
        ),
        (
            "deltaR2_s-1",
            delta_r2_values,
        ),
        (
            "relative_deltaR2",
            relative_r2,
        ),
        (
            "T2_reference_ms",
            ref_t2,
        ),
        (
            "T2_dual_ms",
            dual_t2,
        ),
        (
            "deltaT2_ms",
            delta_t2_values,
        ),
    ]:
        metric_stats = robust_stats(
            values
        )

        for key, value in metric_stats.items():
            summary[
                f"{prefix}_{key}"
            ] = value

    delta_r2 = np.full(
        r2_ref.shape,
        np.nan,
        dtype=np.float32,
    )

    delta_t2 = np.full(
        t2_ref.shape,
        np.nan,
        dtype=np.float32,
    )

    delta_r2[
        common
    ] = delta_r2_values.astype(
        np.float32
    )

    delta_t2[
        common
    ] = delta_t2_values.astype(
        np.float32
    )

    return (
        summary,
        common,
        delta_r2,
        delta_t2,
    )


# -------------------------------------------------------------------------
# QC figure
# -------------------------------------------------------------------------


def percentile_limits(
    data: np.ndarray,
    mask: np.ndarray,
    low: float = 2,
    high: float = 98,
) -> Tuple[float, float]:
    values = data[
        mask
        & np.isfinite(
            data
        )
    ]

    if values.size == 0:
        return (
            0.0,
            1.0,
        )

    vmin = float(
        np.percentile(
            values,
            low,
        )
    )

    vmax = float(
        np.percentile(
            values,
            high,
        )
    )

    if vmax <= vmin:
        vmax = (
            vmin
            + 1.0
        )

    return (
        vmin,
        vmax,
    )


def make_qc_png(
    case: str,
    reference_echo1: np.ndarray,
    dual_echo1_registered: np.ndarray,
    r2_ref: np.ndarray,
    r2_dual: np.ndarray,
    common: np.ndarray,
    delta_r2: np.ndarray,
    summary: Dict[str, Any],
    path: Path,
    max_points: int,
) -> None:
    z_score = np.sum(
        common,
        axis=(0, 1),
    )

    z = int(
        np.argmax(
            z_score
        )
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 10),
    )

    # ---------------------------------------------------------
    # Registration reference
    # ---------------------------------------------------------

    mask_ref = (
        np.isfinite(
            reference_echo1
        )
        & (
            reference_echo1
            > 0
        )
    )

    vmin, vmax = percentile_limits(
        reference_echo1,
        mask_ref,
    )

    axes[0, 0].imshow(
        np.rot90(
            reference_echo1[
                :,
                :,
                z,
            ]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0, 0].set_title(
        "12-echo echo 1\nreference"
    )

    # ---------------------------------------------------------
    # Registered dual echo
    # ---------------------------------------------------------

    vmin, vmax = percentile_limits(
        dual_echo1_registered,
        np.isfinite(
            dual_echo1_registered
        )
        & (
            dual_echo1_registered
            > 0
        ),
    )

    axes[0, 1].imshow(
        np.rot90(
            dual_echo1_registered[
                :,
                :,
                z,
            ]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0, 1].set_title(
        "Dual-TSE echo 1\nregistered"
    )

    # ---------------------------------------------------------
    # R2 reference
    # ---------------------------------------------------------

    vmin, vmax = percentile_limits(
        r2_ref,
        common,
    )

    im = axes[0, 2].imshow(
        np.rot90(
            r2_ref[
                :,
                :,
                z,
            ]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0, 2].set_title(
        "12-echo drop1 R2 [s⁻¹]"
    )

    fig.colorbar(
        im,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
    )

    # ---------------------------------------------------------
    # Dual R2
    # ---------------------------------------------------------

    im = axes[0, 3].imshow(
        np.rot90(
            r2_dual[
                :,
                :,
                z,
            ]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0, 3].set_title(
        "Dual R2app [s⁻¹]\nin 12-echo space"
    )

    fig.colorbar(
        im,
        ax=axes[0, 3],
        fraction=0.046,
        pad=0.04,
    )

    # ---------------------------------------------------------
    # Common mask
    # ---------------------------------------------------------

    axes[1, 0].imshow(
        np.rot90(
            common[
                :,
                :,
                z,
            ]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Common valid mask"
    )

    # ---------------------------------------------------------
    # Delta R2 map
    # ---------------------------------------------------------

    delta_values = delta_r2[
        common
    ]

    limit = float(
        np.percentile(
            np.abs(
                delta_values
            ),
            98,
        )
    )

    if not np.isfinite(
        limit
    ) or limit <= 0:
        limit = 1.0

    im = axes[1, 1].imshow(
        np.rot90(
            delta_r2[
                :,
                :,
                z,
            ]
        ),
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )

    axes[1, 1].set_title(
        "ΔR2 = dual - reference [s⁻¹]"
    )

    fig.colorbar(
        im,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    # ---------------------------------------------------------
    # Scatter
    # ---------------------------------------------------------

    x = r2_ref[
        common
    ].astype(
        np.float64
    )

    y = r2_dual[
        common
    ].astype(
        np.float64
    )

    rng = np.random.default_rng(
        DEFAULT_RANDOM_SEED
    )

    if x.size > max_points:
        indices = rng.choice(
            x.size,
            size=max_points,
            replace=False,
        )

        x_plot = x[
            indices
        ]

        y_plot = y[
            indices
        ]

    else:
        x_plot = x
        y_plot = y

    axes[1, 2].scatter(
        x_plot,
        y_plot,
        s=1,
        alpha=0.15,
    )

    low = float(
        np.percentile(
            np.concatenate(
                [
                    x_plot,
                    y_plot,
                ]
            ),
            1,
        )
    )

    high = float(
        np.percentile(
            np.concatenate(
                [
                    x_plot,
                    y_plot,
                ]
            ),
            99,
        )
    )

    axes[1, 2].plot(
        [
            low,
            high,
        ],
        [
            low,
            high,
        ],
        linestyle="--",
        linewidth=1,
    )

    slope = summary[
        "regression_slope"
    ]

    intercept = summary[
        "regression_intercept"
    ]

    xx = np.array(
        [
            low,
            high,
        ]
    )

    axes[1, 2].plot(
        xx,
        intercept
        + slope
        * xx,
        linewidth=1,
    )

    axes[1, 2].set_xlabel(
        "12-echo drop1 R2 [s⁻¹]"
    )

    axes[1, 2].set_ylabel(
        "Dual R2app [s⁻¹]"
    )

    axes[1, 2].set_title(
        "Voxelwise comparison\n"
        f"r={summary['pearson_r']:.3f}, "
        f"ρ={summary['spearman_rho']:.3f}"
    )

    # ---------------------------------------------------------
    # Bland-Altman
    # ---------------------------------------------------------

    mean_values = (
        x
        + y
    ) / 2.0

    differences = (
        y
        - x
    )

    if x.size > max_points:
        indices = rng.choice(
            x.size,
            size=max_points,
            replace=False,
        )

        mean_plot = mean_values[
            indices
        ]

        diff_plot = differences[
            indices
        ]

    else:
        mean_plot = mean_values
        diff_plot = differences

    axes[1, 3].scatter(
        mean_plot,
        diff_plot,
        s=1,
        alpha=0.15,
    )

    bias = summary[
        "bland_altman_bias_R2_s-1"
    ]

    lower = summary[
        "bland_altman_lower95_R2_s-1"
    ]

    upper = summary[
        "bland_altman_upper95_R2_s-1"
    ]

    axes[1, 3].axhline(
        bias,
        linewidth=1,
    )

    axes[1, 3].axhline(
        lower,
        linestyle="--",
        linewidth=1,
    )

    axes[1, 3].axhline(
        upper,
        linestyle="--",
        linewidth=1,
    )

    axes[1, 3].set_xlabel(
        "Mean R2 [s⁻¹]"
    )

    axes[1, 3].set_ylabel(
        "Dual - reference [s⁻¹]"
    )

    axes[1, 3].set_title(
        "Bland-Altman\n"
        f"bias={bias:.2f} s⁻¹"
    )

    for ax in axes[
        0,
        :2,
    ]:
        ax.axis(
            "off"
        )

    for ax in [
        axes[0, 2],
        axes[0, 3],
        axes[1, 0],
        axes[1, 1],
    ]:
        ax.axis(
            "off"
        )

    fig.suptitle(
        f"{TAG}: dual-echo R2app validation against "
        f"12-echo drop1\n{case}, z={z}"
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# -------------------------------------------------------------------------
# Case processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    overwrite: bool,
    dry_run: bool,
    max_scatter_points: int,
) -> Dict[str, Any]:
    print(
        "=" * 72
    )

    print(
        f"{TAG}: {case}"
    )

    print(
        "=" * 72
    )

    check_fsl()

    start = now_iso()

    # ---------------------------------------------------------
    # Resolve inputs
    # ---------------------------------------------------------

    rlx02 = load_rlx02_reference(
        paths,
        case,
    )

    dual = load_dual_echo1(
        paths,
        case,
    )

    reference = load_rlx03_reference_paths(
        paths,
        case,
    )

    candidate = load_rlx05_paths(
        paths,
        case,
    )

    out_dir = rlx06_dir(
        paths,
        case,
    )

    metadata_path = (
        out_dir
        / f"rlx06_{case}_metadata.json"
    )

    if (
        metadata_path.exists()
        and not overwrite
        and not dry_run
    ):
        raise FileExistsError(
            f"rlx06 outputs already exist:\n"
            f"  {metadata_path}\n"
            "Use --overwrite to replace them."
        )

    reference_echo1_path = (
        out_dir
        / f"rlx06_{case}_12echo_echo1.nii.gz"
    )

    registered_echo1_path = (
        out_dir
        / f"rlx06_{case}_dualEcho1_in_12echo.nii.gz"
    )

    matrix_path = (
        out_dir
        / f"rlx06_{case}_dualEcho1_to_12echoEcho1.mat"
    )

    r2_dual_registered_path = (
        out_dir
        / f"rlx06_{case}_R2app_in_12echo.nii.gz"
    )

    t2_dual_registered_path = (
        out_dir
        / f"rlx06_{case}_T2app_in_12echo.nii.gz"
    )

    valid_dual_registered_path = (
        out_dir
        / f"rlx06_{case}_dual_valid_in_12echo.nii.gz"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  12-echo stack: "
        f"{rlx02['stack_path']}"
    )

    print(
        f"  Dual echo 1: "
        f"{dual['image_path']}"
    )

    print(
        f"  Reference R2: "
        f"{reference['R2']}"
    )

    print(
        f"  Candidate R2app: "
        f"{candidate['R2']}"
    )

    if dry_run:
        return {
            "case": case,
            "success": True,
            "dry_run": True,
            "rlx02_stack":
                str(
                    rlx02[
                        "stack_path"
                    ]
                ),
            "dual_echo1":
                str(
                    dual[
                        "image_path"
                    ]
                ),
            "reference_R2":
                str(
                    reference[
                        "R2"
                    ]
                ),
            "candidate_R2app":
                str(
                    candidate[
                        "R2"
                    ]
                ),
        }

    # ---------------------------------------------------------
    # Extract first multi-echo volume
    # ---------------------------------------------------------

    reference_echo1_img = extract_reference_echo1(
        stack_path=
            rlx02[
                "stack_path"
            ],

        output_path=
            reference_echo1_path,
    )

    # ---------------------------------------------------------
    # Rigid registration
    # ---------------------------------------------------------

    register_dual_to_reference(
        dual_echo1=
            dual[
                "image_path"
            ],

        reference_echo1=
            reference_echo1_path,

        registered_echo1=
            registered_echo1_path,

        matrix_path=
            matrix_path,
    )

    # ---------------------------------------------------------
    # Apply transformation
    # ---------------------------------------------------------

    apply_transform(
        input_path=
            candidate[
                "R2"
            ],

        reference_path=
            reference_echo1_path,

        matrix_path=
            matrix_path,

        output_path=
            r2_dual_registered_path,

        interpolation=
            "trilinear",
    )

    apply_transform(
        input_path=
            candidate[
                "T2"
            ],

        reference_path=
            reference_echo1_path,

        matrix_path=
            matrix_path,

        output_path=
            t2_dual_registered_path,

        interpolation=
            "trilinear",
    )

    apply_transform(
        input_path=
            candidate[
                "valid"
            ],

        reference_path=
            reference_echo1_path,

        matrix_path=
            matrix_path,

        output_path=
            valid_dual_registered_path,

        interpolation=
            "nearestneighbour",
    )

    # ---------------------------------------------------------
    # Load transformed/reference data
    # ---------------------------------------------------------

    ref_r2_img = nib.load(
        str(
            reference[
                "R2"
            ]
        )
    )

    ref_t2_img = nib.load(
        str(
            reference[
                "T2"
            ]
        )
    )

    ref_valid_img = nib.load(
        str(
            reference[
                "valid"
            ]
        )
    )

    dual_r2_img = nib.load(
        str(
            r2_dual_registered_path
        )
    )

    dual_t2_img = nib.load(
        str(
            t2_dual_registered_path
        )
    )

    dual_valid_img = nib.load(
        str(
            valid_dual_registered_path
        )
    )

    registered_echo1_img = nib.load(
        str(
            registered_echo1_path
        )
    )

    # Geometry must now be identical.
    shapes = {
        ref_r2_img.shape,
        ref_t2_img.shape,
        ref_valid_img.shape,
        dual_r2_img.shape,
        dual_t2_img.shape,
        dual_valid_img.shape,
        reference_echo1_img.shape,
        registered_echo1_img.shape,
    }

    if len(
        shapes
    ) != 1:
        raise ValueError(
            "Registered/reference output shapes do not match."
        )

    r2_ref = np.asarray(
        ref_r2_img.dataobj,
        dtype=np.float32,
    )

    t2_ref = np.asarray(
        ref_t2_img.dataobj,
        dtype=np.float32,
    )

    valid_ref = (
        np.asarray(
            ref_valid_img.dataobj
        )
        > 0
    )

    r2_dual = np.asarray(
        dual_r2_img.dataobj,
        dtype=np.float32,
    )

    t2_dual = np.asarray(
        dual_t2_img.dataobj,
        dtype=np.float32,
    )

    valid_dual = (
        np.asarray(
            dual_valid_img.dataobj
        )
        > 0
    )

    reference_echo1 = np.asarray(
        reference_echo1_img.dataobj,
        dtype=np.float32,
    )

    registered_echo1 = np.asarray(
        registered_echo1_img.dataobj,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Quantitative comparison
    # ---------------------------------------------------------

    (
        comparison,
        common,
        delta_r2,
        delta_t2,
    ) = compare_maps(
        r2_ref=r2_ref,
        t2_ref=t2_ref,
        valid_ref=valid_ref,
        r2_dual=r2_dual,
        t2_dual=t2_dual,
        valid_dual=valid_dual,
    )

    comparison[
        "case"
    ] = case

    common_path = (
        out_dir
        / f"rlx06_{case}_common_valid_mask.nii.gz"
    )

    delta_r2_path = (
        out_dir
        / f"rlx06_{case}_deltaR2_s-1.nii.gz"
    )

    delta_t2_path = (
        out_dir
        / f"rlx06_{case}_deltaT2_ms.nii.gz"
    )

    save_nifti(
        common.astype(
            np.uint8
        ),
        ref_r2_img,
        common_path,
        np.uint8,
    )

    save_nifti(
        delta_r2,
        ref_r2_img,
        delta_r2_path,
        np.float32,
    )

    save_nifti(
        delta_t2,
        ref_t2_img,
        delta_t2_path,
        np.float32,
    )

    comparison_csv = (
        out_dir
        / f"rlx06_{case}_comparison.csv"
    )

    write_csv(
        comparison_csv,
        [
            comparison
        ],
    )

    # ---------------------------------------------------------
    # QC
    # ---------------------------------------------------------

    qc_png = (
        out_dir
        / f"rlx06_{case}_qc.png"
    )

    make_qc_png(
        case=case,
        reference_echo1=
            reference_echo1,

        dual_echo1_registered=
            registered_echo1,

        r2_ref=r2_ref,
        r2_dual=r2_dual,
        common=common,
        delta_r2=
            delta_r2,

        summary=
            comparison,

        path=
            qc_png,

        max_points=
            max_scatter_points,
    )

    end = now_iso()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "script":
            SCRIPT,

        "tag":
            TAG,

        "version":
            VERSION,

        "case":
            case,

        "created":
            end,

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

            "command":
                " ".join(
                    sys.argv
                ),
        },

        "scientific_role": {
            "reference":
                (
                    "rlx03 drop1 mono-exponential "
                    "12-echo R2"
                ),

            "candidate":
                (
                    "rlx05 high-resolution "
                    "dual-echo R2app"
                ),

            "comparison_space":
                (
                    "native 12-echo acquisition space"
                ),

            "interpretation":
                (
                    "Tests absolute agreement, association, "
                    "proportional bias, and spatial agreement "
                    "to determine whether dual-echo R2app is "
                    "suitable as a high-resolution surrogate."
                ),
        },

        "registration": {
            "moving":
                str(
                    dual[
                        "image_path"
                    ]
                ),

            "reference":
                str(
                    reference_echo1_path
                ),

            "method":
                "FSL FLIRT",

            "degrees_of_freedom":
                6,

            "cost":
                "normalized mutual information",

            "matrix":
                str(
                    matrix_path
                ),

            "registered_dual_echo1":
                str(
                    registered_echo1_path
                ),

            "continuous_map_interpolation":
                "trilinear",

            "mask_interpolation":
                "nearest neighbour",
        },

        "inputs": {
            "rlx02_metadata":
                str(
                    rlx02[
                        "metadata_path"
                    ]
                ),

            "rlx02_stack":
                str(
                    rlx02[
                        "stack_path"
                    ]
                ),

            "reference_R2":
                str(
                    reference[
                        "R2"
                    ]
                ),

            "reference_T2":
                str(
                    reference[
                        "T2"
                    ]
                ),

            "reference_valid_mask":
                str(
                    reference[
                        "valid"
                    ]
                ),

            "dual_R2app":
                str(
                    candidate[
                        "R2"
                    ]
                ),

            "dual_T2app":
                str(
                    candidate[
                        "T2"
                    ]
                ),

            "dual_valid_mask":
                str(
                    candidate[
                        "valid"
                    ]
                ),
        },

        "definitions": {
            "deltaR2_s-1":
                (
                    "R2app_dual_in_12echo_space "
                    "- R2_drop1_reference"
                ),

            "deltaT2_ms":
                (
                    "T2app_dual_in_12echo_space "
                    "- T2_drop1_reference"
                ),

            "common_valid_mask":
                (
                    "valid drop1 reference AND "
                    "valid transformed dual-echo estimate "
                    "AND finite R2/T2 values"
                ),

            "bland_altman_difference":
                "dual - reference",
        },

        "comparison":
            comparison,

        "outputs": {
            "reference_echo1":
                str(
                    reference_echo1_path
                ),

            "registration_matrix":
                str(
                    matrix_path
                ),

            "registered_dual_echo1":
                str(
                    registered_echo1_path
                ),

            "dual_R2app_in_reference":
                str(
                    r2_dual_registered_path
                ),

            "dual_T2app_in_reference":
                str(
                    t2_dual_registered_path
                ),

            "dual_valid_in_reference":
                str(
                    valid_dual_registered_path
                ),

            "common_valid_mask":
                str(
                    common_path
                ),

            "deltaR2":
                str(
                    delta_r2_path
                ),

            "deltaT2":
                str(
                    delta_t2_path
                ),

            "comparison_csv":
                str(
                    comparison_csv
                ),

            "qc_png":
                str(
                    qc_png
                ),
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------

    print(
        f"  Common valid voxels: "
        f"{comparison['common_valid_voxels']}"
    )

    print(
        f"  Reference median R2: "
        f"{comparison['R2_reference_s-1_median']:.3f} s^-1"
    )

    print(
        f"  Dual median R2app: "
        f"{comparison['R2_dual_s-1_median']:.3f} s^-1"
    )

    print(
        f"  Median ΔR2: "
        f"{comparison['deltaR2_s-1_median']:.3f} s^-1"
    )

    print(
        f"  Pearson r: "
        f"{comparison['pearson_r']:.3f}"
    )

    print(
        f"  Spearman rho: "
        f"{comparison['spearman_rho']:.3f}"
    )

    print(
        f"  Regression: "
        f"dual = "
        f"{comparison['regression_intercept']:.3f} "
        f"+ "
        f"{comparison['regression_slope']:.3f} "
        f"* reference"
    )

    print(
        f"  Bland-Altman bias: "
        f"{comparison['bland_altman_bias_R2_s-1']:.3f} s^-1"
    )

    print(
        f"  DONE: {case}"
    )

    return metadata


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF rlx06: validate high-resolution "
            "dual-echo R2app against 12-echo drop1."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rlx06 outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs without running registration/comparison.",
    )

    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=DEFAULT_SCATTER_MAX_POINTS,
        help=(
            "Maximum randomly sampled voxels displayed "
            "in scatter/Bland-Altman plots."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.max_scatter_points < 1000:
        raise SystemExit(
            "--max-scatter-points must be >= 1000."
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )

    print_selected_cases(
        cases
    )

    cohort_rows: List[
        Dict[str, Any]
    ] = []

    failures = 0

    for case in cases:
        try:
            metadata = process_case(
                case=case,
                paths=paths,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                max_scatter_points=
                    args.max_scatter_points,
            )

            if args.dry_run:
                cohort_rows.append(
                    {
                        "case":
                            case,

                        "success":
                            True,

                        "dry_run":
                            True,
                    }
                )

            else:
                cohort_rows.append(
                    {
                        "success":
                            True,

                        **metadata[
                            "comparison"
                        ],
                    }
                )

        except Exception as exc:
            failures += 1

            print(
                f"ERROR in {case}: {exc}",
                file=sys.stderr,
            )

            cohort_rows.append(
                {
                    "case":
                        case,

                    "success":
                        False,

                    "error":
                        str(
                            exc
                        ),
                }
            )

            if len(
                cases
            ) == 1:
                raise

    if not args.dry_run:
        out = cohort_dir(
            paths
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            out
            / "rlx06_cohort_comparison.csv",
            cohort_rows,
        )

        write_json(
            out
            / "rlx06_cohort_metadata.json",
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

                "n_cases_requested":
                    len(
                        cases
                    ),

                "n_failures":
                    failures,

                "rows":
                    cohort_rows,
            },
        )

    if failures:
        raise SystemExit(
            f"rlx06 completed with "
            f"{failures} failed case(s)."
        )


if __name__ == "__main__":
    main()