#!/usr/bin/env python3
"""
APF rlx05: High-resolution dual-echo apparent T2/R2 estimation.

Scientific purpose
------------------
rlx05 derives a high-resolution apparent transverse relaxation estimate from
the dual-echo TSE acquisition:

    S(TE) = S0 * exp(-R2_app * TE)

With exactly two echoes:

    R2_app = ln(S1 / S2) / (TE2 - TE1)

and:

    T2_app = 1 / R2_app

The result is deliberately labelled R2_app / T2_app rather than R2 / T2.
A two-echo acquisition exactly determines the two-parameter mono-exponential
model and therefore cannot independently establish goodness-of-fit.

The validated 12-echo drop1 estimate from rlx03 remains the quantitative
reference. A later validation stage will compare the high-resolution dual-echo
R2_app against that reference.

Upstream contracts
------------------
rlx05 consumes:

1. rlx01 dual-TSE selection:

    derivatives/<case>/relaxometry/rlx01_inventory/
        rlx01_<case>_selected_inputs.json

2. rlx05a native-space brain masks:

    derivatives/<case>/relaxometry/rlx05a/
        rlx05a_<case>_brain_mask.nii.gz
        rlx05a_<case>_brain_mask_ero1.nii.gz

Masking strategy
----------------
Primary fitting domain:
    rlx05a HD-BET brain mask

Secondary conservative summary domain:
    rlx05a one-voxel-eroded brain mask

The R2app/T2app maps are calculated once in the full HD-BET brain mask.
The eroded mask is retained for secondary/sensitivity summary statistics.

Validity logic
--------------
A voxel enters the dual-echo candidate domain if:

    - it lies inside the primary HD-BET brain mask
    - both signals are finite
    - S1 > min_signal
    - S2 > min_signal

A physically positive mono-exponential R2_app additionally requires:

    S1 > S2

Candidate voxels are rejected if:

    - S2 >= S1       -> non-decaying / non-positive R2_app
    - calculated R2_app lies outside configured plausibility bounds
    - calculated outputs are non-finite

Outputs
-------
    R2app_s-1
    T2app_ms
    S0app
    signal_ratio_S1_S2
    candidate_mask
    fit_valid_mask
    nondecay_mask
    out_of_range_mask
    failed_fit_mask

QC
--
    - per-case summary CSV
    - metadata JSON
    - compact brain-restricted QC PNG
    - cohort summary CSV

Run:
    ./apf scripts/relaxometry/05_fit_dual_echo.py ANISO_00_000_InS
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT = "05_fit_dual_echo.py"
TAG = "rlx05"
VERSION = "0.2.0"

DEFAULT_MIN_SIGNAL = 1.0
DEFAULT_R2_MIN = 0.5
DEFAULT_R2_MAX = 500.0
DEFAULT_AFFINE_ATOL = 1e-5


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("rlx05 requires nibabel.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("rlx05 requires matplotlib.") from exc

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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n"
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


def robust_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]

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
        "mean": float(np.mean(values)),
        "sd": (
            float(np.std(values, ddof=1))
            if values.size > 1
            else 0.0
        ),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def resolve_project_path(
    raw: str,
    project_root: Path,
) -> Path:
    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = project_root / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist:\n  {path}"
        )

    return path


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
    header.set_data_dtype(dtype)

    image = nib.Nifti1Image(
        data.astype(dtype, copy=False),
        reference.affine,
        header,
    )

    nib.save(
        image,
        str(path),
    )


# -------------------------------------------------------------------------
# APF paths
# -------------------------------------------------------------------------


def rlx01_selected_inputs_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx01_inventory"
        / f"rlx01_{case}_selected_inputs.json"
    )


def rlx05a_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05a"
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


def summary_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "rlx05"
    )


# -------------------------------------------------------------------------
# Input contracts
# -------------------------------------------------------------------------


def load_rlx01_dual_tse(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    metadata_path = rlx01_selected_inputs_path(
        paths,
        case,
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Required rlx01 selected-input metadata not found:\n"
            f"  {metadata_path}\n"
            "Run rlx01 first."
        )

    metadata = json.loads(
        metadata_path.read_text()
    )

    dual = metadata.get("dual_tse")

    if not isinstance(dual, dict):
        raise ValueError(
            f"rlx01 metadata has no dual_tse object:\n"
            f"  {metadata_path}"
        )

    if not dual.get("ready", False):
        raise RuntimeError(
            f"dual_tse is not READY for {case}: "
            f"{dual.get('readiness_reason', 'unknown reason')}"
        )

    if not dual.get("selected", False):
        raise RuntimeError(
            f"dual_tse is not selected for {case}."
        )

    if dual.get("n_echoes") != 2:
        raise ValueError(
            f"Expected exactly 2 dual-TSE echoes, got "
            f"{dual.get('n_echoes')}."
        )

    echoes = dual.get("echoes")

    if not isinstance(echoes, list) or len(echoes) != 2:
        raise ValueError(
            "dual_tse.echoes must contain exactly two entries."
        )

    try:
        echoes = sorted(
            echoes,
            key=lambda x: int(x["echo_number"]),
        )
    except Exception as exc:
        raise ValueError(
            "Could not order dual-TSE echoes by echo_number."
        ) from exc

    resolved = []

    for echo in echoes:
        raw_image = echo.get("image")
        raw_te = echo.get("echo_time_s")
        echo_number = echo.get("echo_number")

        if not isinstance(raw_image, str):
            raise ValueError(
                f"Missing image path for echo {echo_number}."
            )

        try:
            te_s = float(raw_te)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid echo_time_s for echo {echo_number}: "
                f"{raw_te}"
            ) from exc

        if not np.isfinite(te_s) or te_s <= 0:
            raise ValueError(
                f"Echo time must be finite and > 0: {te_s}"
            )

        image_path = resolve_project_path(
            raw_image,
            paths.root,
        )

        resolved.append(
            {
                "echo_number": int(echo_number),
                "echo_time_s": te_s,
                "image": image_path,
                "json_sidecar": echo.get("json_sidecar"),
            }
        )

    if resolved[1]["echo_time_s"] <= resolved[0]["echo_time_s"]:
        raise ValueError(
            "Dual-TSE echo times must be strictly increasing."
        )

    return {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "dual_tse": dual,
        "echoes": resolved,
    }


def load_rlx05a_masks(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    stage_dir = rlx05a_dir(
        paths,
        case,
    )

    metadata_path = (
        stage_dir
        / f"rlx05a_{case}_metadata.json"
    )

    brain_mask_path = (
        stage_dir
        / f"rlx05a_{case}_brain_mask.nii.gz"
    )

    brain_mask_ero1_path = (
        stage_dir
        / f"rlx05a_{case}_brain_mask_ero1.nii.gz"
    )

    required = [
        metadata_path,
        brain_mask_path,
        brain_mask_ero1_path,
    ]

    missing = [
        path for path in required
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required rlx05a outputs are missing:\n"
            f"{formatted}\n"
            "Run rlx05a before rlx05."
        )

    metadata = json.loads(
        metadata_path.read_text()
    )

    return {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "brain_mask_path": brain_mask_path,
        "brain_mask_ero1_path": brain_mask_ero1_path,
    }


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------


def validate_geometry(
    reference: nib.spatialimages.SpatialImage,
    other: nib.spatialimages.SpatialImage,
    label_reference: str,
    label_other: str,
    affine_atol: float,
) -> None:
    if reference.shape != other.shape:
        raise ValueError(
            "Image dimensions do not match:\n"
            f"  {label_reference}: {reference.shape}\n"
            f"  {label_other}: {other.shape}"
        )

    if len(reference.shape) != 3:
        raise ValueError(
            f"Expected 3D images, got {reference.shape}."
        )

    if not np.allclose(
        reference.affine,
        other.affine,
        rtol=0,
        atol=affine_atol,
    ):
        raise ValueError(
            "Image affines do not match:\n"
            f"  {label_reference}\n"
            f"  {label_other}"
        )


# -------------------------------------------------------------------------
# Dual-echo model
# -------------------------------------------------------------------------


def calculate_dual_echo_maps(
    s1: np.ndarray,
    s2: np.ndarray,
    brain_mask: np.ndarray,
    te1_s: float,
    te2_s: float,
    min_signal: float,
    r2_min: float,
    r2_max: float,
) -> Dict[str, np.ndarray]:
    """
    Calculate analytical two-point mono-exponential estimates.

        R2app = ln(S1/S2) / (TE2 - TE1)

    The primary candidate domain is restricted to the authoritative rlx05a
    HD-BET brain mask.

    No goodness-of-fit metric exists because two observations exactly
    determine the two model parameters S0 and R2app.
    """

    if s1.shape != s2.shape:
        raise ValueError(
            "Signal arrays must have identical shapes."
        )

    if brain_mask.shape != s1.shape:
        raise ValueError(
            "Brain mask does not match signal geometry."
        )

    delta_te = te2_s - te1_s

    if delta_te <= 0:
        raise ValueError(
            "Require TE2 > TE1."
        )

    finite = (
        np.isfinite(s1)
        & np.isfinite(s2)
    )

    candidate = (
        brain_mask
        & finite
        & (s1 > min_signal)
        & (s2 > min_signal)
    )

    nondecay = (
        candidate
        & (s2 >= s1)
    )

    decaying = (
        candidate
        & (s1 > s2)
    )

    shape = s1.shape

    ratio = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    r2 = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    t2 = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    s0 = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    ratio[decaying] = (
        s1[decaying]
        / s2[decaying]
    )

    log_s1 = np.full(
        shape,
        np.nan,
        dtype=np.float64,
    )

    log_s2 = np.full(
        shape,
        np.nan,
        dtype=np.float64,
    )

    log_s1[decaying] = np.log(
        s1[decaying].astype(np.float64)
    )

    log_s2[decaying] = np.log(
        s2[decaying].astype(np.float64)
    )

    r2_raw = np.full(
        shape,
        np.nan,
        dtype=np.float64,
    )

    r2_raw[decaying] = (
        log_s1[decaying]
        - log_s2[decaying]
    ) / delta_te

    log_s0_raw = np.full(
        shape,
        np.nan,
        dtype=np.float64,
    )

    log_s0_raw[decaying] = (
        log_s1[decaying]
        + r2_raw[decaying] * te1_s
    )

    finite_model = (
        decaying
        & np.isfinite(r2_raw)
        & np.isfinite(log_s0_raw)
    )

    out_of_range = (
        finite_model
        & (
            (r2_raw < r2_min)
            | (r2_raw > r2_max)
        )
    )

    valid = (
        finite_model
        & ~out_of_range
    )

    r2[valid] = (
        r2_raw[valid]
        .astype(np.float32)
    )

    t2[valid] = (
        1000.0
        / r2_raw[valid]
    ).astype(
        np.float32
    )

    failed = np.zeros(
        shape,
        dtype=bool,
    )

    valid_flat = np.flatnonzero(
        valid
    )

    if valid_flat.size:
        s0_raw = np.exp(
            log_s0_raw.flat[
                valid_flat
            ]
        )

        s0_finite = np.isfinite(
            s0_raw
        )

        if np.any(
            ~s0_finite
        ):
            bad_flat = valid_flat[
                ~s0_finite
            ]

            failed.flat[
                bad_flat
            ] = True

            valid.flat[
                bad_flat
            ] = False

            r2.flat[
                bad_flat
            ] = np.nan

            t2.flat[
                bad_flat
            ] = np.nan

        good_flat = valid_flat[
            s0_finite
        ]

        s0.flat[
            good_flat
        ] = s0_raw[
            s0_finite
        ].astype(
            np.float32
        )

    failed |= (
        candidate
        & ~nondecay
        & ~out_of_range
        & ~valid
    )

    return {
        "R2app_s-1": r2,
        "T2app_ms": t2,
        "S0app": s0,
        "signal_ratio_S1_S2": ratio,
        "candidate_mask": candidate.astype(np.uint8),
        "fit_valid_mask": valid.astype(np.uint8),
        "nondecay_mask": nondecay.astype(np.uint8),
        "out_of_range_mask": out_of_range.astype(np.uint8),
        "failed_fit_mask": failed.astype(np.uint8),
    }


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------


def add_domain_summary(
    row: Dict[str, Any],
    prefix: str,
    outputs: Dict[str, np.ndarray],
    domain_mask: np.ndarray,
) -> None:
    candidate = (
        outputs["candidate_mask"] > 0
    ) & domain_mask

    valid = (
        outputs["fit_valid_mask"] > 0
    ) & domain_mask

    nondecay = (
        outputs["nondecay_mask"] > 0
    ) & domain_mask

    out_of_range = (
        outputs["out_of_range_mask"] > 0
    ) & domain_mask

    failed = (
        outputs["failed_fit_mask"] > 0
    ) & domain_mask

    n_domain = int(
        np.count_nonzero(
            domain_mask
        )
    )

    n_candidate = int(
        np.count_nonzero(
            candidate
        )
    )

    row[
        f"{prefix}_domain_voxels"
    ] = n_domain

    row[
        f"{prefix}_candidate_voxels"
    ] = n_candidate

    row[
        f"{prefix}_candidate_fraction_of_domain"
    ] = (
        float(
            n_candidate / n_domain
        )
        if n_domain > 0
        else None
    )

    masks = {
        "valid": valid,
        "nondecay": nondecay,
        "out_of_range": out_of_range,
        "failed_fit": failed,
    }

    for name, mask in masks.items():
        count = int(
            np.count_nonzero(
                mask
            )
        )

        row[
            f"{prefix}_{name}_voxels"
        ] = count

        row[
            f"{prefix}_{name}_fraction_of_candidate"
        ] = (
            float(
                count / n_candidate
            )
            if n_candidate > 0
            else None
        )

    for metric in [
        "R2app_s-1",
        "T2app_ms",
        "S0app",
        "signal_ratio_S1_S2",
    ]:
        values = outputs[
            metric
        ][
            valid
        ]

        metric_stats = robust_stats(
            values
        )

        for key, value in metric_stats.items():
            row[
                f"{prefix}_{metric}_{key}"
            ] = value


def summarize_case(
    case: str,
    outputs: Dict[str, np.ndarray],
    brain_mask: np.ndarray,
    brain_mask_ero1: np.ndarray,
    te1_s: float,
    te2_s: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case": case,
        "TE1_ms": te1_s * 1000.0,
        "TE2_ms": te2_s * 1000.0,
        "deltaTE_ms": (
            te2_s - te1_s
        ) * 1000.0,
    }

    add_domain_summary(
        row=row,
        prefix="brain",
        outputs=outputs,
        domain_mask=brain_mask,
    )

    add_domain_summary(
        row=row,
        prefix="ero1",
        outputs=outputs,
        domain_mask=brain_mask_ero1,
    )

    return row


# -------------------------------------------------------------------------
# Output paths
# -------------------------------------------------------------------------


def output_paths(
    out_dir: Path,
    case: str,
) -> Dict[str, Path]:
    stem = f"rlx05_{case}"

    return {
        "R2app_s-1":
            out_dir
            / f"{stem}_R2app_s-1.nii.gz",

        "T2app_ms":
            out_dir
            / f"{stem}_T2app_ms.nii.gz",

        "S0app":
            out_dir
            / f"{stem}_S0app.nii.gz",

        "signal_ratio_S1_S2":
            out_dir
            / f"{stem}_signal_ratio_S1_S2.nii.gz",

        "candidate_mask":
            out_dir
            / f"{stem}_candidate_mask.nii.gz",

        "fit_valid_mask":
            out_dir
            / f"{stem}_fit_valid_mask.nii.gz",

        "nondecay_mask":
            out_dir
            / f"{stem}_nondecay_mask.nii.gz",

        "out_of_range_mask":
            out_dir
            / f"{stem}_out_of_range_mask.nii.gz",

        "failed_fit_mask":
            out_dir
            / f"{stem}_failed_fit_mask.nii.gz",
    }


# -------------------------------------------------------------------------
# QC
# -------------------------------------------------------------------------


def percentile_limits(
    data: np.ndarray,
    mask: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Tuple[float, float]:
    values = data[
        mask
        & np.isfinite(data)
    ]

    if values.size == 0:
        return 0.0, 1.0

    low = float(
        np.percentile(
            values,
            p_low,
        )
    )

    high = float(
        np.percentile(
            values,
            p_high,
        )
    )

    if high <= low:
        high = low + 1.0

    return low, high


def make_qc_png(
    case: str,
    s1: np.ndarray,
    s2: np.ndarray,
    brain_mask: np.ndarray,
    brain_mask_ero1: np.ndarray,
    outputs: Dict[str, np.ndarray],
    te1_s: float,
    te2_s: float,
    path: Path,
) -> None:
    candidate = (
        outputs["candidate_mask"] > 0
    )

    valid = (
        outputs["fit_valid_mask"] > 0
    )

    z_scores = np.sum(
        brain_mask,
        axis=(0, 1),
    )

    if np.any(
        z_scores > 0
    ):
        z = int(
            np.argmax(
                z_scores
            )
        )
    else:
        z = (
            s1.shape[2]
            // 2
        )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 13),
    )

    panels = axes.flat

    vmin, vmax = percentile_limits(
        s1,
        brain_mask,
    )

    im = panels[0].imshow(
        np.rot90(
            s1[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    panels[0].contour(
        np.rot90(
            brain_mask[:, :, z]
        ),
        levels=[0.5],
        linewidths=0.7,
    )

    panels[0].set_title(
        f"Echo 1 + brain mask\n"
        f"TE={te1_s * 1000:.1f} ms"
    )

    fig.colorbar(
        im,
        ax=panels[0],
        fraction=0.046,
        pad=0.04,
    )

    vmin, vmax = percentile_limits(
        s2,
        brain_mask,
    )

    im = panels[1].imshow(
        np.rot90(
            s2[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    panels[1].contour(
        np.rot90(
            brain_mask[:, :, z]
        ),
        levels=[0.5],
        linewidths=0.7,
    )

    panels[1].set_title(
        f"Echo 2 + brain mask\n"
        f"TE={te2_s * 1000:.1f} ms"
    )

    fig.colorbar(
        im,
        ax=panels[1],
        fraction=0.046,
        pad=0.04,
    )

    ratio = outputs[
        "signal_ratio_S1_S2"
    ]

    vmin, vmax = percentile_limits(
        ratio,
        valid,
    )

    im = panels[2].imshow(
        np.rot90(
            ratio[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    panels[2].set_title(
        "Signal ratio S1/S2"
    )

    fig.colorbar(
        im,
        ax=panels[2],
        fraction=0.046,
        pad=0.04,
    )

    r2 = outputs[
        "R2app_s-1"
    ]

    vmin, vmax = percentile_limits(
        r2,
        valid,
    )

    im = panels[3].imshow(
        np.rot90(
            r2[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    panels[3].set_title(
        "R2app [s⁻¹]"
    )

    fig.colorbar(
        im,
        ax=panels[3],
        fraction=0.046,
        pad=0.04,
    )

    t2 = outputs[
        "T2app_ms"
    ]

    vmin, vmax = percentile_limits(
        t2,
        valid,
    )

    im = panels[4].imshow(
        np.rot90(
            t2[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    panels[4].set_title(
        "T2app [ms]"
    )

    fig.colorbar(
        im,
        ax=panels[4],
        fraction=0.046,
        pad=0.04,
    )

    panels[5].imshow(
        np.rot90(
            outputs[
                "fit_valid_mask"
            ][:, :, z]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    panels[5].contour(
        np.rot90(
            brain_mask[
                :, :, z
            ]
        ),
        levels=[0.5],
        linewidths=0.7,
    )

    panels[5].set_title(
        "Valid dual-echo estimate"
    )

    panels[6].imshow(
        np.rot90(
            outputs[
                "nondecay_mask"
            ][:, :, z]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    panels[6].contour(
        np.rot90(
            brain_mask[
                :, :, z
            ]
        ),
        levels=[0.5],
        linewidths=0.7,
    )

    panels[6].set_title(
        "Non-decay mask\nS2 ≥ S1"
    )

    r2_values = r2[
        valid
        & np.isfinite(r2)
    ]

    if r2_values.size:
        lo, hi = np.percentile(
            r2_values,
            [1, 99],
        )

        hist_values = r2_values[
            (r2_values >= lo)
            & (r2_values <= hi)
        ]

        panels[7].hist(
            hist_values,
            bins=100,
        )

        panels[7].axvline(
            np.median(
                r2_values
            ),
            linewidth=1,
        )

    panels[7].set_xlabel(
        "R2app [s⁻¹]"
    )

    panels[7].set_ylabel(
        "Voxel count"
    )

    panels[7].set_title(
        "R2app distribution\n"
        "primary brain-mask domain"
    )

    t2_values = t2[
        valid
        & np.isfinite(t2)
    ]

    if t2_values.size:
        lo, hi = np.percentile(
            t2_values,
            [1, 99],
        )

        hist_values = t2_values[
            (t2_values >= lo)
            & (t2_values <= hi)
        ]

        panels[8].hist(
            hist_values,
            bins=100,
        )

        panels[8].axvline(
            np.median(
                t2_values
            ),
            linewidth=1,
        )

    panels[8].set_xlabel(
        "T2app [ms]"
    )

    panels[8].set_ylabel(
        "Voxel count"
    )

    panels[8].set_title(
        "T2app distribution\n"
        "primary brain-mask domain"
    )

    for ax in panels[:7]:
        ax.axis("off")

    n_brain = int(
        np.count_nonzero(
            brain_mask
        )
    )

    n_ero1 = int(
        np.count_nonzero(
            brain_mask_ero1
        )
    )

    fig.suptitle(
        f"{TAG} high-resolution dual-echo T2/R2 QC\n"
        f"{case}, z={z}, "
        f"TE={te1_s * 1000:.1f}/{te2_s * 1000:.1f} ms\n"
        f"brain={n_brain:,} vox, "
        f"ero1={n_ero1:,} vox"
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

    plt.close(fig)


# -------------------------------------------------------------------------
# Case processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    overwrite: bool,
    dry_run: bool,
    min_signal: float,
    r2_min: float,
    r2_max: float,
    affine_atol: float,
) -> Dict[str, Any]:
    start = now_iso()

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    selection = load_rlx01_dual_tse(
        paths,
        case,
    )

    masks = load_rlx05a_masks(
        paths,
        case,
    )

    dual = selection[
        "dual_tse"
    ]

    echoes = selection[
        "echoes"
    ]

    echo1 = echoes[0]
    echo2 = echoes[1]

    image1_path = echo1[
        "image"
    ]

    image2_path = echo2[
        "image"
    ]

    te1_s = echo1[
        "echo_time_s"
    ]

    te2_s = echo2[
        "echo_time_s"
    ]

    print(
        f"  Echo 1: {image1_path}"
    )

    print(
        f"  Echo 2: {image2_path}"
    )

    print(
        f"  TE: "
        f"{te1_s * 1000:.3f} / "
        f"{te2_s * 1000:.3f} ms"
    )

    print(
        "  Primary analysis mask: "
        f"{masks['brain_mask_path']}"
    )

    print(
        "  Secondary ero1 mask: "
        f"{masks['brain_mask_ero1_path']}"
    )

    image1 = nib.load(
        str(image1_path)
    )

    image2 = nib.load(
        str(image2_path)
    )

    brain_mask_img = nib.load(
        str(
            masks[
                "brain_mask_path"
            ]
        )
    )

    brain_mask_ero1_img = nib.load(
        str(
            masks[
                "brain_mask_ero1_path"
            ]
        )
    )

    validate_geometry(
        image1,
        image2,
        "echo 1",
        "echo 2",
        affine_atol,
    )

    validate_geometry(
        image1,
        brain_mask_img,
        "echo 1",
        "rlx05a brain mask",
        affine_atol,
    )

    validate_geometry(
        image1,
        brain_mask_ero1_img,
        "echo 1",
        "rlx05a ero1 mask",
        affine_atol,
    )

    voxel_size = (
        image1.header
        .get_zooms()[:3]
    )

    print(
        f"  Native shape: "
        f"{image1.shape}"
    )

    print(
        "  Native voxel size: "
        f"{voxel_size[0]:.3f} x "
        f"{voxel_size[1]:.3f} x "
        f"{voxel_size[2]:.3f} mm"
    )

    brain_mask = (
        np.asarray(
            brain_mask_img.dataobj
        ) > 0
    )

    brain_mask_ero1 = (
        np.asarray(
            brain_mask_ero1_img.dataobj
        ) > 0
    )

    if np.any(
        brain_mask_ero1
        & ~brain_mask
    ):
        raise ValueError(
            "rlx05a ero1 mask contains voxels outside "
            "the primary brain mask."
        )

    print(
        f"  Brain-mask voxels: "
        f"{np.count_nonzero(brain_mask)}"
    )

    print(
        f"  Ero1 voxels: "
        f"{np.count_nonzero(brain_mask_ero1)}"
    )

    out_dir = rlx05_dir(
        paths,
        case,
    )

    metadata_path = (
        out_dir
        / f"rlx05_{case}_metadata.json"
    )

    if (
        metadata_path.exists()
        and not overwrite
        and not dry_run
    ):
        raise FileExistsError(
            f"rlx05 output already exists:\n"
            f"  {metadata_path}\n"
            "Use --overwrite to replace it."
        )

    if dry_run:
        return {
            "case": case,
            "success": True,
            "dry_run": True,
            "echo1": str(
                image1_path
            ),
            "echo2": str(
                image2_path
            ),
            "brain_mask": str(
                masks[
                    "brain_mask_path"
                ]
            ),
            "brain_mask_ero1": str(
                masks[
                    "brain_mask_ero1_path"
                ]
            ),
            "TE1_ms":
                te1_s * 1000.0,
            "TE2_ms":
                te2_s * 1000.0,
            "shape":
                list(image1.shape),
            "voxel_size_mm":
                list(voxel_size),
        }

    s1 = np.asarray(
        image1.dataobj,
        dtype=np.float32,
    )

    s2 = np.asarray(
        image2.dataobj,
        dtype=np.float32,
    )

    outputs = calculate_dual_echo_maps(
        s1=s1,
        s2=s2,
        brain_mask=brain_mask,
        te1_s=te1_s,
        te2_s=te2_s,
        min_signal=min_signal,
        r2_min=r2_min,
        r2_max=r2_max,
    )

    paths_out = output_paths(
        out_dir,
        case,
    )

    for name, path in paths_out.items():
        dtype = (
            np.uint8
            if name.endswith("_mask")
            else np.float32
        )

        save_nifti(
            outputs[name],
            image1,
            path,
            dtype,
        )

    summary = summarize_case(
        case=case,
        outputs=outputs,
        brain_mask=brain_mask,
        brain_mask_ero1=
            brain_mask_ero1,
        te1_s=te1_s,
        te2_s=te2_s,
    )

    summary_csv = (
        out_dir
        / f"rlx05_{case}_summary.csv"
    )

    write_csv(
        summary_csv,
        [summary],
    )

    qc_png = (
        out_dir
        / f"rlx05_{case}_qc.png"
    )

    make_qc_png(
        case=case,
        s1=s1,
        s2=s2,
        brain_mask=brain_mask,
        brain_mask_ero1=
            brain_mask_ero1,
        outputs=outputs,
        te1_s=te1_s,
        te2_s=te2_s,
        path=qc_png,
    )

    end = now_iso()

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
            "command": " ".join(sys.argv),
            "dry_run": dry_run,
        },

        "scientific_role": {
            "estimate":
                (
                    "apparent dual-echo "
                    "transverse relaxation"
                ),

            "labels": [
                "R2app",
                "T2app",
            ],

            "quantitative_reference":
                (
                    "rlx03 drop1 12-echo "
                    "mono-exponential estimate"
                ),

            "interpretation":
                (
                    "The dual-echo result is retained "
                    "as an apparent relaxation estimate "
                    "until validated against the "
                    "12-echo drop1 reference."
                ),
        },

        "upstream": {
            "rlx01_selected_inputs":
                str(
                    selection[
                        "metadata_path"
                    ]
                ),

            "rlx05a_metadata":
                str(
                    masks[
                        "metadata_path"
                    ]
                ),

            "brain_mask":
                str(
                    masks[
                        "brain_mask_path"
                    ]
                ),

            "brain_mask_ero1":
                str(
                    masks[
                        "brain_mask_ero1_path"
                    ]
                ),

            "acquisition_id":
                dual.get(
                    "acquisition_id"
                ),

            "classification":
                dual.get(
                    "classification"
                ),

            "echo1_image":
                str(
                    image1_path
                ),

            "echo2_image":
                str(
                    image2_path
                ),

            "echo1_json":
                echo1.get(
                    "json_sidecar"
                ),

            "echo2_json":
                echo2.get(
                    "json_sidecar"
                ),

            "TE1_s":
                te1_s,

            "TE2_s":
                te2_s,

            "TE1_ms":
                te1_s * 1000.0,

            "TE2_ms":
                te2_s * 1000.0,

            "deltaTE_ms":
                (
                    te2_s - te1_s
                ) * 1000.0,

            "native_shape":
                list(
                    image1.shape
                ),

            "native_voxel_size_mm":
                [
                    float(v)
                    for v in voxel_size
                ],

            "series_number":
                dual.get(
                    "series_number"
                ),

            "series_description":
                dual.get(
                    "series_description"
                ),

            "protocol_name":
                dual.get(
                    "protocol_name"
                ),
        },

        "masking": {
            "primary_fitting_domain":
                "rlx05a HD-BET brain mask",

            "primary_mask_voxels":
                int(
                    np.count_nonzero(
                        brain_mask
                    )
                ),

            "secondary_summary_domain":
                (
                    "rlx05a one-voxel-eroded "
                    "brain mask"
                ),

            "ero1_mask_voxels":
                int(
                    np.count_nonzero(
                        brain_mask_ero1
                    )
                ),

            "rationale":
                (
                    "The native HD-BET mask preserves "
                    "the full 1-mm cortical boundary "
                    "for high-resolution R2app fitting. "
                    "The one-voxel-eroded mask is retained "
                    "for conservative secondary statistics "
                    "and sensitivity analyses."
                ),
        },

        "model": {
            "equation":
                (
                    "S(TE) = S0 * "
                    "exp(-R2app * TE)"
                ),

            "R2app_equation":
                (
                    "ln(S1/S2) / "
                    "(TE2-TE1)"
                ),

            "T2app_equation":
                "1 / R2app",

            "fit_method":
                (
                    "analytical two-point "
                    "mono-exponential estimate"
                ),

            "goodness_of_fit":
                (
                    "Not defined: two measurements "
                    "exactly determine the two model "
                    "parameters S0 and R2app."
                ),

            "minimum_signal":
                min_signal,

            "R2app_bounds_s-1": [
                r2_min,
                r2_max,
            ],

            "validity": {
                "candidate":
                    (
                        "inside primary HD-BET brain "
                        "mask, finite S1 and S2, and "
                        "both signals above "
                        "minimum_signal"
                    ),

                "positive_decay":
                    "S1 > S2",

                "valid":
                    (
                        "candidate with S1>S2, finite "
                        "analytical solution, and "
                        "R2app within configured bounds"
                    ),

                "nondecay":
                    (
                        "candidate voxel with "
                        "S2 >= S1"
                    ),

                "out_of_range":
                    (
                        "finite positive decay but "
                        "R2app outside configured bounds"
                    ),
            },
        },

        "summary":
            summary,

        "outputs": {
            key: str(value)
            for key, value
            in paths_out.items()
        },

        "summary_csv":
            str(summary_csv),

        "qc_png":
            str(qc_png),
    }

    write_json(
        metadata_path,
        metadata,
    )

    print(
        "  Primary brain domain:"
    )

    print(
        f"    candidate: "
        f"{summary['brain_candidate_voxels']} / "
        f"{summary['brain_domain_voxels']} "
        f"({100 * summary['brain_candidate_fraction_of_domain']:.2f}%)"
    )

    print(
        f"    valid: "
        f"{summary['brain_valid_voxels']} "
        f"({100 * summary['brain_valid_fraction_of_candidate']:.2f}%)"
    )

    print(
        f"    non-decay: "
        f"{summary['brain_nondecay_voxels']} "
        f"({100 * summary['brain_nondecay_fraction_of_candidate']:.2f}%)"
    )

    if (
        summary[
            "brain_R2app_s-1_median"
        ]
        is not None
    ):
        print(
            f"    median R2app: "
            f"{summary['brain_R2app_s-1_median']:.3f} s^-1"
        )

    if (
        summary[
            "brain_T2app_ms_median"
        ]
        is not None
    ):
        print(
            f"    median T2app: "
            f"{summary['brain_T2app_ms_median']:.3f} ms"
        )

    print(
        "  Conservative ero1 domain:"
    )

    print(
        f"    valid: "
        f"{summary['ero1_valid_voxels']}"
    )

    if (
        summary[
            "ero1_R2app_s-1_median"
        ]
        is not None
    ):
        print(
            f"    median R2app: "
            f"{summary['ero1_R2app_s-1_median']:.3f} s^-1"
        )

    if (
        summary[
            "ero1_T2app_ms_median"
        ]
        is not None
    ):
        print(
            f"    median T2app: "
            f"{summary['ero1_T2app_ms_median']:.3f} ms"
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
            "APF rlx05: high-resolution dual-echo "
            "apparent T2/R2 estimation."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rlx05 outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and validate inputs "
            "without calculating maps."
        ),
    )

    parser.add_argument(
        "--min-signal",
        type=float,
        default=DEFAULT_MIN_SIGNAL,
        help=(
            "Minimum signal required independently "
            "in both echoes "
            f"(default: {DEFAULT_MIN_SIGNAL})."
        ),
    )

    parser.add_argument(
        "--r2-min",
        type=float,
        default=DEFAULT_R2_MIN,
        help=(
            "Minimum accepted R2app in s^-1 "
            f"(default: {DEFAULT_R2_MIN})."
        ),
    )

    parser.add_argument(
        "--r2-max",
        type=float,
        default=DEFAULT_R2_MAX,
        help=(
            "Maximum accepted R2app in s^-1 "
            f"(default: {DEFAULT_R2_MAX})."
        ),
    )

    parser.add_argument(
        "--affine-atol",
        type=float,
        default=DEFAULT_AFFINE_ATOL,
        help=(
            "Absolute tolerance for echo/mask "
            "affine comparison "
            f"(default: {DEFAULT_AFFINE_ATOL})."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.min_signal < 0:
        raise SystemExit(
            "--min-signal must be >= 0."
        )

    if args.r2_min <= 0:
        raise SystemExit(
            "--r2-min must be > 0."
        )

    if args.r2_max <= args.r2_min:
        raise SystemExit(
            "Require --r2-max > --r2-min."
        )

    if args.affine_atol <= 0:
        raise SystemExit(
            "--affine-atol must be > 0."
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
                min_signal=args.min_signal,
                r2_min=args.r2_min,
                r2_max=args.r2_max,
                affine_atol=args.affine_atol,
            )

            if args.dry_run:
                cohort_rows.append(
                    {
                        "case": case,
                        "success": True,
                        "dry_run": True,
                        "echo1":
                            metadata[
                                "echo1"
                            ],
                        "echo2":
                            metadata[
                                "echo2"
                            ],
                        "brain_mask":
                            metadata[
                                "brain_mask"
                            ],
                        "TE1_ms":
                            metadata[
                                "TE1_ms"
                            ],
                        "TE2_ms":
                            metadata[
                                "TE2_ms"
                            ],
                    }
                )

            else:
                cohort_rows.append(
                    {
                        "success": True,
                        **metadata[
                            "summary"
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
                    "case": case,
                    "success": False,
                    "error": str(exc),
                }
            )

            if len(cases) == 1:
                raise

    if not args.dry_run:
        cohort_out = summary_dir(
            paths
        )

        cohort_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            cohort_out
            / "rlx05_cohort_summary.csv",
            cohort_rows,
        )

        write_json(
            cohort_out
            / "rlx05_cohort_metadata.json",
            {
                "script": SCRIPT,
                "tag": TAG,
                "version": VERSION,
                "created": now_iso(),
                "command": " ".join(
                    sys.argv
                ),
                "n_cases_requested":
                    len(cases),
                "n_failures":
                    failures,
                "rows":
                    cohort_rows,
            },
        )

    if failures:
        raise SystemExit(
            f"rlx05 completed with "
            f"{failures} failed case(s)."
        )


if __name__ == "__main__":
    main()