#!/usr/bin/env python3
"""
APF GRE04d v0.1.0
Odd/even echo phase diagnostic.

Purpose
-------
Test whether multi-echo GRE phase contains a systematic alternating odd/even
echo component and determine whether omission of that component materially
biases the GRE04b WLS total-field estimate.

The production GRE04b model is:

    phase = beta0 + omega * TE

GRE04d additionally fits:

    phase = beta0 + omega * TE + beta_OE * parity

with:

    parity = +1 for echoes 1,3,5,...
             -1 for echoes 2,4,6,...

The diagnostic deliberately reproduces GRE04b's:
- ROMEO-unwrapped phase input
- GRE02 magnitude input
- GRE03 effective mask
- magnitude-squared weights
- weight-floor rule
- minimum-valid-echo rule

GRE04d does NOT overwrite GRE04b and is not a downstream production module.

Outputs
-------
Per case, under GRE/04d_odd_even_diagnostic/:

- parity-aware total field [Hz]
- parity-aware omega [rad/s]
- odd/even nuisance coefficient beta_OE [rad]
- parity-aware weighted phase RMSE [rad]
- relative weighted-RMSE reduction
- parity-aware valid-fit mask
- parity-aware minus current GRE04b field [Hz]
- odd-minus-even residual-template map from current GRE04b residuals [rad]
- optional standardized QC figure [PNG/PDF]
- QC JSON

Cohort summaries:

- derivatives/summary/GRE/gre04d_odd_even_echo_diagnostic_qc.csv
- derivatives/summary/GRE/gre04d_odd_even_echo_diagnostic_run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "0.1.0"
MODULE_NAME = "gre04d"
METHOD_NAME = "odd_even_echo_phase_diagnostic"

OUTPUT_SUBDIR = Path("GRE") / "04d_odd_even_diagnostic"
SUMMARY_SUBDIR = Path("derivatives") / "summary" / "GRE"

DEFAULT_MIN_VALID_ECHOES = 3
DEFAULT_CHUNK_VOXELS = 200000
DEFAULT_WEIGHT_FLOOR_FRACTION = 1e-6


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser().resolve()

    if os.environ.get("APF_ROOT"):
        return Path(os.environ["APF_ROOT"]).expanduser().resolve()

    return Path.cwd().resolve()


def list_cases(root: Path) -> list[str]:
    deriv = root / "derivatives"
    return [
        p.name
        for p in sorted(deriv.glob("ANISO_*_InS"))
        if p.is_dir()
    ]


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def input_paths(root: Path, case: str) -> dict[str, Path]:
    base = root / "derivatives" / case / "GRE"

    return {
        "phase": require_file(
            base / "03_romeo_unwrap" /
            f"{case}_GRE_phase_unwrapped_romeo_rad_4d.nii.gz",
            "GRE03 unwrapped phase",
        ),
        "magnitude": require_file(
            base / "phase_preparation_gre02" /
            f"gre02_{case}_magnitude_4d.nii.gz",
            "GRE02 magnitude",
        ),
        "mask": require_file(
            base / "03_romeo_unwrap" /
            f"{case}_GRE_romeo_mask.nii.gz",
            "GRE03 ROMEO mask",
        ),
        "echo_times": require_file(
            base / "phase_preparation_gre02" /
            f"gre02_{case}_echo_times_ms.txt",
            "GRE02 echo times",
        ),
        "current_field": require_file(
            base / "04b_field_wls" /
            f"{case}_GRE04b_total_field_hz.nii.gz",
            "GRE04b total field",
        ),
        "current_wrmse": require_file(
            base / "04b_field_wls" /
            f"{case}_GRE04b_phase_weighted_rmse_rad.nii.gz",
            "GRE04b weighted RMSE",
        ),
        "current_valid": require_file(
            base / "04b_field_wls" /
            f"{case}_GRE04b_valid_fit_mask.nii.gz",
            "GRE04b valid-fit mask",
        ),
        "current_residuals": require_file(
            base / "04b_field_wls" /
            f"{case}_GRE04b_phase_residuals_rad_4d.nii.gz",
            "GRE04b residual series",
        ),
    }


def output_paths(root: Path, case: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case / OUTPUT_SUBDIR
    prefix = f"{case}_GRE04d"

    return {
        "dir": outdir,
        "field_parity_hz": outdir / f"{prefix}_parity_aware_total_field_hz.nii.gz",
        "omega_parity": outdir / f"{prefix}_parity_aware_omega_rad_s.nii.gz",
        "beta_oe": outdir / f"{prefix}_beta_odd_even_rad.nii.gz",
        "wrmse_parity": outdir / f"{prefix}_parity_aware_weighted_rmse_rad.nii.gz",
        "wrmse_reduction": outdir / f"{prefix}_relative_weighted_rmse_reduction.nii.gz",
        "valid": outdir / f"{prefix}_parity_aware_valid_fit_mask.nii.gz",
        "delta_field": outdir / f"{prefix}_parity_minus_GRE04b_field_hz.nii.gz",
        "oe_template": outdir / f"{prefix}_odd_minus_even_residual_template_rad.nii.gz",
        "plot_png": outdir / f"{prefix}_odd_even_diagnostic.png",
        "plot_pdf": outdir / f"{prefix}_odd_even_diagnostic.pdf",
        "qc": outdir / f"{prefix}_qc.json",
    }


def load_3d(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {data.shape}: {path}")
    return img, data


def validate_geometry(
    reference: nib.Nifti1Image,
    other: nib.Nifti1Image,
    label: str,
) -> None:
    if tuple(reference.shape[:3]) != tuple(other.shape[:3]):
        raise ValueError(
            f"{label} shape mismatch: {other.shape} vs {reference.shape}"
        )

    if not np.allclose(
        reference.affine,
        other.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise ValueError(f"{label} affine mismatch")


def save_nifti(
    data: np.ndarray,
    reference: nib.Nifti1Image,
    path: Path,
    dtype,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        header,
    )
    nib.save(img, path)


def fit_spatial_plane(
    data: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | str]:
    finite = mask & np.isfinite(data)

    if np.count_nonzero(finite) < 4:
        return {
            "intercept": float("nan"),
            "x_slope": float("nan"),
            "y_slope": float("nan"),
            "z_slope": float("nan"),
            "r2": float("nan"),
            "dominant_axis": "none",
        }

    coords = np.indices(data.shape, dtype=np.float64)
    xyz = []

    for axis in range(3):
        c = coords[axis][finite]
        lo = float(c.min())
        hi = float(c.max())

        if hi > lo:
            c = 2.0 * (c - lo) / (hi - lo) - 1.0
        else:
            c = np.zeros_like(c)

        xyz.append(c)

    y = data[finite].astype(np.float64)

    X = np.column_stack([
        np.ones(y.size),
        xyz[0],
        xyz[1],
        xyz[2],
    ])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0.0
        else float("nan")
    )

    slopes = beta[1:]
    dominant = int(np.argmax(np.abs(slopes)))

    return {
        "intercept": float(beta[0]),
        "x_slope": float(beta[1]),
        "y_slope": float(beta[2]),
        "z_slope": float(beta[3]),
        "r2": float(r2),
        "dominant_axis": ("voxel-x", "voxel-y", "voxel-z")[dominant],
    }


def finite_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "median": float("nan"),
            "mae": float("nan"),
            "rms": float("nan"),
            "p01": float("nan"),
            "p99": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }

    return {
        "median": float(np.median(x)),
        "mae": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
        "p01": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def fit_current_and_parity_wls(
    phase_4d: np.ndarray,
    magnitude_4d: np.ndarray,
    mask_3d: np.ndarray,
    tes_s: np.ndarray,
    *,
    min_valid_echoes: int,
    chunk_voxels: int,
    weight_floor_fraction: float,
) -> dict[str, np.ndarray]:
    """
    Fit both:
        phase = b0 + omega*TE
    and:
        phase = b0 + omega*TE + beta_OE*parity

    The two-parameter fit reproduces GRE04b's production WLS equations.
    """
    spatial_shape = phase_4d.shape[:3]
    n_echoes = phase_4d.shape[3]
    n_voxels = int(np.prod(spatial_shape))

    y_all = phase_4d.reshape(n_voxels, n_echoes)
    mag_all = magnitude_4d.reshape(n_voxels, n_echoes)
    mask_flat = mask_3d.reshape(n_voxels)

    current_omega = np.zeros(n_voxels, dtype=np.float32)
    current_wrmse = np.zeros(n_voxels, dtype=np.float32)
    current_valid = np.zeros(n_voxels, dtype=np.uint8)

    parity_omega = np.zeros(n_voxels, dtype=np.float32)
    beta_oe = np.zeros(n_voxels, dtype=np.float32)
    parity_wrmse = np.zeros(n_voxels, dtype=np.float32)
    parity_valid = np.zeros(n_voxels, dtype=np.uint8)

    masked_indices = np.flatnonzero(mask_flat)

    t = tes_s.astype(np.float64)
    parity = np.where(
        np.arange(n_echoes) % 2 == 0,
        1.0,
        -1.0,
    ).astype(np.float64)

    for start in range(0, masked_indices.size, chunk_voxels):
        idx = masked_indices[start:start + chunk_voxels]

        y = np.asarray(y_all[idx, :], dtype=np.float64)
        mag = np.asarray(mag_all[idx, :], dtype=np.float64)

        finite = np.isfinite(y) & np.isfinite(mag) & (mag > 0)
        n = finite.sum(axis=1).astype(np.int32)

        raw_weights = np.where(finite, mag * mag, 0.0)
        max_weight = np.max(raw_weights, axis=1, keepdims=True)
        floor = weight_floor_fraction * max_weight
        weights = np.where(
            finite,
            np.maximum(raw_weights, floor),
            0.0,
        )

        y0 = np.where(finite, y, 0.0)

        # ----------------------------------------------------------
        # Reproduce current GRE04b 2-parameter WLS fit.
        # ----------------------------------------------------------
        sw = np.sum(weights, axis=1)
        swt = np.sum(weights * t[None, :], axis=1)
        swy = np.sum(weights * y0, axis=1)
        swtt = np.sum(weights * t[None, :] * t[None, :], axis=1)
        swty = np.sum(weights * y0 * t[None, :], axis=1)

        denom = sw * swtt - swt * swt

        good_current = (
            (n >= min_valid_echoes)
            & np.isfinite(sw)
            & (sw > 0)
            & np.isfinite(denom)
            & (denom > 0)
        )

        b1 = np.zeros(idx.size, dtype=np.float64)
        b0 = np.zeros(idx.size, dtype=np.float64)

        b1[good_current] = (
            sw[good_current] * swty[good_current]
            - swt[good_current] * swy[good_current]
        ) / denom[good_current]

        b0[good_current] = (
            swy[good_current]
            - b1[good_current] * swt[good_current]
        ) / sw[good_current]

        fitted_current = (
            b0[:, None]
            + b1[:, None] * t[None, :]
        )

        resid_current = np.where(
            finite,
            y - fitted_current,
            0.0,
        )

        weighted_sse_current = np.sum(
            weights * resid_current * resid_current,
            axis=1,
        )

        wrmse_current = np.sqrt(
            np.divide(
                weighted_sse_current,
                sw,
                out=np.zeros_like(weighted_sse_current),
                where=sw > 0,
            )
        )

        good_current &= (
            np.isfinite(b0)
            & np.isfinite(b1)
            & np.isfinite(wrmse_current)
        )

        current_omega[idx[good_current]] = (
            b1[good_current].astype(np.float32)
        )
        current_wrmse[idx[good_current]] = (
            wrmse_current[good_current].astype(np.float32)
        )
        current_valid[idx[good_current]] = 1

        # ----------------------------------------------------------
        # Parity-aware 3-parameter WLS fit.
        #
        # Normal equations for design columns:
        #     [1, TE, parity]
        #
        # Requiring at least one odd and one even valid echo prevents
        # parity from becoming collinear with the intercept.
        # ----------------------------------------------------------
        odd_valid = np.any(
            finite[:, np.arange(n_echoes) % 2 == 0],
            axis=1,
        )
        even_valid = np.any(
            finite[:, np.arange(n_echoes) % 2 == 1],
            axis=1,
        )

        sws = np.sum(weights * parity[None, :], axis=1)
        swts = np.sum(
            weights * t[None, :] * parity[None, :],
            axis=1,
        )
        swsy = np.sum(
            weights * y0 * parity[None, :],
            axis=1,
        )

        # parity^2 == 1, therefore A22 == sw.
        A = np.empty((idx.size, 3, 3), dtype=np.float64)

        A[:, 0, 0] = sw
        A[:, 0, 1] = swt
        A[:, 0, 2] = sws

        A[:, 1, 0] = swt
        A[:, 1, 1] = swtt
        A[:, 1, 2] = swts

        A[:, 2, 0] = sws
        A[:, 2, 1] = swts
        A[:, 2, 2] = sw

        B = np.column_stack([
            swy,
            swty,
            swsy,
        ])

        candidate = (
            (n >= min_valid_echoes)
            & odd_valid
            & even_valid
            & np.isfinite(sw)
            & (sw > 0)
        )

        cand = np.flatnonzero(candidate)

        if cand.size > 0:
            A_c = A[cand]
            B_c = B[cand]

            det = np.linalg.det(A_c)
            nonsingular = np.isfinite(det) & (det != 0.0)

            solve_idx = cand[nonsingular]

            if solve_idx.size > 0:
                beta = np.linalg.solve(
                    A[solve_idx],
                    B[solve_idx, :, None],
                )[..., 0]

                finite_beta = np.all(np.isfinite(beta), axis=1)

                solve_idx = solve_idx[finite_beta]
                beta = beta[finite_beta]

                if solve_idx.size > 0:
                    fitted = (
                        beta[:, 0, None]
                        + beta[:, 1, None] * t[None, :]
                        + beta[:, 2, None] * parity[None, :]
                    )

                    resid = np.where(
                        finite[solve_idx],
                        y[solve_idx] - fitted,
                        0.0,
                    )

                    weighted_sse = np.sum(
                        weights[solve_idx] * resid * resid,
                        axis=1,
                    )

                    wrmse = np.sqrt(
                        np.divide(
                            weighted_sse,
                            sw[solve_idx],
                            out=np.zeros_like(weighted_sse),
                            where=sw[solve_idx] > 0,
                        )
                    )

                    finite_result = (
                        np.isfinite(wrmse)
                        & np.all(np.isfinite(beta), axis=1)
                    )

                    solve_idx = solve_idx[finite_result]
                    beta = beta[finite_result]
                    wrmse = wrmse[finite_result]

                    global_idx = idx[solve_idx]

                    parity_omega[global_idx] = (
                        beta[:, 1].astype(np.float32)
                    )
                    beta_oe[global_idx] = (
                        beta[:, 2].astype(np.float32)
                    )
                    parity_wrmse[global_idx] = (
                        wrmse.astype(np.float32)
                    )
                    parity_valid[global_idx] = 1

    return {
        "current_omega_rad_s": current_omega.reshape(spatial_shape),
        "current_weighted_rmse_rad": current_wrmse.reshape(spatial_shape),
        "current_valid_fit_mask": current_valid.reshape(spatial_shape),
        "parity_omega_rad_s": parity_omega.reshape(spatial_shape),
        "beta_oe_rad": beta_oe.reshape(spatial_shape),
        "parity_weighted_rmse_rad": parity_wrmse.reshape(spatial_shape),
        "parity_valid_fit_mask": parity_valid.reshape(spatial_shape),
    }


def current_residual_diagnostic(
    residuals: np.ndarray,
    mask: np.ndarray,
) -> dict:
    n_echoes = residuals.shape[3]

    odd_idx = np.arange(0, n_echoes, 2)
    even_idx = np.arange(1, n_echoes, 2)

    if odd_idx.size != even_idx.size:
        raise ValueError(
            "Odd/even residual diagnostic currently requires paired echoes"
        )

    odd_mean = np.mean(residuals[..., odd_idx], axis=3)
    even_mean = np.mean(residuals[..., even_idx], axis=3)
    template = odd_mean - even_mean

    echo_rows = []

    template_flat = template[mask].astype(np.float64)

    for e in range(n_echoes):
        x = residuals[..., e][mask].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(template_flat)

        xf = x[finite]
        tf = template_flat[finite]

        if xf.size >= 2 and np.std(xf) > 0 and np.std(tf) > 0:
            corr = float(np.corrcoef(xf, tf)[0, 1])
        else:
            corr = float("nan")

        med = float(np.median(xf))
        mad = float(np.median(np.abs(xf - med)))
        sd = float(np.std(xf))

        echo_rows.append({
            "echo": e + 1,
            "parity": "odd" if e % 2 == 0 else "even",
            "median_rad": med,
            "mad_rad": mad,
            "sd_rad": sd,
            "template_correlation": corr,
        })

    odd_corr = np.asarray([
        r["template_correlation"]
        for r in echo_rows
        if r["parity"] == "odd"
    ])

    even_corr = np.asarray([
        r["template_correlation"]
        for r in echo_rows
        if r["parity"] == "even"
    ])

    strict_sign_alternation = bool(
        np.all(odd_corr > 0)
        and np.all(even_corr < 0)
    )

    template_plane = fit_spatial_plane(template, mask)

    return {
        "template_rad": template.astype(np.float32),
        "template_summary": finite_summary(template[mask]),
        "template_plane": template_plane,
        "echoes": echo_rows,
        "strict_template_sign_alternation": strict_sign_alternation,
        "median_odd_template_correlation": float(np.median(odd_corr)),
        "median_even_template_correlation": float(np.median(even_corr)),
    }


def reproducible_sample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size != y.size:
        raise ValueError("Sampling arrays differ in length")

    if x.size <= maximum:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=maximum, replace=False)
    return x[idx], y[idx]


def write_plot(
    paths: dict[str, Path],
    case: str,
    current_field: np.ndarray,
    parity_field: np.ndarray,
    delta_field: np.ndarray,
    beta_oe: np.ndarray,
    common: np.ndarray,
    metrics: dict,
    *,
    maximum_voxels: int,
    seed: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--write-plots requested but matplotlib unavailable"
        ) from exc

    support_per_slice = np.count_nonzero(common, axis=(0, 1))
    z = int(np.argmax(support_per_slice))

    delta_slice = np.where(
        common[:, :, z],
        delta_field[:, :, z],
        np.nan,
    )

    beta_slice = np.where(
        common[:, :, z],
        beta_oe[:, :, z],
        np.nan,
    )

    delta_values = delta_field[common].astype(np.float64)
    beta_values = beta_oe[common].astype(np.float64)

    dmax = float(np.percentile(np.abs(delta_values), 99))
    bmax = float(np.percentile(np.abs(beta_values), 99))

    dmax = max(dmax, np.finfo(float).eps)
    bmax = max(bmax, np.finfo(float).eps)

    x = current_field[common].astype(np.float64)
    y = parity_field[common].astype(np.float64)

    xs, ys = reproducible_sample(
        x,
        y,
        maximum=maximum_voxels,
        seed=seed,
    )

    low = float(
        min(
            np.percentile(xs, 0.5),
            np.percentile(ys, 0.5),
        )
    )
    high = float(
        max(
            np.percentile(xs, 99.5),
            np.percentile(ys, 99.5),
        )
    )

    relative_reduction = np.zeros_like(delta_field, dtype=np.float32)

    # Reconstructed from scalar arrays already written into metrics is not
    # sufficient for a map, therefore panel D is the field scatter.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(
        np.rot90(delta_slice),
        cmap="RdBu_r",
        vmin=-dmax,
        vmax=dmax,
        interpolation="nearest",
    )
    axes[0, 0].set_title("A  Parity-aware − GRE04b field")
    axes[0, 0].set_axis_off()

    cb0 = fig.colorbar(
        im0,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.04,
    )
    cb0.set_label("Field difference (Hz)")

    im1 = axes[0, 1].imshow(
        np.rot90(beta_slice),
        cmap="RdBu_r",
        vmin=-bmax,
        vmax=bmax,
        interpolation="nearest",
    )
    axes[0, 1].set_title("B  Odd/even nuisance coefficient")
    axes[0, 1].set_axis_off()

    cb1 = fig.colorbar(
        im1,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
    )
    cb1.set_label(r"$\beta_{OE}$ (rad)")

    axes[1, 0].hist(
        delta_values,
        bins=120,
    )
    axes[1, 0].axvline(0.0, linewidth=1)
    axes[1, 0].set_xlabel("Parity-aware − GRE04b field (Hz)")
    axes[1, 0].set_ylabel("Voxels")
    axes[1, 0].set_title("C  Field-difference distribution")

    ax = axes[1, 1]
    ax.scatter(
        xs,
        ys,
        s=2,
        alpha=0.15,
        rasterized=True,
    )
    ax.plot(
        [low, high],
        [low, high],
        linewidth=1.2,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Current GRE04b total field (Hz)")
    ax.set_ylabel("Parity-aware total field (Hz)")
    ax.set_title("D  Current versus parity-aware field")

    annotation = (
        f"r = {metrics['field_pearson_r']:.6f}\n"
        f"RMS Δf = {metrics['delta_field_rms_hz']:.3f} Hz\n"
        f"median RMSE reduction = "
        f"{metrics['median_relative_wrmse_reduction']*100:.1f}%\n"
        f"Δf plane R² = {metrics['delta_field_plane_r2']:.3f}"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    fig.suptitle(
        f"{case}: GRE04 odd/even echo phase diagnostic",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        paths["plot_png"],
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        paths["plot_pdf"],
        bbox_inches="tight",
    )
    plt.close(fig)


def compare_case(
    root: Path,
    case: str,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()

    inp = input_paths(root, case)
    out = output_paths(root, case)

    if out["qc"].exists() and not args.force:
        raise FileExistsError(
            f"Output exists; use --force: {out['qc']}"
        )

    phase_img = nib.load(inp["phase"])
    magnitude_img = nib.load(inp["magnitude"])
    mask_img = nib.load(inp["mask"])
    current_field_img = nib.load(inp["current_field"])
    current_wrmse_img = nib.load(inp["current_wrmse"])
    current_valid_img = nib.load(inp["current_valid"])
    residual_img = nib.load(inp["current_residuals"])

    phase = np.asanyarray(
        phase_img.dataobj,
        dtype=np.float32,
    )
    magnitude = np.asanyarray(
        magnitude_img.dataobj,
        dtype=np.float32,
    )
    mask = np.asanyarray(mask_img.dataobj) > 0

    current_field_saved = np.asanyarray(
        current_field_img.dataobj,
        dtype=np.float32,
    )
    current_wrmse_saved = np.asanyarray(
        current_wrmse_img.dataobj,
        dtype=np.float32,
    )
    current_valid_saved = (
        np.asanyarray(current_valid_img.dataobj) > 0
    )
    residuals = np.asanyarray(
        residual_img.dataobj,
        dtype=np.float32,
    )

    tes_ms = np.loadtxt(
        inp["echo_times"],
        dtype=float,
    ).reshape(-1)

    tes_s = tes_ms / 1000.0

    if phase.ndim != 4:
        raise ValueError(f"Expected 4D phase, got {phase.shape}")

    if magnitude.shape != phase.shape:
        raise ValueError(
            f"Magnitude shape {magnitude.shape} != phase {phase.shape}"
        )

    if residuals.shape != phase.shape:
        raise ValueError(
            f"Residual shape {residuals.shape} != phase {phase.shape}"
        )

    if mask.shape != phase.shape[:3]:
        raise ValueError("Mask/phase spatial shape mismatch")

    if tes_s.size != phase.shape[3]:
        raise ValueError("Echo-time count mismatch")

    for img, label in (
        (magnitude_img, "magnitude"),
        (mask_img, "mask"),
        (current_field_img, "current field"),
        (current_wrmse_img, "current weighted RMSE"),
        (current_valid_img, "current valid mask"),
        (residual_img, "current residuals"),
    ):
        validate_geometry(phase_img, img, label)

    fit = fit_current_and_parity_wls(
        phase,
        magnitude,
        mask,
        tes_s,
        min_valid_echoes=args.min_valid_echoes,
        chunk_voxels=args.chunk_voxels,
        weight_floor_fraction=args.weight_floor_fraction,
    )

    current_field_reproduced = (
        fit["current_omega_rad_s"] / (2.0 * np.pi)
    )

    parity_field = (
        fit["parity_omega_rad_s"] / (2.0 * np.pi)
    )

    current_valid_reproduced = (
        fit["current_valid_fit_mask"] > 0
    )
    parity_valid = (
        fit["parity_valid_fit_mask"] > 0
    )

    # Verify the internal two-parameter reproduction against the saved
    # production GRE04b result.
    reproduce_common = (
        mask
        & current_valid_saved
        & current_valid_reproduced
        & np.isfinite(current_field_saved)
        & np.isfinite(current_field_reproduced)
        & np.isfinite(current_wrmse_saved)
        & np.isfinite(fit["current_weighted_rmse_rad"])
    )

    if not np.any(reproduce_common):
        raise RuntimeError("No common voxels for GRE04b reproduction check")

    reproduction_field_diff = (
        current_field_reproduced[reproduce_common].astype(np.float64)
        - current_field_saved[reproduce_common].astype(np.float64)
    )

    reproduction_wrmse_diff = (
        fit["current_weighted_rmse_rad"][reproduce_common].astype(np.float64)
        - current_wrmse_saved[reproduce_common].astype(np.float64)
    )

    # Main current-versus-parity comparison.
    common = (
        mask
        & current_valid_saved
        & parity_valid
        & np.isfinite(current_field_saved)
        & np.isfinite(parity_field)
        & np.isfinite(current_wrmse_saved)
        & np.isfinite(fit["parity_weighted_rmse_rad"])
    )

    if not np.any(common):
        raise RuntimeError("No common valid voxels")

    old_field = current_field_saved[common].astype(np.float64)
    new_field = parity_field[common].astype(np.float64)

    delta_field = (
        parity_field - current_field_saved
    ).astype(np.float32)

    beta_oe = fit["beta_oe_rad"]

    old_wrmse = current_wrmse_saved[common].astype(np.float64)
    new_wrmse = fit["parity_weighted_rmse_rad"][common].astype(np.float64)

    relative_reduction = np.zeros(
        phase.shape[:3],
        dtype=np.float32,
    )

    relative_values = np.divide(
        old_wrmse - new_wrmse,
        old_wrmse,
        out=np.zeros_like(old_wrmse),
        where=old_wrmse > 0,
    )

    relative_reduction[common] = relative_values.astype(np.float32)

    field_r = float(np.corrcoef(old_field, new_field)[0, 1])
    slope, intercept = np.polyfit(old_field, new_field, 1)

    delta_values = new_field - old_field
    delta_summary = finite_summary(delta_values)
    beta_summary = finite_summary(beta_oe[common])

    delta_plane = fit_spatial_plane(
        delta_field,
        common,
    )
    beta_plane = fit_spatial_plane(
        beta_oe,
        common,
    )

    residual_diag = current_residual_diagnostic(
        residuals,
        current_valid_saved & mask,
    )

    metrics = {
        "case_id": case,
        "status": "OK",
        "n_echoes": int(phase.shape[3]),
        "mask_voxels": int(np.count_nonzero(mask)),
        "common_valid_voxels": int(np.count_nonzero(common)),
        "current_reproduction_field_rms_hz": float(
            np.sqrt(np.mean(reproduction_field_diff ** 2))
        ),
        "current_reproduction_field_maxabs_hz": float(
            np.max(np.abs(reproduction_field_diff))
        ),
        "current_reproduction_wrmse_rms_rad": float(
            np.sqrt(np.mean(reproduction_wrmse_diff ** 2))
        ),
        "field_pearson_r": field_r,
        "field_slope_parity_on_current": float(slope),
        "field_intercept_hz": float(intercept),
        "delta_field_median_hz": delta_summary["median"],
        "delta_field_mae_hz": delta_summary["mae"],
        "delta_field_rms_hz": delta_summary["rms"],
        "delta_field_p01_hz": delta_summary["p01"],
        "delta_field_p99_hz": delta_summary["p99"],
        "current_wrmse_median_rad": float(np.median(old_wrmse)),
        "parity_wrmse_median_rad": float(np.median(new_wrmse)),
        "voxels_lower_wrmse_fraction": float(
            np.mean(new_wrmse < old_wrmse)
        ),
        "median_relative_wrmse_reduction": float(
            np.median(relative_values)
        ),
        "beta_oe_median_rad": beta_summary["median"],
        "beta_oe_mae_rad": beta_summary["mae"],
        "beta_oe_rms_rad": beta_summary["rms"],
        "beta_oe_p01_rad": beta_summary["p01"],
        "beta_oe_p99_rad": beta_summary["p99"],
        "delta_field_plane_r2": delta_plane["r2"],
        "delta_field_plane_dominant_axis": delta_plane["dominant_axis"],
        "delta_field_plane_x_slope_hz": delta_plane["x_slope"],
        "delta_field_plane_y_slope_hz": delta_plane["y_slope"],
        "delta_field_plane_z_slope_hz": delta_plane["z_slope"],
        "beta_oe_plane_r2": beta_plane["r2"],
        "beta_oe_plane_dominant_axis": beta_plane["dominant_axis"],
        "strict_odd_even_template_sign_alternation": (
            residual_diag["strict_template_sign_alternation"]
        ),
        "median_odd_template_correlation": (
            residual_diag["median_odd_template_correlation"]
        ),
        "median_even_template_correlation": (
            residual_diag["median_even_template_correlation"]
        ),
        "odd_even_template_rms_rad": (
            residual_diag["template_summary"]["rms"]
        ),
        "odd_even_template_plane_r2": (
            residual_diag["template_plane"]["r2"]
        ),
        "odd_even_template_plane_dominant_axis": (
            residual_diag["template_plane"]["dominant_axis"]
        ),
        "runtime_s": float(time.perf_counter() - started),
    }

    out["dir"].mkdir(parents=True, exist_ok=True)

    save_nifti(
        parity_field,
        phase_img,
        out["field_parity_hz"],
        np.float32,
    )
    save_nifti(
        fit["parity_omega_rad_s"],
        phase_img,
        out["omega_parity"],
        np.float32,
    )
    save_nifti(
        beta_oe,
        phase_img,
        out["beta_oe"],
        np.float32,
    )
    save_nifti(
        fit["parity_weighted_rmse_rad"],
        phase_img,
        out["wrmse_parity"],
        np.float32,
    )
    save_nifti(
        relative_reduction,
        phase_img,
        out["wrmse_reduction"],
        np.float32,
    )
    save_nifti(
        parity_valid.astype(np.uint8),
        phase_img,
        out["valid"],
        np.uint8,
    )
    save_nifti(
        np.where(common, delta_field, 0.0),
        phase_img,
        out["delta_field"],
        np.float32,
    )
    save_nifti(
        residual_diag["template_rad"],
        phase_img,
        out["oe_template"],
        np.float32,
    )

    qc = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "created_utc": now_iso(),
        "case_id": case,
        "inputs": {
            key: str(value)
            for key, value in inp.items()
        },
        "configuration": {
            "model_current": "phase = beta0 + omega * TE",
            "model_parity": (
                "phase = beta0 + omega * TE + beta_OE * parity"
            ),
            "parity_definition": (
                "+1 for echoes 1,3,5,...; -1 for echoes 2,4,6,..."
            ),
            "min_valid_echoes": args.min_valid_echoes,
            "chunk_voxels": args.chunk_voxels,
            "weight_floor_fraction": args.weight_floor_fraction,
            "weight_definition": "magnitude squared",
            "echo_times_ms": tes_ms.tolist(),
        },
        "metrics": metrics,
        "delta_field_plane": delta_plane,
        "beta_oe_plane": beta_plane,
        "current_residual_diagnostic": {
            key: value
            for key, value in residual_diag.items()
            if key != "template_rad"
        },
        "outputs": {
            key: str(value)
            for key, value in out.items()
            if key != "dir"
        },
    }

    out["qc"].write_text(
        json.dumps(qc, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.write_plots:
        write_plot(
            out,
            case,
            current_field_saved,
            parity_field,
            delta_field,
            beta_oe,
            common,
            metrics,
            maximum_voxels=args.plot_max_voxels,
            seed=args.plot_seed,
        )

    return metrics


def numeric_summary(
    rows: Sequence[dict],
    key: str,
) -> dict[str, float]:
    x = np.asarray(
        [float(row[key]) for row in rows],
        dtype=np.float64,
    )

    return {
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "APF GRE04d v0.1.0: diagnose odd/even echo phase "
            "and quantify its effect on GRE04b total-field fitting"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root", type=Path)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--all-gre-ready", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument(
        "--min-valid-echoes",
        type=int,
        default=DEFAULT_MIN_VALID_ECHOES,
        help="Minimum finite echoes required for a voxel-wise fit",
    )
    p.add_argument(
        "--chunk-voxels",
        type=int,
        default=DEFAULT_CHUNK_VOXELS,
        help="Number of masked voxels processed per chunk",
    )
    p.add_argument(
        "--weight-floor-fraction",
        type=float,
        default=DEFAULT_WEIGHT_FLOOR_FRACTION,
        help=(
            "Minimum positive weight as fraction of each voxel's "
            "maximum magnitude-squared echo weight"
        ),
    )

    p.add_argument(
        "--write-plots",
        action="store_true",
        help="Write standardized diagnostic PNG/PDF for every selected case",
    )
    p.add_argument(
        "--plot-max-voxels",
        type=int,
        default=100000,
        help="Maximum voxels used in field scatter",
    )
    p.add_argument(
        "--plot-seed",
        type=int,
        default=20260818,
        help="Seed for reproducible scatter subsampling",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.min_valid_echoes < 3:
        raise SystemExit("--min-valid-echoes must be at least 3")

    if args.chunk_voxels < 1:
        raise SystemExit("--chunk-voxels must be positive")

    if (
        not math.isfinite(args.weight_floor_fraction)
        or args.weight_floor_fraction < 0
    ):
        raise SystemExit(
            "--weight-floor-fraction must be finite and non-negative"
        )

    root = resolve_root(args.root)

    if args.all_gre_ready:
        cases = list_cases(root)
    else:
        cases = list(dict.fromkeys(args.case))

    if not cases:
        raise SystemExit(
            "Select --case CASE_ID or use --all-gre-ready"
        )

    summary_dir = root / SUMMARY_SUBDIR
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        summary_dir /
        "gre04d_odd_even_echo_diagnostic_qc.csv"
    )
    json_path = (
        summary_dir /
        "gre04d_odd_even_echo_diagnostic_run.json"
    )

    rows: list[dict] = []
    failures: list[dict] = []

    print(f"APF GRE04d v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Cases requested: {len(cases)}")
    print(
        "Model: phase = beta0 + omega*TE + beta_OE*parity"
    )
    print(
        "Parity: odd echoes +1; even echoes -1"
    )
    print()

    for case in cases:
        try:
            row = compare_case(
                root,
                case,
                args,
            )
            rows.append(row)

            print(
                f"[OK] {case}: "
                f"Δf RMS={row['delta_field_rms_hz']:.4f} Hz, "
                f"RMSE reduction="
                f"{row['median_relative_wrmse_reduction']*100:.1f}%, "
                f"OE sign-alt="
                f"{row['strict_odd_even_template_sign_alternation']}, "
                f"Δf plane={row['delta_field_plane_dominant_axis']} "
                f"(R²={row['delta_field_plane_r2']:.3f})"
            )

        except Exception as exc:
            failures.append({
                "case_id": case,
                "status": "FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            })

            print(
                f"[FAILED] {case}: "
                f"{type(exc).__name__}: {exc}"
            )

    write_csv(csv_path, rows)

    summary_keys = [
        "current_reproduction_field_rms_hz",
        "delta_field_rms_hz",
        "delta_field_mae_hz",
        "field_pearson_r",
        "median_relative_wrmse_reduction",
        "voxels_lower_wrmse_fraction",
        "beta_oe_rms_rad",
        "delta_field_plane_r2",
        "odd_even_template_rms_rad",
        "odd_even_template_plane_r2",
    ]

    cohort_summary = {
        key: numeric_summary(rows, key)
        for key in summary_keys
    } if rows else {}

    sign_alt_count = sum(
        bool(
            row["strict_odd_even_template_sign_alternation"]
        )
        for row in rows
    )

    dominant_axes = {
        axis: sum(
            row["delta_field_plane_dominant_axis"] == axis
            for row in rows
        )
        for axis in ("voxel-x", "voxel-y", "voxel-z")
    }

    run = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "created_utc": now_iso(),
        "root": str(root),
        "configuration": {
            "min_valid_echoes": args.min_valid_echoes,
            "chunk_voxels": args.chunk_voxels,
            "weight_floor_fraction": args.weight_floor_fraction,
            "write_plots": bool(args.write_plots),
            "plot_max_voxels": args.plot_max_voxels,
            "plot_seed": args.plot_seed,
        },
        "cases_requested": len(cases),
        "cases_ok": len(rows),
        "cases_failed": len(failures),
        "strict_sign_alternation_cases": sign_alt_count,
        "delta_field_dominant_axes": dominant_axes,
        "cohort_summary": cohort_summary,
        "failures": failures,
        "outputs": {
            "qc_csv": str(csv_path),
            "run_json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(run, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("COHORT SUMMARY")
    print("=" * 92)

    if rows:
        for key in summary_keys:
            s = cohort_summary[key]
            print(
                f"{key:38s} "
                f"median={s['median']:.8f} "
                f"min={s['min']:.8f} "
                f"max={s['max']:.8f}"
            )

        print()
        print(
            f"Strict odd/even sign alternation: "
            f"{sign_alt_count}/{len(rows)} cases"
        )
        print(
            "Dominant Δfield plane axis: "
            + ", ".join(
                f"{axis}={n}"
                for axis, n in dominant_axes.items()
            )
        )

    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(
        f"Completed: OK={len(rows)}, "
        f"FAILED={len(failures)}, "
        f"TOTAL={len(cases)}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
