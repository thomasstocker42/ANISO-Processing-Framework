#!/usr/bin/env python3
"""
APF GRE04b: magnitude-weighted least-squares multi-echo total-field estimation.

Scientific purpose
------------------
Fit unwrapped phase versus echo time voxel-wise using magnitude-squared weights
while modelling the cohort-validated odd/even echo phase term:

    phase(TE, echo) = intercept + omega * TE + beta_OE * parity(echo)
    parity(echo) = +1 for echoes 1,3,5,... and -1 for echoes 2,4,6,...
    weight(TE) = magnitude(TE)^2

and convert the fitted angular-frequency slope to:
- frequency offset in Hz
- relative total field in ppm

This module does not remove background fields and does not perform QSM.

Inputs
------
- GRE03 ROMEO-unwrapped 4D phase
- GRE03 effective mask
- GRE02 multi-echo magnitude
- GRE02 echo times
- explicit scanner field strength via --b0-t

Outputs
-------
derivatives/<CASE>/GRE/04b_field_wls/
  <CASE>_GRE04b_omega_rad_s.nii.gz
  <CASE>_GRE04b_total_field_hz.nii.gz
  <CASE>_GRE04b_total_field_ppm.nii.gz
  <CASE>_GRE04b_phase_intercept_rad.nii.gz
  <CASE>_GRE04b_beta_odd_even_rad.nii.gz
  <CASE>_GRE04b_phase_rmse_rad.nii.gz
  <CASE>_GRE04b_phase_weighted_rmse_rad.nii.gz
  <CASE>_GRE04b_phase_r2.nii.gz
  <CASE>_GRE04b_phase_weighted_r2.nii.gz
  <CASE>_GRE04b_phase_maxabs_residual_rad.nii.gz
  <CASE>_GRE04b_phase_linearity_error_rad.nii.gz
  <CASE>_GRE04b_frequency_se_hz.nii.gz
  <CASE>_GRE04b_valid_fit_mask.nii.gz
  <CASE>_GRE04b_n_valid_echoes.nii.gz
  <CASE>_GRE04b_phase_predicted_rad_4d.nii.gz  [optional]
  <CASE>_GRE04b_phase_residuals_rad_4d.nii.gz  [optional]
  <CASE>_GRE04b_qc.json

Cohort outputs
--------------
derivatives/summary/GRE/
  gre04b_field_wls_qc.csv
  gre04b_field_wls_run.json

run
----
python scripts/gre/04b_estimate_total_field_wls.py \
  --all-gre-ready \
  --b0-t 3.0 \
  --write-residuals \
  --force

"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from _field_fit_common import (
    GYROMAGNETIC_RATIO_HZ_PER_T,
    case_search_roots,
    discover_echo_times,
    discover_romeo_mask,
    discover_unwrapped_phase,
    list_cases,
    now_iso,
    parse_echo_times_ms,
    resolve_root,
    save_nifti,
    validate_geometry,
    write_case_summary_json,
    write_cohort_csv,
)


SCRIPT_VERSION = "0.2.0"
MODULE_NAME = "gre04b"
METHOD_NAME = "magnitude_squared_weighted_least_squares_with_odd_even_parity"
OUTPUT_SUBDIR = Path("GRE") / "04b_field_wls"


@dataclass(frozen=True)
class Inputs:
    case_id: str
    phase_4d: Path
    magnitude_4d: Path
    mask: Path
    echo_times_file: Path


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None
    n_echoes: int | None = None
    te_min_ms: float | None = None
    te_max_ms: float | None = None
    b0_t: float | None = None
    mask_voxels: int | None = None
    valid_fit_voxels: int | None = None
    valid_fit_fraction_masked: float | None = None
    all_echoes_fraction_valid: float | None = None
    median_field_hz: float | None = None
    p01_field_hz: float | None = None
    p99_field_hz: float | None = None
    median_field_ppm: float | None = None
    median_beta_odd_even_rad: float | None = None
    median_rmse_rad: float | None = None
    median_r2: float | None = None
    median_weighted_rmse_rad: float | None = None
    median_weighted_r2: float | None = None
    median_frequency_se_hz: float | None = None
    output_dir: str | None = None


def discover_magnitude_4d(roots: Sequence[Path]) -> Path:
    """Discover the GRE02 multi-echo magnitude stack deterministically."""
    candidates: list[Path] = []
    patterns = (
        "*GRE02*magnitude*4d*.nii*",
        "*gre02*magnitude*4d*.nii*",
        "*magnitude_4d*.nii*",
    )
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())

    unique = sorted({path.resolve() for path in candidates})
    unique = [
        path for path in unique
        if "relaxometr" not in str(path).lower()
        and "04a_field_ols" not in str(path).lower()
        and "04b_field_wls" not in str(path).lower()
    ]
    if not unique:
        raise FileNotFoundError("No GRE02 multi-echo magnitude 4D image found")

    def rank(path: Path) -> tuple[int, int, str]:
        text = str(path).lower()
        score = 0
        score += 100 if "gre02" in text else 0
        score += 50 if "magnitude_4d" in path.name.lower() else 0
        score += 20 if "/gre/" in text else 0
        score += 10 if "derivatives" in text else 0
        return (-score, len(str(path)), str(path))

    ranked = sorted(unique, key=rank)
    best_score = rank(ranked[0])[:2]
    tied = [path for path in ranked if rank(path)[:2] == best_score]
    if len(tied) > 1:
        listing = "\n".join(f"  - {path}" for path in tied)
        raise RuntimeError(
            "Ambiguous GRE02 magnitude discovery; use --magnitude explicitly:\n"
            + listing
        )
    return ranked[0]


def discover_inputs(
    root: Path,
    case_id: str,
    phase: Path | None,
    magnitude: Path | None,
    mask: Path | None,
    echo_times: Path | None,
) -> Inputs:
    roots = case_search_roots(root, case_id)
    if not roots:
        raise FileNotFoundError(f"No case directories found for {case_id}")

    return Inputs(
        case_id=case_id,
        phase_4d=phase.resolve() if phase else discover_unwrapped_phase(roots),
        magnitude_4d=(
            magnitude.resolve() if magnitude else discover_magnitude_4d(roots)
        ),
        mask=mask.resolve() if mask else discover_romeo_mask(roots),
        echo_times_file=(
            echo_times.resolve() if echo_times else discover_echo_times(roots)
        ),
    )


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case_id / OUTPUT_SUBDIR
    prefix = f"{case_id}_GRE04b"
    return {
        "dir": outdir,
        "omega": outdir / f"{prefix}_omega_rad_s.nii.gz",
        "hz": outdir / f"{prefix}_total_field_hz.nii.gz",
        "ppm": outdir / f"{prefix}_total_field_ppm.nii.gz",
        "intercept": outdir / f"{prefix}_phase_intercept_rad.nii.gz",
        "beta_odd_even": outdir / f"{prefix}_beta_odd_even_rad.nii.gz",
        "rmse": outdir / f"{prefix}_phase_rmse_rad.nii.gz",
        "weighted_rmse": outdir / f"{prefix}_phase_weighted_rmse_rad.nii.gz",
        "r2": outdir / f"{prefix}_phase_r2.nii.gz",
        "weighted_r2": outdir / f"{prefix}_phase_weighted_r2.nii.gz",
        "maxabs": outdir / f"{prefix}_phase_maxabs_residual_rad.nii.gz",
        "linearity": outdir / f"{prefix}_phase_linearity_error_rad.nii.gz",
        "se_hz": outdir / f"{prefix}_frequency_se_hz.nii.gz",
        "valid": outdir / f"{prefix}_valid_fit_mask.nii.gz",
        "n_valid": outdir / f"{prefix}_n_valid_echoes.nii.gz",
        "predicted": outdir / f"{prefix}_phase_predicted_rad_4d.nii.gz",
        "residuals": outdir / f"{prefix}_phase_residuals_rad_4d.nii.gz",
        "qc": outdir / f"{prefix}_qc.json",
    }


def fit_wls_chunked(
    phase_4d: np.ndarray,
    magnitude_4d: np.ndarray,
    mask_3d: np.ndarray,
    tes_s: np.ndarray,
    *,
    min_valid_echoes: int,
    chunk_voxels: int,
    weight_floor_fraction: float,
    write_residuals: bool,
) -> dict[str, np.ndarray]:
    """Fit phase against TE with magnitude-squared weights and an odd/even term."""
    spatial_shape = phase_4d.shape[:3]
    n_echoes = phase_4d.shape[3]
    n_voxels = int(np.prod(spatial_shape))

    y_all = phase_4d.reshape(n_voxels, n_echoes)
    magnitude_all = magnitude_4d.reshape(n_voxels, n_echoes)
    mask_flat = mask_3d.reshape(n_voxels)

    omega = np.zeros(n_voxels, dtype=np.float32)
    intercept = np.zeros(n_voxels, dtype=np.float32)
    beta_odd_even = np.zeros(n_voxels, dtype=np.float32)
    rmse = np.zeros(n_voxels, dtype=np.float32)
    weighted_rmse = np.zeros(n_voxels, dtype=np.float32)
    r2 = np.zeros(n_voxels, dtype=np.float32)
    weighted_r2 = np.zeros(n_voxels, dtype=np.float32)
    maxabs = np.zeros(n_voxels, dtype=np.float32)
    slope_se = np.zeros(n_voxels, dtype=np.float32)
    n_valid = np.zeros(n_voxels, dtype=np.uint8)
    valid_fit = np.zeros(n_voxels, dtype=np.uint8)

    predicted_all = None
    residuals_all = None
    if write_residuals:
        predicted_all = np.zeros((n_voxels, n_echoes), dtype=np.float32)
        residuals_all = np.zeros((n_voxels, n_echoes), dtype=np.float32)

    masked_indices = np.flatnonzero(mask_flat)
    t = tes_s[None, :]
    parity = np.where(
        np.arange(n_echoes) % 2 == 0,
        1.0,
        -1.0,
    )[None, :]

    for start in range(0, masked_indices.size, chunk_voxels):
        idx = masked_indices[start:start + chunk_voxels]
        y = np.asarray(y_all[idx, :], dtype=np.float64)
        magnitude = np.asarray(magnitude_all[idx, :], dtype=np.float64)

        finite = np.isfinite(y) & np.isfinite(magnitude) & (magnitude > 0)
        n = finite.sum(axis=1).astype(np.int32)
        n_valid[idx] = np.minimum(n, 255).astype(np.uint8)

        raw_weights = np.where(finite, magnitude * magnitude, 0.0)
        max_weight = np.max(raw_weights, axis=1, keepdims=True)
        floor = weight_floor_fraction * max_weight
        weights = np.where(finite, np.maximum(raw_weights, floor), 0.0)

        y0 = np.where(finite, y, 0.0)

        # Weighted normal equations for design columns [1, TE, parity].
        sw = np.sum(weights, axis=1)
        swt = np.sum(weights * t, axis=1)
        sws = np.sum(weights * parity, axis=1)
        swtt = np.sum(weights * t * t, axis=1)
        swts = np.sum(weights * t * parity, axis=1)

        swy = np.sum(weights * y0, axis=1)
        swty = np.sum(weights * y0 * t, axis=1)
        swsy = np.sum(weights * y0 * parity, axis=1)

        # parity^2 == 1 for every valid echo, hence sum(w*s^2) == sw.
        normal = np.empty((idx.size, 3, 3), dtype=np.float64)
        normal[:, 0, 0] = sw
        normal[:, 0, 1] = swt
        normal[:, 0, 2] = sws
        normal[:, 1, 0] = swt
        normal[:, 1, 1] = swtt
        normal[:, 1, 2] = swts
        normal[:, 2, 0] = sws
        normal[:, 2, 1] = swts
        normal[:, 2, 2] = sw

        rhs = np.column_stack((swy, swty, swsy))

        odd_valid = np.any(finite[:, 0::2], axis=1)
        even_valid = np.any(finite[:, 1::2], axis=1)

        determinant = np.linalg.det(normal)
        good = (
            (n >= min_valid_echoes)
            & odd_valid
            & even_valid
            & np.isfinite(sw)
            & (sw > 0)
            & np.isfinite(determinant)
            & (determinant > 0)
        )

        b0 = np.zeros(idx.size, dtype=np.float64)
        b1 = np.zeros(idx.size, dtype=np.float64)
        b2 = np.zeros(idx.size, dtype=np.float64)

        solve_idx = np.flatnonzero(good)
        if solve_idx.size:
            # Explicit singleton RHS dimension is required for reliable
            # batched np.linalg.solve behaviour across supported NumPy versions.
            beta = np.linalg.solve(
                normal[solve_idx],
                rhs[solve_idx, :, None],
            )[..., 0]

            finite_beta = np.all(np.isfinite(beta), axis=1)
            bad_solve_idx = solve_idx[~finite_beta]
            good[bad_solve_idx] = False

            solve_idx = solve_idx[finite_beta]
            beta = beta[finite_beta]

            b0[solve_idx] = beta[:, 0]
            b1[solve_idx] = beta[:, 1]
            b2[solve_idx] = beta[:, 2]

        fitted = (
            b0[:, None]
            + b1[:, None] * t
            + b2[:, None] * parity
        )
        resid = np.where(finite, y - fitted, np.nan)

        sse = np.nansum(resid * resid, axis=1)
        weighted_sse = np.sum(
            weights * np.where(finite, resid * resid, 0.0),
            axis=1,
        )

        mean_y = np.divide(
            np.nansum(y0, axis=1),
            n,
            out=np.zeros(idx.size, dtype=np.float64),
            where=n > 0,
        )
        weighted_mean_y = np.divide(
            swy,
            sw,
            out=np.zeros_like(swy),
            where=sw > 0,
        )

        centered = np.where(finite, y - mean_y[:, None], np.nan)
        sst = np.nansum(centered * centered, axis=1)
        weighted_sst = np.sum(
            weights
            * np.where(
                finite,
                (y - weighted_mean_y[:, None]) ** 2,
                0.0,
            ),
            axis=1,
        )

        rmse_chunk = np.sqrt(
            np.divide(
                sse,
                n,
                out=np.zeros_like(sse),
                where=n > 0,
            )
        )
        weighted_rmse_chunk = np.sqrt(
            np.divide(
                weighted_sse,
                sw,
                out=np.zeros_like(weighted_sse),
                where=sw > 0,
            )
        )

        r2_chunk = np.zeros(idx.size, dtype=np.float64)
        informative = good & (sst > 0)
        r2_chunk[informative] = (
            1.0 - sse[informative] / sst[informative]
        )

        weighted_r2_chunk = np.zeros(idx.size, dtype=np.float64)
        weighted_informative = good & (weighted_sst > 0)
        weighted_r2_chunk[weighted_informative] = (
            1.0
            - weighted_sse[weighted_informative]
            / weighted_sst[weighted_informative]
        )

        maxabs_chunk = np.zeros(idx.size, dtype=np.float64)
        has_finite = np.any(finite, axis=1)
        maxabs_chunk[has_finite] = np.nanmax(
            np.abs(resid[has_finite, :]),
            axis=1,
        )

        # Approximate WLS covariance under
        # Var(phase_i) proportional to 1 / weight_i.
        #
        # Cov(beta) = residual_scale * inv(X'WX).
        # For beta[1] (omega), the relevant inverse-normal diagonal is:
        #   inv(X'WX)[1,1] = (sw*sw - sws^2) / det(X'WX)
        dof = n - 3
        residual_scale = np.divide(
            weighted_sse,
            dof,
            out=np.zeros_like(weighted_sse),
            where=dof > 0,
        )
        inv11 = np.divide(
            sw * sw - sws * sws,
            determinant,
            out=np.zeros_like(determinant),
            where=determinant > 0,
        )
        slope_variance = residual_scale * inv11
        slope_se_chunk = np.sqrt(
            np.maximum(slope_variance, 0.0)
        )

        finite_coefficients = (
            np.isfinite(b0)
            & np.isfinite(b1)
            & np.isfinite(b2)
            & np.isfinite(rmse_chunk)
            & np.isfinite(weighted_rmse_chunk)
            & np.isfinite(maxabs_chunk)
            & np.isfinite(slope_se_chunk)
        )
        good &= finite_coefficients

        omega[idx[good]] = b1[good].astype(np.float32)
        intercept[idx[good]] = b0[good].astype(np.float32)
        beta_odd_even[idx[good]] = b2[good].astype(np.float32)
        rmse[idx[good]] = rmse_chunk[good].astype(np.float32)
        weighted_rmse[idx[good]] = weighted_rmse_chunk[good].astype(np.float32)
        r2[idx[good]] = r2_chunk[good].astype(np.float32)
        weighted_r2[idx[good]] = weighted_r2_chunk[good].astype(np.float32)
        maxabs[idx[good]] = maxabs_chunk[good].astype(np.float32)
        slope_se[idx[good]] = slope_se_chunk[good].astype(np.float32)
        valid_fit[idx[good]] = 1

        if residuals_all is not None and predicted_all is not None:
            predicted_all[idx, :] = np.where(
                good[:, None] & finite,
                fitted,
                0.0,
            ).astype(np.float32)
            residuals_all[idx, :] = np.where(
                good[:, None] & finite,
                resid,
                0.0,
            ).astype(np.float32)

    result = {
        "omega_rad_s": omega.reshape(spatial_shape),
        "intercept_rad": intercept.reshape(spatial_shape),
        "beta_odd_even_rad": beta_odd_even.reshape(spatial_shape),
        "rmse_rad": rmse.reshape(spatial_shape),
        "weighted_rmse_rad": weighted_rmse.reshape(spatial_shape),
        "r2": r2.reshape(spatial_shape),
        "weighted_r2": weighted_r2.reshape(spatial_shape),
        "maxabs_residual_rad": maxabs.reshape(spatial_shape),
        "omega_se_rad_s": slope_se.reshape(spatial_shape),
        "valid_fit_mask": valid_fit.reshape(spatial_shape),
        "n_valid_echoes": n_valid.reshape(spatial_shape),
    }
    if residuals_all is not None and predicted_all is not None:
        result["predicted_rad_4d"] = predicted_all.reshape(
            spatial_shape + (n_echoes,)
        )
        result["residuals_rad_4d"] = residuals_all.reshape(
            spatial_shape + (n_echoes,)
        )
    return result

def masked_stats(data: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(data)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "median": None,
            "p01": None,
            "p99": None,
        }
    return {
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def process_case(
    root: Path,
    case_id: str,
    args: argparse.Namespace,
) -> CaseResult:
    started = time.perf_counter()
    paths = output_paths(root, case_id)

    try:
        explicit_allowed = len(args.case) == 1
        inputs = discover_inputs(
            root,
            case_id,
            args.phase if explicit_allowed else None,
            args.magnitude if explicit_allowed else None,
            args.mask if explicit_allowed else None,
            args.echo_times if explicit_allowed else None,
        )

        tes_ms = parse_echo_times_ms(inputs.echo_times_file)
        tes_s = np.asarray(tes_ms, dtype=np.float64) / 1000.0

        phase_img = nib.load(str(inputs.phase_4d))
        magnitude_img = nib.load(str(inputs.magnitude_4d))
        mask_img = nib.load(str(inputs.mask))
        phase_shape = validate_geometry(phase_img, mask_img, tes_ms)
        if magnitude_img.shape != phase_img.shape:
            raise ValueError(
                f"Magnitude shape {magnitude_img.shape} does not match phase "
                f"shape {phase_img.shape}"
            )
        if not np.allclose(magnitude_img.affine, phase_img.affine, atol=1e-5):
            raise ValueError("Magnitude and phase affines do not match")

        if paths["hz"].exists() and not args.force:
            return CaseResult(
                case_id=case_id,
                status="SKIPPED_EXISTS",
                message="Primary output exists; use --force to overwrite",
                n_echoes=len(tes_ms),
                te_min_ms=min(tes_ms),
                te_max_ms=max(tes_ms),
                b0_t=args.b0_t,
                output_dir=str(paths["dir"]),
            )

        if args.dry_run:
            return CaseResult(
                case_id=case_id,
                status="DRY_RUN",
                message=(
                    f"phase={inputs.phase_4d}; magnitude={inputs.magnitude_4d}; "
                    f"mask={inputs.mask}; "
                    f"TE={inputs.echo_times_file}; B0={args.b0_t} T"
                ),
                n_echoes=len(tes_ms),
                te_min_ms=min(tes_ms),
                te_max_ms=max(tes_ms),
                b0_t=args.b0_t,
                output_dir=str(paths["dir"]),
            )

        paths["dir"].mkdir(parents=True, exist_ok=True)
        if args.force:
            for key, path in paths.items():
                if key != "dir":
                    path.unlink(missing_ok=True)

        phase = np.asanyarray(phase_img.dataobj, dtype=np.float32)
        magnitude = np.asanyarray(magnitude_img.dataobj, dtype=np.float32)
        mask = np.asanyarray(mask_img.dataobj) > 0

        fit = fit_wls_chunked(
            phase,
            magnitude,
            mask,
            tes_s,
            min_valid_echoes=args.min_valid_echoes,
            chunk_voxels=args.chunk_voxels,
            weight_floor_fraction=args.weight_floor_fraction,
            write_residuals=args.write_residuals,
        )

        omega = fit["omega_rad_s"]
        field_hz = omega / (2.0 * math.pi)
        field_ppm = (
            field_hz
            / (GYROMAGNETIC_RATIO_HZ_PER_T * args.b0_t)
            * 1e6
        )
        frequency_se_hz = fit["omega_se_rad_s"] / (2.0 * math.pi)

        valid = fit["valid_fit_mask"] > 0
        valid &= np.isfinite(field_hz)
        valid &= np.isfinite(field_ppm)

        # Ensure all outputs are zero outside the final valid mask.
        for key in (
            "omega_rad_s",
            "intercept_rad",
            "beta_odd_even_rad",
            "rmse_rad",
            "r2",
            "maxabs_residual_rad",
            "weighted_rmse_rad",
            "weighted_r2",
            "omega_se_rad_s",
        ):
            fit[key] = np.where(valid, fit[key], 0).astype(np.float32)

        field_hz = np.where(valid, field_hz, 0).astype(np.float32)
        field_ppm = np.where(valid, field_ppm, 0).astype(np.float32)
        frequency_se_hz = np.where(valid, frequency_se_hz, 0).astype(np.float32)
        fit["valid_fit_mask"] = valid.astype(np.uint8)

        if "predicted_rad_4d" in fit:
            fit["predicted_rad_4d"] = np.where(
                valid[..., None],
                fit["predicted_rad_4d"],
                0,
            ).astype(np.float32)

        if "residuals_rad_4d" in fit:
            fit["residuals_rad_4d"] = np.where(
                valid[..., None],
                fit["residuals_rad_4d"],
                0,
            ).astype(np.float32)

        save_nifti(fit["omega_rad_s"], phase_img, paths["omega"], np.float32)
        save_nifti(field_hz, phase_img, paths["hz"], np.float32)
        save_nifti(field_ppm, phase_img, paths["ppm"], np.float32)
        save_nifti(fit["intercept_rad"], phase_img, paths["intercept"], np.float32)
        save_nifti(
            fit["beta_odd_even_rad"], phase_img, paths["beta_odd_even"], np.float32
        )
        save_nifti(fit["rmse_rad"], phase_img, paths["rmse"], np.float32)
        save_nifti(
            fit["weighted_rmse_rad"], phase_img, paths["weighted_rmse"], np.float32
        )
        save_nifti(fit["r2"], phase_img, paths["r2"], np.float32)
        save_nifti(
            fit["weighted_r2"], phase_img, paths["weighted_r2"], np.float32
        )
        save_nifti(
            fit["maxabs_residual_rad"],
            phase_img,
            paths["maxabs"],
            np.float32,
        )
        save_nifti(
            fit["maxabs_residual_rad"],
            phase_img,
            paths["linearity"],
            np.float32,
        )
        save_nifti(frequency_se_hz, phase_img, paths["se_hz"], np.float32)
        save_nifti(fit["valid_fit_mask"], phase_img, paths["valid"], np.uint8)
        save_nifti(
            fit["n_valid_echoes"],
            phase_img,
            paths["n_valid"],
            np.uint8,
        )
        if args.write_residuals:
            save_nifti(
                fit["predicted_rad_4d"],
                phase_img,
                paths["predicted"],
                np.float32,
            )
            save_nifti(
                fit["residuals_rad_4d"],
                phase_img,
                paths["residuals"],
                np.float32,
            )

        mask_voxels = int(mask.sum())
        valid_voxels = int(valid.sum())
        valid_fraction = (
            float(valid_voxels / mask_voxels)
            if mask_voxels > 0
            else 0.0
        )
        all_echoes_fraction = (
            float(
                np.mean(
                    fit["n_valid_echoes"][valid]
                    == phase_shape[3]
                )
            )
            if valid_voxels > 0
            else 0.0
        )

        field_hz_stats = masked_stats(field_hz, valid)
        field_ppm_stats = masked_stats(field_ppm, valid)
        beta_odd_even_stats = masked_stats(fit["beta_odd_even_rad"], valid)
        rmse_stats = masked_stats(fit["rmse_rad"], valid)
        r2_stats = masked_stats(fit["r2"], valid)
        weighted_rmse_stats = masked_stats(fit["weighted_rmse_rad"], valid)
        weighted_r2_stats = masked_stats(fit["weighted_r2"], valid)
        se_stats = masked_stats(frequency_se_hz, valid)

        runtime_s = time.perf_counter() - started

        provenance = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "method": METHOD_NAME,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": runtime_s,
            "inputs": {
                "phase_unwrapped_rad_4d": str(inputs.phase_4d),
                "magnitude_4d": str(inputs.magnitude_4d),
                "effective_mask": str(inputs.mask),
                "echo_times_file": str(inputs.echo_times_file),
            },
            "configuration": {
                "echo_times_ms": tes_ms,
                "echo_times_s": tes_s.tolist(),
                "n_echoes": len(tes_ms),
                "minimum_valid_echoes": args.min_valid_echoes,
                "chunk_voxels": args.chunk_voxels,
                "weight_floor_fraction": args.weight_floor_fraction,
                "b0_t": args.b0_t,
                "gyromagnetic_ratio_hz_per_t": GYROMAGNETIC_RATIO_HZ_PER_T,
                "write_predicted_phase": args.write_residuals,
                "write_residuals": args.write_residuals,
                "fit_model": (
                    "phase_rad = intercept_rad + omega_rad_s * TE_s "
                    "+ beta_odd_even_rad * parity"
                ),
                "parity_definition": (
                    "+1 for echoes 1,3,5,...; -1 for echoes 2,4,6,..."
                ),
                "weighting": "per-echo magnitude squared",
                "weight_floor_definition": (
                    "minimum positive weight equals weight_floor_fraction "
                    "times the voxel maximum echo weight"
                ),
                "rmse_definition": "unweighted sqrt(mean(residual_rad^2))",
                "r2_definition": "1 - SSE/SST",
                "weighted_rmse_definition": (
                    "sqrt(sum(weight * residual_rad^2) / sum(weight))"
                ),
                "slope_se_model": (
                    "WLS covariance with Var(phase_i) proportional to 1/weight_i"
                ),
            },
            "outputs": {
                key: str(value)
                for key, value in paths.items()
                if key != "dir" and (value.exists() or key == "residuals")
            },
            "qc": {
                "phase_shape": list(phase_shape),
                "mask_voxels": mask_voxels,
                "valid_fit_voxels": valid_voxels,
                "valid_fit_fraction_masked": valid_fraction,
                "all_echoes_fraction_valid": all_echoes_fraction,
                "field_hz": field_hz_stats,
                "field_ppm": field_ppm_stats,
                "beta_odd_even_rad": beta_odd_even_stats,
                "rmse_rad": rmse_stats,
                "r2": r2_stats,
                "weighted_rmse_rad": weighted_rmse_stats,
                "weighted_r2": weighted_r2_stats,
                "frequency_se_hz": se_stats,
            },
        }
        write_case_summary_json(paths["qc"], provenance)

        return CaseResult(
            case_id=case_id,
            status="OK",
            message="Magnitude-squared WLS total-field estimation completed",
            runtime_s=runtime_s,
            n_echoes=len(tes_ms),
            te_min_ms=min(tes_ms),
            te_max_ms=max(tes_ms),
            b0_t=args.b0_t,
            mask_voxels=mask_voxels,
            valid_fit_voxels=valid_voxels,
            valid_fit_fraction_masked=valid_fraction,
            all_echoes_fraction_valid=all_echoes_fraction,
            median_field_hz=field_hz_stats["median"],
            p01_field_hz=field_hz_stats["p01"],
            p99_field_hz=field_hz_stats["p99"],
            median_field_ppm=field_ppm_stats["median"],
            median_beta_odd_even_rad=beta_odd_even_stats["median"],
            median_rmse_rad=rmse_stats["median"],
            median_r2=r2_stats["median"],
            median_weighted_rmse_rad=weighted_rmse_stats["median"],
            median_weighted_r2=weighted_r2_stats["median"],
            median_frequency_se_hz=se_stats["median"],
            output_dir=str(paths["dir"]),
        )

    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - started,
            b0_t=args.b0_t,
            output_dir=str(paths["dir"]),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "APF GRE04b: voxel-wise magnitude-squared WLS fitting of ROMEO-unwrapped "
            "multi-echo phase with an odd/even parity nuisance regressor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        help="APF repository root; defaults to APF_ROOT or cwd",
    )
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID; repeat for multiple cases",
    )
    p.add_argument(
        "--all-gre-ready",
        action="store_true",
        help="Attempt all discovered ANISO_*_InS cases",
    )
    p.add_argument(
        "--b0-t",
        type=float,
        required=True,
        help="Scanner field strength in tesla; required for ppm conversion",
    )
    p.add_argument(
        "--min-valid-echoes",
        type=int,
        default=4,
        help=(
            "Minimum finite echoes required for the three-parameter parity-aware fit"
        ),
    )
    p.add_argument(
        "--chunk-voxels",
        type=int,
        default=200000,
        help="Number of masked voxels processed per numerical chunk",
    )
    p.add_argument(
        "--weight-floor-fraction",
        type=float,
        default=1e-6,
        help=(
            "Per-voxel floor applied to magnitude-squared weights as a "
            "fraction of that voxel's maximum echo weight"
        ),
    )
    p.add_argument(
        "--write-residuals",
        action="store_true",
        help=(
            "Write the optional 4D predicted-phase and phase-residual series"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GRE04b outputs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery and print case inputs without fitting",
    )

    # Explicit single-case overrides.
    p.add_argument(
        "--phase",
        type=Path,
        help="Explicit GRE03 unwrapped 4D phase; single case only",
    )
    p.add_argument(
        "--magnitude",
        type=Path,
        help="Explicit GRE02 multi-echo magnitude 4D image; single case only",
    )
    p.add_argument(
        "--mask",
        type=Path,
        help="Explicit GRE03 effective mask; single case only",
    )
    p.add_argument(
        "--echo-times",
        type=Path,
        help="Explicit GRE02 echo-time file; single case only",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    if not math.isfinite(args.b0_t) or args.b0_t <= 0:
        raise SystemExit("--b0-t must be finite and positive")
    if args.min_valid_echoes < 4:
        raise SystemExit("--min-valid-echoes must be at least 4")
    if args.chunk_voxels < 1:
        raise SystemExit("--chunk-voxels must be positive")
    if (
        not math.isfinite(args.weight_floor_fraction)
        or args.weight_floor_fraction < 0
        or args.weight_floor_fraction > 1
    ):
        raise SystemExit("--weight-floor-fraction must be finite and within [0, 1]")

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit("Select at least one --case CASE_ID or use --all-gre-ready")

    explicit_inputs = [args.phase, args.magnitude, args.mask, args.echo_times]
    if any(x is not None for x in explicit_inputs) and len(cases) != 1:
        raise SystemExit(
            "Explicit --phase/--magnitude/--mask/--echo-times require exactly one case"
        )

    print(f"APF GRE04b v{SCRIPT_VERSION}")
    print(f"Root:   {root}")
    print(f"Method: {METHOD_NAME}")
    print(f"B0:     {args.b0_t} T")
    print(f"Cases:  {len(cases)}")

    started_utc = now_iso()
    results: list[CaseResult] = []

    for case_id in cases:
        result = process_case(root, case_id, args)
        results.append(result)
        print(f"[{result.status}] {case_id}: {result.message}")

    summary_dir = root / "derivatives" / "summary" / "GRE"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_cohort_csv(
        summary_dir / "gre04b_field_wls_qc.csv",
        results,
    )

    run_payload = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "started_utc": started_utc,
        "finished_utc": now_iso(),
        "root": str(root),
        "arguments": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "results": [asdict(result) for result in results],
    }
    (
        summary_dir / "gre04b_field_wls_run.json"
    ).write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

    n_failed = sum(r.status == "FAILED" for r in results)
    n_ok = sum(r.status == "OK" for r in results)
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
