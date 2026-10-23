#!/usr/bin/env python3
"""
APF GRE04c: four-scenario comparison of multi-echo GRE total-field fits.

Scientific purpose
------------------
Quantify the separate effects of magnitude-squared weighting and odd/even echo
parity modelling on the fitted multi-echo GRE total field.

Four finalized estimators are compared on one common four-way valid support:

    A  OLS legacy   : equal weighting, no parity term
    B  WLS legacy   : magnitude-squared weighting, no parity term
    C  OLS parity   : equal weighting, odd/even parity term
    D  WLS parity   : magnitude-squared weighting, odd/even parity term

Primary contrasts
-----------------
    B - A : weighting effect without parity
    C - A : parity effect under OLS
    D - B : parity effect under WLS
    D - C : weighting effect with parity

The factorial interaction is also reported:

    (D - C) - (B - A)

which quantifies whether the odd/even parity term changes the OLS-versus-WLS
field difference.

GRE04c does not fit phase data itself.

Inputs
------
Per case:
- GRE/04a_field_ols_v010_legacy/    OLS legacy
- GRE/04b_field_wls_v010_legacy/    WLS legacy
- GRE/04a_field_ols/                parity-aware OLS
- GRE/04b_field_wls/                parity-aware WLS

Outputs
-------
derivatives/<CASE>/GRE/04c_field_fit_model_comparison/
  <CASE>_GRE04c_common_four_way_valid_mask.nii.gz
  <CASE>_GRE04c_weighting_legacy_delta_hz.nii.gz
  <CASE>_GRE04c_weighting_legacy_delta_ppm.nii.gz
  <CASE>_GRE04c_parity_ols_delta_hz.nii.gz
  <CASE>_GRE04c_parity_ols_delta_ppm.nii.gz
  <CASE>_GRE04c_parity_wls_delta_hz.nii.gz
  <CASE>_GRE04c_parity_wls_delta_ppm.nii.gz
  <CASE>_GRE04c_weighting_parity_delta_hz.nii.gz
  <CASE>_GRE04c_weighting_parity_delta_ppm.nii.gz
  <CASE>_GRE04c_weighting_by_parity_interaction_hz.nii.gz
  <CASE>_GRE04c_weighting_by_parity_interaction_ppm.nii.gz
  <CASE>_GRE04c_beta_wls_minus_ols_rad.nii.gz
  <CASE>_GRE04c_qc.json

Optional plots:
  <CASE>_GRE04c_four_contrast_histograms_hz.(png|pdf)
  <CASE>_GRE04c_parity_ols_vs_wls_scatter_hz.(png|pdf)
  <CASE>_GRE04c_interaction_histogram_hz.(png|pdf)

Cohort outputs
--------------
derivatives/summary/GRE/
  gre04c_field_fit_model_comparison_qc.csv
  gre04c_field_fit_model_comparison_run.json

Run
---
./apf scripts/gre/04c_compare_ols_wls.py \
  --all-gre-ready \
  --write-plots \
  --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from _field_fit_common import (
    list_cases,
    now_iso,
    resolve_root,
    save_nifti,
    write_case_summary_json,
    write_cohort_csv,
)


SCRIPT_VERSION = "0.2.0"
MODULE_NAME = "gre04c"
METHOD_NAME = "four_scenario_ols_wls_parity_model_comparison"
OUTPUT_SUBDIR = Path("GRE") / "04c_field_fit_model_comparison"


@dataclass(frozen=True)
class Inputs:
    case_id: str

    ols_legacy_hz: Path
    ols_legacy_ppm: Path
    ols_legacy_valid: Path
    ols_legacy_rmse: Path
    ols_legacy_r2: Path
    ols_legacy_se_hz: Path

    wls_legacy_hz: Path
    wls_legacy_ppm: Path
    wls_legacy_valid: Path
    wls_legacy_rmse: Path
    wls_legacy_r2: Path
    wls_legacy_se_hz: Path
    wls_legacy_weighted_rmse: Path
    wls_legacy_weighted_r2: Path

    ols_parity_hz: Path
    ols_parity_ppm: Path
    ols_parity_valid: Path
    ols_parity_rmse: Path
    ols_parity_r2: Path
    ols_parity_se_hz: Path
    ols_parity_beta: Path

    wls_parity_hz: Path
    wls_parity_ppm: Path
    wls_parity_valid: Path
    wls_parity_rmse: Path
    wls_parity_r2: Path
    wls_parity_se_hz: Path
    wls_parity_weighted_rmse: Path
    wls_parity_weighted_r2: Path
    wls_parity_beta: Path


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None

    common_valid_voxels: int | None = None
    common_fraction_ols_legacy: float | None = None
    common_fraction_wls_legacy: float | None = None
    common_fraction_ols_parity: float | None = None
    common_fraction_wls_parity: float | None = None

    weighting_legacy_rms_hz: float | None = None
    weighting_legacy_mae_hz: float | None = None
    weighting_legacy_median_hz: float | None = None
    weighting_legacy_pearson_r: float | None = None

    parity_ols_rms_hz: float | None = None
    parity_ols_mae_hz: float | None = None
    parity_ols_median_hz: float | None = None
    parity_ols_pearson_r: float | None = None

    parity_wls_rms_hz: float | None = None
    parity_wls_mae_hz: float | None = None
    parity_wls_median_hz: float | None = None
    parity_wls_pearson_r: float | None = None

    weighting_parity_rms_hz: float | None = None
    weighting_parity_mae_hz: float | None = None
    weighting_parity_median_hz: float | None = None
    weighting_parity_pearson_r: float | None = None

    interaction_rms_hz: float | None = None
    interaction_mae_hz: float | None = None
    interaction_median_hz: float | None = None

    beta_ols_wls_pearson_r: float | None = None
    beta_wls_minus_ols_rms_rad: float | None = None
    beta_wls_minus_ols_median_rad: float | None = None

    median_ols_legacy_rmse_rad: float | None = None
    median_wls_legacy_rmse_rad: float | None = None
    median_ols_parity_rmse_rad: float | None = None
    median_wls_parity_rmse_rad: float | None = None

    median_ols_legacy_r2: float | None = None
    median_wls_legacy_r2: float | None = None
    median_ols_parity_r2: float | None = None
    median_wls_parity_r2: float | None = None

    median_ols_legacy_se_hz: float | None = None
    median_wls_legacy_se_hz: float | None = None
    median_ols_parity_se_hz: float | None = None
    median_wls_parity_se_hz: float | None = None

    fraction_parity_lower_rmse_ols: float | None = None
    fraction_parity_lower_rmse_wls: float | None = None
    fraction_wls_lower_rmse_parity: float | None = None

    fraction_parity_higher_r2_ols: float | None = None
    fraction_parity_higher_r2_wls: float | None = None
    fraction_wls_higher_r2_parity: float | None = None

    median_wls_legacy_weighted_rmse_rad: float | None = None
    median_wls_parity_weighted_rmse_rad: float | None = None
    median_wls_legacy_weighted_r2: float | None = None
    median_wls_parity_weighted_r2: float | None = None
    fraction_parity_lower_weighted_rmse_wls: float | None = None
    fraction_parity_higher_weighted_r2_wls: float | None = None

    output_dir: str | None = None


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def model_paths(case_root: Path, case_id: str, subdir: str, prefix: str) -> dict[str, Path]:
    root = case_root / subdir
    return {
        "hz": root / f"{case_id}_{prefix}_total_field_hz.nii.gz",
        "ppm": root / f"{case_id}_{prefix}_total_field_ppm.nii.gz",
        "valid": root / f"{case_id}_{prefix}_valid_fit_mask.nii.gz",
        "rmse": root / f"{case_id}_{prefix}_phase_rmse_rad.nii.gz",
        "r2": root / f"{case_id}_{prefix}_phase_r2.nii.gz",
        "se": root / f"{case_id}_{prefix}_frequency_se_hz.nii.gz",
    }


def discover_inputs(root: Path, case_id: str) -> Inputs:
    case_root = root / "derivatives" / case_id / "GRE"

    ols_l = model_paths(case_root, case_id, "04a_field_ols_v010_legacy", "GRE04a")
    wls_l = model_paths(case_root, case_id, "04b_field_wls_v010_legacy", "GRE04b")
    ols_p = model_paths(case_root, case_id, "04a_field_ols", "GRE04a")
    wls_p = model_paths(case_root, case_id, "04b_field_wls", "GRE04b")

    return Inputs(
        case_id=case_id,
        ols_legacy_hz=require_file(ols_l["hz"], "OLS legacy field Hz"),
        ols_legacy_ppm=require_file(ols_l["ppm"], "OLS legacy field ppm"),
        ols_legacy_valid=require_file(ols_l["valid"], "OLS legacy valid mask"),
        ols_legacy_rmse=require_file(ols_l["rmse"], "OLS legacy RMSE"),
        ols_legacy_r2=require_file(ols_l["r2"], "OLS legacy R2"),
        ols_legacy_se_hz=require_file(ols_l["se"], "OLS legacy frequency SE"),
        wls_legacy_hz=require_file(wls_l["hz"], "WLS legacy field Hz"),
        wls_legacy_ppm=require_file(wls_l["ppm"], "WLS legacy field ppm"),
        wls_legacy_valid=require_file(wls_l["valid"], "WLS legacy valid mask"),
        wls_legacy_rmse=require_file(wls_l["rmse"], "WLS legacy RMSE"),
        wls_legacy_r2=require_file(wls_l["r2"], "WLS legacy R2"),
        wls_legacy_se_hz=require_file(wls_l["se"], "WLS legacy frequency SE"),
        wls_legacy_weighted_rmse=require_file(
            case_root / "04b_field_wls_v010_legacy" /
            f"{case_id}_GRE04b_phase_weighted_rmse_rad.nii.gz",
            "WLS legacy weighted RMSE",
        ),
        wls_legacy_weighted_r2=require_file(
            case_root / "04b_field_wls_v010_legacy" /
            f"{case_id}_GRE04b_phase_weighted_r2.nii.gz",
            "WLS legacy weighted R2",
        ),
        ols_parity_hz=require_file(ols_p["hz"], "OLS parity field Hz"),
        ols_parity_ppm=require_file(ols_p["ppm"], "OLS parity field ppm"),
        ols_parity_valid=require_file(ols_p["valid"], "OLS parity valid mask"),
        ols_parity_rmse=require_file(ols_p["rmse"], "OLS parity RMSE"),
        ols_parity_r2=require_file(ols_p["r2"], "OLS parity R2"),
        ols_parity_se_hz=require_file(ols_p["se"], "OLS parity frequency SE"),
        ols_parity_beta=require_file(
            case_root / "04a_field_ols" /
            f"{case_id}_GRE04a_beta_odd_even_rad.nii.gz",
            "OLS parity beta",
        ),
        wls_parity_hz=require_file(wls_p["hz"], "WLS parity field Hz"),
        wls_parity_ppm=require_file(wls_p["ppm"], "WLS parity field ppm"),
        wls_parity_valid=require_file(wls_p["valid"], "WLS parity valid mask"),
        wls_parity_rmse=require_file(wls_p["rmse"], "WLS parity RMSE"),
        wls_parity_r2=require_file(wls_p["r2"], "WLS parity R2"),
        wls_parity_se_hz=require_file(wls_p["se"], "WLS parity frequency SE"),
        wls_parity_weighted_rmse=require_file(
            case_root / "04b_field_wls" /
            f"{case_id}_GRE04b_phase_weighted_rmse_rad.nii.gz",
            "WLS parity weighted RMSE",
        ),
        wls_parity_weighted_r2=require_file(
            case_root / "04b_field_wls" /
            f"{case_id}_GRE04b_phase_weighted_r2.nii.gz",
            "WLS parity weighted R2",
        ),
        wls_parity_beta=require_file(
            case_root / "04b_field_wls" /
            f"{case_id}_GRE04b_beta_odd_even_rad.nii.gz",
            "WLS parity beta",
        ),
    )


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case_id / OUTPUT_SUBDIR
    prefix = f"{case_id}_GRE04c"
    return {
        "dir": outdir,
        "common_mask": outdir / f"{prefix}_common_four_way_valid_mask.nii.gz",
        "weighting_legacy_hz": outdir / f"{prefix}_weighting_legacy_delta_hz.nii.gz",
        "weighting_legacy_ppm": outdir / f"{prefix}_weighting_legacy_delta_ppm.nii.gz",
        "parity_ols_hz": outdir / f"{prefix}_parity_ols_delta_hz.nii.gz",
        "parity_ols_ppm": outdir / f"{prefix}_parity_ols_delta_ppm.nii.gz",
        "parity_wls_hz": outdir / f"{prefix}_parity_wls_delta_hz.nii.gz",
        "parity_wls_ppm": outdir / f"{prefix}_parity_wls_delta_ppm.nii.gz",
        "weighting_parity_hz": outdir / f"{prefix}_weighting_parity_delta_hz.nii.gz",
        "weighting_parity_ppm": outdir / f"{prefix}_weighting_parity_delta_ppm.nii.gz",
        "interaction_hz": outdir / f"{prefix}_weighting_by_parity_interaction_hz.nii.gz",
        "interaction_ppm": outdir / f"{prefix}_weighting_by_parity_interaction_ppm.nii.gz",
        "beta_difference": outdir / f"{prefix}_beta_wls_minus_ols_rad.nii.gz",
        "contrast_hist_png": outdir / f"{prefix}_four_contrast_histograms_hz.png",
        "contrast_hist_pdf": outdir / f"{prefix}_four_contrast_histograms_hz.pdf",
        "parity_scatter_png": outdir / f"{prefix}_parity_ols_vs_wls_scatter_hz.png",
        "parity_scatter_pdf": outdir / f"{prefix}_parity_ols_vs_wls_scatter_hz.pdf",
        "interaction_hist_png": outdir / f"{prefix}_interaction_histogram_hz.png",
        "interaction_hist_pdf": outdir / f"{prefix}_interaction_histogram_hz.pdf",
        "qc": outdir / f"{prefix}_qc.json",
    }


def load_image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {data.shape}: {path}")
    return img, data


def validate_geometry(reference_img: nib.Nifti1Image, other_img: nib.Nifti1Image, label: str) -> None:
    if tuple(reference_img.shape) != tuple(other_img.shape):
        raise ValueError(f"{label} shape mismatch: {other_img.shape} vs {reference_img.shape}")
    if not np.allclose(reference_img.affine, other_img.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{label} affine mismatch")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def finite_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot summarize an empty finite array")
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mae": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def agreement_summary(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    s = finite_summary(diff)
    return {
        "definition": "second minus first",
        "pearson_r": safe_pearson(a, b),
        "difference": s,
        "bland_altman": {
            "bias": s["mean"],
            "difference_sd": s["sd"],
            "lower_95_limit": s["mean"] - 1.96 * s["sd"],
            "upper_95_limit": s["mean"] + 1.96 * s["sd"],
        },
    }


def reproducible_sample(*arrays: np.ndarray, maximum: int, seed: int) -> tuple[np.ndarray, ...]:
    n = arrays[0].size
    if any(array.size != n for array in arrays):
        raise ValueError("Sample arrays have inconsistent lengths")
    if n <= maximum:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=maximum, replace=False)
    return tuple(array[idx] for array in arrays)


def contrast_map(shape: tuple[int, ...], common: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    out[common] = values.astype(np.float32)
    return out


def write_plots(
    paths: dict[str, Path],
    case_id: str,
    ols_parity_hz: np.ndarray,
    wls_parity_hz: np.ndarray,
    contrasts_hz: dict[str, np.ndarray],
    interaction_hz: np.ndarray,
    *,
    maximum_voxels: int,
    seed: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--write-plots requested but matplotlib is unavailable") from exc

    labels = {
        "weighting_legacy": "WLS legacy − OLS legacy",
        "parity_ols": "OLS parity − OLS legacy",
        "parity_wls": "WLS parity − WLS legacy",
        "weighting_parity": "WLS parity − OLS parity",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key in zip(axes.ravel(), labels):
        sampled, = reproducible_sample(
            contrasts_hz[key], maximum=maximum_voxels, seed=seed
        )
        lim = max(float(np.percentile(np.abs(sampled), 99.5)), np.finfo(float).eps)
        ax.hist(sampled, bins=np.linspace(-lim, lim, 121))
        ax.axvline(0.0, linewidth=1)
        ax.set_title(labels[key])
        ax.set_xlabel("Field difference (Hz)")
        ax.set_ylabel("Sampled voxels")
    fig.suptitle(f"{case_id}: GRE04 four-scenario field contrasts")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(paths["contrast_hist_png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["contrast_hist_pdf"], bbox_inches="tight")
    plt.close(fig)

    x, y = reproducible_sample(
        ols_parity_hz, wls_parity_hz, maximum=maximum_voxels, seed=seed
    )
    low = float(min(np.percentile(x, 0.5), np.percentile(y, 0.5)))
    high = float(max(np.percentile(x, 99.5), np.percentile(y, 99.5)))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=2, alpha=0.15, rasterized=True)
    ax.plot([low, high], [low, high], linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Parity-aware OLS total field (Hz)")
    ax.set_ylabel("Parity-aware WLS total field (Hz)")
    ax.set_title(f"{case_id}: final parity-aware OLS versus WLS")
    fig.tight_layout()
    fig.savefig(paths["parity_scatter_png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["parity_scatter_pdf"], bbox_inches="tight")
    plt.close(fig)

    sampled, = reproducible_sample(
        interaction_hz, maximum=maximum_voxels, seed=seed
    )
    lim = max(float(np.percentile(np.abs(sampled), 99.5)), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sampled, bins=np.linspace(-lim, lim, 121))
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("(WLS parity − OLS parity) − (WLS legacy − OLS legacy) (Hz)")
    ax.set_ylabel("Sampled voxels")
    ax.set_title(f"{case_id}: weighting-by-parity interaction")
    fig.tight_layout()
    fig.savefig(paths["interaction_hist_png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["interaction_hist_pdf"], bbox_inches="tight")
    plt.close(fig)


def process_case(root: Path, case_id: str, args: argparse.Namespace) -> CaseResult:
    started = time.perf_counter()
    paths = output_paths(root, case_id)

    try:
        inputs = discover_inputs(root, case_id)

        if paths["qc"].exists() and not args.force and not args.dry_run:
            return CaseResult(
                case_id=case_id,
                status="SKIPPED_EXISTS",
                message="GRE04c output exists; use --force to overwrite",
                output_dir=str(paths["dir"]),
            )

        if args.dry_run:
            return CaseResult(
                case_id=case_id,
                status="DRY_RUN",
                message=(
                    f"OLS legacy={inputs.ols_legacy_hz}; "
                    f"WLS legacy={inputs.wls_legacy_hz}; "
                    f"OLS parity={inputs.ols_parity_hz}; "
                    f"WLS parity={inputs.wls_parity_hz}"
                ),
                output_dir=str(paths["dir"]),
            )

        paths["dir"].mkdir(parents=True, exist_ok=True)
        if args.force:
            for key, path in paths.items():
                if key != "dir":
                    path.unlink(missing_ok=True)

        reference_img, ols_legacy_hz_map = load_image(inputs.ols_legacy_hz)
        loaded: dict[str, np.ndarray] = {"ols_legacy_hz": ols_legacy_hz_map}

        for label, path_value in asdict(inputs).items():
            if label in {"case_id", "ols_legacy_hz"}:
                continue
            img, data = load_image(Path(path_value))
            validate_geometry(reference_img, img, label)
            loaded[label] = data

        common = np.ones(reference_img.shape, dtype=bool)
        for key in (
            "ols_legacy_valid",
            "wls_legacy_valid",
            "ols_parity_valid",
            "wls_parity_valid",
        ):
            common &= loaded[key] > 0

        for key, array in loaded.items():
            if not key.endswith("_valid"):
                common &= np.isfinite(array)

        common_voxels = int(np.count_nonzero(common))
        if common_voxels < 2:
            raise ValueError(
                f"Insufficient four-way common valid voxels: {common_voxels}"
            )

        valid_counts = {
            "ols_legacy": int(np.count_nonzero(loaded["ols_legacy_valid"] > 0)),
            "wls_legacy": int(np.count_nonzero(loaded["wls_legacy_valid"] > 0)),
            "ols_parity": int(np.count_nonzero(loaded["ols_parity_valid"] > 0)),
            "wls_parity": int(np.count_nonzero(loaded["wls_parity_valid"] > 0)),
        }

        def v(key: str) -> np.ndarray:
            return loaded[key][common].astype(np.float64)

        A_hz, B_hz = v("ols_legacy_hz"), v("wls_legacy_hz")
        C_hz, D_hz = v("ols_parity_hz"), v("wls_parity_hz")
        A_ppm, B_ppm = v("ols_legacy_ppm"), v("wls_legacy_ppm")
        C_ppm, D_ppm = v("ols_parity_ppm"), v("wls_parity_ppm")

        contrasts_hz = {
            "weighting_legacy": B_hz - A_hz,
            "parity_ols": C_hz - A_hz,
            "parity_wls": D_hz - B_hz,
            "weighting_parity": D_hz - C_hz,
        }
        contrasts_ppm = {
            "weighting_legacy": B_ppm - A_ppm,
            "parity_ols": C_ppm - A_ppm,
            "parity_wls": D_ppm - B_ppm,
            "weighting_parity": D_ppm - C_ppm,
        }

        interaction_hz = contrasts_hz["weighting_parity"] - contrasts_hz["weighting_legacy"]
        interaction_ppm = contrasts_ppm["weighting_parity"] - contrasts_ppm["weighting_legacy"]

        beta_ols = v("ols_parity_beta")
        beta_wls = v("wls_parity_beta")
        beta_diff = beta_wls - beta_ols

        agreement = {
            "weighting_without_parity": agreement_summary(A_hz, B_hz),
            "parity_under_ols": agreement_summary(A_hz, C_hz),
            "parity_under_wls": agreement_summary(B_hz, D_hz),
            "weighting_with_parity": agreement_summary(C_hz, D_hz),
        }
        ppm_lookup = {
            "weighting_without_parity": contrasts_ppm["weighting_legacy"],
            "parity_under_ols": contrasts_ppm["parity_ols"],
            "parity_under_wls": contrasts_ppm["parity_wls"],
            "weighting_with_parity": contrasts_ppm["weighting_parity"],
        }
        for key, values in ppm_lookup.items():
            agreement[key]["difference_ppm"] = finite_summary(values)

        interaction_summary_hz = finite_summary(interaction_hz)
        interaction_summary_ppm = finite_summary(interaction_ppm)

        rmse_A, rmse_B = v("ols_legacy_rmse"), v("wls_legacy_rmse")
        rmse_C, rmse_D = v("ols_parity_rmse"), v("wls_parity_rmse")
        r2_A, r2_B = v("ols_legacy_r2"), v("wls_legacy_r2")
        r2_C, r2_D = v("ols_parity_r2"), v("wls_parity_r2")
        se_A, se_B = v("ols_legacy_se_hz"), v("wls_legacy_se_hz")
        se_C, se_D = v("ols_parity_se_hz"), v("wls_parity_se_hz")
        wrmse_B = v("wls_legacy_weighted_rmse")
        wrmse_D = v("wls_parity_weighted_rmse")
        wr2_B = v("wls_legacy_weighted_r2")
        wr2_D = v("wls_parity_weighted_r2")

        quality = {
            "models": {
                "ols_legacy": {
                    "rmse_rad": finite_summary(rmse_A),
                    "r2": finite_summary(r2_A),
                    "frequency_se_hz": finite_summary(se_A),
                },
                "wls_legacy": {
                    "rmse_rad": finite_summary(rmse_B),
                    "r2": finite_summary(r2_B),
                    "frequency_se_hz": finite_summary(se_B),
                    "weighted_rmse_rad": finite_summary(wrmse_B),
                    "weighted_r2": finite_summary(wr2_B),
                },
                "ols_parity": {
                    "rmse_rad": finite_summary(rmse_C),
                    "r2": finite_summary(r2_C),
                    "frequency_se_hz": finite_summary(se_C),
                },
                "wls_parity": {
                    "rmse_rad": finite_summary(rmse_D),
                    "r2": finite_summary(r2_D),
                    "frequency_se_hz": finite_summary(se_D),
                    "weighted_rmse_rad": finite_summary(wrmse_D),
                    "weighted_r2": finite_summary(wr2_D),
                },
            },
            "fractions": {
                "parity_lower_rmse_ols": float(np.mean(rmse_C < rmse_A)),
                "parity_lower_rmse_wls": float(np.mean(rmse_D < rmse_B)),
                "wls_lower_rmse_parity": float(np.mean(rmse_D < rmse_C)),
                "parity_higher_r2_ols": float(np.mean(r2_C > r2_A)),
                "parity_higher_r2_wls": float(np.mean(r2_D > r2_B)),
                "wls_higher_r2_parity": float(np.mean(r2_D > r2_C)),
                "parity_lower_weighted_rmse_wls": float(np.mean(wrmse_D < wrmse_B)),
                "parity_higher_weighted_r2_wls": float(np.mean(wr2_D > wr2_B)),
            },
        }

        beta_agreement = {
            "pearson_r": safe_pearson(beta_ols, beta_wls),
            "ols_beta_rad": finite_summary(beta_ols),
            "wls_beta_rad": finite_summary(beta_wls),
            "wls_minus_ols_beta_rad": finite_summary(beta_diff),
        }

        save_nifti(common.astype(np.uint8), reference_img, paths["common_mask"], np.uint8)
        for key in ("weighting_legacy", "parity_ols", "parity_wls", "weighting_parity"):
            save_nifti(
                contrast_map(reference_img.shape, common, contrasts_hz[key]),
                reference_img,
                paths[f"{key}_hz"],
                np.float32,
            )
            save_nifti(
                contrast_map(reference_img.shape, common, contrasts_ppm[key]),
                reference_img,
                paths[f"{key}_ppm"],
                np.float32,
            )
        save_nifti(
            contrast_map(reference_img.shape, common, interaction_hz),
            reference_img,
            paths["interaction_hz"],
            np.float32,
        )
        save_nifti(
            contrast_map(reference_img.shape, common, interaction_ppm),
            reference_img,
            paths["interaction_ppm"],
            np.float32,
        )
        save_nifti(
            contrast_map(reference_img.shape, common, beta_diff),
            reference_img,
            paths["beta_difference"],
            np.float32,
        )

        if args.write_plots:
            write_plots(
                paths,
                case_id,
                C_hz,
                D_hz,
                contrasts_hz,
                interaction_hz,
                maximum_voxels=args.plot_sample_voxels,
                seed=args.random_seed,
            )

        runtime_s = time.perf_counter() - started

        provenance = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "method": METHOD_NAME,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": runtime_s,
            "model_definitions": {
                "A_ols_legacy": "equal weighting, no parity term",
                "B_wls_legacy": "magnitude-squared weighting, no parity term",
                "C_ols_parity": "equal weighting, odd/even parity term",
                "D_wls_parity": "magnitude-squared weighting, odd/even parity term",
            },
            "contrast_definitions": {
                "weighting_without_parity": "B - A",
                "parity_under_ols": "C - A",
                "parity_under_wls": "D - B",
                "weighting_with_parity": "D - C",
                "weighting_by_parity_interaction": "(D - C) - (B - A)",
            },
            "inputs": {
                key: str(value)
                for key, value in asdict(inputs).items()
                if key != "case_id"
            },
            "configuration": {
                "voxel_support": (
                    "single four-way common valid mask across A/B/C/D and all loaded comparison metrics"
                ),
                "bland_altman_limits": "bias ± 1.96 * sample SD of paired field differences",
                "write_plots": args.write_plots,
                "plot_sample_voxels": args.plot_sample_voxels,
                "random_seed": args.random_seed,
            },
            "voxel_support": {
                "ols_legacy_valid_voxels": valid_counts["ols_legacy"],
                "wls_legacy_valid_voxels": valid_counts["wls_legacy"],
                "ols_parity_valid_voxels": valid_counts["ols_parity"],
                "wls_parity_valid_voxels": valid_counts["wls_parity"],
                "common_four_way_valid_voxels": common_voxels,
                "common_fraction_ols_legacy": common_voxels / valid_counts["ols_legacy"],
                "common_fraction_wls_legacy": common_voxels / valid_counts["wls_legacy"],
                "common_fraction_ols_parity": common_voxels / valid_counts["ols_parity"],
                "common_fraction_wls_parity": common_voxels / valid_counts["wls_parity"],
            },
            "agreement": agreement,
            "interaction": {
                "field_difference_hz": interaction_summary_hz,
                "field_difference_ppm": interaction_summary_ppm,
            },
            "parity_coefficient_agreement": beta_agreement,
            "quality_metrics": quality,
            "outputs": {
                key: str(value)
                for key, value in paths.items()
                if key != "dir" and (value.exists() or key == "qc")
            },
        }
        write_case_summary_json(paths["qc"], provenance)

        return CaseResult(
            case_id=case_id,
            status="OK",
            message="Four-scenario GRE04 field-fit comparison completed",
            runtime_s=runtime_s,
            common_valid_voxels=common_voxels,
            common_fraction_ols_legacy=common_voxels / valid_counts["ols_legacy"],
            common_fraction_wls_legacy=common_voxels / valid_counts["wls_legacy"],
            common_fraction_ols_parity=common_voxels / valid_counts["ols_parity"],
            common_fraction_wls_parity=common_voxels / valid_counts["wls_parity"],
            weighting_legacy_rms_hz=agreement["weighting_without_parity"]["difference"]["rms"],
            weighting_legacy_mae_hz=agreement["weighting_without_parity"]["difference"]["mae"],
            weighting_legacy_median_hz=agreement["weighting_without_parity"]["difference"]["median"],
            weighting_legacy_pearson_r=agreement["weighting_without_parity"]["pearson_r"],
            parity_ols_rms_hz=agreement["parity_under_ols"]["difference"]["rms"],
            parity_ols_mae_hz=agreement["parity_under_ols"]["difference"]["mae"],
            parity_ols_median_hz=agreement["parity_under_ols"]["difference"]["median"],
            parity_ols_pearson_r=agreement["parity_under_ols"]["pearson_r"],
            parity_wls_rms_hz=agreement["parity_under_wls"]["difference"]["rms"],
            parity_wls_mae_hz=agreement["parity_under_wls"]["difference"]["mae"],
            parity_wls_median_hz=agreement["parity_under_wls"]["difference"]["median"],
            parity_wls_pearson_r=agreement["parity_under_wls"]["pearson_r"],
            weighting_parity_rms_hz=agreement["weighting_with_parity"]["difference"]["rms"],
            weighting_parity_mae_hz=agreement["weighting_with_parity"]["difference"]["mae"],
            weighting_parity_median_hz=agreement["weighting_with_parity"]["difference"]["median"],
            weighting_parity_pearson_r=agreement["weighting_with_parity"]["pearson_r"],
            interaction_rms_hz=interaction_summary_hz["rms"],
            interaction_mae_hz=interaction_summary_hz["mae"],
            interaction_median_hz=interaction_summary_hz["median"],
            beta_ols_wls_pearson_r=beta_agreement["pearson_r"],
            beta_wls_minus_ols_rms_rad=beta_agreement["wls_minus_ols_beta_rad"]["rms"],
            beta_wls_minus_ols_median_rad=beta_agreement["wls_minus_ols_beta_rad"]["median"],
            median_ols_legacy_rmse_rad=float(np.median(rmse_A)),
            median_wls_legacy_rmse_rad=float(np.median(rmse_B)),
            median_ols_parity_rmse_rad=float(np.median(rmse_C)),
            median_wls_parity_rmse_rad=float(np.median(rmse_D)),
            median_ols_legacy_r2=float(np.median(r2_A)),
            median_wls_legacy_r2=float(np.median(r2_B)),
            median_ols_parity_r2=float(np.median(r2_C)),
            median_wls_parity_r2=float(np.median(r2_D)),
            median_ols_legacy_se_hz=float(np.median(se_A)),
            median_wls_legacy_se_hz=float(np.median(se_B)),
            median_ols_parity_se_hz=float(np.median(se_C)),
            median_wls_parity_se_hz=float(np.median(se_D)),
            fraction_parity_lower_rmse_ols=quality["fractions"]["parity_lower_rmse_ols"],
            fraction_parity_lower_rmse_wls=quality["fractions"]["parity_lower_rmse_wls"],
            fraction_wls_lower_rmse_parity=quality["fractions"]["wls_lower_rmse_parity"],
            fraction_parity_higher_r2_ols=quality["fractions"]["parity_higher_r2_ols"],
            fraction_parity_higher_r2_wls=quality["fractions"]["parity_higher_r2_wls"],
            fraction_wls_higher_r2_parity=quality["fractions"]["wls_higher_r2_parity"],
            median_wls_legacy_weighted_rmse_rad=float(np.median(wrmse_B)),
            median_wls_parity_weighted_rmse_rad=float(np.median(wrmse_D)),
            median_wls_legacy_weighted_r2=float(np.median(wr2_B)),
            median_wls_parity_weighted_r2=float(np.median(wr2_D)),
            fraction_parity_lower_weighted_rmse_wls=quality["fractions"]["parity_lower_weighted_rmse_wls"],
            fraction_parity_higher_weighted_r2_wls=quality["fractions"]["parity_higher_weighted_r2_wls"],
            output_dir=str(paths["dir"]),
        )

    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - started,
            output_dir=str(paths["dir"]),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "APF GRE04c: compare four GRE total-field fit scenarios "
            "(OLS/WLS x no-parity/parity)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, help="APF repository root; defaults to APF_ROOT or cwd")
    parser.add_argument("--case", action="append", default=[], help="Case ID; repeat for multiple cases")
    parser.add_argument("--all-gre-ready", action="store_true", help="Attempt all discovered ANISO_*_InS cases")
    parser.add_argument("--write-plots", action="store_true", help="Write standardized four-scenario diagnostic plots")
    parser.add_argument("--plot-sample-voxels", type=int, default=100000, help="Maximum reproducibly sampled voxels per plot")
    parser.add_argument("--random-seed", type=int, default=20260819, help="Random seed for reproducible plot sampling")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GRE04c v0.2 outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate four-scenario input discovery without loading image data")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    if args.plot_sample_voxels < 100:
        raise SystemExit("--plot-sample-voxels must be at least 100")

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit("Select at least one --case CASE_ID or use --all-gre-ready")

    print(f"APF GRE04c v{SCRIPT_VERSION}")
    print(f"Root:   {root}")
    print(f"Method: {METHOD_NAME}")
    print(f"Cases:  {len(cases)}")
    print(f"Plots:  {'yes' if args.write_plots else 'no'}")
    print("Models: A=OLS legacy, B=WLS legacy, C=OLS parity, D=WLS parity")

    started_utc = now_iso()
    results: list[CaseResult] = []

    for case_id in cases:
        result = process_case(root, case_id, args)
        results.append(result)
        if result.status == "OK":
            print(
                f"[OK] {case_id}: "
                f"Wlegacy={result.weighting_legacy_rms_hz:.3f} Hz; "
                f"POLS={result.parity_ols_rms_hz:.3f} Hz; "
                f"PWLS={result.parity_wls_rms_hz:.3f} Hz; "
                f"Wparity={result.weighting_parity_rms_hz:.3f} Hz; "
                f"interaction={result.interaction_rms_hz:.3f} Hz"
            )
        else:
            print(f"[{result.status}] {case_id}: {result.message}")

    summary_dir = root / "derivatives" / "summary" / "GRE"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "gre04c_field_fit_model_comparison_qc.csv"
    run_path = summary_dir / "gre04c_field_fit_model_comparison_run.json"

    write_cohort_csv(csv_path, results)

    ok_rows = [r for r in results if r.status == "OK"]
    cohort_summary = {}
    if ok_rows:
        keys = (
            "weighting_legacy_rms_hz",
            "parity_ols_rms_hz",
            "parity_wls_rms_hz",
            "weighting_parity_rms_hz",
            "interaction_rms_hz",
            "beta_ols_wls_pearson_r",
            "beta_wls_minus_ols_rms_rad",
            "median_ols_legacy_rmse_rad",
            "median_wls_legacy_rmse_rad",
            "median_ols_parity_rmse_rad",
            "median_wls_parity_rmse_rad",
            "fraction_parity_lower_rmse_ols",
            "fraction_parity_lower_rmse_wls",
            "fraction_wls_lower_rmse_parity",
            "fraction_parity_lower_weighted_rmse_wls",
        )
        for key in keys:
            values = np.asarray([getattr(r, key) for r in ok_rows], dtype=np.float64)
            cohort_summary[key] = {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }

    run_payload = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "started_utc": started_utc,
        "finished_utc": now_iso(),
        "root": str(root),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_definitions": {
            "A": "OLS legacy: equal weighting, no parity",
            "B": "WLS legacy: magnitude-squared weighting, no parity",
            "C": "OLS parity: equal weighting, odd/even parity",
            "D": "WLS parity: magnitude-squared weighting, odd/even parity",
        },
        "contrast_definitions": {
            "weighting_without_parity": "B - A",
            "parity_under_ols": "C - A",
            "parity_under_wls": "D - B",
            "weighting_with_parity": "D - C",
            "weighting_by_parity_interaction": "(D - C) - (B - A)",
        },
        "cohort_summary": cohort_summary,
        "results": [asdict(result) for result in results],
    }
    run_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

    n_failed = sum(r.status == "FAILED" for r in results)
    n_ok = sum(r.status == "OK" for r in results)

    print()
    if ok_rows:
        print("COHORT MEDIANS")
        print("=" * 72)
        for key in (
            "weighting_legacy_rms_hz",
            "parity_ols_rms_hz",
            "parity_wls_rms_hz",
            "weighting_parity_rms_hz",
            "interaction_rms_hz",
        ):
            print(f"{key:34s}: {cohort_summary[key]['median']:.6f}")

    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {run_path}")
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
