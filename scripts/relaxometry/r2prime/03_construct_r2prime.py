#!/usr/bin/env python3
"""
APF R2PRIME03: construct R2-prime in native GRE space.

Definition
----------
    R2prime = R2star - R2app_GRE

Production support
------------------
The raw R2-prime map is evaluated only within the intersection of:

    - R2STAR02 final fit mask;
    - R2PRIME02 final GRE-space R2app-valid mask;
    - finite R2star values;
    - finite R2app values.

No clipping of negative R2-prime values is performed in this module.
Negative values are retained and quantified for QC so that any later
non-negativity policy can be justified explicitly before chi-separation.

Outputs
-------
Per case:
    derivatives/<case>/relaxometry/r2prime/r2prime03_construct/

Cohort:
    derivatives/summary/relaxometry/r2prime/r2prime03_construct/
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np

from apf_paths import APFPaths, get_paths
from apf_case_selection import (
    add_case_selection_args,
    print_selected_cases,
    resolve_cases,
)


SCRIPT = "03_construct_r2prime.py"
TAG = "r2prime03"
VERSION = "0.2.0"

DEFAULT_AFFINE_ATOL = 1e-5


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}:\n  {path}"
        )
    return path


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


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)

    img = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        header=header,
    )

    nib.save(img, str(path))


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


def stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------


def r2star02_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2star"
        / "r2star02_loglinear"
    )


def r2star_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2star02_dir(paths, case)
        / f"r2star02_{case}_r2star_s-1.nii.gz"
    )


def r2star_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2star02_dir(paths, case)
        / f"r2star02_{case}_fit_mask.nii.gz"
    )


def r2star_qc_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2star02_dir(paths, case)
        / f"r2star02_{case}_qc.json"
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


def r2app_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(paths, case)
        / f"r2prime02_{case}_R2app_in_GRE_s-1.nii.gz"
    )


def r2app_mask_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(paths, case)
        / f"r2prime02_{case}_valid_mask_in_GRE.nii.gz"
    )


def r2prime02_metadata_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        r2prime02_dir(paths, case)
        / f"r2prime02_{case}_metadata.json"
    )


def gre_brain_mask_path(
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


def output_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2prime"
        / "r2prime03_construct"
    )


def cohort_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "r2prime"
        / "r2prime03_construct"
    )


# -------------------------------------------------------------------------
# QC figure
# -------------------------------------------------------------------------


def robust_limits(
    data: np.ndarray,
    mask: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[float, float]:
    values = data[
        mask
        & np.isfinite(data)
    ]

    if values.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(
        values,
        [p_low, p_high],
    )

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.min(values)), float(np.max(values))

    return float(lo), float(hi)


def make_qc_figure(
    case: str,
    r2star: np.ndarray,
    r2app: np.ndarray,
    r2prime: np.ndarray,
    common_mask: np.ndarray,
    negative_mask: np.ndarray,
    negative_fraction_pct: float,
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Coronal central slice of common support extent.
    coords = np.argwhere(common_mask)

    if coords.size == 0:
        raise RuntimeError("Cannot create QC: common support is empty.")

    y = int(
        np.round(
            np.median(coords[:, 1])
        )
    )

    r2star_lo, r2star_hi = robust_limits(
        r2star,
        common_mask,
    )

    r2app_lo, r2app_hi = robust_limits(
        r2app,
        common_mask,
    )

    rp_lo, rp_hi = robust_limits(
        r2prime,
        common_mask,
        1,
        99,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    axes[0, 0].imshow(
        r2star[:, y, :].T,
        origin="lower",
        cmap="gray",
        vmin=r2star_lo,
        vmax=r2star_hi,
    )
    axes[0, 0].set_title("R2* [s$^{-1}$]")

    axes[0, 1].imshow(
        r2app[:, y, :].T,
        origin="lower",
        cmap="gray",
        vmin=r2app_lo,
        vmax=r2app_hi,
    )
    axes[0, 1].set_title("R2app in GRE [s$^{-1}$]")

    axes[0, 2].imshow(
        r2prime[:, y, :].T,
        origin="lower",
        cmap="gray",
        vmin=rp_lo,
        vmax=rp_hi,
    )
    axes[0, 2].set_title("Raw R2' [s$^{-1}$]")

    axes[1, 0].imshow(
        common_mask[:, y, :].T,
        origin="lower",
        cmap="gray",
    )
    axes[1, 0].set_title("Common valid support")

    axes[1, 1].imshow(
        common_mask[:, y, :].T,
        origin="lower",
        cmap="gray",
    )
    axes[1, 1].imshow(
        np.ma.masked_where(
            ~negative_mask[:, y, :].T,
            negative_mask[:, y, :].T,
        ),
        origin="lower",
        alpha=0.65,
    )
    axes[1, 1].set_title(
        f"Negative R2' ({negative_fraction_pct:.2f}% of support)"
    )

    values = r2prime[
        common_mask
        & np.isfinite(r2prime)
    ]

    axes[1, 2].hist(
        values,
        bins=150,
    )
    axes[1, 2].axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    axes[1, 2].set_title("R2' distribution")
    axes[1, 2].set_xlabel("R2' [s$^{-1}$]")
    axes[1, 2].set_ylabel("Voxels")

    for ax in axes.ravel()[:5]:
        ax.axis("off")

    fig.suptitle(
        f"R2PRIME03 — {case}\n"
        f"Raw R2' = R2* - R2app; "
        f"negative fraction = {negative_fraction_pct:.3f}%"
    )

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


# -------------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    affine_atol: float,
    overwrite: bool,
) -> Dict[str, Any]:

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    start = now_iso()

    p_r2star = require_file(
        r2star_path(paths, case),
        "R2STAR02 R2star map",
    )

    p_r2star_mask = require_file(
        r2star_mask_path(paths, case),
        "R2STAR02 fit mask",
    )

    p_r2star_qc = require_file(
        r2star_qc_path(paths, case),
        "R2STAR02 QC metadata",
    )

    p_r2app = require_file(
        r2app_path(paths, case),
        "R2PRIME02 R2app map",
    )

    p_r2app_mask = require_file(
        r2app_mask_path(paths, case),
        "R2PRIME02 valid mask",
    )

    p_r2prime02_meta = require_file(
        r2prime02_metadata_path(paths, case),
        "R2PRIME02 metadata",
    )

    p_gre_mask = require_file(
        gre_brain_mask_path(paths, case),
        "GRE02b brain mask",
    )

    out = output_dir(
        paths,
        case,
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = f"r2prime03_{case}"

    p_common = out / f"{prefix}_common_mask.nii.gz"
    p_raw = out / f"{prefix}_R2prime_raw_s-1.nii.gz"
    p_negative = out / f"{prefix}_negative_mask.nii.gz"
    p_zero = out / f"{prefix}_zero_mask.nii.gz"
    p_qc_png = out / f"{prefix}_qc.png"
    p_qc_pdf = out / f"{prefix}_qc.pdf"
    p_metadata = out / f"{prefix}_metadata.json"
    p_summary = out / f"{prefix}_summary.csv"

    if (
        p_metadata.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {p_metadata}\n"
            "Use --overwrite to replace."
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    r2star_img = nib.load(
        str(p_r2star)
    )

    r2star_mask_img = nib.load(
        str(p_r2star_mask)
    )

    r2app_img = nib.load(
        str(p_r2app)
    )

    r2app_mask_img = nib.load(
        str(p_r2app_mask)
    )

    gre_mask_img = nib.load(
        str(p_gre_mask)
    )

    references = {
        "R2STAR02 fit mask":
            r2star_mask_img,

        "R2PRIME02 R2app":
            r2app_img,

        "R2PRIME02 valid mask":
            r2app_mask_img,

        "GRE02b brain mask":
            gre_mask_img,
    }

    for label, img in references.items():
        if not geometry_equal(
            r2star_img,
            img,
            affine_atol,
        ):
            raise RuntimeError(
                f"Geometry mismatch: R2STAR02 versus {label}"
            )

    r2star = np.asarray(
        r2star_img.dataobj,
        dtype=np.float32,
    )

    r2star_mask = (
        np.asarray(
            r2star_mask_img.dataobj
        )
        > 0
    )

    r2app = np.asarray(
        r2app_img.dataobj,
        dtype=np.float32,
    )

    r2app_mask = (
        np.asarray(
            r2app_mask_img.dataobj
        )
        > 0
    )

    gre_mask = (
        np.asarray(
            gre_mask_img.dataobj
        )
        > 0
    )

    # ------------------------------------------------------------------
    # Common support
    # ------------------------------------------------------------------

    common = (
        r2star_mask
        & r2app_mask
        & np.isfinite(r2star)
        & np.isfinite(r2app)
    )

    n_common = int(
        np.count_nonzero(common)
    )

    if n_common == 0:
        raise RuntimeError(
            "Common R2star/R2app support is empty."
        )

    n_gre = int(
        np.count_nonzero(gre_mask)
    )

    n_common_in_gre = int(
        np.count_nonzero(
            common
            & gre_mask
        )
    )

    # ------------------------------------------------------------------
    # Raw R2-prime
    # ------------------------------------------------------------------

    r2prime = np.full(
        r2star.shape,
        np.nan,
        dtype=np.float32,
    )

    r2prime[common] = (
        r2star[common]
        - r2app[common]
    )

    negative = (
        common
        & (r2prime < 0)
    )

    zero = (
        common
        & (r2prime == 0)
    )

    n_negative = int(
        np.count_nonzero(negative)
    )

    n_zero = int(
        np.count_nonzero(zero)
    )

    negative_fraction_pct = (
        100.0
        * n_negative
        / n_common
    )

    zero_fraction_pct = (
        100.0
        * n_zero
        / n_common
    )

    common_gre_coverage_pct = (
        100.0
        * n_common_in_gre
        / n_gre
    )

    r2star_stats = stats(
        r2star[common]
    )

    r2app_stats = stats(
        r2app[common]
    )

    r2prime_stats = stats(
        r2prime[common]
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_nifti(
        common.astype(np.uint8),
        r2star_img,
        p_common,
        np.uint8,
    )

    save_nifti(
        r2prime,
        r2star_img,
        p_raw,
        np.float32,
    )

    save_nifti(
        negative.astype(np.uint8),
        r2star_img,
        p_negative,
        np.uint8,
    )

    save_nifti(
        zero.astype(np.uint8),
        r2star_img,
        p_zero,
        np.uint8,
    )

    make_qc_figure(
        case=case,
        r2star=r2star,
        r2app=r2app,
        r2prime=r2prime,
        common_mask=common,
        negative_mask=negative,
        negative_fraction_pct=negative_fraction_pct,
        output_png=p_qc_png,
        output_pdf=p_qc_pdf,
    )

    end = now_iso()

    row = {
        "case":
            case,

        "status":
            "PASS",

        "common_voxels":
            n_common,

        "GRE_brain_voxels":
            n_gre,

        "common_in_GRE_voxels":
            n_common_in_gre,

        "common_GRE_coverage_pct":
            common_gre_coverage_pct,

        "negative_R2prime_voxels":
            n_negative,

        "negative_R2prime_fraction_pct":
            negative_fraction_pct,

        "zero_R2prime_voxels":
            n_zero,

        "zero_R2prime_fraction_pct":
            zero_fraction_pct,

        "R2star_median_s-1":
            r2star_stats["median"],

        "R2app_median_s-1":
            r2app_stats["median"],

        "R2prime_median_s-1":
            r2prime_stats["median"],

        "R2prime_mean_s-1":
            r2prime_stats["mean"],

        "R2prime_p01_s-1":
            r2prime_stats["p01"],

        "R2prime_p05_s-1":
            r2prime_stats["p05"],

        "R2prime_p95_s-1":
            r2prime_stats["p95"],

        "R2prime_p99_s-1":
            r2prime_stats["p99"],

        "R2prime_min_s-1":
            r2prime_stats["min"],

        "R2prime_max_s-1":
            r2prime_stats["max"],

        "R2prime_raw":
            str(p_raw),

        "common_mask":
            str(p_common),

        "negative_mask":
            str(p_negative),

        "qc_png":
            str(p_qc_png),

        "qc_pdf":
            str(p_qc_pdf),

        "metadata":
            str(p_metadata),
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
            " ".join(sys.argv),

        "case":
            case,

        "status":
            "PASS",

        "purpose": (
            "Construct raw R2-prime in native GRE space from "
            "R2STAR02 R2star and R2PRIME02 registered R2app."
        ),

        "definition": {
            "equation":
                "R2prime = R2star - R2app_GRE",

            "units":
                "s^-1",

            "negative_value_policy":
                "retained; no clipping in R2PRIME03 v0.1.0",
        },

        "inputs": {
            "R2STAR02_r2star":
                str(p_r2star),

            "R2STAR02_fit_mask":
                str(p_r2star_mask),

            "R2STAR02_qc":
                str(p_r2star_qc),

            "R2PRIME02_R2app_GRE":
                str(p_r2app),

            "R2PRIME02_valid_mask":
                str(p_r2app_mask),

            "R2PRIME02_metadata":
                str(p_r2prime02_meta),

            "GRE02b_brain_mask":
                str(p_gre_mask),
        },

        "support": {
            "definition": (
                "R2STAR02 fit mask AND R2PRIME02 valid mask "
                "AND finite R2star AND finite R2app"
            ),

            "common_voxels":
                n_common,

            "common_in_GRE_voxels":
                n_common_in_gre,

            "GRE_brain_voxels":
                n_gre,

            "common_GRE_coverage_pct":
                common_gre_coverage_pct,
        },

        "qc": {
            "negative_R2prime_voxels":
                n_negative,

            "negative_R2prime_fraction_pct":
                negative_fraction_pct,

            "zero_R2prime_voxels":
                n_zero,

            "zero_R2prime_fraction_pct":
                zero_fraction_pct,

            "R2star_s-1":
                r2star_stats,

            "R2app_s-1":
                r2app_stats,

            "R2prime_s-1":
                r2prime_stats,

            "qc_png":
                str(p_qc_png),

            "qc_pdf":
                str(p_qc_pdf),
        },

        "outputs": {
            "common_mask":
                str(p_common),

            "R2prime_raw":
                str(p_raw),

            "negative_mask":
                str(p_negative),

            "zero_mask":
                str(p_zero),

            "qc_png":
                str(p_qc_png),

            "qc_pdf":
                str(p_qc_pdf),

            "summary_csv":
                str(p_summary),
        },

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

            "numpy":
                np.__version__,

            "nibabel":
                nib.__version__,
        },
    }

    write_json(
        p_metadata,
        metadata,
    )

    write_csv(
        p_summary,
        [row],
    )

    print(
        f"  common support        : "
        f"{n_common:,} voxels"
    )

    print(
        f"  GRE brain coverage    : "
        f"{common_gre_coverage_pct:.3f}%"
    )

    print(
        f"  median R2*            : "
        f"{r2star_stats['median']:.3f} s^-1"
    )

    print(
        f"  median R2app          : "
        f"{r2app_stats['median']:.3f} s^-1"
    )

    print(
        f"  median R2'            : "
        f"{r2prime_stats['median']:.3f} s^-1"
    )

    print(
        f"  negative R2'          : "
        f"{n_negative:,} "
        f"({negative_fraction_pct:.3f}%)"
    )

    print(
        f"  R2' p01 / p99         : "
        f"{r2prime_stats['p01']:.3f} / "
        f"{r2prime_stats['p99']:.3f} s^-1"
    )

    print(
        f"  QC PNG                : {p_qc_png}"
    )

    print(
        f"  QC PDF                : {p_qc_pdf}"
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
            "APF R2PRIME03: construct raw R2-prime in native GRE space."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--affine-atol",
        type=float,
        default=DEFAULT_AFFINE_ATOL,
        help=(
            "Absolute tolerance for affine equality checks. "
            "Default: 1e-5."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing R2PRIME03 outputs.",
    )

    return parser.parse_args()


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
                        str(exc),
                }
            )

    summary_dir = cohort_dir(
        paths
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_csv = (
        summary_dir
        / "r2prime03_cohort_summary.csv"
    )

    cohort_json = (
        summary_dir
        / "r2prime03_cohort_metadata.json"
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
                " ".join(sys.argv),

            "negative_value_policy":
                "retained; no clipping",

            "n_cases_requested":
                len(cases),

            "n_completed":
                len(rows),

            "n_failures":
                len(failures),

            "rows":
                rows,

            "failures":
                failures,
        },
    )

    print()
    print("=" * 72)
    print("R2PRIME03 cohort summary")
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
