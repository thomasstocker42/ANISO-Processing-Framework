#!/usr/bin/env python3
"""
APF GRE05c v0.1.2

Validate the finalized APF GRE05b V-SHARP implementation against the
controlled STI Suite V-SHARP reference.

Primary comparison:
    APF GRE05b v0.3.0
    versus
    STI Suite V_SHARP with smvsize=9

APF local fields are stored in Hz.
STI Suite reference fields are stored in ppm.

The comparison is performed only on common spatial support. Support overlap
is quantified separately so differences in mask erosion are not conflated
with differences in local-field values.

This module is validation/QC only and does not generate downstream local-field
inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "0.1.2"
MODULE_NAME = "gre05c"
METHOD_NAME = "vsharp_implementation_comparison"

APF_SUBDIR = Path("GRE") / "05b_vsharp_apf_v030"
STI_SUBDIR = Path("GRE") / "05b_vsharp_sti_smv9"
SUMMARY_SUBDIR = Path("derivatives") / "summary" / "GRE"
OUTPUT_SUBDIR = Path("GRE") / "05c_vsharp_comparison"

GAMMA_HZ_PER_T = 42.577478518e6
B0_T = 3.0
HZ_PER_PPM = GAMMA_HZ_PER_T * B0_T * 1e-6


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

    cases = [
        p.name
        for p in sorted(deriv.glob("ANISO_*_InS"))
        if p.is_dir()
    ]

    return cases


def paths_for_case(root: Path, case: str) -> dict[str, Path]:
    base = root / "derivatives" / case

    return {
        "apf_local": (
            base / APF_SUBDIR /
            f"{case}_GRE05b_local_field_hz.nii.gz"
        ),
        "apf_mask": (
            base / APF_SUBDIR /
            f"{case}_GRE05b_vsharp_mask.nii.gz"
        ),
        "sti_local": (
            base / STI_SUBDIR /
            f"{case}_GRE05b_sti_vsharp_smv9_local_field_ppm.nii.gz"
        ),
    }


def affine_compatible(
    a: nib.Nifti1Image,
    b: nib.Nifti1Image,
    atol: float = 1e-5,
) -> bool:
    return (
        a.shape == b.shape
        and np.allclose(a.affine, b.affine, atol=atol, rtol=0.0)
    )


def compare_case(root: Path, case: str) -> dict:
    p = paths_for_case(root, case)

    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input(s): " + "; ".join(missing)
        )

    apf_img = nib.load(p["apf_local"])
    mask_img = nib.load(p["apf_mask"])
    sti_img = nib.load(p["sti_local"])

    if not affine_compatible(apf_img, mask_img):
        raise RuntimeError("APF local field and APF mask grids differ")

    if not affine_compatible(apf_img, sti_img):
        raise RuntimeError("APF and STI grids differ")

    apf_hz = apf_img.get_fdata(dtype=np.float32)
    apf_mask = mask_img.get_fdata(dtype=np.float32) > 0.5
    sti_ppm = sti_img.get_fdata(dtype=np.float32)

    apf_ppm = apf_hz / HZ_PER_PPM

    sti_mask = np.isfinite(sti_ppm) & (sti_ppm != 0)
    common = apf_mask & sti_mask
    union = apf_mask | sti_mask

    n_apf = int(np.count_nonzero(apf_mask))
    n_sti = int(np.count_nonzero(sti_mask))
    n_common = int(np.count_nonzero(common))
    n_union = int(np.count_nonzero(union))

    if n_common == 0:
        raise RuntimeError("APF/STI common support is empty")

    x = apf_ppm[common].astype(np.float64)
    y = sti_ppm[common].astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise RuntimeError("Non-finite APF values within common support")

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite STI values within common support")

    diff = x - y

    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(y, x, 1)

    affine_residual = x - (slope * y + intercept)

    return {
        "case_id": case,
        "status": "OK",
        "apf_support_voxels": n_apf,
        "sti_support_voxels": n_sti,
        "common_support_voxels": n_common,
        "union_support_voxels": n_union,
        "jaccard": float(n_common / n_union),
        "sti_contained_in_apf_fraction": float(n_common / n_sti),
        "apf_contained_in_sti_fraction": float(n_common / n_apf),
        "pearson_r": r,
        "slope_apf_on_sti": float(slope),
        "intercept_ppm": float(intercept),
        "mean_bias_ppm": float(np.mean(diff)),
        "rmse_ppm": float(np.sqrt(np.mean(diff * diff))),
        "mae_ppm": float(np.mean(np.abs(diff))),
        "affine_rmse_ppm": float(
            np.sqrt(np.mean(affine_residual * affine_residual))
        ),
        "apf_local_field": str(p["apf_local"]),
        "apf_mask": str(p["apf_mask"]),
        "sti_local_field": str(p["sti_local"]),
    }


def numeric_summary(rows: list[dict], key: str) -> dict:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)

    return {
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id",
        "status",
        "apf_support_voxels",
        "sti_support_voxels",
        "common_support_voxels",
        "union_support_voxels",
        "jaccard",
        "sti_contained_in_apf_fraction",
        "apf_contained_in_sti_fraction",
        "pearson_r",
        "slope_apf_on_sti",
        "intercept_ppm",
        "mean_bias_ppm",
        "rmse_ppm",
        "mae_ppm",
        "affine_rmse_ppm",
        "apf_local_field",
        "apf_mask",
        "sti_local_field",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reproducible_sample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size != y.size:
        raise ValueError("Scatter arrays have inconsistent lengths")

    if maximum <= 0:
        raise ValueError("--plot-max-voxels must be > 0")

    if x.size <= maximum:
        return x, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(x.size, size=maximum, replace=False)
    return x[indices], y[indices]


def validation_plot_paths(root: Path, case: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case / OUTPUT_SUBDIR
    prefix = f"{case}_GRE05c_apf_vs_sti_validation"

    return {
        "dir": outdir,
        "png": outdir / f"{prefix}.png",
        "pdf": outdir / f"{prefix}.pdf",
    }


def write_validation_plot(
    root: Path,
    case: str,
    metrics: dict,
    *,
    maximum_voxels: int,
    seed: int,
) -> dict[str, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--write-plots requested but matplotlib is unavailable"
        ) from exc

    p = paths_for_case(root, case)

    apf_img = nib.load(p["apf_local"])
    mask_img = nib.load(p["apf_mask"])
    sti_img = nib.load(p["sti_local"])

    if not affine_compatible(apf_img, mask_img):
        raise RuntimeError("APF local field and APF mask grids differ")

    if not affine_compatible(apf_img, sti_img):
        raise RuntimeError("APF and STI grids differ")

    apf_hz = apf_img.get_fdata(dtype=np.float32)
    apf_mask = mask_img.get_fdata(dtype=np.float32) > 0.5
    sti_ppm = sti_img.get_fdata(dtype=np.float32)

    apf_ppm = apf_hz / HZ_PER_PPM

    sti_mask = np.isfinite(sti_ppm) & (sti_ppm != 0)
    common = apf_mask & sti_mask

    if not np.any(common):
        raise RuntimeError("APF/STI common support is empty")

    x = sti_ppm[common].astype(np.float64)
    y = apf_ppm[common].astype(np.float64)
    diff = y - x

    # Representative axial slice: maximum common-support area.
    support_per_slice = np.count_nonzero(common, axis=(0, 1))
    z = int(np.argmax(support_per_slice))

    apf_slice = np.where(common[:, :, z], apf_ppm[:, :, z], np.nan)
    sti_slice = np.where(common[:, :, z], sti_ppm[:, :, z], np.nan)
    diff_slice = np.where(
        common[:, :, z],
        apf_ppm[:, :, z] - sti_ppm[:, :, z],
        np.nan,
    )

    # Shared symmetric scale for APF/STI.
    joint_abs = np.concatenate((np.abs(x), np.abs(y)))
    field_vmax = float(np.percentile(joint_abs, 99.0))
    field_vmax = max(field_vmax, np.finfo(float).eps)

    # Independent, tighter symmetric difference scale.
    diff_vmax = float(np.percentile(np.abs(diff), 99.0))
    diff_vmax = max(diff_vmax, np.finfo(float).eps)

    x_sample, y_sample = reproducible_sample(
        x,
        y,
        maximum=maximum_voxels,
        seed=seed,
    )

    scatter_low = float(
        min(
            np.percentile(x_sample, 0.5),
            np.percentile(y_sample, 0.5),
        )
    )
    scatter_high = float(
        max(
            np.percentile(x_sample, 99.5),
            np.percentile(y_sample, 99.5),
        )
    )

    paths = validation_plot_paths(root, case)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Rotate identically for display only; quantitative data are unchanged.
    im0 = axes[0, 0].imshow(
        np.rot90(apf_slice),
        cmap="RdBu_r",
        vmin=-field_vmax,
        vmax=field_vmax,
        interpolation="nearest",
    )
    axes[0, 0].set_title("A  APF GRE05b v0.3.0")
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(
        np.rot90(sti_slice),
        cmap="RdBu_r",
        vmin=-field_vmax,
        vmax=field_vmax,
        interpolation="nearest",
    )
    axes[0, 1].set_title("B  STI Suite V-SHARP, smvsize=9")
    axes[0, 1].set_axis_off()

    cbar_field = fig.colorbar(
        im0,
        ax=[axes[0, 0], axes[0, 1]],
        fraction=0.035,
        pad=0.02,
    )
    cbar_field.set_label("Local field (ppm)")

    im2 = axes[1, 0].imshow(
        np.rot90(diff_slice),
        cmap="RdBu_r",
        vmin=-diff_vmax,
        vmax=diff_vmax,
        interpolation="nearest",
    )
    axes[1, 0].set_title("C  APF − STI")
    axes[1, 0].set_axis_off()

    cbar_diff = fig.colorbar(
        im2,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )
    cbar_diff.set_label("Difference (ppm)")

    ax = axes[1, 1]
    ax.scatter(
        x_sample,
        y_sample,
        s=2,
        alpha=0.15,
        rasterized=True,
    )
    ax.plot(
        [scatter_low, scatter_high],
        [scatter_low, scatter_high],
        linewidth=1.2,
        label="Identity",
    )
    ax.set_xlim(scatter_low, scatter_high)
    ax.set_ylim(scatter_low, scatter_high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("STI local field (ppm)")
    ax.set_ylabel("APF local field (ppm)")
    ax.set_title("D  APF versus STI")

    annotation = (
        f"r = {metrics['pearson_r']:.4f}\n"
        f"slope = {metrics['slope_apf_on_sti']:.4f}\n"
        f"bias = {metrics['mean_bias_ppm']:+.5f} ppm\n"
        f"RMSE = {metrics['rmse_ppm']:.5f} ppm"
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
        f"{case}: GRE05 V-SHARP external-reference validation\n"
        f"common support; axial slice z={z}",
        fontsize=14,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.95,
        bottom=0.07,
        top=0.91,
        wspace=0.24,
        hspace=0.20,
    )

    fig.savefig(
        paths["png"],
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        paths["pdf"],
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "plot_case": case,
        "axial_slice_index": z,
        "scatter_voxels": int(x_sample.size),
        "field_display_abs_p99_ppm": field_vmax,
        "difference_display_abs_p99_ppm": diff_vmax,
        "png": str(paths["png"]),
        "pdf": str(paths["pdf"]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "APF GRE05c v0.1.2: compare APF GRE05b v0.3.0 "
            "against STI Suite V-SHARP smvsize=9"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--root", type=Path)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--all-gre-ready", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument(
        "--write-plots",
        action="store_true",
        help="Write APF-versus-STI validation figure for every selected case",
    )
    p.add_argument(
        "--plot-max-voxels",
        type=int,
        default=100000,
        help="Maximum common-support voxels used in the scatter panel",
    )
    p.add_argument(
        "--plot-seed",
        type=int,
        default=20260818,
        help="Random seed for reproducible scatter subsampling",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()
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
        "gre05c_vsharp_implementation_comparison_qc.csv"
    )
    json_path = (
        summary_dir /
        "gre05c_vsharp_implementation_comparison_run.json"
    )

    if not args.force:
        existing = [p for p in (csv_path, json_path) if p.exists()]
        if existing:
            raise SystemExit(
                "Output summary file(s) already exist; use --force: "
                + ", ".join(str(p) for p in existing)
            )

    print(f"APF GRE05c v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Method: {METHOD_NAME}")
    print(f"Cases requested: {len(cases)}")
    print(f"Conversion: 1 ppm = {HZ_PER_PPM:.8f} Hz at {B0_T:g} T")
    print("Reference: STI Suite V-SHARP smvsize=9")
    print()

    rows: list[dict] = []
    failures: list[dict] = []

    for case in cases:
        try:
            row = compare_case(root, case)
            rows.append(row)

            print(
                f"[OK] {case}: "
                f"J={row['jaccard']:.6f}, "
                f"r={row['pearson_r']:.6f}, "
                f"slope={row['slope_apf_on_sti']:.6f}, "
                f"bias={row['mean_bias_ppm']:+.6f} ppm, "
                f"RMSE={row['rmse_ppm']:.6f} ppm"
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

    if rows:
        write_csv(csv_path, rows)

    plot_outputs = []
    if args.write_plots:
        for plot_metrics in rows:
            plot_case = plot_metrics["case_id"]

            try:
                plot_output = write_validation_plot(
                    root,
                    plot_case,
                    plot_metrics,
                    maximum_voxels=args.plot_max_voxels,
                    seed=args.plot_seed,
                )
                plot_outputs.append(plot_output)

                print(
                    f"[PLOT] {plot_case}: "
                    f"{plot_output['png']}"
                )

            except Exception as exc:
                failures.append({
                    "case_id": plot_case,
                    "status": "FAILED",
                    "message": (
                        "Validation plot failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                })
                print(
                    f"[FAILED] Plot for {plot_case}: "
                    f"{type(exc).__name__}: {exc}"
                )

    summary_keys = [
        "jaccard",
        "sti_contained_in_apf_fraction",
        "apf_contained_in_sti_fraction",
        "pearson_r",
        "slope_apf_on_sti",
        "intercept_ppm",
        "mean_bias_ppm",
        "rmse_ppm",
        "mae_ppm",
        "affine_rmse_ppm",
    ]

    cohort_summary = {
        key: numeric_summary(rows, key)
        for key in summary_keys
    } if rows else {}

    run = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "created_utc": now_iso(),
        "root": str(root),
        "configuration": {
            "apf_implementation": "GRE05b v0.3.0",
            "apf_subdir": str(APF_SUBDIR),
            "reference_implementation": "STI Suite V-SHARP",
            "reference_smvsize": 9,
            "reference_subdir": str(STI_SUBDIR),
            "b0_t": B0_T,
            "gamma_hz_per_t": GAMMA_HZ_PER_T,
            "hz_per_ppm": HZ_PER_PPM,
            "comparison_support": "intersection of APF and STI support",
            "support_comparison": "reported separately using Jaccard/containment",
            "write_plots": bool(args.write_plots),
            "plot_max_voxels": args.plot_max_voxels,
            "plot_seed": args.plot_seed,
        },
        "cases_requested": len(cases),
        "cases_ok": len(rows),
        "cases_failed": len(failures),
        "cohort_summary": cohort_summary,
        "failures": failures,
        "outputs": {
            "qc_csv": str(csv_path),
            "run_json": str(json_path),
            "validation_plots": plot_outputs,
        },
    }

    json_path.write_text(
        json.dumps(run, indent=2),
        encoding="utf-8",
    )

    print()
    print("COHORT SUMMARY")
    print("=" * 78)

    if rows:
        for key in [
            "jaccard",
            "pearson_r",
            "slope_apf_on_sti",
            "mean_bias_ppm",
            "rmse_ppm",
            "mae_ppm",
            "affine_rmse_ppm",
        ]:
            s = cohort_summary[key]
            print(
                f"{key:28s} "
                f"median={s['median']:.8f} "
                f"min={s['min']:.8f} "
                f"max={s['max']:.8f}"
            )

    print()
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(
        f"Completed: OK={len(rows)}, "
        f"FAILED={len(failures)}, TOTAL={len(cases)}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
