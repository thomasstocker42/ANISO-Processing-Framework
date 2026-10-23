#!/usr/bin/env python3
"""
APF rlx04: Validate first-echo exclusion for mono-exponential T2 fitting.

Purpose
-------
rlx04 is a scientific validation/QC stage. It does not create a new primary T2
estimator. It tests and documents whether exclusion of echo 1 from the 12-echo
TSE mono-exponential fit is empirically justified.

Primary questions
-----------------
1. Is echo 1 systematically inconsistent with the decay predicted by echoes 2-12?
2. Does drop1 improve fit quality consistently across the analysis domain?
3. How much does dropping echo 1 change T2/R2?

Inputs
------
- authoritative rlx02 4D stack + echo times
- rlx03 all12 and drop1 fit outputs

Outputs
-------
- echo-wise residual summary CSV
- voxel-wise all12-vs-drop1 comparison summary CSV
- delta T2 and delta R2 maps on the common valid domain
- echo-1 deviation map from the drop1 backward prediction
- compact validation figure
- metadata JSON

Convention
----------
delta T2 = T2_drop1 - T2_all12https://aka.ms/vscode-workspace-trust
delta R2 = R2_drop1 - R2_all12

Run:
    ./apf scripts/relaxometry/04_validate_first_echo.py ANISO_00_000_InS
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

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from apf_paths import get_paths
from apf_case_selection import add_case_selection_args, print_selected_cases, resolve_cases

SCRIPT = "04_validate_first_echo.py"
TAG = "rlx04"
VERSION = "0.1.0"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_nifti(data, ref, path, dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data.astype(dtype), ref.affine, hdr), str(path))


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=None, sd=None, median=None, p05=None, p25=None,
                    p75=None, p95=None)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        median=float(np.median(x)),
        p05=float(np.percentile(x, 5)),
        p25=float(np.percentile(x, 25)),
        p75=float(np.percentile(x, 75)),
        p95=float(np.percentile(x, 95)),
    )


def load(path):
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj, dtype=np.float32)


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def process_case(case, paths, overwrite=False):
    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    rlx02_meta_path = paths.case_stage_dir(
        case=case, module="relaxometry", stage="rlx02_prepare_multi_echo_t2"
    ) / f"rlx02_{case}_preparation.json"
    require(rlx02_meta_path)
    rlx02 = json.loads(rlx02_meta_path.read_text())

    stack_path = Path(rlx02["outputs"]["series_4d"])
    mask_path = Path(rlx02["outputs"]["fit_mask"])
    if not stack_path.is_absolute():
        stack_path = paths.root / stack_path
    if not mask_path.is_absolute():
        mask_path = paths.root / mask_path
    require(stack_path)
    require(mask_path)

    tes = np.asarray(rlx02["input"]["echo_times_s"], dtype=float)
    if np.max(tes) > 1:
        tes /= 1000.0

    stack_img, data = load(stack_path)
    _, fit_mask = load(mask_path)
    fit_mask = fit_mask > 0

    if data.ndim != 4 or data.shape[3] != len(tes):
        raise ValueError("rlx02 stack/TE mismatch.")

    r3 = paths.derivatives_dir / case / "relaxometry" / "rlx03"
    def p(model, metric):
        return r3 / model / f"rlx03_{case}_{model}_{metric}.nii.gz"

    _, t2a = load(require(p("all12", "T2_ms")))
    _, r2a = load(require(p("all12", "R2_s-1")))
    _, s0a = load(require(p("all12", "S0")))
    _, nrmsea = load(require(p("all12", "NRMSE")))
    _, r2ca = load(require(p("all12", "R2coef")))
    _, va = load(require(p("all12", "fit_valid_mask")))

    _, t2d = load(require(p("drop1", "T2_ms")))
    _, r2d = load(require(p("drop1", "R2_s-1")))
    _, s0d = load(require(p("drop1", "S0")))
    _, nrmse_d = load(require(p("drop1", "NRMSE")))
    _, r2cd = load(require(p("drop1", "R2coef")))
    _, vd = load(require(p("drop1", "fit_valid_mask")))

    common = fit_mask & (va > 0) & (vd > 0)
    common &= np.isfinite(t2a) & np.isfinite(t2d) & np.isfinite(r2a) & np.isfinite(r2d)

    if not np.any(common):
        raise RuntimeError("No common valid voxels between all12 and drop1.")

    out = paths.derivatives_dir / case / "relaxometry" / "rlx04"
    meta_path = out / f"rlx04_{case}_metadata.json"
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"{meta_path} exists; use --overwrite.")

    # Model predictions across all TEs.
    pred_all = s0a[..., None] * np.exp(-r2a[..., None] * tes)
    pred_drop = s0d[..., None] * np.exp(-r2d[..., None] * tes)

    # Normalize residuals by predicted signal. This makes the echo-1 backward
    # prediction directly interpretable as fractional deviation from drop1.
    eps = 1e-8
    resid_all_frac = (data - pred_all) / np.maximum(np.abs(pred_all), eps)
    resid_drop_frac = (data - pred_drop) / np.maximum(np.abs(pred_drop), eps)

    echo_rows = []
    for i, te in enumerate(tes):
        a = resid_all_frac[..., i][common]
        d = resid_drop_frac[..., i][common]
        row = {
            "case": case,
            "echo": i + 1,
            "TE_ms": float(te * 1000),
        }
        for prefix, vals in [("all12_resid_frac", a), ("drop1_resid_frac", d)]:
            for k, v in stats(vals).items():
                row[f"{prefix}_{k}"] = v
        echo_rows.append(row)

    delta_t2 = np.full(t2a.shape, np.nan, np.float32)
    delta_r2 = np.full(r2a.shape, np.nan, np.float32)
    delta_t2[common] = t2d[common] - t2a[common]
    delta_r2[common] = r2d[common] - r2a[common]

    e1_dev = np.full(t2a.shape, np.nan, np.float32)
    e1_dev[common] = resid_drop_frac[..., 0][common]

    comparison = {
        "case": case,
        "common_valid_voxels": int(np.count_nonzero(common)),
    }
    metrics = {
        "deltaT2_ms": delta_t2[common],
        "deltaR2_s-1": delta_r2[common],
        "deltaNRMSE_drop1_minus_all12": (nrmse_d - nrmsea)[common],
        "deltaR2coef_drop1_minus_all12": (r2cd - r2ca)[common],
        "echo1_fractional_deviation_from_drop1_prediction": e1_dev[common],
    }
    for name, vals in metrics.items():
        for k, v in stats(vals).items():
            comparison[f"{name}_{k}"] = v

    comparison["fraction_drop1_lower_NRMSE"] = float(
        np.mean(nrmse_d[common] < nrmsea[common])
    )
    comparison["fraction_drop1_higher_R2coef"] = float(
        np.mean(r2cd[common] > r2ca[common])
    )
    comparison["fraction_drop1_lower_T2"] = float(
        np.mean(t2d[common] < t2a[common])
    )

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"rlx04_{case}_echo_residuals.csv", echo_rows)
    write_csv(out / f"rlx04_{case}_comparison.csv", [comparison])

    save_nifti(delta_t2, stack_img, out / f"rlx04_{case}_deltaT2_ms.nii.gz")
    save_nifti(delta_r2, stack_img, out / f"rlx04_{case}_deltaR2_s-1.nii.gz")
    save_nifti(e1_dev, stack_img, out / f"rlx04_{case}_echo1_deviation_frac.nii.gz")
    save_nifti(common.astype(np.uint8), stack_img,
               out / f"rlx04_{case}_common_valid_mask.nii.gz", np.uint8)

    # QC figure.
    zscores = np.sum(common, axis=(0, 1))
    z = int(np.argmax(zscores))

    te_ms = tes * 1000
    med_all = np.array([np.nanmedian(resid_all_frac[..., i][common])
                        for i in range(len(tes))])
    q25_all = np.array([np.nanpercentile(resid_all_frac[..., i][common], 25)
                        for i in range(len(tes))])
    q75_all = np.array([np.nanpercentile(resid_all_frac[..., i][common], 75)
                        for i in range(len(tes))])
    med_drop = np.array([np.nanmedian(resid_drop_frac[..., i][common])
                         for i in range(len(tes))])
    q25_drop = np.array([np.nanpercentile(resid_drop_frac[..., i][common], 25)
                         for i in range(len(tes))])
    q75_drop = np.array([np.nanpercentile(resid_drop_frac[..., i][common], 75)
                         for i in range(len(tes))])

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(te_ms, med_all, marker="o", label="all12")
    ax1.fill_between(te_ms, q25_all, q75_all, alpha=0.2)
    ax1.plot(te_ms, med_drop, marker="o", label="drop1")
    ax1.fill_between(te_ms, q25_drop, q75_drop, alpha=0.2)
    ax1.axhline(0, linewidth=1)
    ax1.set_xlabel("TE [ms]")
    ax1.set_ylabel("(observed - predicted) / predicted")
    ax1.set_title("Echo-wise fractional residuals (median + IQR)")
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    vals = e1_dev[common]
    lo, hi = np.nanpercentile(vals, [1, 99])
    ax2.hist(vals[(vals >= lo) & (vals <= hi)], bins=100)
    ax2.axvline(np.nanmedian(vals), linewidth=1)
    ax2.set_xlabel("Echo-1 fractional deviation")
    ax2.set_title("Echo 1 vs drop1 backward prediction")

    def show(ax, arr, title, cmap="gray", symmetric=False):
        panel = np.rot90(arr[:, :, z])
        finite = panel[np.isfinite(panel)]
        kwargs = {}
        if finite.size:
            if symmetric:
                lim = np.nanpercentile(np.abs(finite), 98)
                kwargs = {"vmin": -lim, "vmax": lim}
            else:
                kwargs = {"vmin": np.nanpercentile(finite, 2),
                          "vmax": np.nanpercentile(finite, 98)}
        im = ax.imshow(panel, cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    show(ax3, e1_dev, f"Echo-1 deviation, z={z}", symmetric=True)

    ax4 = fig.add_subplot(2, 3, 4)
    show(ax4, delta_t2, "ΔT2 = drop1 - all12 [ms]", symmetric=True)

    ax5 = fig.add_subplot(2, 3, 5)
    d_nrmse = np.full(t2a.shape, np.nan)
    d_nrmse[common] = nrmse_d[common] - nrmsea[common]
    show(ax5, d_nrmse, "ΔNRMSE = drop1 - all12", symmetric=True)

    ax6 = fig.add_subplot(2, 3, 6)
    d_r2c = np.full(t2a.shape, np.nan)
    d_r2c[common] = r2cd[common] - r2ca[common]
    show(ax6, d_r2c, "ΔR² = drop1 - all12", symmetric=True)

    fig.suptitle(f"rlx04 first-echo exclusion validation: {case}")
    fig.tight_layout()
    qc = out / f"rlx04_{case}_qc.png"
    fig.savefig(qc, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "created": now_iso(),
        "execution": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "command": " ".join(sys.argv),
        },
        "purpose": (
            "Validate and quantify exclusion of echo 1 from the 12-echo "
            "mono-exponential T2 fit; rlx04 does not define a new estimator."
        ),
        "definitions": {
            "deltaT2_ms": "T2_drop1 - T2_all12",
            "deltaR2_s-1": "R2_drop1 - R2_all12",
            "echo1_deviation_frac": (
                "(S1_observed - S1_predicted_from_drop1) / "
                "S1_predicted_from_drop1"
            ),
            "residual_fraction": "(observed - predicted) / predicted",
            "common_valid_mask": "rlx02 mask AND all12 valid AND drop1 valid",
        },
        "inputs": {
            "rlx02_metadata": str(rlx02_meta_path),
            "stack": str(stack_path),
            "echo_times_ms": te_ms.tolist(),
            "rlx03": str(r3),
        },
        "summary": comparison,
        "outputs": {
            "echo_residuals_csv": str(out / f"rlx04_{case}_echo_residuals.csv"),
            "comparison_csv": str(out / f"rlx04_{case}_comparison.csv"),
            "deltaT2_ms": str(out / f"rlx04_{case}_deltaT2_ms.nii.gz"),
            "deltaR2_s-1": str(out / f"rlx04_{case}_deltaR2_s-1.nii.gz"),
            "echo1_deviation_frac": str(out / f"rlx04_{case}_echo1_deviation_frac.nii.gz"),
            "common_valid_mask": str(out / f"rlx04_{case}_common_valid_mask.nii.gz"),
            "qc_png": str(qc),
        },
    }
    write_json(meta_path, metadata)

    print(f"  common valid voxels: {comparison['common_valid_voxels']}")
    print(f"  median ΔT2 [ms]: {comparison['deltaT2_ms_median']:.3f}")
    print(f"  median echo-1 deviation: "
          f"{100*comparison['echo1_fractional_deviation_from_drop1_prediction_median']:.2f}%")
    print(f"  fraction with lower NRMSE after drop1: "
          f"{100*comparison['fraction_drop1_lower_NRMSE']:.2f}%")
    print(f"  DONE: {case}")
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="APF rlx04: validate exclusion of echo 1 from 12-echo T2 fitting."
    )
    add_case_selection_args(parser)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths.ins_registry)
    print_selected_cases(cases)

    rows = []
    failures = 0
    for case in cases:
        try:
            rows.append(process_case(case, paths, args.overwrite))
        except Exception as exc:
            failures += 1
            print(f"ERROR in {case}: {exc}", file=sys.stderr)
            if len(cases) == 1:
                raise

    if rows:
        summary = paths.derivatives_dir / "summary" / "relaxometry" / "rlx04"
        write_csv(summary / "rlx04_cohort_comparison.csv", rows)

    if failures:
        raise SystemExit(f"rlx04 completed with {failures} failed case(s).")


if __name__ == "__main__":
    main()
