#!/usr/bin/env python3
"""
APF CHI04: susceptibility source-separation reconstruction QC.

Version 0.1.0

Purpose
-------
Perform quantitative and visual quality control of finalized CHI03b MEDI
susceptibility source-separation outputs.

CHI04 does not modify reconstruction outputs.

Per-case quantitative QC
------------------------
- required-output completeness
- finite values within solver mask
- exact chi-component algebra:
      chi_tot = chi_para - chi_dia
- chi_tot agreement with frozen GRE06c STI-iLSQR QSM
- chi_para / chi_dia correlation
- CSF support size
- negative raw R2-prime count
- source-component distribution summaries

Per-case visual QC
------------------
Standardized orthogonal montage showing:
- chi_para
- chi_dia
- chi_tot
- GRE06c QSM
- raw R2-prime
- CSF reference overlay

The montage is intended for human QC and does not determine exclusion
automatically.

Outputs
-------
derivatives/<CASE>/chi_separation/chi04_qc/
    chi04_<CASE>_qc.json
    chi04_<CASE>_montage.png

derivatives/summary/chi_separation/chi04_qc/
    chi04_cohort_summary.csv
    chi04_cohort_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from apf_paths import APFPaths, get_paths
from apf_case_selection import (
    add_case_selection_args,
    print_selected_cases,
    resolve_cases,
)


SCRIPT = "04_qc.py"
SCRIPT_ID = "CHI04"
TAG = "chi04"
VERSION = "0.1.0"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def case_paths(paths: APFPaths, case: str) -> dict[str, Path]:
    chi03 = (
        paths.derivatives_dir
        / case
        / "chi_separation"
        / "chi03b_medi"
    )

    chi02 = (
        paths.derivatives_dir
        / case
        / "chi_separation"
        / "chi02_prepare"
        / "qsm_ci_input"
    )

    return {
        "chi_para": chi03 / f"chi03b_{case}_chi_para_ppm.nii.gz",
        "chi_dia": chi03 / f"chi03b_{case}_chi_dia_ppm.nii.gz",
        "chi_tot": chi03 / f"chi03b_{case}_chi_tot_ppm.nii.gz",
        "mask_csf": chi03 / f"chi03b_{case}_mask_CSF.nii.gz",
        "chi03_qc": chi03 / f"chi03b_{case}_qc.json",
        "mask": chi02 / "mask.nii.gz",
        "r2prime": chi02 / "r2prime.nii.gz",
        "qsm": (
            paths.derivatives_dir
            / case
            / "GRE"
            / "06c_sti_ilsqr"
            / f"{case}_GRE06c_sti_ilsqr_chi_ppm.nii.gz"
        ),
    }


def output_dir(paths: APFPaths, case: str) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "chi_separation"
        / "chi04_qc"
    )


def cohort_output_dir(paths: APFPaths) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "chi_separation"
        / "chi04_qc"
    )


def load_float(path: Path) -> np.ndarray:
    return np.asarray(
        nib.load(path).dataobj,
        dtype=np.float64,
    )


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj) > 0


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "min": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    q = np.percentile(
        values,
        [0, 1, 5, 50, 95, 99, 100],
    )

    return {
        "min": float(q[0]),
        "p1": float(q[1]),
        "p5": float(q[2]),
        "median": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(q[6]),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)

    if np.count_nonzero(good) < 2:
        return float("nan")

    aa = a[good]
    bb = b[good]

    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")

    return float(np.corrcoef(aa, bb)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)

    if np.count_nonzero(good) == 0:
        return float("nan")

    d = a[good] - b[good]
    return float(np.sqrt(np.mean(d * d)))


def masked_center(mask: np.ndarray) -> tuple[int, int, int]:
    ijk = np.argwhere(mask)

    if ijk.size == 0:
        return tuple(int(s // 2) for s in mask.shape)

    c = np.median(ijk, axis=0)
    return tuple(int(round(x)) for x in c)


def get_slice(
    vol: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    if axis == 0:
        sl = vol[index, :, :]
    elif axis == 1:
        sl = vol[:, index, :]
    else:
        sl = vol[:, :, index]

    return np.rot90(sl)


def robust_limits(
    vol: np.ndarray,
    mask: np.ndarray,
    symmetric: bool,
) -> tuple[float, float]:
    x = vol[mask & np.isfinite(vol)]

    if x.size == 0:
        return (-1.0, 1.0)

    if symmetric:
        lim = float(np.percentile(np.abs(x), 99))
        if lim <= 0:
            lim = 1.0
        return (-lim, lim)

    lo, hi = np.percentile(x, [1, 99])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (0.0, 1.0)

    return float(lo), float(hi)


def make_montage(
    case: str,
    out_png: Path,
    chi_para: np.ndarray,
    chi_dia: np.ndarray,
    chi_tot: np.ndarray,
    qsm: np.ndarray,
    r2prime: np.ndarray,
    mask: np.ndarray,
    mask_csf: np.ndarray,
) -> None:
    center = masked_center(mask)

    para_lim = robust_limits(chi_para, mask, symmetric=False)
    dia_lim = robust_limits(chi_dia, mask, symmetric=False)

    tot_lim = robust_limits(chi_tot, mask, symmetric=True)
    qsm_lim = robust_limits(qsm, mask, symmetric=True)

    # Common symmetric scale for chi_tot and QSM.
    sym_lim = max(
        abs(tot_lim[0]),
        abs(tot_lim[1]),
        abs(qsm_lim[0]),
        abs(qsm_lim[1]),
    )
    tot_qsm_lim = (-sym_lim, sym_lim)

    r2_mask = mask & np.isfinite(r2prime)
    r2_lim = robust_limits(
        np.where(r2_mask, r2prime, np.nan),
        r2_mask,
        symmetric=False,
    )

    rows = [
        ("chi_para [ppm]", chi_para, para_lim, "gray"),
        ("chi_dia [ppm]", chi_dia, dia_lim, "gray"),
        ("chi_tot [ppm]", chi_tot, tot_qsm_lim, "gray"),
        ("GRE06c QSM [ppm]", qsm, tot_qsm_lim, "gray"),
        ("raw R2prime [s^-1]", r2prime, r2_lim, "gray"),
    ]

    axis_names = ["sagittal", "coronal", "axial"]

    fig, axes = plt.subplots(
        nrows=6,
        ncols=3,
        figsize=(12, 18),
        constrained_layout=True,
    )

    for r, (label, vol, lims, cmap) in enumerate(rows):
        for axis in range(3):
            ax = axes[r, axis]
            sl = get_slice(
                vol,
                axis,
                center[axis],
            )

            ax.imshow(
                sl,
                cmap=cmap,
                vmin=lims[0],
                vmax=lims[1],
                interpolation="nearest",
            )

            ax.axis("off")

            if r == 0:
                ax.set_title(axis_names[axis])

            if axis == 0:
                ax.text(
                    -0.05,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

    # Final row: chi_tot with CSF overlay.
    for axis in range(3):
        ax = axes[5, axis]

        base = get_slice(
            chi_tot,
            axis,
            center[axis],
        )

        csf = get_slice(
            mask_csf.astype(float),
            axis,
            center[axis],
        )

        ax.imshow(
            base,
            cmap="gray",
            vmin=tot_qsm_lim[0],
            vmax=tot_qsm_lim[1],
            interpolation="nearest",
        )

        overlay = np.ma.masked_where(
            csf <= 0,
            csf,
        )

        ax.imshow(
            overlay,
            cmap="autumn",
            alpha=0.8,
            interpolation="nearest",
        )

        ax.axis("off")

        if axis == 0:
            ax.text(
                -0.05,
                0.5,
                "chi_tot + CSF",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=10,
            )

    fig.suptitle(
        f"CHI04 reconstruction QC — {case}",
        fontsize=14,
    )

    fig.savefig(
        out_png,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_case(
    paths: APFPaths,
    case: str,
) -> dict[str, Any]:

    p = case_paths(paths, case)

    missing = [
        key
        for key, path in p.items()
        if not path.is_file()
    ]

    out = output_dir(paths, case)
    out.mkdir(parents=True, exist_ok=True)

    qc_json = out / f"chi04_{case}_qc.json"
    montage_png = out / f"chi04_{case}_montage.png"

    if missing:
        result = {
            "case_id": case,
            "status": "FAILED",
            "reason": "missing_inputs",
            "missing": missing,
        }

        qc_json.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        return result

    chi_para = load_float(p["chi_para"])
    chi_dia = load_float(p["chi_dia"])
    chi_tot = load_float(p["chi_tot"])
    qsm = load_float(p["qsm"])
    r2prime = load_float(p["r2prime"])

    mask = load_mask(p["mask"])
    mask_csf = load_mask(p["mask_csf"])

    shapes = {
        tuple(x.shape)
        for x in [
            chi_para,
            chi_dia,
            chi_tot,
            qsm,
            r2prime,
            mask,
            mask_csf,
        ]
    }

    if len(shapes) != 1:
        result = {
            "case_id": case,
            "status": "FAILED",
            "reason": "geometry_shape_mismatch",
            "shapes": [list(x) for x in sorted(shapes)],
        }

        qc_json.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        return result

    n_mask = int(np.count_nonzero(mask))
    n_csf = int(np.count_nonzero(mask_csf))

    finite_para = np.isfinite(chi_para[mask])
    finite_dia = np.isfinite(chi_dia[mask])
    finite_tot = np.isfinite(chi_tot[mask])
    finite_qsm = np.isfinite(qsm[mask])

    algebra = chi_tot[mask] - (
        chi_para[mask] - chi_dia[mask]
    )

    algebra_good = np.isfinite(algebra)

    if np.count_nonzero(algebra_good):
        algebra_rmse = float(
            np.sqrt(
                np.mean(
                    algebra[algebra_good] ** 2
                )
            )
        )
        algebra_max_abs = float(
            np.max(
                np.abs(
                    algebra[algebra_good]
                )
            )
        )
    else:
        algebra_rmse = float("nan")
        algebra_max_abs = float("nan")

    vt = chi_tot[mask]
    vp = chi_para[mask]
    vd = chi_dia[mask]
    vq = qsm[mask]

    result = {
        "case_id": case,
        "status": "COMPLETED",
        "generated_at": now_iso(),
        "script": SCRIPT,
        "script_id": SCRIPT_ID,
        "version": VERSION,
        "mask_voxels": n_mask,
        "csf_voxels": n_csf,
        "csf_fraction_of_solver_mask": (
            float(n_csf / n_mask)
            if n_mask
            else float("nan")
        ),
        "nonfinite_in_solver_mask": {
            "chi_para": int(
                np.count_nonzero(~finite_para)
            ),
            "chi_dia": int(
                np.count_nonzero(~finite_dia)
            ),
            "chi_tot": int(
                np.count_nonzero(~finite_tot)
            ),
            "qsm": int(
                np.count_nonzero(~finite_qsm)
            ),
        },
        "r2prime": {
            "nonfinite_in_solver_mask": int(
                np.count_nonzero(
                    ~np.isfinite(r2prime[mask])
                )
            ),
            "negative_voxels_in_solver_mask": int(
                np.count_nonzero(
                    r2prime[mask] < 0
                )
            ),
        },
        "algebra": {
            "definition": (
                "chi_tot = chi_para - chi_dia"
            ),
            "rmse_ppm": algebra_rmse,
            "max_abs_error_ppm": algebra_max_abs,
        },
        "qsm_comparison": {
            "pearson_r": pearson(vt, vq),
            "rmse_ppm": rmse(vt, vq),
        },
        "component_relationship": {
            "para_dia_pearson_r": pearson(vp, vd),
        },
        "chi_para": finite_stats(vp),
        "chi_dia": finite_stats(vd),
        "chi_tot": finite_stats(vt),
        "visual_qc": {
            "montage": str(montage_png),
            "review_status": "PENDING",
        },
        "automatic_exclusion": False,
        "notes": (
            "Quantitative metrics are descriptive QC. "
            "No case is automatically excluded from "
            "biological analysis by CHI04."
        ),
    }

    make_montage(
        case=case,
        out_png=montage_png,
        chi_para=chi_para,
        chi_dia=chi_dia,
        chi_tot=chi_tot,
        qsm=qsm,
        r2prime=r2prime,
        mask=mask,
        mask_csf=mask_csf,
    )

    qc_json.write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )

    return result


def write_cohort_summary(
    paths: APFPaths,
    results: list[dict[str, Any]],
) -> None:

    out = cohort_output_dir(paths)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "chi04_cohort_summary.csv"
    json_path = out / "chi04_cohort_summary.json"

    rows = []

    for r in results:
        if r.get("status") != "COMPLETED":
            rows.append({
                "case_id": r.get("case_id"),
                "status": r.get("status"),
            })
            continue

        rows.append({
            "case_id": r["case_id"],
            "status": r["status"],
            "mask_voxels": r["mask_voxels"],
            "csf_voxels": r["csf_voxels"],
            "csf_fraction_of_solver_mask":
                r["csf_fraction_of_solver_mask"],
            "r2prime_negative_voxels_in_mask":
                r["r2prime"][
                    "negative_voxels_in_solver_mask"
                ],
            "algebra_rmse_ppm":
                r["algebra"]["rmse_ppm"],
            "algebra_max_abs_error_ppm":
                r["algebra"]["max_abs_error_ppm"],
            "qsm_rmse_ppm":
                r["qsm_comparison"]["rmse_ppm"],
            "qsm_pearson_r":
                r["qsm_comparison"]["pearson_r"],
            "para_dia_pearson_r":
                r["component_relationship"][
                    "para_dia_pearson_r"
                ],
            "visual_review_status":
                r["visual_qc"]["review_status"],
        })

    fields = [
        "case_id",
        "status",
        "mask_voxels",
        "csf_voxels",
        "csf_fraction_of_solver_mask",
        "r2prime_negative_voxels_in_mask",
        "algebra_rmse_ppm",
        "algebra_max_abs_error_ppm",
        "qsm_rmse_ppm",
        "qsm_pearson_r",
        "para_dia_pearson_r",
        "visual_review_status",
    ]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()

        for row in rows:
            w.writerow(row)

    payload = {
        "generated_at": now_iso(),
        "script": SCRIPT,
        "script_id": SCRIPT_ID,
        "version": VERSION,
        "n_cases": len(results),
        "n_completed": sum(
            r.get("status") == "COMPLETED"
            for r in results
        ),
        "cases": results,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )

    print()
    print(f"Cohort CSV:  {csv_path}")
    print(f"Cohort JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF CHI04 susceptibility "
            "source-separation QC"
        )
    )

    add_case_selection_args(parser)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )

    print_selected_cases(cases)

    results = []

    for case in cases:
        print(f"[CHI04] {case}")

        result = run_case(
            paths,
            case,
        )

        results.append(result)

        if result["status"] == "COMPLETED":
            print(
                "  "
                f"CSF={result['csf_voxels']} | "
                f"QSM r="
                f"{result['qsm_comparison']['pearson_r']:.4f} | "
                f"QSM RMSE="
                f"{result['qsm_comparison']['rmse_ppm']:.4f} ppm | "
                f"para-dia r="
                f"{result['component_relationship']['para_dia_pearson_r']:.4f}"
            )
        else:
            print(
                f"  FAILED: "
                f"{result.get('reason', 'unknown')}"
            )

    write_cohort_summary(
        paths,
        results,
    )


if __name__ == "__main__":
    main()
