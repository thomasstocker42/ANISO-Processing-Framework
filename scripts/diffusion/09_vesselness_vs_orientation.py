#!/usr/bin/env python3
"""
APF dwi09: Continuous vesselness vs fiber orientation relative to B0.

Primary analysis:
    continuous Sato vesselness vs dwi08 GRE-grid orientation maps

Binning analyses:
    1) quantile bins, default n=20
    2) equal-width bins, default n=20
    3) existing s17 4-bin labelmap as supplementary comparison

Valid analysis domain:
    finite vesselness
    finite angle/cos2
    innerMask_dist4_from_s11b > 0
    optional vesselness threshold >= --min-vesselness
    optional threshold sweep via --min-vesselness-values
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT = "09_vesselness_vs_orientation.py"
TAG = "dwi09"
VERSION = "0.1.3"

DEFAULT_SHELL = "b2000"
DEFAULT_QUANTILE_BINS = 20
DEFAULT_EQUAL_WIDTH_BINS = 20


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("dwi09 requires nibabel.") from exc


try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("dwi09 requires matplotlib.") from exc


try:
    from scipy.stats import spearmanr, pearsonr, kruskal
except ImportError as exc:
    raise ImportError("dwi09 requires scipy.") from exc


try:
    from apf_paths import APFPaths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError("Could not import APF infrastructure. Run via ./apf.") from exc


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dwi08_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "dwi08_orientation_GREgrid"


def dwi09_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "vesselness_orientation" / "dwi09"


def vesselness_threshold_label(min_vesselness: float) -> str:
    """Return stable output-folder label for the vesselness threshold.

    Examples
    --------
    0.0  -> minV000
    0.05 -> minV005
    0.10 -> minV010
    """
    scaled = int(round(float(min_vesselness) * 100))
    return f"minV{scaled:03d}"


def find_vesselness(paths: APFPaths, case: str) -> Path:
    base = paths.derivatives_dir / case / "vessel_segmentation"
    candidates = [
        base / "GRE_e7_2D_Sato_s13_from_s12" / f"s13_{case}_GRE_e7_2D_Sato_vesselness_from_s12.nii.gz",
        base / "GRE_e7_2D_Sato" / f"{case}_GRE_e7_2D_Sato_vesselness.nii.gz",
    ]
    candidates.extend(sorted(base.glob("**/*vesselness*.nii.gz")))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find continuous vesselness map for {case}")


def find_inner_mask(paths: APFPaths, case: str) -> Path:
    base = paths.derivatives_dir / case / "vessel_segmentation"
    candidates = [
        base / "GRE_e7_Sato_s17_vesselnessBins_from_s13" / f"s17_{case}_innerMask_dist4_from_s11b.nii.gz",
    ]
    candidates.extend(sorted(base.glob("**/*innerMask_dist4*.nii.gz")))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find s17 innerMask_dist4 for {case}")


def find_s17_labelmap(paths: APFPaths, case: str) -> Optional[Path]:
    base = paths.derivatives_dir / case / "vessel_segmentation"
    candidates = [
        base / "GRE_e7_Sato_s17_vesselnessBins_from_s13" / f"s17_{case}_GRE_e7_Sato_vesselnessBins_labelmap_dist4_from_s13.nii.gz",
    ]
    candidates.extend(sorted(base.glob("**/*vesselnessBins_labelmap*.nii.gz")))
    for p in candidates:
        if p.exists():
            return p
    return None


def find_dwi08_map(paths: APFPaths, case: str, shell: str, name: str) -> Path:
    p = dwi08_dir(paths, case) / shell / f"dwi08_{case}_{shell}_{name}_GREgrid.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"Missing dwi08 GRE-grid map: {p}")
    return p


def find_dwi08_shells(paths: APFPaths, case: str) -> List[str]:
    base = dwi08_dir(paths, case)
    if not base.exists():
        raise FileNotFoundError(f"No dwi08 directory found: {base}")
    shells = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("b"))
    if not shells:
        raise FileNotFoundError(f"No dwi08 shell directories found in {base}")
    return shells


def load_data(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float64)


def check_same_shape(arrays: Dict[str, np.ndarray]) -> None:
    shapes = {k: v.shape for k, v in arrays.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"Shape mismatch: {shapes}")


def safe_corr(x: np.ndarray, y: np.ndarray) -> Dict:
    if x.size < 3:
        return {
            "spearman_rho": None,
            "spearman_p": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    sp = spearmanr(x, y)
    pr = pearsonr(x, y)

    return {
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def summarize_values(values: np.ndarray) -> Dict:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "iqr": None,
        }

    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
    }


def make_quantile_bins(vesselness: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Return integer bins 1..n_bins for valid vesselness values.
    Empty/unusable bins remain possible if many identical values exist.
    """
    ranks = np.argsort(np.argsort(vesselness, kind="mergesort"), kind="mergesort")
    bins = np.floor(ranks / max(1, vesselness.size) * n_bins).astype(int) + 1
    bins = np.clip(bins, 1, n_bins)
    return bins


def make_equal_width_bins(vesselness: np.ndarray, n_bins: int) -> np.ndarray:
    vmin = float(np.min(vesselness))
    vmax = float(np.max(vesselness))

    if vmax <= vmin:
        return np.ones_like(vesselness, dtype=int)

    edges = np.linspace(vmin, vmax, n_bins + 1)
    bins = np.digitize(vesselness, edges[1:-1], right=False) + 1
    bins = np.clip(bins, 1, n_bins)
    return bins


def summarize_bins(
    vesselness: np.ndarray,
    theta: np.ndarray,
    cos2: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    binning_name: str,
) -> List[Dict]:
    rows = []

    for b in range(1, n_bins + 1):
        m = bins == b
        if not np.any(m):
            rows.append(
                {
                    "binning": binning_name,
                    "bin": b,
                    "n_voxels": 0,
                    "vesselness_min": None,
                    "vesselness_max": None,
                    "vesselness_median": None,
                    "theta_mean": None,
                    "theta_median": None,
                    "theta_iqr": None,
                    "cos2_mean": None,
                    "cos2_median": None,
                    "cos2_iqr": None,
                }
            )
            continue

        v_sum = summarize_values(vesselness[m])
        t_sum = summarize_values(theta[m])
        c_sum = summarize_values(cos2[m])

        rows.append(
            {
                "binning": binning_name,
                "bin": b,
                "n_voxels": int(m.sum()),
                "vesselness_min": float(np.min(vesselness[m])),
                "vesselness_max": float(np.max(vesselness[m])),
                "vesselness_median": v_sum["median"],
                "theta_mean": t_sum["mean"],
                "theta_median": t_sum["median"],
                "theta_iqr": t_sum["iqr"],
                "cos2_mean": c_sum["mean"],
                "cos2_median": c_sum["median"],
                "cos2_iqr": c_sum["iqr"],
            }
        )

    return rows


def summarize_s17_labels(
    labels: np.ndarray,
    theta: np.ndarray,
    cos2: np.ndarray,
    vesselness: np.ndarray,
) -> List[Dict]:
    label_names = {
        1: "low_p50-p75",
        2: "medium_p75-p90",
        3: "high_p90-p97p5",
        4: "veryHigh_p97p5-p100",
    }

    rows = []
    for label in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        m = labels == label
        v_sum = summarize_values(vesselness[m])
        t_sum = summarize_values(theta[m])
        c_sum = summarize_values(cos2[m])

        rows.append(
            {
                "label": label,
                "label_name": label_names.get(label, f"label_{label}"),
                "n_voxels": int(m.sum()),
                "vesselness_median": v_sum["median"],
                "theta_mean": t_sum["mean"],
                "theta_median": t_sum["median"],
                "theta_iqr": t_sum["iqr"],
                "cos2_mean": c_sum["mean"],
                "cos2_median": c_sum["median"],
                "cos2_iqr": c_sum["iqr"],
            }
        )

    return rows


def kruskal_by_bins(values: np.ndarray, bins: np.ndarray) -> Dict:
    groups = [values[bins == b] for b in sorted(np.unique(bins))]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return {"h": None, "p": None, "n_groups": len(groups)}
    stat = kruskal(*groups)
    return {
        "h": float(stat.statistic),
        "p": float(stat.pvalue),
        "n_groups": len(groups),
    }


def make_qc_png(
    quantile_rows: List[Dict],
    equal_rows: List[Dict],
    vesselness: np.ndarray,
    theta: np.ndarray,
    cos2: np.ndarray,
    out_png: Path,
) -> None:
    q_x = [r["vesselness_median"] for r in quantile_rows if r["n_voxels"] > 0]
    q_theta = [r["theta_median"] for r in quantile_rows if r["n_voxels"] > 0]
    q_cos2 = [r["cos2_median"] for r in quantile_rows if r["n_voxels"] > 0]

    e_x = [r["vesselness_median"] for r in equal_rows if r["n_voxels"] > 0]
    e_theta = [r["theta_median"] for r in equal_rows if r["n_voxels"] > 0]
    e_cos2 = [r["cos2_median"] for r in equal_rows if r["n_voxels"] > 0]
    e_n = [r["n_voxels"] for r in equal_rows if r["n_voxels"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(q_x, q_theta, marker="o")
    axes[0, 0].set_title("Quantile bins: median θ")
    axes[0, 0].set_xlabel("Median vesselness")
    axes[0, 0].set_ylabel("Median θ to B0 [deg]")

    axes[0, 1].plot(q_x, q_cos2, marker="o")
    axes[0, 1].set_title("Quantile bins: median cos²θ")
    axes[0, 1].set_xlabel("Median vesselness")
    axes[0, 1].set_ylabel("Median cos²θ")

    axes[1, 0].plot(e_x, e_theta, marker="o")
    axes[1, 0].set_title("Equal-width bins: median θ")
    axes[1, 0].set_xlabel("Median vesselness")
    axes[1, 0].set_ylabel("Median θ to B0 [deg]")

    axes[1, 1].bar(range(1, len(e_n) + 1), e_n)
    axes[1, 1].set_title("Equal-width bin voxel counts")
    axes[1, 1].set_xlabel("Non-empty equal-width bin")
    axes[1, 1].set_ylabel("Voxels")

    fig.suptitle("dwi09 vesselness vs orientation QC")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_vesselness_cdf_png(
    vesselness_all_valid_pre_threshold: np.ndarray,
    min_vesselness: float,
    out_png: Path,
) -> Dict:
    """Create vesselness CDF/QC plot and return threshold summary statistics."""
    v = vesselness_all_valid_pre_threshold[np.isfinite(vesselness_all_valid_pre_threshold)]

    if v.size == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Vesselness distribution: no valid voxels")
        ax.set_xlabel("Vesselness")
        ax.set_ylabel("Cumulative fraction")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return {
            "n_pre_threshold": 0,
            "n_post_threshold": 0,
            "fraction_retained": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    v_sorted = np.sort(v)
    cdf = np.arange(1, v_sorted.size + 1, dtype=float) / float(v_sorted.size)
    retained = v >= min_vesselness

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v_sorted, cdf)
    ax.axvline(min_vesselness, linestyle="--")
    ax.set_title("Vesselness cumulative distribution")
    ax.set_xlabel("Vesselness")
    ax.set_ylabel("Cumulative fraction of valid voxels")
    ax.text(
        0.98,
        0.05,
        f"threshold ≥ {min_vesselness:.3f}\nretained {retained.sum():,}/{v.size:,} ({retained.mean() * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "n_pre_threshold": int(v.size),
        "n_post_threshold": int(retained.sum()),
        "fraction_retained": float(retained.mean()),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }


def process_shell(
    case: str,
    shell: str,
    paths: APFPaths,
    out_case_dir: Path,
    vesselness_img: Path,
    inner_mask_img: Path,
    s17_labelmap_img: Optional[Path],
    quantile_bins_n: int,
    equal_width_bins_n: int,
    use_equal_width: bool,
    min_vesselness: float,
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    print(f"  Shell: {shell}")

    out_dir = out_case_dir / shell
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_img = find_dwi08_map(paths, case, shell, "angle_to_B0_deg")
    cos2_img = find_dwi08_map(paths, case, shell, "cos2_theta")
    cos_img = find_dwi08_map(paths, case, shell, "cos_theta")

    vesselness = load_data(vesselness_img)
    inner_mask = load_data(inner_mask_img) > 0
    theta = load_data(theta_img)
    cos2 = load_data(cos2_img)
    cos_theta = load_data(cos_img)

    arrays = {
        "vesselness": vesselness,
        "inner_mask": inner_mask.astype(float),
        "theta": theta,
        "cos2": cos2,
        "cos_theta": cos_theta,
    }
    check_same_shape(arrays)

    valid_pre_threshold = (
        np.isfinite(vesselness)
        & np.isfinite(theta)
        & np.isfinite(cos2)
        & np.isfinite(cos_theta)
        & inner_mask
        & (theta >= 0)
        & (theta <= 90)
        & (cos2 >= 0)
        & (cos2 <= 1)
    )

    valid = valid_pre_threshold & (vesselness >= min_vesselness)

    v_pre_threshold = vesselness[valid_pre_threshold]
    v = vesselness[valid]
    t = theta[valid]
    c2 = cos2[valid]
    ct = cos_theta[valid]

    if v.size == 0:
        raise ValueError(
            f"No valid voxels remain after applying min_vesselness={min_vesselness}"
        )

    continuous_stats = {
        "case": case,
        "shell": shell,
        "n_voxels": int(v.size),
        "n_voxels_pre_vesselness_threshold": int(v_pre_threshold.size),
        "min_vesselness": float(min_vesselness),
        "fraction_retained_after_vesselness_threshold": float(v.size / v_pre_threshold.size) if v_pre_threshold.size else None,
        "vesselness": summarize_values(v),
        "theta": summarize_values(t),
        "cos_theta": summarize_values(ct),
        "cos2_theta": summarize_values(c2),
        "correlation_vesselness_theta": safe_corr(v, t),
        "correlation_vesselness_cos2": safe_corr(v, c2),
    }

    q_bins = make_quantile_bins(v, quantile_bins_n)
    quantile_rows = summarize_bins(v, t, c2, q_bins, quantile_bins_n, "quantile")

    if use_equal_width:
        e_bins = make_equal_width_bins(v, equal_width_bins_n)
        equal_rows = summarize_bins(v, t, c2, e_bins, equal_width_bins_n, "equal_width")
    else:
        e_bins = np.ones_like(q_bins)
        equal_rows = []

    label_rows = []
    label_stats = None
    if s17_labelmap_img is not None:
        labels_full = load_data(s17_labelmap_img)
        check_same_shape({"labels": labels_full, "theta": theta})
        labels = labels_full[valid].astype(int)

        label_rows = summarize_s17_labels(labels, t, c2, v)

        positive = labels > 0
        if np.any(positive):
            label_stats = {
                "kruskal_theta_by_s17": kruskal_by_bins(t[positive], labels[positive]),
                "kruskal_cos2_by_s17": kruskal_by_bins(c2[positive], labels[positive]),
            }

    stats = {
        "continuous": continuous_stats,
        "quantile_bins": {
            "n_bins": quantile_bins_n,
            "kruskal_theta": kruskal_by_bins(t, q_bins),
            "kruskal_cos2": kruskal_by_bins(c2, q_bins),
        },
        "equal_width_bins": {
            "enabled": use_equal_width,
            "n_bins": equal_width_bins_n,
            "kruskal_theta": kruskal_by_bins(t, e_bins) if use_equal_width else None,
            "kruskal_cos2": kruskal_by_bins(c2, e_bins) if use_equal_width else None,
        },
        "s17_labels": label_stats,
    }

    continuous_csv = out_dir / f"dwi09_{case}_{shell}_continuous_stats.csv"
    quantile_csv = out_dir / f"dwi09_{case}_{shell}_quantile{quantile_bins_n}_summary.csv"
    equal_csv = out_dir / f"dwi09_{case}_{shell}_equalwidth{equal_width_bins_n}_summary.csv"
    label_csv = out_dir / f"dwi09_{case}_{shell}_s17_label_summary.csv"
    qc_png = out_dir / f"dwi09_{case}_{shell}_vesselness_orientation_qc.png"
    cdf_png = out_dir / f"dwi09_{case}_{shell}_vesselness_cdf.png"

    cdf_stats = None

    if not dry_run:
        write_csv(
            continuous_csv,
            [
                {
                    "case": case,
                    "shell": shell,
                    "n_voxels": continuous_stats["n_voxels"],
                    "spearman_vesselness_theta": continuous_stats["correlation_vesselness_theta"]["spearman_rho"],
                    "spearman_p_vesselness_theta": continuous_stats["correlation_vesselness_theta"]["spearman_p"],
                    "pearson_vesselness_theta": continuous_stats["correlation_vesselness_theta"]["pearson_r"],
                    "pearson_p_vesselness_theta": continuous_stats["correlation_vesselness_theta"]["pearson_p"],
                    "spearman_vesselness_cos2": continuous_stats["correlation_vesselness_cos2"]["spearman_rho"],
                    "spearman_p_vesselness_cos2": continuous_stats["correlation_vesselness_cos2"]["spearman_p"],
                    "pearson_vesselness_cos2": continuous_stats["correlation_vesselness_cos2"]["pearson_r"],
                    "pearson_p_vesselness_cos2": continuous_stats["correlation_vesselness_cos2"]["pearson_p"],
                }
            ],
        )
        write_csv(quantile_csv, quantile_rows)

        if use_equal_width:
            write_csv(equal_csv, equal_rows)

        if label_rows:
            write_csv(label_csv, label_rows)

        make_qc_png(quantile_rows, equal_rows, v, t, c2, qc_png)
        cdf_stats = make_vesselness_cdf_png(v_pre_threshold, min_vesselness, cdf_png)

    return {
        "shell": shell,
        "inputs": {
            "vesselness": str(vesselness_img),
            "inner_mask_dist4": str(inner_mask_img),
            "s17_labelmap": str(s17_labelmap_img) if s17_labelmap_img else None,
            "theta": str(theta_img),
            "cos_theta": str(cos_img),
            "cos2_theta": str(cos2_img),
        },
        "outputs": {
            "continuous_stats_csv": str(continuous_csv),
            "quantile_summary_csv": str(quantile_csv),
            "equal_width_summary_csv": str(equal_csv) if use_equal_width else None,
            "s17_label_summary_csv": str(label_csv) if label_rows else None,
            "qc_png": str(qc_png),
            "vesselness_cdf_png": str(cdf_png),
        },
        "valid_voxels": int(v.size),
        "valid_voxels_pre_vesselness_threshold": int(v_pre_threshold.size),
        "min_vesselness": float(min_vesselness),
        "vesselness_cdf": cdf_stats,
        "statistics": stats,
    }


def process_case(
    case: str,
    paths: APFPaths,
    selected_shells: Optional[List[str]],
    quantile_bins_n: int,
    equal_width_bins_n: int,
    use_equal_width: bool,
    min_vesselness: float,
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    start = now_iso()

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    out_case_dir = dwi09_dir(paths, case) / vesselness_threshold_label(min_vesselness)
    out_case_dir.mkdir(parents=True, exist_ok=True)

    vesselness_img = find_vesselness(paths, case)
    inner_mask_img = find_inner_mask(paths, case)
    s17_labelmap_img = find_s17_labelmap(paths, case)

    all_shells = find_dwi08_shells(paths, case)
    shells = all_shells if selected_shells is None else selected_shells

    shell_results = {}
    qc_rows = []

    for shell in shells:
        result = process_shell(
            case=case,
            shell=shell,
            paths=paths,
            out_case_dir=out_case_dir,
            vesselness_img=vesselness_img,
            inner_mask_img=inner_mask_img,
            s17_labelmap_img=s17_labelmap_img,
            quantile_bins_n=quantile_bins_n,
            equal_width_bins_n=equal_width_bins_n,
            use_equal_width=use_equal_width,
            min_vesselness=min_vesselness,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        shell_results[shell] = result

        cont = result["statistics"]["continuous"]
        qc_rows.append(
            {
                "case": case,
                "shell": shell,
                "valid_voxels": result["valid_voxels"],
                "valid_voxels_pre_vesselness_threshold": result["valid_voxels_pre_vesselness_threshold"],
                "min_vesselness": result["min_vesselness"],
                "fraction_retained_after_vesselness_threshold": cont["fraction_retained_after_vesselness_threshold"],
                "spearman_vesselness_theta": cont["correlation_vesselness_theta"]["spearman_rho"],
                "spearman_p_vesselness_theta": cont["correlation_vesselness_theta"]["spearman_p"],
                "spearman_vesselness_cos2": cont["correlation_vesselness_cos2"]["spearman_rho"],
                "spearman_p_vesselness_cos2": cont["correlation_vesselness_cos2"]["spearman_p"],
                "quantile_bins": quantile_bins_n,
                "equal_width_bins": equal_width_bins_n if use_equal_width else None,
            }
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
        "scientific_objective": "Quantify the relationship between continuous vesselness and fiber orientation relative to B0.",
        "analysis_domain": {
            "primary_predictor": "continuous Sato vesselness",
            "primary_orientation_metrics": [
                "angle_to_B0_deg",
                "cos_theta",
                "cos2_theta",
            ],
            "mask": "s17 innerMask_dist4_from_s11b to reduce border effects",
            "valid_voxel_rule": "finite vesselness and orientation maps inside innerMask_dist4 and vesselness >= min_vesselness",
        },
        "vesselness_thresholding": {
            "min_vesselness": float(min_vesselness),
            "threshold_label": vesselness_threshold_label(min_vesselness),
            "threshold_applied": bool(min_vesselness > 0),
            "interpretation": "Analysis restricted to voxels with vesselness >= threshold. minV000 is the unthresholded baseline.",
        },
        "binning": {
            "quantile_bins": quantile_bins_n,
            "equal_width_bins": equal_width_bins_n if use_equal_width else None,
            "primary_binning": "quantile",
            "equal_width_used_for_comparison": use_equal_width,
            "s17_4bin_labelmap_used_as_supplementary": s17_labelmap_img is not None,
        },
        "inputs": {
            "continuous_vesselness": str(vesselness_img),
            "inner_mask_dist4": str(inner_mask_img),
            "s17_labelmap": str(s17_labelmap_img) if s17_labelmap_img else None,
        },
        "processed_shells": shells,
        "shell_results": shell_results,
    }

    meta_path = out_case_dir / f"dwi09_{case}_metadata.json"
    qc_csv = out_case_dir / f"dwi09_{case}_qc_summary.csv"

    if not dry_run:
        write_json(meta_path, metadata)
        write_csv(qc_csv, qc_rows)

    print(f"  DONE: {case}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF dwi09: continuous vesselness vs fiber orientation."
    )

    add_case_selection_args(parser)

    parser.add_argument("--shell", default=DEFAULT_SHELL, help=f"Single shell. Default: {DEFAULT_SHELL}.")
    parser.add_argument("--all-shells", action="store_true", help="Process all available dwi08 GRE-grid shells.")
    parser.add_argument("--quantile-bins", type=int, default=DEFAULT_QUANTILE_BINS)
    parser.add_argument("--equal-width-bins", type=int, default=DEFAULT_EQUAL_WIDTH_BINS)
    parser.add_argument("--no-equal-width", action="store_true", help="Disable equal-width bin comparison.")
    parser.add_argument("--min-vesselness", type=float, default=0.0, help="Minimum vesselness threshold for analysis. Default: 0.0 (baseline).")
    parser.add_argument(
        "--min-vesselness-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Run a threshold sweep with one or more minimum vesselness thresholds. "
            "Example: --min-vesselness-values 0 0.05 0.10 0.20 0.30 0.50. "
            "If provided, this overrides --min-vesselness."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = APFPaths(root=Path.cwd())

    registry_csv = getattr(paths, "config_dir", paths.root / "config") / "InS_cases.csv"
    cases = resolve_cases(args=args, registry_csv=registry_csv)
    print_selected_cases(cases)

    selected_shells = None if args.all_shells else [args.shell]
    use_equal_width = not args.no_equal_width

    # Remove duplicates while preserving order, and guard against invalid values.
    seen_thresholds = set()
    thresholds = []
    for value in (args.min_vesselness_values if args.min_vesselness_values is not None else [args.min_vesselness]):
        value = float(value)
        if value < 0 or value > 1:
            raise SystemExit(f"Invalid --min-vesselness value: {value}. Expected range: 0..1")
        key = round(value, 6)
        if key not in seen_thresholds:
            seen_thresholds.add(key)
            thresholds.append(value)

    for min_vesselness in thresholds:
        label = vesselness_threshold_label(min_vesselness)
        print("=" * 72)
        print(f"dwi09 threshold: {label}  (min vesselness = {min_vesselness:.6g})")
        print("=" * 72)

        all_rows = []

        for case in cases:
            try:
                meta = process_case(
                    case=case,
                    paths=paths,
                    selected_shells=selected_shells,
                    quantile_bins_n=args.quantile_bins,
                    equal_width_bins_n=args.equal_width_bins,
                    use_equal_width=use_equal_width,
                    min_vesselness=min_vesselness,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
                all_rows.append(
                    {
                        "case": case,
                        "success": True,
                        "processed_shells": ";".join(meta["processed_shells"]),
                        "threshold_label": label,
                        "min_vesselness": min_vesselness,
                        "metadata": str(dwi09_dir(paths, case) / label / f"dwi09_{case}_metadata.json"),
                    }
                )
            except Exception as exc:
                print(f"ERROR in {case} at {label}: {exc}", file=sys.stderr)
                all_rows.append(
                    {
                        "case": case,
                        "success": False,
                        "threshold_label": label,
                        "min_vesselness": min_vesselness,
                        "error": str(exc),
                    }
                )
                if len(cases) == 1 and len(thresholds) == 1:
                    raise

        if len(cases) > 1 and not args.dry_run:
            summary_dir = paths.derivatives_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            write_csv(summary_dir / f"dwi09_vesselness_orientation_summary_{label}.csv", all_rows)

    if len(thresholds) > 1 and not args.dry_run:
        sweep_rows = [
            {
                "threshold_label": vesselness_threshold_label(t),
                "min_vesselness": t,
                "summary_csv": str(paths.derivatives_dir / "summary" / f"dwi09_vesselness_orientation_summary_{vesselness_threshold_label(t)}.csv"),
            }
            for t in thresholds
        ]
        summary_dir = paths.derivatives_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_csv(summary_dir / "dwi09_vesselness_orientation_threshold_sweep_index.csv", sweep_rows)


if __name__ == "__main__":
    main()