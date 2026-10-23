#!/usr/bin/env python3
"""
APF dwi09b: Cohort-level summary of dwi09 vesselness-threshold sensitivity.

This script aggregates existing dwi09 outputs. It does not perform voxelwise
analysis. It reads per-case/per-shell threshold-specific dwi09 CSV files and
produces cohort-level tables and figures for threshold sensitivity.

Inputs
------
derivatives/<case>/DWI/vesselness_orientation/dwi09/minV*/<shell>/
    dwi09_<case>_<shell>_quantile*_summary.csv

derivatives/<case>/DWI/vesselness_orientation/dwi09/minV*/
    dwi09_<case>_qc_summary.csv

Outputs
-------
derivatives/summary/dwi09_threshold_sensitivity/
    dwi09_threshold_sensitivity_per_case_shell.csv
    dwi09_threshold_sensitivity_cohort_summary.csv
    dwi09_threshold_sensitivity_retention_effectsize.png
    dwi09_threshold_sensitivity_spearman.png
    dwi09_quantile_curves_theta_continuous_ci95.png
    dwi09_quantile_curves_cos2_continuous_ci95.png
    dwi09_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

SCRIPT = "09b_summarize_vesselness_orientation_threshold_sweep.py"
TAG = "dwi09b"
VERSION = "0.3.0"

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("dwi09b requires matplotlib.") from exc

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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan", "na"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def as_int(value: object) -> Optional[int]:
    x = as_float(value)
    if x is None:
        return None
    return int(round(x))


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return None
    return float(np.mean(xs))


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return None
    return float(np.median(xs))


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if len(xs) < 2:
        return None
    return float(np.std(xs, ddof=1))


def sem(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if len(xs) < 2:
        return None
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def ci95(values: Iterable[Optional[float]]) -> Optional[float]:
    s = sem(values)
    if s is None:
        return None
    return float(1.96 * s)


def threshold_label(value: float) -> str:
    return f"minV{int(round(value * 100)):03d}"


def threshold_from_label(label: str) -> float:
    m = re.fullmatch(r"minV(\d+)", label)
    if not m:
        raise ValueError(f"Could not parse threshold label: {label}")
    return int(m.group(1)) / 100.0


def dwi09_root(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "vesselness_orientation" / "dwi09"


def find_threshold_dirs(paths: APFPaths, case: str) -> List[Path]:
    root = dwi09_root(paths, case)
    if not root.exists():
        raise FileNotFoundError(f"Missing dwi09 directory: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("minV"))
    if not dirs:
        raise FileNotFoundError(f"No threshold directories found in {root}")
    return dirs


def find_quantile_csv(threshold_dir: Path, case: str, shell: str) -> Path:
    shell_dir = threshold_dir / shell
    candidates = sorted(shell_dir.glob(f"dwi09_{case}_{shell}_quantile*_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No quantile summary CSV found in {shell_dir}")
    return candidates[0]


def collect_rows(cases: List[str], paths: APFPaths) -> tuple[List[Dict], List[Dict]]:
    """Return per-case/shell rows and quantile-curve rows."""
    per_case_shell_rows: List[Dict] = []
    quantile_curve_rows: List[Dict] = []

    for case in cases:
        for t_dir in find_threshold_dirs(paths, case):
            t_label = t_dir.name
            threshold = threshold_from_label(t_label)
            qc_csv = t_dir / f"dwi09_{case}_qc_summary.csv"
            if not qc_csv.exists():
                raise FileNotFoundError(f"Missing QC summary: {qc_csv}")

            for row in read_csv(qc_csv):
                shell = row["shell"]
                valid_voxels = as_int(row.get("valid_voxels"))
                pre_voxels = as_int(row.get("valid_voxels_pre_vesselness_threshold"))
                frac_retained = as_float(row.get("fraction_retained_after_vesselness_threshold"))
                percent_retained = None if frac_retained is None else 100.0 * frac_retained
                rho_theta = as_float(row.get("spearman_vesselness_theta"))
                rho_cos2 = as_float(row.get("spearman_vesselness_cos2"))

                q_csv = find_quantile_csv(t_dir, case, shell)
                q_rows = read_csv(q_csv)
                theta_medians = [as_float(q.get("theta_median")) for q in q_rows]
                cos2_medians = [as_float(q.get("cos2_median")) for q in q_rows]
                theta_vals = [x for x in theta_medians if x is not None and np.isfinite(x)]
                cos2_vals = [x for x in cos2_medians if x is not None and np.isfinite(x)]
                delta_theta = float(np.max(theta_vals) - np.min(theta_vals)) if theta_vals else None
                delta_cos2 = float(np.max(cos2_vals) - np.min(cos2_vals)) if cos2_vals else None

                per_case_shell_rows.append(
                    {
                        "case": case,
                        "shell": shell,
                        "threshold": threshold,
                        "threshold_label": t_label,
                        "valid_voxels": valid_voxels,
                        "valid_voxels_pre_vesselness_threshold": pre_voxels,
                        "percent_retained": percent_retained,
                        "spearman_vesselness_theta": rho_theta,
                        "spearman_vesselness_cos2": rho_cos2,
                        "delta_theta_quantile_median_deg": delta_theta,
                        "delta_cos2_quantile_median": delta_cos2,
                        "quantile_summary_csv": str(q_csv),
                    }
                )

                for q in q_rows:
                    quantile_curve_rows.append(
                        {
                            "case": case,
                            "shell": shell,
                            "threshold": threshold,
                            "threshold_label": t_label,
                            "bin": as_int(q.get("bin")),
                            "n_voxels": as_int(q.get("n_voxels")),
                            "vesselness_median": as_float(q.get("vesselness_median")),
                            "theta_median": as_float(q.get("theta_median")),
                            "cos2_median": as_float(q.get("cos2_median")),
                        }
                    )

    return per_case_shell_rows, quantile_curve_rows


def add_delta_vs_baseline(rows: List[Dict]) -> None:
    by_shell: Dict[str, Dict[float, Dict]] = defaultdict(dict)
    # Baseline is cohort-level threshold 0, not per-case.
    baseline_theta = None
    baseline_cos2 = None
    for row in rows:
        if abs(float(row["threshold"])) < 1e-12:
            baseline_theta = row.get("mean_spearman_vesselness_theta")
            baseline_cos2 = row.get("mean_spearman_vesselness_cos2")
            break

    for row in rows:
        rho_t = row.get("mean_spearman_vesselness_theta")
        rho_c = row.get("mean_spearman_vesselness_cos2")
        row["mean_delta_spearman_theta_vs_threshold0"] = (
            None if baseline_theta is None or rho_t is None else rho_t - baseline_theta
        )
        row["mean_delta_spearman_cos2_vs_threshold0"] = (
            None if baseline_cos2 is None or rho_c is None else rho_c - baseline_cos2
        )


def summarize_cohort(per_case_shell_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[float, List[Dict]] = defaultdict(list)
    for row in per_case_shell_rows:
        grouped[float(row["threshold"])].append(row)

    summary: List[Dict] = []
    for threshold in sorted(grouped):
        rows = grouped[threshold]
        pct_vals = [r.get("percent_retained") for r in rows]
        rho_theta_vals = [r.get("spearman_vesselness_theta") for r in rows]
        rho_cos2_vals = [r.get("spearman_vesselness_cos2") for r in rows]
        dtheta_vals = [r.get("delta_theta_quantile_median_deg") for r in rows]
        dcos2_vals = [r.get("delta_cos2_quantile_median") for r in rows]
        summary.append(
            {
                "threshold": threshold,
                "threshold_label": threshold_label(threshold),
                "n_case_shells": len(rows),
                "mean_voxels_retained": mean([r.get("valid_voxels") for r in rows]),
                "median_voxels_retained": median([r.get("valid_voxels") for r in rows]),
                "mean_percent_retained": mean(pct_vals),
                "median_percent_retained": median(pct_vals),
                "sd_percent_retained": std(pct_vals),
                "sem_percent_retained": sem(pct_vals),
                "ci95_percent_retained": ci95(pct_vals),
                "mean_spearman_vesselness_theta": mean(rho_theta_vals),
                "median_spearman_vesselness_theta": median(rho_theta_vals),
                "sd_spearman_vesselness_theta": std(rho_theta_vals),
                "sem_spearman_vesselness_theta": sem(rho_theta_vals),
                "ci95_spearman_vesselness_theta": ci95(rho_theta_vals),
                "mean_spearman_vesselness_cos2": mean(rho_cos2_vals),
                "median_spearman_vesselness_cos2": median(rho_cos2_vals),
                "sd_spearman_vesselness_cos2": std(rho_cos2_vals),
                "sem_spearman_vesselness_cos2": sem(rho_cos2_vals),
                "ci95_spearman_vesselness_cos2": ci95(rho_cos2_vals),
                "mean_delta_theta_quantile_median_deg": mean(dtheta_vals),
                "median_delta_theta_quantile_median_deg": median(dtheta_vals),
                "sd_delta_theta_quantile_median_deg": std(dtheta_vals),
                "sem_delta_theta_quantile_median_deg": sem(dtheta_vals),
                "ci95_delta_theta_quantile_median_deg": ci95(dtheta_vals),
                "mean_delta_cos2_quantile_median": mean(dcos2_vals),
                "median_delta_cos2_quantile_median": median(dcos2_vals),
                "sd_delta_cos2_quantile_median": std(dcos2_vals),
                "sem_delta_cos2_quantile_median": sem(dcos2_vals),
                "ci95_delta_cos2_quantile_median": ci95(dcos2_vals),
            }
        )

    add_delta_vs_baseline(summary)
    return summary

def summarize_quantile_curves(quantile_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple[float, int], List[Dict]] = defaultdict(list)
    for row in quantile_rows:
        b = row.get("bin")
        if b is None:
            continue
        grouped[(float(row["threshold"]), int(b))].append(row)

    out: List[Dict] = []
    for (threshold, bin_id), rows in sorted(grouped.items()):
        theta_vals = [r.get("theta_median") for r in rows]
        cos2_vals = [r.get("cos2_median") for r in rows]
        v_vals = [r.get("vesselness_median") for r in rows]
        out.append(
            {
                "threshold": threshold,
                "threshold_label": threshold_label(threshold),
                "bin": bin_id,
                "n_case_shells": len(rows),
                "mean_vesselness_median": mean(v_vals),
                "median_vesselness_median": median(v_vals),
                "sd_vesselness_median": std(v_vals),
                "sem_vesselness_median": sem(v_vals),
                "ci95_vesselness_median": ci95(v_vals),
                "mean_theta_median": mean(theta_vals),
                "median_theta_median": median(theta_vals),
                "sd_theta_median": std(theta_vals),
                "sem_theta_median": sem(theta_vals),
                "ci95_theta_median": ci95(theta_vals),
                "mean_cos2_median": mean(cos2_vals),
                "median_cos2_median": median(cos2_vals),
                "sd_cos2_median": std(cos2_vals),
                "sem_cos2_median": sem(cos2_vals),
                "ci95_cos2_median": ci95(cos2_vals),
            }
        )
    return out

def as_plot_array(values: List[Optional[float]], default: float = 0.0) -> np.ndarray:
    return np.asarray([default if v is None else float(v) for v in values], dtype=float)


def plot_retention_effectsize(summary_rows: List[Dict], out_png: Path) -> None:
    thresholds = [float(r["threshold"]) for r in summary_rows]
    pct = [float(r["mean_percent_retained"]) for r in summary_rows]
    pct_ci = as_plot_array([r.get("ci95_percent_retained") for r in summary_rows])
    dtheta = [float(r["mean_delta_theta_quantile_median_deg"]) for r in summary_rows]
    dtheta_ci = as_plot_array([r.get("ci95_delta_theta_quantile_median_deg") for r in summary_rows])
    dcos2 = [float(r["mean_delta_cos2_quantile_median"]) for r in summary_rows]
    dcos2_ci = as_plot_array([r.get("ci95_delta_cos2_quantile_median") for r in summary_rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].errorbar(thresholds, pct, yerr=pct_ci, marker="o", capsize=3)
    axes[0].set_ylabel("Mean voxels retained [%]")
    axes[0].set_title("A. Data retention")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(thresholds, dtheta, yerr=dtheta_ci, marker="o", capsize=3, label="Mean Δθ ±95% CI")
    axes[1].set_xlabel("Minimum vesselness threshold")
    axes[1].set_ylabel("Mean Δθ across quantile bins [deg]")
    axes[1].set_title("B. Orientation effect size")
    axes[1].grid(True, alpha=0.3)

    ax2 = axes[1].twinx()
    ax2.errorbar(thresholds, dcos2, yerr=dcos2_ci, marker="s", linestyle="--", capsize=3, label="Mean Δcos²θ ±95% CI")
    ax2.set_ylabel("Mean Δcos²θ")

    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle("dwi09 threshold sensitivity: retention and effect size")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_spearman(summary_rows: List[Dict], out_png: Path) -> None:
    thresholds = [float(r["threshold"]) for r in summary_rows]
    rho_t = [float(r["mean_spearman_vesselness_theta"]) for r in summary_rows]
    rho_t_ci = as_plot_array([r.get("ci95_spearman_vesselness_theta") for r in summary_rows])
    rho_c = [float(r["mean_spearman_vesselness_cos2"]) for r in summary_rows]
    rho_c_ci = as_plot_array([r.get("ci95_spearman_vesselness_cos2") for r in summary_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, linewidth=1)
    ax.errorbar(thresholds, rho_t, yerr=rho_t_ci, marker="o", capsize=3, label="ρ(vesselness, θ) ±95% CI")
    ax.errorbar(thresholds, rho_c, yerr=rho_c_ci, marker="s", linestyle="--", capsize=3, label="ρ(vesselness, cos²θ) ±95% CI")
    ax.set_xlabel("Minimum vesselness threshold")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title("dwi09 threshold sensitivity: monotonic correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def _curve_instances_for_threshold(rows: List[Dict], metric: str) -> List[tuple[np.ndarray, np.ndarray]]:
    value_key = "theta_median" if metric == "theta" else "cos2_median"
    grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case"]), str(row["shell"]))].append(row)

    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for _, curve_rows in grouped.items():
        pts = []
        for r in curve_rows:
            x = as_float(r.get("vesselness_median"))
            y = as_float(r.get(value_key))
            if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
        if len(pts) < 2:
            continue
        pts = sorted(pts)
        x = np.asarray([p[0] for p in pts], dtype=float)
        y = np.asarray([p[1] for p in pts], dtype=float)
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        if unique_x.size >= 2:
            curves.append((unique_x, unique_y))
    return curves


def _interpolate_curves(curves: List[tuple[np.ndarray, np.ndarray]], n_grid: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    x_min = min(float(x[0]) for x, _ in curves)
    x_max = max(float(x[-1]) for x, _ in curves)
    if x_max <= x_min:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    grid = np.linspace(x_min, x_max, n_grid)
    y_stack = []
    for x, y in curves:
        yi = np.full(grid.shape, np.nan, dtype=float)
        inside = (grid >= x[0]) & (grid <= x[-1])
        if np.any(inside):
            yi[inside] = np.interp(grid[inside], x, y)
        y_stack.append(yi)

    arr = np.vstack(y_stack)
    n = np.sum(np.isfinite(arr), axis=0)
    mean_y = np.nanmean(arr, axis=0)
    sd_y = np.nanstd(arr, axis=0, ddof=1)
    sem_y = sd_y / np.sqrt(np.maximum(n, 1))
    ci_y = 1.96 * sem_y
    mean_y[n < 2] = np.nan
    ci_y[n < 2] = np.nan
    return grid, mean_y, ci_y, sd_y, n.astype(float)


def plot_quantile_curves_continuous_ci(quantile_rows: List[Dict], metric: str, out_png: Path) -> None:
    ylabel = "Median θ to B0 [deg]" if metric == "theta" else "Median cos²θ"

    grouped: Dict[float, List[Dict]] = defaultdict(list)
    for row in quantile_rows:
        grouped[float(row["threshold"])].append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    for threshold in sorted(grouped):
        curves = _curve_instances_for_threshold(grouped[threshold], metric)
        x, y_mean, y_ci, _, n = _interpolate_curves(curves)
        if x.size == 0:
            continue
        valid = np.isfinite(y_mean) & np.isfinite(y_ci)
        if not np.any(valid):
            continue
        label = f"{threshold_label(threshold)} (n={len(curves)})"
        ax.plot(x[valid], y_mean[valid], linewidth=2.0, label=label)
        ax.fill_between(x[valid], y_mean[valid] - y_ci[valid], y_mean[valid] + y_ci[valid], alpha=0.18)

    ax.set_xlabel("Vesselness (continuous interpolation of quantile curves)")
    ax.set_ylabel(f"Mean {ylabel} ±95% CI")
    ax.set_title(f"dwi09 continuous quantile-curve summary across thresholds: {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Threshold", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF dwi09b: summarize dwi09 threshold-sensitivity outputs."
    )
    add_case_selection_args(parser)
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: derivatives/summary/dwi09_threshold_sensitivity")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    start = now_iso()
    args = parse_args()
    paths = APFPaths(root=Path.cwd())

    registry_csv = getattr(paths, "config_dir", paths.root / "config") / "InS_cases.csv"
    cases = resolve_cases(args=args, registry_csv=registry_csv)
    print_selected_cases(cases)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = paths.derivatives_dir / "summary" / "dwi09_threshold_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_case_shell_rows, quantile_rows = collect_rows(cases, paths)
    cohort_rows = summarize_cohort(per_case_shell_rows)
    cohort_curve_rows = summarize_quantile_curves(quantile_rows)

    outputs = {
        "per_case_shell_csv": out_dir / "dwi09_threshold_sensitivity_per_case_shell.csv",
        "cohort_summary_csv": out_dir / "dwi09_threshold_sensitivity_cohort_summary.csv",
        "quantile_curve_summary_csv": out_dir / "dwi09_threshold_sensitivity_quantile_curve_summary.csv",
        "retention_effectsize_png": out_dir / "dwi09_threshold_sensitivity_retention_effectsize.png",
        "spearman_png": out_dir / "dwi09_threshold_sensitivity_spearman.png",
        "theta_curves_png": out_dir / "dwi09_quantile_curves_theta_continuous_ci95.png",
        "cos2_curves_png": out_dir / "dwi09_quantile_curves_cos2_continuous_ci95.png",
        "summary_json": out_dir / "dwi09_summary.json",
    }

    if not args.dry_run:
        write_csv(outputs["per_case_shell_csv"], per_case_shell_rows)
        write_csv(outputs["cohort_summary_csv"], cohort_rows)
        write_csv(outputs["quantile_curve_summary_csv"], cohort_curve_rows)
        plot_retention_effectsize(cohort_rows, outputs["retention_effectsize_png"])
        plot_spearman(cohort_rows, outputs["spearman_png"])
        plot_quantile_curves_continuous_ci(quantile_rows, "theta", outputs["theta_curves_png"])
        plot_quantile_curves_continuous_ci(quantile_rows, "cos2", outputs["cos2_curves_png"])

        metadata = {
            "script": SCRIPT,
            "tag": TAG,
            "version": VERSION,
            "created": now_iso(),
            "execution": {
                "start": start,
                "end": now_iso(),
                "host": platform.node(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "command": " ".join(sys.argv),
            },
            "cases": cases,
            "n_cases": len(cases),
            "n_case_shell_threshold_rows": len(per_case_shell_rows),
            "thresholds": sorted({float(r["threshold"]) for r in per_case_shell_rows}),
            "metrics": [
                "valid_voxels",
                "percent_retained",
                "spearman_vesselness_theta",
                "spearman_vesselness_cos2",
                "delta_theta_quantile_median_deg",
                "delta_cos2_quantile_median",
                "sem",
                "ci95",
                "continuous_interpolated_quantile_curves",
            ],
            "outputs": {k: str(v) for k, v in outputs.items()},
        }
        write_json(outputs["summary_json"], metadata)

    for label, path in outputs.items():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
