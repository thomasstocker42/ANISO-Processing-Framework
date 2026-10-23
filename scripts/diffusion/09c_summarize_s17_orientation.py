#!/usr/bin/env python3
"""
APF dwi09c: Cohort-level S17 vesselness-class versus fibre-orientation analysis.

This script aggregates frozen dwi09 S17 label summaries. It performs
subject-level cohort analysis using case as the statistical unit.

Primary analysis:
- shell: b2000
- vesselness classes:
    low       p50-p75
    medium    p75-p90
    high      p90-p97.5
    veryHigh  p97.5-p100
- primary contrast: veryHigh - low

Sensitivity:
- b1000
- b3000

Inputs
------
derivatives/<case>/DWI/vesselness_orientation/dwi09/minV000/<shell>/
    dwi09_<case>_<shell>_s17_label_summary.csv

Outputs
-------
derivatives/summary/dwi09_s17_orientation/
    dwi09c_s17_orientation_per_case.csv
    dwi09c_s17_orientation_class_summary.csv
    dwi09c_s17_orientation_paired_contrasts.csv
    dwi09c_s17_orientation_class_profile.png
    dwi09c_s17_orientation_paired_low_vs_veryHigh.png
    dwi09c_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("dwi09c requires matplotlib.") from exc

try:
    from apf_paths import APFPaths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError("Could not import APF infrastructure.") from exc


SCRIPT = "09c_summarize_s17_orientation.py"
TAG = "dwi09c"
VERSION = "0.1.0"

SHELLS = ["b1000", "b2000", "b3000"]
PRIMARY_SHELL = "b2000"

CLASS_ORDER = [1, 2, 3, 4]
CLASS_NAMES = {
    1: "low_p50-p75",
    2: "medium_p75-p90",
    3: "high_p90-p97p5",
    4: "veryHigh_p97p5-p100",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def input_csv(paths: APFPaths, case: str, shell: str) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "DWI"
        / "vesselness_orientation"
        / "dwi09"
        / "minV000"
        / shell
        / f"dwi09_{case}_{shell}_s17_label_summary.csv"
    )


def collect_per_case(cases: List[str], paths: APFPaths) -> List[Dict]:
    rows = []

    for case in cases:
        for shell in SHELLS:
            f = input_csv(paths, case, shell)

            if not f.exists():
                raise FileNotFoundError(f"Missing input: {f}")

            data = read_csv(f)
            by_label = {int(r["label"]): r for r in data}

            for label in CLASS_ORDER:
                if label not in by_label:
                    raise ValueError(
                        f"Missing S17 class {label} in {f}"
                    )

                r = by_label[label]

                rows.append(
                    {
                        "case": case,
                        "shell": shell,
                        "label": label,
                        "label_name": r["label_name"],
                        "n_voxels": int(float(r["n_voxels"])),
                        "vesselness_median": float(r["vesselness_median"]),
                        "theta_mean_deg": float(r["theta_mean"]),
                        "theta_median_deg": float(r["theta_median"]),
                        "theta_iqr_deg": float(r["theta_iqr"]),
                        "cos2_mean": float(r["cos2_mean"]),
                        "cos2_median": float(r["cos2_median"]),
                        "cos2_iqr": float(r["cos2_iqr"]),
                        "source_csv": str(f),
                    }
                )

    return rows


def summarize_classes(rows: List[Dict]) -> List[Dict]:
    out = []

    for shell in SHELLS:
        for label in CLASS_ORDER:
            subset = [
                r for r in rows
                if r["shell"] == shell and r["label"] == label
            ]

            theta = np.array(
                [r["theta_mean_deg"] for r in subset],
                dtype=float,
            )
            cos2 = np.array(
                [r["cos2_mean"] for r in subset],
                dtype=float,
            )

            n = len(subset)

            out.append(
                {
                    "shell": shell,
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                    "n_cases": n,
                    "theta_mean_across_cases_deg": float(np.mean(theta)),
                    "theta_sd_across_cases_deg": float(np.std(theta, ddof=1)),
                    "theta_sem_deg": float(stats.sem(theta)),
                    "cos2_mean_across_cases": float(np.mean(cos2)),
                    "cos2_sd_across_cases": float(np.std(cos2, ddof=1)),
                    "cos2_sem": float(stats.sem(cos2)),
                }
            )

    return out


def paired_contrast(
    rows: List[Dict],
    shell: str,
    metric: str,
    label_a: int = 1,
    label_b: int = 4,
) -> Dict:

    by_case = {}

    for r in rows:
        if r["shell"] != shell:
            continue
        by_case.setdefault(r["case"], {})[r["label"]] = r

    a = []
    b = []
    cases = []

    for case in sorted(by_case):
        if label_a not in by_case[case] or label_b not in by_case[case]:
            continue

        a.append(float(by_case[case][label_a][metric]))
        b.append(float(by_case[case][label_b][metric]))
        cases.append(case)

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    delta = b - a
    n = len(delta)

    mean_delta = float(np.mean(delta))
    sd_delta = float(np.std(delta, ddof=1))
    sem_delta = float(stats.sem(delta))

    ci_low, ci_high = stats.t.interval(
        0.95,
        df=n - 1,
        loc=mean_delta,
        scale=sem_delta,
    )

    ttest = stats.ttest_rel(b, a)
    wilcoxon = stats.wilcoxon(b, a)

    return {
        "shell": shell,
        "metric": metric,
        "contrast": f"{CLASS_NAMES[label_b]} - {CLASS_NAMES[label_a]}",
        "n_cases": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "sem_delta": sem_delta,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n_delta_negative": int(np.sum(delta < 0)),
        "n_delta_positive": int(np.sum(delta > 0)),
        "n_delta_zero": int(np.sum(delta == 0)),
        "paired_t_statistic": float(ttest.statistic),
        "paired_t_pvalue": float(ttest.pvalue),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
        "cases": ";".join(cases),
    }


def make_contrasts(rows: List[Dict]) -> List[Dict]:
    out = []

    for shell in SHELLS:
        out.append(
            paired_contrast(
                rows,
                shell,
                metric="theta_mean_deg",
            )
        )
        out.append(
            paired_contrast(
                rows,
                shell,
                metric="cos2_mean",
            )
        )

    return out


def plot_class_profile(rows: List[Dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(CLASS_ORDER))

    for shell in SHELLS:
        means = []
        sems = []

        for label in CLASS_ORDER:
            vals = np.array(
                [
                    r["theta_mean_deg"]
                    for r in rows
                    if r["shell"] == shell and r["label"] == label
                ],
                dtype=float,
            )

            means.append(np.mean(vals))
            sems.append(stats.sem(vals))

        ax.errorbar(
            x,
            means,
            yerr=sems,
            marker="o",
            capsize=4,
            label=shell,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["low\np50–75", "medium\np75–90", "high\np90–97.5", "very high\np97.5–100"]
    )
    ax.set_ylabel("Mean fibre angle to B0 [deg]")
    ax.set_xlabel("S17 vesselness class")
    ax.set_title("Fibre orientation across vesselness classes")
    ax.grid(True, alpha=0.3)
    ax.legend(title="DTI shell")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_paired_primary(rows: List[Dict], out_png: Path) -> None:
    by_case = {}

    for r in rows:
        if r["shell"] != PRIMARY_SHELL:
            continue
        by_case.setdefault(r["case"], {})[r["label"]] = r

    fig, ax = plt.subplots(figsize=(7, 6))

    for case in sorted(by_case):
        low = by_case[case][1]["theta_mean_deg"]
        very = by_case[case][4]["theta_mean_deg"]

        ax.plot(
            [0, 1],
            [low, very],
            marker="o",
            alpha=0.7,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low\np50–75", "very high\np97.5–100"])
    ax.set_ylabel("Mean fibre angle to B0 [deg]")
    ax.set_title("b2000 paired S17 vesselness-class comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF dwi09c: cohort-level S17 vesselness-orientation analysis."
    )
    add_case_selection_args(parser)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. Default: "
            "derivatives/summary/dwi09_s17_orientation"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    start = now_iso()
    args = parse_args()

    paths = APFPaths(root=Path.cwd())

    registry_csv = (
        getattr(paths, "config_dir", paths.root / "config")
        / "InS_cases.csv"
    )

    cases = resolve_cases(
        args=args,
        registry_csv=registry_csv,
    )

    print_selected_cases(cases)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            paths.derivatives_dir
            / "summary"
            / "dwi09_s17_orientation"
        )

    outputs = {
        "per_case_csv":
            out_dir / "dwi09c_s17_orientation_per_case.csv",
        "class_summary_csv":
            out_dir / "dwi09c_s17_orientation_class_summary.csv",
        "paired_contrasts_csv":
            out_dir / "dwi09c_s17_orientation_paired_contrasts.csv",
        "class_profile_png":
            out_dir / "dwi09c_s17_orientation_class_profile.png",
        "paired_primary_png":
            out_dir / "dwi09c_s17_orientation_paired_low_vs_veryHigh.png",
        "summary_json":
            out_dir / "dwi09c_summary.json",
    }

    rows = collect_per_case(cases, paths)
    class_summary = summarize_classes(rows)
    contrasts = make_contrasts(rows)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

        write_csv(outputs["per_case_csv"], rows)
        write_csv(outputs["class_summary_csv"], class_summary)
        write_csv(outputs["paired_contrasts_csv"], contrasts)

        plot_class_profile(
            rows,
            outputs["class_profile_png"],
        )

        plot_paired_primary(
            rows,
            outputs["paired_primary_png"],
        )

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
            "shells": SHELLS,
            "primary_shell": PRIMARY_SHELL,
            "vesselness_classes": CLASS_NAMES,
            "statistical_unit": "case",
            "primary_contrast": "veryHigh_p97p5-p100 - low_p50-p75",
            "outputs": {
                k: str(v)
                for k, v in outputs.items()
            },
        }

        write_json(
            outputs["summary_json"],
            metadata,
        )

    for label, path in outputs.items():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
