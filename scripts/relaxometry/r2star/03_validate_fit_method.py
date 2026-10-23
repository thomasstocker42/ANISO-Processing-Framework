#!/usr/bin/env python3
"""
R2STAR03 — validation of log-linear GRE R2* fitting

Compares the production R2STAR02 log-linear fit against direct nonlinear
least-squares fitting of the original GRE magnitude signal:

    S(TE) = S0 * exp(-R2star * TE)

Validation is performed on a deterministic stratified sample of voxels.

This module is validation-only and does not replace the production
R2STAR02 reconstruction.

In addition to global agreement metrics, R2STAR03 evaluates method agreement
as a function of fit quality using:

  - R2STAR02 log-domain RMSE quartiles;
  - direct-fit magnitude-domain NRMSE quartiles;
  - predefined exploratory fit-quality thresholds.

The fit-quality thresholds are used to characterize method agreement only.
They are not production exclusion criteria.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.optimize import least_squares


SCRIPT_ID = "R2STAR03"
VERSION = "0.2.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[3]

RNG_SEED = 20260909
DEFAULT_SAMPLE_SIZE = 5000

R2_MIN = 1e-6
R2_MAX = 1000.0
MIN_VALID_ECHOES = 6

LOG_RMSE_THRESHOLDS = (0.05, 0.10, 0.20)
DIRECT_NRMSE_THRESHOLDS = (0.10, 0.20)


class ValidationError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_inputs(root: Path, case: str):
    mag_dir = root / "nifti" / case / "GRE" / "mag"

    echo_files = []
    json_files = []

    for echo in range(1, 13):
        candidates = sorted(
            list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii.gz"))
            + list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii"))
        )

        if len(candidates) != 1:
            raise ValidationError(
                f"{case}: expected one magnitude image for echo {echo}, "
                f"found {len(candidates)}"
            )

        nii = candidates[0]

        if nii.name.endswith(".nii.gz"):
            js = nii.with_name(nii.name[:-7] + ".json")
        else:
            js = nii.with_suffix(".json")

        if not js.is_file():
            raise ValidationError(f"Missing JSON: {js}")

        echo_files.append(nii)
        json_files.append(js)

    te = []

    for expected_echo, js in enumerate(json_files, start=1):
        metadata = json.loads(js.read_text())

        echo_number = int(metadata["EchoNumber"])
        te_s = float(metadata["EchoTime"])

        if echo_number != expected_echo:
            raise ValidationError(
                f"EchoNumber mismatch in {js}: {echo_number}"
            )

        te.append(te_s)

    te = np.asarray(te, dtype=np.float64)

    r2star02_dir = (
        root
        / "derivatives"
        / case
        / "relaxometry"
        / "r2star"
        / "r2star02_loglinear"
    )

    r2_file = (
        r2star02_dir
        / f"r2star02_{case}_r2star_s-1.nii.gz"
    )

    s0_file = (
        r2star02_dir
        / f"r2star02_{case}_s0.nii.gz"
    )

    fit_mask_file = (
        r2star02_dir
        / f"r2star02_{case}_fit_mask.nii.gz"
    )

    rmse_log_file = (
        r2star02_dir
        / f"r2star02_{case}_log_rmse.nii.gz"
    )

    for p in (
        r2_file,
        s0_file,
        fit_mask_file,
        rmse_log_file,
    ):
        if not p.is_file():
            raise ValidationError(f"Missing R2STAR02 input: {p}")

    ref = nib.load(str(echo_files[0]))

    signal = []

    for p in echo_files:
        img = nib.load(str(p))

        if img.shape != ref.shape:
            raise ValidationError(f"Shape mismatch: {p}")

        if not np.allclose(
            img.affine,
            ref.affine,
            atol=1e-5,
            rtol=0,
        ):
            raise ValidationError(f"Affine mismatch: {p}")

        signal.append(
            np.asarray(img.dataobj, dtype=np.float32)
        )

    signal = np.stack(signal, axis=0)

    r2 = np.asarray(
        nib.load(str(r2_file)).dataobj,
        dtype=np.float32,
    )

    s0 = np.asarray(
        nib.load(str(s0_file)).dataobj,
        dtype=np.float32,
    )

    fit_mask = (
        np.asarray(
            nib.load(str(fit_mask_file)).dataobj
        ) > 0
    )

    rmse_log = np.asarray(
        nib.load(str(rmse_log_file)).dataobj,
        dtype=np.float32,
    )

    return (
        te,
        signal,
        r2,
        s0,
        fit_mask,
        rmse_log,
    )


def stratified_sample(
    r2: np.ndarray,
    mask: np.ndarray,
    n_total: int,
    seed: int,
):
    """
    Deterministic sampling across R2* deciles.

    This prevents the validation sample from being dominated by the
    dense middle of the R2* distribution.
    """
    flat_indices = np.flatnonzero(mask)

    values = r2.ravel()[flat_indices]

    finite = (
        np.isfinite(values)
        & (values > 0)
    )

    flat_indices = flat_indices[finite]
    values = values[finite]

    if flat_indices.size == 0:
        raise ValidationError(
            "No valid R2STAR02 voxels available"
        )

    rng = np.random.default_rng(seed)

    quantiles = np.quantile(
        values,
        np.linspace(0, 1, 11),
    )

    sampled = []

    per_bin = max(1, n_total // 10)

    for i in range(10):
        if i == 9:
            sel = (
                (values >= quantiles[i])
                & (values <= quantiles[i + 1])
            )
        else:
            sel = (
                (values >= quantiles[i])
                & (values < quantiles[i + 1])
            )

        candidates = flat_indices[sel]

        if candidates.size == 0:
            continue

        n = min(per_bin, candidates.size)

        sampled.append(
            rng.choice(
                candidates,
                size=n,
                replace=False,
            )
        )

    if not sampled:
        raise ValidationError(
            "Stratified sampling returned no voxels"
        )

    indices = np.concatenate(sampled)

    rng.shuffle(indices)

    return indices


def direct_fit(
    te: np.ndarray,
    signal: np.ndarray,
    r2_init: float,
    s0_init: float,
):
    valid = (
        np.isfinite(signal)
        & (signal > 0)
    )

    if valid.sum() < MIN_VALID_ECHOES:
        return None

    t = te[valid]
    y = signal[valid].astype(np.float64)

    if (
        not np.isfinite(r2_init)
        or r2_init <= 0
    ):
        r2_init = 30.0

    if (
        not np.isfinite(s0_init)
        or s0_init <= 0
    ):
        s0_init = float(np.max(y))

    x0 = np.array(
        [
            max(s0_init, 1e-6),
            np.clip(r2_init, R2_MIN, R2_MAX),
        ],
        dtype=np.float64,
    )

    def residuals(x):
        s0, r2 = x
        pred = s0 * np.exp(-r2 * t)
        return pred - y

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(
            [1e-12, R2_MIN],
            [np.inf, R2_MAX],
        ),
        method="trf",
        max_nfev=200,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    if not result.success:
        return None

    s0_fit, r2_fit = result.x

    pred = s0_fit * np.exp(-r2_fit * t)

    rmse_signal = float(
        np.sqrt(
            np.mean(
                (y - pred) ** 2
            )
        )
    )

    mean_signal = float(np.mean(y))

    nrmse_signal = (
        rmse_signal / mean_signal
        if mean_signal > 0
        else np.nan
    )

    return {
        "s0_direct": float(s0_fit),
        "r2star_direct": float(r2_fit),
        "rmse_signal": rmse_signal,
        "nrmse_signal": float(nrmse_signal),
        "n_echo": int(valid.sum()),
        "nfev": int(result.nfev),
    }


def agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {
            "n": len(rows),
            "median_loglinear_s_inv": None,
            "median_direct_s_inv": None,
            "mean_difference_s_inv": None,
            "median_difference_s_inv": None,
            "mae_s_inv": None,
            "rmse_s_inv": None,
            "median_abs_difference_s_inv": None,
            "p95_abs_difference_s_inv": None,
            "pearson_r": None,
            "direct_on_loglinear_slope": None,
            "direct_on_loglinear_intercept": None,
        }

    log_values = np.asarray(
        [r["r2star_loglinear"] for r in rows],
        dtype=np.float64,
    )

    direct_values = np.asarray(
        [r["r2star_direct"] for r in rows],
        dtype=np.float64,
    )

    diff = direct_values - log_values
    absdiff = np.abs(diff)

    if (
        np.std(log_values) > 0
        and np.std(direct_values) > 0
    ):
        pearson_r = float(
            np.corrcoef(
                log_values,
                direct_values,
            )[0, 1]
        )
    else:
        pearson_r = None

    slope, intercept = np.polyfit(
        log_values,
        direct_values,
        1,
    )

    return {
        "n": len(rows),
        "median_loglinear_s_inv":
            float(np.median(log_values)),
        "median_direct_s_inv":
            float(np.median(direct_values)),
        "mean_difference_s_inv":
            float(np.mean(diff)),
        "median_difference_s_inv":
            float(np.median(diff)),
        "mae_s_inv":
            float(np.mean(absdiff)),
        "rmse_s_inv":
            float(np.sqrt(np.mean(diff ** 2))),
        "median_abs_difference_s_inv":
            float(np.median(absdiff)),
        "p95_abs_difference_s_inv":
            float(np.percentile(absdiff, 95)),
        "pearson_r":
            pearson_r,
        "direct_on_loglinear_slope":
            float(slope),
        "direct_on_loglinear_intercept":
            float(intercept),
    }


def split_quantiles(
    rows: list[dict[str, Any]],
    field: str,
    n_bins: int = 4,
):
    values = np.asarray(
        [float(r[field]) for r in rows],
        dtype=np.float64,
    )

    edges = np.quantile(
        values,
        np.linspace(0, 1, n_bins + 1),
    )

    output = []

    for i in range(n_bins):
        if i == n_bins - 1:
            selected = [
                r for r in rows
                if float(r[field]) >= edges[i]
                and float(r[field]) <= edges[i + 1]
            ]
        else:
            selected = [
                r for r in rows
                if float(r[field]) >= edges[i]
                and float(r[field]) < edges[i + 1]
            ]

        metrics = agreement_metrics(selected)

        output.append(
            {
                "bin": i + 1,
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "median_quality_value":
                    float(
                        np.median(
                            [float(r[field]) for r in selected]
                        )
                    )
                    if selected else None,
                **metrics,
            }
        )

    return output


def threshold_subset(
    rows: list[dict[str, Any]],
    label: str,
    predicate,
):
    selected = [
        r for r in rows
        if predicate(r)
    ]

    return {
        "label": label,
        **agreement_metrics(selected),
    }


def process_case(
    root: Path,
    case: str,
    sample_size: int,
    seed: int,
):
    (
        te,
        signal,
        r2_log,
        s0_log,
        fit_mask,
        rmse_log_map,
    ) = load_inputs(
        root,
        case,
    )

    indices = stratified_sample(
        r2_log,
        fit_mask,
        sample_size,
        seed,
    )

    shape = r2_log.shape

    rows = []

    for count, flat_idx in enumerate(
        indices,
        start=1,
    ):
        xyz = np.unravel_index(
            flat_idx,
            shape,
        )

        y = signal[
            :,
            xyz[0],
            xyz[1],
            xyz[2],
        ]

        r2_initial = float(
            r2_log[xyz]
        )

        s0_initial = float(
            s0_log[xyz]
        )

        rmse_loglinear = float(
            rmse_log_map[xyz]
        )

        fit = direct_fit(
            te,
            y,
            r2_initial,
            s0_initial,
        )

        if fit is None:
            continue

        direct = fit["r2star_direct"]

        rows.append(
            {
                "case": case,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "r2star_loglinear":
                    r2_initial,
                "r2star_direct":
                    direct,
                "difference_direct_minus_loglinear":
                    direct - r2_initial,
                "relative_difference_percent":
                    100.0
                    * (direct - r2_initial)
                    / r2_initial,
                "s0_loglinear":
                    s0_initial,
                "s0_direct":
                    fit["s0_direct"],
                "rmse_loglinear":
                    rmse_loglinear,
                "rmse_signal_direct":
                    fit["rmse_signal"],
                "nrmse_signal_direct":
                    fit["nrmse_signal"],
                "n_echo":
                    fit["n_echo"],
                "nfev":
                    fit["nfev"],
            }
        )

        if count % 500 == 0:
            print(
                f"  fitted {count:,}/{len(indices):,}"
            )

    if not rows:
        raise ValidationError(
            f"{case}: no successful direct fits"
        )

    overall = agreement_metrics(rows)

    log_rmse_quartiles = split_quantiles(
        rows,
        "rmse_loglinear",
        4,
    )

    direct_nrmse_quartiles = split_quantiles(
        rows,
        "nrmse_signal_direct",
        4,
    )

    threshold_results = []

    for threshold in LOG_RMSE_THRESHOLDS:
        threshold_results.append(
            threshold_subset(
                rows,
                f"logRMSE <= {threshold:.2f}",
                lambda r, t=threshold:
                    float(r["rmse_loglinear"]) <= t,
            )
        )

    for threshold in DIRECT_NRMSE_THRESHOLDS:
        threshold_results.append(
            threshold_subset(
                rows,
                f"direct NRMSE <= {threshold:.2f}",
                lambda r, t=threshold:
                    float(r["nrmse_signal_direct"]) <= t,
            )
        )

    threshold_results.append(
        threshold_subset(
            rows,
            "both good",
            lambda r:
                float(r["rmse_loglinear"]) <= 0.10
                and float(r["nrmse_signal_direct"]) <= 0.20,
        )
    )

    summary = {
        "case": case,
        "requested_sample_size":
            sample_size,
        "successful_fits":
            len(rows),
        "overall":
            overall,
        "log_rmse_quartiles":
            log_rmse_quartiles,
        "direct_nrmse_quartiles":
            direct_nrmse_quartiles,
        "quality_thresholds":
            threshold_results,
    }

    return rows, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate R2STAR02 log-linear fitting "
            "against direct magnitude-domain "
            "mono-exponential fitting."
        )
    )

    parser.add_argument(
        "cases",
        nargs="+",
        help="Case IDs to validate.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Approximate stratified voxel sample "
            f"per case (default {DEFAULT_SAMPLE_SIZE})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    return parser.parse_args()


def print_metrics(prefix: str, metrics: dict[str, Any]):
    if metrics["n"] < 2:
        print(
            f"{prefix}: N={metrics['n']} "
            "(insufficient for agreement statistics)"
        )
        return

    print(
        f"{prefix}: "
        f"N={metrics['n']:4d} "
        f"medΔ={metrics['median_difference_s_inv']:+.3f} "
        f"MAE={metrics['mae_s_inv']:.3f} "
        f"RMSE={metrics['rmse_s_inv']:.3f} "
        f"r={metrics['pearson_r']:.5f} "
        f"p95|Δ|={metrics['p95_abs_difference_s_inv']:.3f}"
    )


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()

    out_dir = (
        root
        / "derivatives"
        / "summary"
        / "relaxometry"
        / "r2star"
        / "r2star03_fit_method_validation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "R2STAR03 — direct magnitude-domain "
        "fit validation"
    )
    print(f"Version:     {VERSION}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seed:        {args.seed}")
    print("=" * 72)

    all_rows = []
    summaries = []

    for case in args.cases:
        print()
        print(case)

        rows, summary = process_case(
            root,
            case,
            args.sample_size,
            args.seed,
        )

        all_rows.extend(rows)
        summaries.append(summary)

        overall = summary["overall"]

        print(
            f"  successful fits: "
            f"{summary['successful_fits']:,}"
        )
        print(
            f"  median log-linear: "
            f"{overall['median_loglinear_s_inv']:.4f} s^-1"
        )
        print(
            f"  median direct:     "
            f"{overall['median_direct_s_inv']:.4f} s^-1"
        )
        print()
        print_metrics(
            "  overall",
            overall,
        )

        print()
        print("  by log-linear RMSE quartile:")

        for q in summary["log_rmse_quartiles"]:
            print_metrics(
                (
                    f"    Q{q['bin']} "
                    f"(median RMSE="
                    f"{q['median_quality_value']:.4f})"
                ),
                q,
            )

        print()
        print("  quality-threshold subsets:")

        for q in summary["quality_thresholds"]:
            print_metrics(
                f"    {q['label']}",
                q,
            )

    voxel_csv = (
        out_dir
        / "r2star03_voxelwise_validation.csv"
    )

    fields = list(all_rows[0].keys())

    with voxel_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_json = (
        out_dir
        / "r2star03_summary.json"
    )

    payload = {
        "script_id":
            SCRIPT_ID,
        "version":
            VERSION,
        "timestamp":
            now_iso(),
        "fit_model":
            "S(TE)=S0*exp(-R2star*TE)",
        "fit_domain":
            "original magnitude signal",
        "optimizer":
            "scipy.optimize.least_squares",
        "sample_strategy":
            "deterministic stratified sampling "
            "over R2STAR02 R2star deciles",
        "sample_size_requested_per_case":
            args.sample_size,
        "seed":
            args.seed,
        "quality_analysis": {
            "log_rmse_quartiles":
                True,
            "direct_nrmse_quartiles":
                True,
            "log_rmse_thresholds":
                list(LOG_RMSE_THRESHOLDS),
            "direct_nrmse_thresholds":
                list(DIRECT_NRMSE_THRESHOLDS),
            "thresholds_are_production_exclusions":
                False,
        },
        "cases":
            summaries,
    }

    summary_json.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    # Compact flat summary for easy spreadsheet inspection.
    summary_csv = (
        out_dir
        / "r2star03_summary.csv"
    )

    flat_rows = []

    for summary in summaries:
        case = summary["case"]

        flat_rows.append(
            {
                "case":
                    case,
                "subset":
                    "overall",
                **summary["overall"],
            }
        )

        for q in summary["log_rmse_quartiles"]:
            flat_rows.append(
                {
                    "case":
                        case,
                    "subset":
                        f"logRMSE_Q{q['bin']}",
                    **{
                        k: v
                        for k, v in q.items()
                        if k not in {
                            "bin",
                            "lower",
                            "upper",
                            "median_quality_value",
                        }
                    },
                }
            )

        for q in summary["direct_nrmse_quartiles"]:
            flat_rows.append(
                {
                    "case":
                        case,
                    "subset":
                        f"directNRMSE_Q{q['bin']}",
                    **{
                        k: v
                        for k, v in q.items()
                        if k not in {
                            "bin",
                            "lower",
                            "upper",
                            "median_quality_value",
                        }
                    },
                }
            )

        for q in summary["quality_thresholds"]:
            flat_rows.append(
                {
                    "case":
                        case,
                    "subset":
                        q["label"],
                    **{
                        k: v
                        for k, v in q.items()
                        if k != "label"
                    },
                }
            )

    csv_fields = [
        "case",
        "subset",
        "n",
        "median_loglinear_s_inv",
        "median_direct_s_inv",
        "mean_difference_s_inv",
        "median_difference_s_inv",
        "mae_s_inv",
        "rmse_s_inv",
        "median_abs_difference_s_inv",
        "p95_abs_difference_s_inv",
        "pearson_r",
        "direct_on_loglinear_slope",
        "direct_on_loglinear_intercept",
    ]

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )
        writer.writeheader()

        for row in flat_rows:
            writer.writerow(
                {
                    field:
                        row.get(field, "")
                    for field in csv_fields
                }
            )

    print()
    print(f"Voxel CSV:   {voxel_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON:{summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())