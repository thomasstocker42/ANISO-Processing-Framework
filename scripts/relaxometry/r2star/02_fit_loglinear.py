#!/usr/bin/env python3
"""
R2STAR02 — GRE magnitude log-linear R2* reconstruction

Purpose
-------
Compute voxel-wise R2* from the 12-echo GRE magnitude series using the
legacy-equivalent mono-exponential log-linear model

    S(TE) = S0 exp(-R2* TE)
    ln(S) = ln(S0) - R2* TE

Scientific fitting behaviour is intentionally kept equivalent to legacy s19:
  - ordinary least-squares regression in log-signal space;
  - finite, positive magnitude samples are valid;
  - minimum 6 valid echoes;
  - final plausible range 0 < R2* <= 1000 s^-1.

APF improvements:
  - registry-driven case selection;
  - EchoTime read from JSON metadata;
  - finalized GRE02b non-eroded brain mask;
  - shape and affine validation;
  - explicit candidate and final-fit masks;
  - finite-value validation;
  - structured QC/provenance;
  - structured case/cohort output locations.

Per-case outputs
----------------
derivatives/<case>/relaxometry/r2star/r2star02_loglinear/

Cohort summary
--------------
derivatives/summary/relaxometry/r2star/r2star02_loglinear/
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


SCRIPT_ID = "R2STAR02"
SCRIPT_NAME = Path(__file__).name
VERSION = "0.1.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_ECHOES = tuple(range(1, 13))
MIN_VALID_ECHOES = 6

R2STAR_MIN_S_INV = 0.0
R2STAR_MAX_S_INV = 1000.0

AFFINE_ATOL = 1e-5


class ProcessingError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_registry_cases(root: Path) -> list[str]:
    """Use APF registry, with CSV fallback."""
    try:
        from apf_registry import get_cases  # type: ignore

        records = get_cases()
        cases: list[str] = []

        for record in records:
            if isinstance(record, str):
                case = record
            elif isinstance(record, dict):
                case = record.get("case") or record.get("case_id") or ""
            else:
                case = (
                    getattr(record, "case", "")
                    or getattr(record, "case_id", "")
                )

            if case:
                cases.append(str(case))

        if cases:
            return sorted(set(cases))

    except (ImportError, TypeError):
        pass

    csv_path = root / "config" / "InS_cases.csv"

    if not csv_path.is_file():
        raise ProcessingError(
            f"APF registry unavailable and case CSV not found: {csv_path}"
        )

    cases: list[str] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lowered = {
                str(k).lower(): str(v).strip()
                for k, v in row.items()
            }

            case = lowered.get("case") or lowered.get("case_id") or ""
            active = lowered.get("active", "true").lower()

            if case and active not in {
                "0", "false", "no", "inactive", "excluded"
            }:
                cases.append(case)

    return sorted(set(cases))


def load_echo_table(root: Path, case: str) -> list[dict[str, Any]]:
    mag_dir = root / "nifti" / case / "GRE" / "mag"

    if not mag_dir.is_dir():
        raise ProcessingError(f"Missing GRE magnitude directory: {mag_dir}")

    rows: list[dict[str, Any]] = []

    for echo in EXPECTED_ECHOES:
        candidates = sorted(
            list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii.gz"))
            + list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii"))
        )

        if len(candidates) != 1:
            raise ProcessingError(
                f"{case}: expected exactly one magnitude NIfTI for echo "
                f"{echo}, found {len(candidates)}."
            )

        nii = candidates[0]

        if nii.name.endswith(".nii.gz"):
            js = nii.with_name(nii.name[:-7] + ".json")
        else:
            js = nii.with_suffix(".json")

        if not js.is_file():
            raise ProcessingError(f"Missing JSON sidecar: {js}")

        try:
            metadata = json.loads(js.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProcessingError(
                f"Could not parse JSON {js}: {exc}"
            ) from exc

        echo_json = metadata.get("EchoNumber")
        te_s = metadata.get("EchoTime")

        if echo_json is None or te_s is None:
            raise ProcessingError(
                f"Missing EchoNumber/EchoTime in {js}"
            )

        if int(echo_json) != echo:
            raise ProcessingError(
                f"EchoNumber mismatch in {js}: "
                f"filename echo={echo}, JSON EchoNumber={echo_json}"
            )

        te_s = float(te_s)

        if not np.isfinite(te_s) or te_s <= 0:
            raise ProcessingError(
                f"Invalid EchoTime in {js}: {te_s}"
            )

        rows.append(
            {
                "echo": echo,
                "te_s": te_s,
                "te_ms": te_s * 1000.0,
                "nii": nii,
                "json": js,
            }
        )

    te = [row["te_s"] for row in rows]

    if not all(te[i] < te[i + 1] for i in range(len(te) - 1)):
        raise ProcessingError(
            f"{case}: EchoTime values are not strictly increasing."
        )

    return rows


def brain_mask_path(root: Path, case: str) -> Path:
    return (
        root
        / "derivatives"
        / case
        / "GRE"
        / "02b_brain_mask"
        / f"{case}_GRE02b_brain_mask.nii.gz"
    )


def case_output_dir(root: Path, case: str) -> Path:
    return (
        root
        / "derivatives"
        / case
        / "relaxometry"
        / "r2star"
        / "r2star02_loglinear"
    )


def summary_output_dir(root: Path) -> Path:
    return (
        root
        / "derivatives"
        / "summary"
        / "relaxometry"
        / "r2star"
        / "r2star02_loglinear"
    )


def assert_same_geometry(
    reference: nib.Nifti1Image,
    image: nib.Nifti1Image,
    label: str,
) -> None:
    if image.shape != reference.shape:
        raise ProcessingError(
            f"Shape mismatch for {label}: "
            f"{image.shape} != {reference.shape}"
        )

    if not np.allclose(
        image.affine,
        reference.affine,
        atol=AFFINE_ATOL,
        rtol=0,
    ):
        raise ProcessingError(
            f"Affine mismatch for {label}."
        )


def save_nifti(
    data: np.ndarray,
    reference: nib.Nifti1Image,
    path: Path,
    dtype: np.dtype,
) -> None:
    image = nib.Nifti1Image(
        data.astype(dtype),
        reference.affine,
        reference.header,
    )
    image.set_data_dtype(dtype)
    nib.save(image, str(path))


def percentile_or_nan(
    values: np.ndarray,
    percentile: float,
) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def process_case(
    root: Path,
    case: str,
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()
    started_iso = now_iso()

    echo_rows = load_echo_table(root, case)
    mask_file = brain_mask_path(root, case)

    if not mask_file.is_file():
        raise ProcessingError(
            f"Missing GRE02b brain mask: {mask_file}"
        )

    out_dir = case_output_dir(root, case)

    if out_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"r2star02_{case}"

    outputs = {
        "r2star_s_inv":
            out_dir / f"{prefix}_r2star_s-1.nii.gz",
        "r2star_ms_inv":
            out_dir / f"{prefix}_r2star_ms-1.nii.gz",
        "t2star_ms":
            out_dir / f"{prefix}_t2star_ms.nii.gz",
        "s0":
            out_dir / f"{prefix}_s0.nii.gz",
        "rmse_log":
            out_dir / f"{prefix}_log_rmse.nii.gz",
        "n_echo_used":
            out_dir / f"{prefix}_n_echo_used.nii.gz",
        "candidate_mask":
            out_dir / f"{prefix}_candidate_fit_mask.nii.gz",
        "fit_mask":
            out_dir / f"{prefix}_fit_mask.nii.gz",
        "qc":
            out_dir / f"{prefix}_qc.json",
    }

    if (
        not overwrite
        and outputs["r2star_s_inv"].is_file()
        and outputs["qc"].is_file()
    ):
        raise ProcessingError(
            f"Outputs already exist for {case}. "
            f"Use --overwrite to replace them."
        )

    reference = nib.load(str(echo_rows[0]["nii"]))
    reference_shape = reference.shape

    mask_img = nib.load(str(mask_file))
    assert_same_geometry(reference, mask_img, "GRE02b brain mask")

    mask_data = np.asarray(mask_img.dataobj)

    if not np.isfinite(mask_data).all():
        raise ProcessingError(
            f"{case}: brain mask contains non-finite values."
        )

    brain_mask = mask_data > 0

    if not brain_mask.any():
        raise ProcessingError(
            f"{case}: GRE02b brain mask is empty."
        )

    echo_data: list[np.ndarray] = []

    for row in echo_rows:
        img = nib.load(str(row["nii"]))
        assert_same_geometry(
            reference,
            img,
            f"echo {row['echo']}",
        )

        data = np.asarray(
            img.dataobj,
            dtype=np.float32,
        )

        echo_data.append(data)

    signal = np.stack(
        echo_data,
        axis=0,
    ).astype(np.float32)

    te_s = np.asarray(
        [row["te_s"] for row in echo_rows],
        dtype=np.float64,
    )

    # Legacy-equivalent valid sample definition, with explicit finite check.
    valid = (
        np.isfinite(signal)
        & (signal > 0)
        & brain_mask[None, ...]
    )

    n_valid = valid.sum(axis=0).astype(np.uint8)

    candidate_mask = (
        brain_mask
        & (n_valid >= MIN_VALID_ECHOES)
    )

    # Log signal is populated only for valid samples.
    log_signal = np.zeros_like(
        signal,
        dtype=np.float64,
    )
    log_signal[valid] = np.log(
        signal[valid].astype(np.float64)
    )

    x = te_s[:, None, None, None]
    v = valid.astype(np.float64)

    n = v.sum(axis=0)
    sum_x = (v * x).sum(axis=0)
    sum_y = log_signal.sum(axis=0)
    sum_xx = (v * x * x).sum(axis=0)
    sum_xy = (v * x * log_signal).sum(axis=0)

    denominator = n * sum_xx - sum_x * sum_x

    slope = np.zeros(
        reference_shape,
        dtype=np.float64,
    )
    intercept = np.zeros(
        reference_shape,
        dtype=np.float64,
    )

    regression_ok = (
        candidate_mask
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    slope[regression_ok] = (
        n[regression_ok] * sum_xy[regression_ok]
        - sum_x[regression_ok] * sum_y[regression_ok]
    ) / denominator[regression_ok]

    intercept[regression_ok] = (
        sum_y[regression_ok]
        - slope[regression_ok] * sum_x[regression_ok]
    ) / n[regression_ok]

    r2star_raw = -slope

    final_fit_mask = (
        regression_ok
        & np.isfinite(r2star_raw)
        & (r2star_raw > R2STAR_MIN_S_INV)
        & (r2star_raw <= R2STAR_MAX_S_INV)
        & np.isfinite(intercept)
    )

    r2star_s_inv = np.zeros(
        reference_shape,
        dtype=np.float32,
    )
    r2star_s_inv[final_fit_mask] = (
        r2star_raw[final_fit_mask]
        .astype(np.float32)
    )

    r2star_ms_inv = (
        r2star_s_inv / 1000.0
    ).astype(np.float32)

    t2star_ms = np.zeros(
        reference_shape,
        dtype=np.float32,
    )
    t2star_ms[final_fit_mask] = (
        1000.0
        / r2star_s_inv[final_fit_mask]
    )

    s0 = np.zeros(
        reference_shape,
        dtype=np.float32,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        s0_values = np.exp(
            intercept[final_fit_mask]
        )

    finite_s0 = np.isfinite(s0_values)

    fit_indices = np.flatnonzero(final_fit_mask)
    flat_s0 = s0.ravel()
    flat_s0[
        fit_indices[finite_s0]
    ] = s0_values[finite_s0].astype(np.float32)

    # Legacy-equivalent log-domain RMSE.
    prediction = (
        intercept[None, ...]
        + slope[None, ...] * x
    )

    residual = np.zeros_like(
        log_signal,
        dtype=np.float64,
    )

    residual[valid] = (
        log_signal[valid]
        - prediction[valid]
    )

    sse = (
        residual
        * residual
        * v
    ).sum(axis=0)

    rmse_log = np.zeros(
        reference_shape,
        dtype=np.float32,
    )

    rmse_log[regression_ok] = np.sqrt(
        sse[regression_ok]
        / n[regression_ok]
    ).astype(np.float32)

    rmse_log[~final_fit_mask] = 0

    # Final numerical checks.
    for name, array in (
        ("r2star_s_inv", r2star_s_inv),
        ("r2star_ms_inv", r2star_ms_inv),
        ("t2star_ms", t2star_ms),
        ("s0", s0),
        ("rmse_log", rmse_log),
    ):
        if not np.isfinite(array).all():
            raise ProcessingError(
                f"{case}: {name} contains non-finite values."
            )

    save_nifti(
        r2star_s_inv,
        reference,
        outputs["r2star_s_inv"],
        np.float32,
    )
    save_nifti(
        r2star_ms_inv,
        reference,
        outputs["r2star_ms_inv"],
        np.float32,
    )
    save_nifti(
        t2star_ms,
        reference,
        outputs["t2star_ms"],
        np.float32,
    )
    save_nifti(
        s0,
        reference,
        outputs["s0"],
        np.float32,
    )
    save_nifti(
        rmse_log,
        reference,
        outputs["rmse_log"],
        np.float32,
    )
    save_nifti(
        n_valid,
        reference,
        outputs["n_echo_used"],
        np.uint8,
    )
    save_nifti(
        candidate_mask.astype(np.uint8),
        reference,
        outputs["candidate_mask"],
        np.uint8,
    )
    save_nifti(
        final_fit_mask.astype(np.uint8),
        reference,
        outputs["fit_mask"],
        np.uint8,
    )

    r2_values = r2star_s_inv[final_fit_mask]
    t2_values = t2star_ms[final_fit_mask]
    rmse_values = rmse_log[final_fit_mask]

    brain_voxels = int(brain_mask.sum())
    candidate_voxels = int(candidate_mask.sum())
    fit_voxels = int(final_fit_mask.sum())

    summary = {
        "case": case,
        "status": "OK",
        "brain_voxels": brain_voxels,
        "candidate_voxels": candidate_voxels,
        "fit_voxels": fit_voxels,
        "candidate_fraction_of_brain":
            candidate_voxels / brain_voxels,
        "fit_fraction_of_brain":
            fit_voxels / brain_voxels,
        "r2star_s_inv_median":
            float(np.median(r2_values)),
        "r2star_s_inv_p05":
            percentile_or_nan(r2_values, 5),
        "r2star_s_inv_p95":
            percentile_or_nan(r2_values, 95),
        "t2star_ms_median":
            float(np.median(t2_values)),
        "t2star_ms_p05":
            percentile_or_nan(t2_values, 5),
        "t2star_ms_p95":
            percentile_or_nan(t2_values, 95),
        "rmse_log_median":
            float(np.median(rmse_values)),
        "rmse_log_p95":
            percentile_or_nan(rmse_values, 95),
        "n_echo_used_median":
            float(np.median(n_valid[final_fit_mask])),
        "n_echo_used_min":
            int(np.min(n_valid[final_fit_mask])),
        "output_r2star_s_inv":
            str(outputs["r2star_s_inv"]),
        "output_fit_mask":
            str(outputs["fit_mask"]),
    }

    finished = time.time()

    provenance = {
        "script": SCRIPT_NAME,
        "script_id": SCRIPT_ID,
        "version": VERSION,
        "timestamp": now_iso(),
        "case": case,
        "status": "OK",
        "model":
            "ln(S) = ln(S0) - R2star * TE",
        "fit_method":
            "ordinary least squares in log-signal space",
        "legacy_equivalent_to":
            "s19 19_GRE_R2star_fit_from_mag_all_cases.py",
        "input": {
            "magnitude_echoes": [
                str(row["nii"])
                for row in echo_rows
            ],
            "json_sidecars": [
                str(row["json"])
                for row in echo_rows
            ],
            "brain_mask": str(mask_file),
            "brain_mask_definition":
                "GRE02b non-eroded brain mask",
        },
        "echoes": [
            {
                "echo": row["echo"],
                "te_s": row["te_s"],
                "te_ms": row["te_ms"],
            }
            for row in echo_rows
        ],
        "parameters": {
            "expected_echoes":
                list(EXPECTED_ECHOES),
            "min_valid_echoes":
                MIN_VALID_ECHOES,
            "valid_signal":
                "finite and > 0",
            "r2star_min_s_inv":
                R2STAR_MIN_S_INV,
            "r2star_max_s_inv":
                R2STAR_MAX_S_INV,
            "affine_atol":
                AFFINE_ATOL,
            "rmse_definition":
                "sqrt(mean((ln(S)-ln(S_pred))^2)) "
                "over valid echoes",
            "susceptibility_or_vesselness_qc":
                "none",
        },
        "outputs": {
            key: str(value)
            for key, value in outputs.items()
        },
        "summary": summary,
        "execution": {
            "start": started_iso,
            "end": now_iso(),
            "runtime_seconds":
                finished - started,
            "host":
                socket.gethostname(),
            "platform":
                platform.platform(),
            "python":
                sys.version.split()[0],
            "numpy":
                np.__version__,
            "command":
                " ".join(sys.argv),
        },
    }

    with outputs["qc"].open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )
        f.write("\n")

    return summary


def write_summary(
    root: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    out_dir = summary_output_dir(root)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "r2star02_cohort_summary.csv"
    )
    json_path = (
        out_dir
        / "r2star02_cohort_summary.json"
    )

    fields = [
        "case",
        "status",
        "brain_voxels",
        "candidate_voxels",
        "fit_voxels",
        "candidate_fraction_of_brain",
        "fit_fraction_of_brain",
        "r2star_s_inv_median",
        "r2star_s_inv_p05",
        "r2star_s_inv_p95",
        "t2star_ms_median",
        "t2star_ms_p05",
        "t2star_ms_p95",
        "rmse_log_median",
        "rmse_log_p95",
        "n_echo_used_median",
        "n_echo_used_min",
        "output_r2star_s_inv",
        "output_fit_mask",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fields
                }
            )

    payload = {
        "script": SCRIPT_NAME,
        "script_id": SCRIPT_ID,
        "version": VERSION,
        "timestamp": now_iso(),
        "n_cases": len(rows),
        "n_successful": sum(
            row.get("status") == "OK"
            for row in rows
        ),
        "cases": rows,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )
        f.write("\n")

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF GRE magnitude log-linear "
            "R2* reconstruction."
        )
    )

    parser.add_argument(
        "cases",
        nargs="*",
        help="Case IDs.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all active registry cases.",
    )

    parser.add_argument(
        "--all-ready",
        action="store_true",
        help=(
            "With --all, process only cases "
            "that pass the R2STAR01-style "
            "input requirements."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing R2STAR02 outputs.",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    args = parser.parse_args()

    if args.all and args.cases:
        parser.error(
            "Use explicit cases or --all, not both."
        )

    if args.all_ready and not args.all:
        parser.error(
            "--all-ready requires --all."
        )

    if not args.all and not args.cases:
        parser.error(
            "Provide case ID(s) or use --all."
        )

    return args


def case_is_ready(
    root: Path,
    case: str,
) -> bool:
    try:
        rows = load_echo_table(root, case)
        mask = brain_mask_path(root, case)

        if not mask.is_file():
            return False

        ref = nib.load(str(rows[0]["nii"]))
        mask_img = nib.load(str(mask))

        assert_same_geometry(
            ref,
            mask_img,
            "brain mask",
        )

        for row in rows[1:]:
            image = nib.load(str(row["nii"]))
            assert_same_geometry(
                ref,
                image,
                f"echo {row['echo']}",
            )

        return True

    except Exception:
        return False


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    cases = (
        get_registry_cases(root)
        if args.all
        else sorted(set(args.cases))
    )

    if args.all_ready:
        cases = [
            case
            for case in cases
            if case_is_ready(root, case)
        ]

    print("=" * 72)
    print(
        f"{SCRIPT_ID} — GRE magnitude "
        f"log-linear R2* reconstruction"
    )
    print(f"Version: {VERSION}")
    print(f"Root:    {root}")
    print(
        "Model:   ln(S) = ln(S0) - R2* TE"
    )
    print(
        f"Cases:   {len(cases)}"
    )
    print("=" * 72)

    rows: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        print()
        print("=" * 72)
        print(case)
        print("=" * 72)

        try:
            summary = process_case(
                root,
                case,
                args.overwrite,
            )

            rows.append(summary)

            print(
                f"Median R2*: "
                f"{summary['r2star_s_inv_median']:.3f} s^-1"
            )
            print(
                f"Median T2*: "
                f"{summary['t2star_ms_median']:.3f} ms"
            )
            print(
                f"Median log-RMSE: "
                f"{summary['rmse_log_median']:.5f}"
            )
            print(
                f"Fit support: "
                f"{summary['fit_voxels']:,} / "
                f"{summary['brain_voxels']:,} "
                f"({100 * summary['fit_fraction_of_brain']:.2f}%)"
            )

        except Exception as exc:
            failures += 1

            print(
                f"ERROR: {exc}",
                file=sys.stderr,
            )

            rows.append(
                {
                    "case": case,
                    "status": "FAILED",
                    "message": str(exc),
                }
            )

    csv_path, json_path = write_summary(
        root,
        rows,
    )

    print()
    print("=" * 72)
    print(
        f"{SCRIPT_ID} complete: "
        f"{len(cases) - failures}/{len(cases)} successful"
    )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 72)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())