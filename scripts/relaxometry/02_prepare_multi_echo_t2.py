#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
Relaxometry 02 — Prepare multi-echo T2 data

Purpose
-------
Consume the authoritative multi-echo T2 selection produced by rlx01 and
prepare deterministic native-space inputs for relaxometry fitting.

Main outputs
------------
- Echo-sorted 4D multi-echo NIfTI series.
- Echo-time vectors in seconds and milliseconds.
- First-echo and early-echo-mean reference images.
- Native-space fitting mask and one-voxel eroded QC mask.
- Per-case provenance/QC JSON and cohort summary CSV.

The source echoes are never resampled. All preparation remains on the native
multi-echo acquisition grid.

Version
-------
0.2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from apf_case_selection import (
    add_case_selection_args,
    print_selected_cases,
    resolve_cases,
)
from apf_paths import APFPaths, get_paths


SCRIPT = "02_prepare_multi_echo_t2.py"
TAG = "rlx02"
VERSION = "0.2.0"
MODULE = "relaxometry"
INPUT_STAGE = "rlx01_inventory"
OUTPUT_STAGE = "rlx02_prepare_multi_echo_t2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_selection_args(parser)
    parser.add_argument(
        "--n-early",
        type=int,
        default=3,
        help="Number of earliest echoes used for the reference mean (default: 3).",
    )
    parser.add_argument(
        "--noise-shell-width-mm",
        type=float,
        default=8.0,
        help=(
            "Physical width in millimetres of the outer image shell used for "
            "robust noise estimation. The width is converted independently "
            "for each image axis (default: 8.0 mm)."
        ),
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=5.0,
        help=(
            "Foreground threshold in robust background-noise standard "
            "deviations above the background median (default: 5.0)."
        ),
    )
    parser.add_argument(
        "--min-threshold-fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum mask threshold as a fraction of the early-echo-mean "
            "99th percentile (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--closing-iterations",
        type=int,
        default=1,
        help="Binary-closing iterations for the fit mask (default: 1).",
    )
    parser.add_argument(
        "--erosions",
        type=int,
        default=1,
        help="Number of erosions used for the QC mask (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rlx02 outputs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_early < 1:
        raise ValueError("--n-early must be at least 1.")
    if args.noise_shell_width_mm <= 0:
        raise ValueError("--noise-shell-width-mm must be positive.")
    if args.snr_threshold <= 0:
        raise ValueError("--snr-threshold must be positive.")
    if not 0 <= args.min_threshold_fraction < 1:
        raise ValueError("--min-threshold-fraction must be in [0, 1).")
    if args.closing_iterations < 0:
        raise ValueError("--closing-iterations cannot be negative.")
    if args.erosions < 0:
        raise ValueError("--erosions cannot be negative.")


def project_relative(path: Path, root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_project_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_vector(path: Path, values: list[float], precision: int = 9) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value:.{precision}g}\n")
    temporary.replace(path)


def selected_inputs_path(paths: APFPaths, case: str) -> Path:
    return (
        paths.case_stage_dir(
            case=case,
            module=MODULE,
            stage=INPUT_STAGE,
        )
        / f"rlx01_{case}_selected_inputs.json"
    )


def output_dir(paths: APFPaths, case: str) -> Path:
    return paths.case_stage_dir(
        case=case,
        module=MODULE,
        stage=OUTPUT_STAGE,
    )


def load_selected_multi_echo(paths: APFPaths, case: str) -> dict[str, Any]:
    selection_path = selected_inputs_path(paths, case)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing rlx01 selected-input file: {selection_path}"
        )

    payload = load_json(selection_path)
    selected = payload.get("multi_echo_t2")
    if not isinstance(selected, dict):
        raise ValueError(
            f"{case}: rlx01 did not select a multi_echo_t2 acquisition."
        )
    if not bool(selected.get("selected")):
        raise ValueError(
            f"{case}: selected multi_echo_t2 record is not marked selected."
        )
    if not bool(selected.get("ready")):
        raise ValueError(
            f"{case}: selected multi_echo_t2 record is not marked ready."
        )

    echoes = selected.get("echoes")
    if not isinstance(echoes, list) or len(echoes) < 4:
        raise ValueError(
            f"{case}: selected multi_echo_t2 contains fewer than four echoes."
        )
    return selected


def load_echo_series(
    paths: APFPaths,
    case: str,
    selected: dict[str, Any],
) -> tuple[list[Path], list[float], list[int | None], list[nib.Nifti1Image], np.ndarray]:
    records = selected["echoes"]

    try:
        ordered = sorted(
            records,
            key=lambda record: (
                float(record["echo_time_s"]),
                int(record["echo_number"])
                if record.get("echo_number") is not None
                else 0,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{case}: invalid echo metadata in rlx01 selection.") from exc

    echo_paths: list[Path] = []
    echo_times_s: list[float] = []
    echo_numbers: list[int | None] = []
    images: list[nib.Nifti1Image] = []
    arrays: list[np.ndarray] = []

    reference_shape: tuple[int, ...] | None = None
    reference_zooms: tuple[float, ...] | None = None
    reference_affine: np.ndarray | None = None

    for index, record in enumerate(ordered, start=1):
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"{case}: echo {index} has no valid image path.")

        image_path = resolve_project_path(image_value, paths.root)
        if not image_path.is_file():
            raise FileNotFoundError(f"{case}: missing echo image: {image_path}")

        echo_time_s = float(record["echo_time_s"])
        if not np.isfinite(echo_time_s) or echo_time_s <= 0:
            raise ValueError(
                f"{case}: invalid EchoTime for {image_path}: {echo_time_s}"
            )

        image = nib.load(str(image_path))
        if len(image.shape) != 3:
            raise ValueError(
                f"{case}: expected a 3D echo, got shape {image.shape}: {image_path}"
            )

        shape = tuple(int(value) for value in image.shape)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])

        if reference_shape is None:
            reference_shape = shape
            reference_zooms = zooms
            reference_affine = image.affine.copy()
        else:
            if shape != reference_shape:
                raise ValueError(
                    f"{case}: echo shape mismatch: {shape} != {reference_shape}"
                )
            if not np.allclose(zooms, reference_zooms, atol=1e-5, rtol=0):
                raise ValueError(
                    f"{case}: echo voxel-size mismatch: {zooms} != {reference_zooms}"
                )
            if not np.allclose(
                image.affine,
                reference_affine,
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError(f"{case}: echo affine mismatch: {image_path}")

        data = np.asarray(image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{case}: non-finite signal values in {image_path}")

        echo_paths.append(image_path)
        echo_times_s.append(echo_time_s)
        echo_number_value = record.get("echo_number")
        echo_numbers.append(
            int(echo_number_value) if echo_number_value is not None else None
        )
        images.append(image)
        arrays.append(data)

    if any(
        later <= earlier
        for earlier, later in zip(echo_times_s, echo_times_s[1:])
    ):
        raise ValueError(f"{case}: echo times are not strictly increasing.")

    stacked = np.stack(arrays, axis=3).astype(np.float32, copy=False)
    return echo_paths, echo_times_s, echo_numbers, images, stacked


def shell_width_voxels(
    width_mm: float,
    voxel_size_mm: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Convert a physical shell width to a safe voxel width per axis."""
    widths: list[int] = []
    for requested_mm, voxel_mm, size in zip(
        (width_mm, width_mm, width_mm),
        voxel_size_mm,
        shape,
    ):
        if not np.isfinite(voxel_mm) or voxel_mm <= 0:
            raise ValueError(f"Invalid voxel size for noise shell: {voxel_mm}")
        width = max(1, int(np.ceil(requested_mm / voxel_mm)))
        # Preserve at least one voxel in the interior on each side.
        width = min(width, max(1, (size - 1) // 2))
        widths.append(width)
    return tuple(widths)


def outer_shell(
    shape: tuple[int, int, int],
    widths: tuple[int, int, int],
) -> np.ndarray:
    """Return the union of axis-specific peripheral image slabs."""
    shell = np.ones(shape, dtype=bool)
    inner = tuple(
        slice(width, size - width)
        for size, width in zip(shape, widths)
    )
    if all(item.stop > item.start for item in inner):
        shell[inner] = False
    return shell


def robust_noise_statistics(
    image: np.ndarray,
    shell_width_vox: tuple[int, int, int],
) -> tuple[float, float, int]:
    shell_values = image[outer_shell(image.shape, shell_width_vox)]
    shell_values = shell_values[np.isfinite(shell_values)]
    if shell_values.size == 0:
        raise ValueError("No finite voxels available for background estimation.")

    median = float(np.median(shell_values))
    mad = float(np.median(np.abs(shell_values - median)))
    robust_sigma = 1.4826 * mad

    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        q25, q75 = np.percentile(shell_values, [25.0, 75.0])
        robust_sigma = float((q75 - q25) / 1.349)
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.std(shell_values))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.finfo(np.float32).eps)

    return median, robust_sigma, int(shell_values.size)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    labels, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labels == largest


def create_fit_masks(
    early_mean: np.ndarray,
    shell_width_vox: tuple[int, int, int],
    snr_threshold: float,
    minimum_fraction: float,
    closing_iterations: int,
    erosions: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    background_median, noise_sigma, n_noise = robust_noise_statistics(
        early_mean,
        shell_width_vox,
    )

    positive = early_mean[np.isfinite(early_mean) & (early_mean > 0)]
    if positive.size == 0:
        raise ValueError("Early-echo mean contains no positive finite signal.")

    p99 = float(np.percentile(positive, 99.0))
    noise_threshold = background_median + snr_threshold * noise_sigma
    fractional_threshold = minimum_fraction * p99
    threshold = float(max(noise_threshold, fractional_threshold))

    mask = np.isfinite(early_mean) & (early_mean > threshold)
    mask = keep_largest_component(mask)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    if closing_iterations > 0:
        mask = ndimage.binary_closing(
            mask,
            structure=structure,
            iterations=closing_iterations,
        )
    mask = ndimage.binary_fill_holes(mask)
    mask = keep_largest_component(mask)

    if not np.any(mask):
        raise ValueError(
            "Automatic fit-mask generation produced an empty mask. "
            "Review acquisition coverage or mask parameters."
        )

    if erosions > 0:
        eroded = ndimage.binary_erosion(
            mask,
            structure=structure,
            iterations=erosions,
            border_value=0,
        )
    else:
        eroded = mask.copy()

    if not np.any(eroded):
        eroded = mask.copy()
        effective_erosions = 0
    else:
        effective_erosions = erosions

    qc = {
        "background_median": background_median,
        "background_robust_sigma": noise_sigma,
        "noise_shell_width_vox": [int(value) for value in shell_width_vox],
        "noise_shell_voxels": n_noise,
        "positive_signal_p99": p99,
        "noise_based_threshold": float(noise_threshold),
        "fractional_threshold": float(fractional_threshold),
        "applied_threshold": threshold,
        "fit_mask_voxels": int(np.count_nonzero(mask)),
        "fit_mask_fraction": float(np.mean(mask)),
        "qc_mask_voxels": int(np.count_nonzero(eroded)),
        "effective_erosions": int(effective_erosions),
    }
    return mask.astype(np.uint8), eroded.astype(np.uint8), qc


def output_paths(
    directory: Path,
    case: str,
    erosions: int,
) -> dict[str, Path]:
    prefix = f"{TAG}_{case}"
    return {
        "series_4d": directory / f"{prefix}_multi_echo_t2.nii.gz",
        "echo_times_s": directory / f"{prefix}_echo_times_s.txt",
        "echo_times_ms": directory / f"{prefix}_echo_times_ms.txt",
        "echo1": directory / f"{prefix}_echo1.nii.gz",
        "early_mean": directory / f"{prefix}_early_echo_mean.nii.gz",
        "fit_mask": directory / f"{prefix}_fit_mask.nii.gz",
        "fit_mask_ero": directory / (
            f"{prefix}_fit_mask_ero{erosions}.nii.gz"
            if erosions > 0
            else f"{prefix}_fit_mask_qc.nii.gz"
        ),
        "metadata": directory / f"{prefix}_preparation.json",
    }


def save_nifti(
    path: Path,
    data: np.ndarray,
    reference: nib.Nifti1Image,
    dtype: np.dtype[Any],
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        header,
    )
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(path))


def process_case(
    paths: APFPaths,
    case: str,
    args: argparse.Namespace,
    command: list[str],
) -> dict[str, Any]:
    started = time.time()
    selected = load_selected_multi_echo(paths, case)

    directory = output_dir(paths, case)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(directory, case, args.erosions)

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{case}: rlx02 outputs already exist; use --overwrite. "
            f"First existing output: {existing[0]}"
        )

    (
        echo_paths,
        echo_times_s,
        echo_numbers,
        images,
        series_4d,
    ) = load_echo_series(paths, case, selected)

    n_early = min(args.n_early, series_4d.shape[3])
    echo1 = series_4d[..., 0]
    early_mean = np.mean(series_4d[..., :n_early], axis=3, dtype=np.float64).astype(
        np.float32
    )

    reference = images[0]
    voxel_size_tuple = tuple(
        float(value) for value in reference.header.get_zooms()[:3]
    )
    shell_width_vox = shell_width_voxels(
        width_mm=args.noise_shell_width_mm,
        voxel_size_mm=voxel_size_tuple,
        shape=tuple(int(value) for value in reference.shape),
    )

    fit_mask, qc_mask, mask_qc = create_fit_masks(
        early_mean=early_mean,
        shell_width_vox=shell_width_vox,
        snr_threshold=args.snr_threshold,
        minimum_fraction=args.min_threshold_fraction,
        closing_iterations=args.closing_iterations,
        erosions=args.erosions,
    )
    save_nifti(outputs["series_4d"], series_4d, reference, np.float32)
    save_nifti(outputs["echo1"], echo1, reference, np.float32)
    save_nifti(outputs["early_mean"], early_mean, reference, np.float32)
    save_nifti(outputs["fit_mask"], fit_mask, reference, np.uint8)
    save_nifti(outputs["fit_mask_ero"], qc_mask, reference, np.uint8)

    echo_times_ms = [1000.0 * value for value in echo_times_s]
    write_vector(outputs["echo_times_s"], echo_times_s, precision=12)
    write_vector(outputs["echo_times_ms"], echo_times_ms, precision=12)

    qc_warnings: list[str] = []
    mask_fraction = float(mask_qc["fit_mask_fraction"])
    if mask_fraction < 0.01:
        qc_warnings.append(
            "Automatic fit mask occupies less than 1% of the image volume."
        )
    if mask_fraction > 0.80:
        qc_warnings.append(
            "Automatic fit mask occupies more than 80% of the image volume."
        )
    if int(mask_qc["effective_erosions"]) != args.erosions:
        qc_warnings.append(
            "Requested erosion produced an empty QC mask; the un-eroded fit "
            "mask was retained as the QC mask."
        )

    finished = time.time()
    voxel_size_mm = [
        float(value) for value in reference.header.get_zooms()[:3]
    ]

    payload = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "status": "ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "command": command,
        },
        "timing": {
            "start_epoch": started,
            "end_epoch": finished,
            "runtime_seconds": finished - started,
        },
        "input": {
            "rlx01_selected_inputs": project_relative(
                selected_inputs_path(paths, case),
                paths.root,
            ),
            "acquisition_id": selected.get("acquisition_id"),
            "classification": selected.get("classification"),
            "series_number": selected.get("series_number"),
            "series_description": selected.get("series_description"),
            "protocol_name": selected.get("protocol_name"),
            "echo_images": [
                project_relative(path, paths.root) for path in echo_paths
            ],
            "echo_numbers": echo_numbers,
            "echo_times_s": echo_times_s,
            "echo_times_ms": echo_times_ms,
        },
        "geometry": {
            "matrix_3d": [int(value) for value in reference.shape],
            "matrix_4d": [int(value) for value in series_4d.shape],
            "voxel_size_mm": voxel_size_mm,
            "native_space_preserved": True,
            "resampled": False,
        },
        "preparation": {
            "n_echoes": int(series_4d.shape[3]),
            "n_early_requested": int(args.n_early),
            "n_early_used": int(n_early),
            "noise_shell_width_mm": float(args.noise_shell_width_mm),
            "noise_shell_width_vox": [
                int(value) for value in shell_width_vox
            ],
            "snr_threshold": float(args.snr_threshold),
            "minimum_threshold_fraction": float(
                args.min_threshold_fraction
            ),
            "closing_iterations": int(args.closing_iterations),
            "requested_erosions": int(args.erosions),
        },
        "mask_qc": mask_qc,
        "warnings": qc_warnings,
        "signal_qc": {
            "echo1_min": float(np.min(echo1)),
            "echo1_max": float(np.max(echo1)),
            "echo1_median_in_mask": float(np.median(echo1[fit_mask > 0])),
            "early_mean_min": float(np.min(early_mean)),
            "early_mean_max": float(np.max(early_mean)),
            "early_mean_median_in_mask": float(
                np.median(early_mean[fit_mask > 0])
            ),
        },
        "outputs": {
            key: project_relative(path, paths.root)
            for key, path in outputs.items()
            if key != "metadata"
        },
    }
    write_json(outputs["metadata"], payload)

    return {
        "case": case,
        "status": "ready",
        "acquisition_id": selected.get("acquisition_id", ""),
        "n_echoes": int(series_4d.shape[3]),
        "first_te_ms": round(echo_times_ms[0], 6),
        "last_te_ms": round(echo_times_ms[-1], 6),
        "matrix": "x".join(str(int(value)) for value in reference.shape),
        "voxel_size_mm": "x".join(f"{value:g}" for value in voxel_size_mm),
        "fit_mask_voxels": int(mask_qc["fit_mask_voxels"]),
        "fit_mask_fraction": round(float(mask_qc["fit_mask_fraction"]), 6),
        "runtime_seconds": round(finished - started, 3),
    }


def write_cohort_summary(
    paths: APFPaths,
    rows: list[dict[str, Any]],
) -> Path:
    summary_dir = paths.summary_module_dir(MODULE)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"{TAG}_multi_echo_preparation_summary.csv"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    fieldnames = [
        "case",
        "status",
        "acquisition_id",
        "n_echoes",
        "first_te_ms",
        "last_te_ms",
        "matrix",
        "voxel_size_mm",
        "fit_mask_voxels",
        "fit_mask_fraction",
        "runtime_seconds",
    ]

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(summary_path)
    return summary_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = get_paths()
    cases = resolve_cases(args, paths.ins_registry)

    print_selected_cases(cases)
    print(f"{TAG} | {SCRIPT} | version {VERSION}")
    print(f"APF root: {paths.root}")

    results: list[dict[str, Any]] = []
    failures = 0

    for case in cases:
        print("\n" + "=" * 72)
        print(f"{TAG}: {case}")
        print("=" * 72)

        try:
            result = process_case(
                paths=paths,
                case=case,
                args=args,
                command=list(sys.argv),
            )
            results.append(result)
            print(
                f"status={result['status']} | "
                f"echoes={result['n_echoes']} | "
                f"TE={result['first_te_ms']:g}–{result['last_te_ms']:g} ms | "
                f"matrix={result['matrix']} | "
                f"mask={result['fit_mask_voxels']} voxels"
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR in {case}: {exc}", file=sys.stderr)

    if results:
        summary_path = write_cohort_summary(paths, results)
        print(f"\nSummary: {summary_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
