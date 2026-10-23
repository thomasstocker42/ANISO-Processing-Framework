#!/usr/bin/env python3
"""
APF rlx05a: Native-space brain masking for the high-resolution dual-echo TSE.

Purpose
-------
Create an authoritative brain mask in the native high-resolution dual-echo
TSE space for subsequent rlx05 R2app/T2app estimation.

The first dual-TSE echo is used as the brain-extraction image.

Execution profiles
------------------
laptop (default)
    HD-BET on CPU with test-time augmentation disabled:
        -device cpu --disable_tta

workstation
    HD-BET on CUDA GPU with the default test-time augmentation enabled:
        -device cuda

Both profiles produce the same APF output contract. The workstation profile
is intended for later higher-quality reruns.

Outputs
-------
    rlx05a_<case>_brain_mask.nii.gz
    rlx05a_<case>_brain_mask_ero1.nii.gz
    rlx05a_<case>_brain.nii.gz
    rlx05a_<case>_qc.png
    rlx05a_<case>_summary.csv
    rlx05a_<case>_metadata.json

Notes
-----
HD-BET expects standard/canonical orientation. To preserve the original
dual-TSE acquisition grid, rlx05a:

    1. reorients a temporary copy of echo 1 to closest canonical orientation;
    2. runs HD-BET on that temporary image;
    3. transforms the binary mask back to the exact original voxel ordering;
    4. writes all final outputs with the original native affine/header.

The native image itself is never resampled.

Run
---
Laptop:
    ./apf scripts/relaxometry/05a_prepare_dual_echo_mask.py \
        ANISO_00_000_InS

Workstation:
    ./apf scripts/relaxometry/05a_prepare_dual_echo_mask.py \
        ANISO_00_000_InS \
        --profile workstation
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT = "05a_prepare_dual_echo_mask.py"
TAG = "rlx05a"
VERSION = "0.1.0"

PROFILE_LAPTOP = "laptop"
PROFILE_WORKSTATION = "workstation"

DEFAULT_PROFILE = PROFILE_LAPTOP


try:
    import nibabel as nib
    from nibabel.orientations import (
        apply_orientation,
        io_orientation,
        ornt_transform,
    )
except ImportError as exc:
    raise ImportError("rlx05a requires nibabel.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("rlx05a requires matplotlib.") from exc

try:
    from scipy.ndimage import binary_erosion, generate_binary_structure
except ImportError as exc:
    raise ImportError("rlx05a requires scipy.") from exc

try:
    from apf_paths import APFPaths, get_paths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run through ./apf."
    ) from exc


# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n"
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference.header.copy()
    header.set_data_dtype(dtype)

    image = nib.Nifti1Image(
        data.astype(dtype, copy=False),
        reference.affine,
        header,
    )

    nib.save(
        image,
        str(path),
    )


def resolve_project_path(
    raw: str,
    project_root: Path,
) -> Path:
    path = Path(raw).expanduser()

    if not path.is_absolute():
        path = project_root / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist:\n  {path}"
        )

    return path


# -------------------------------------------------------------------------
# APF paths
# -------------------------------------------------------------------------


def rlx01_selected_inputs_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx01_inventory"
        / f"rlx01_{case}_selected_inputs.json"
    )


def rlx05a_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05a"
    )


def summary_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "rlx05a"
    )


# -------------------------------------------------------------------------
# Input contract
# -------------------------------------------------------------------------


def load_dual_echo1(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    metadata_path = rlx01_selected_inputs_path(
        paths,
        case,
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Required rlx01 selected-input metadata not found:\n"
            f"  {metadata_path}\n"
            "Run rlx01 first."
        )

    metadata = json.loads(
        metadata_path.read_text()
    )

    dual = metadata.get("dual_tse")

    if not isinstance(dual, dict):
        raise ValueError(
            f"rlx01 metadata lacks dual_tse:\n"
            f"  {metadata_path}"
        )

    if not dual.get("ready", False):
        raise RuntimeError(
            f"dual_tse is not READY for {case}: "
            f"{dual.get('readiness_reason', 'unknown')}"
        )

    if not dual.get("selected", False):
        raise RuntimeError(
            f"dual_tse is not selected for {case}."
        )

    if dual.get("n_echoes") != 2:
        raise ValueError(
            "Expected exactly two dual-TSE echoes."
        )

    echoes = dual.get("echoes")

    if not isinstance(echoes, list) or len(echoes) != 2:
        raise ValueError(
            "dual_tse.echoes must contain exactly two entries."
        )

    echoes = sorted(
        echoes,
        key=lambda x: int(x["echo_number"]),
    )

    echo1 = echoes[0]

    if int(echo1["echo_number"]) != 1:
        raise ValueError(
            "Could not resolve dual-TSE echo 1."
        )

    image_path = resolve_project_path(
        echo1["image"],
        paths.root,
    )

    return {
        "metadata_path": metadata_path,
        "dual_tse": dual,
        "echo1": echo1,
        "image_path": image_path,
    }


# -------------------------------------------------------------------------
# HD-BET execution
# -------------------------------------------------------------------------


def hd_bet_command(
    input_path: Path,
    output_path: Path,
    profile: str,
) -> List[str]:
    command = [
        "hd-bet",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--save_bet_mask",
    ]

    if profile == PROFILE_LAPTOP:
        command.extend(
            [
                "-device",
                "cpu",
                "--disable_tta",
            ]
        )

    elif profile == PROFILE_WORKSTATION:
        command.extend(
            [
                "-device",
                "cuda",
            ]
        )

    else:
        raise ValueError(
            f"Unknown execution profile: {profile}"
        )

    return command


def looks_binary(data: np.ndarray) -> bool:
    finite = data[np.isfinite(data)]

    if finite.size == 0:
        return False

    unique = np.unique(finite)

    if unique.size > 4:
        return False

    return bool(
        np.all(
            np.isclose(
                unique,
                np.round(unique),
            )
        )
        and np.all(
            (unique >= 0)
            & (unique <= 1)
        )
    )


def find_hdbet_mask(
    work_dir: Path,
    requested_output: Path,
) -> Path:
    """
    Handle HD-BET output naming defensively.

    Depending on HD-BET release/output mode, the requested path itself or a
    companion *_mask NIfTI may contain the binary segmentation.
    """

    candidates: List[Path] = []

    if requested_output.exists():
        candidates.append(
            requested_output
        )

    stem = requested_output.name

    if stem.endswith(".nii.gz"):
        base = stem[:-7]
    elif stem.endswith(".nii"):
        base = stem[:-4]
    else:
        base = stem

    candidates.extend(
        [
            work_dir / f"{base}_mask.nii.gz",
            work_dir / f"{base}_mask.nii",
        ]
    )

    candidates.extend(
        sorted(
            work_dir.glob("*.nii.gz")
        )
    )

    candidates.extend(
        sorted(
            work_dir.glob("*.nii")
        )
    )

    seen = set()

    for candidate in candidates:
        if candidate in seen:
            continue

        seen.add(candidate)

        if not candidate.exists():
            continue

        try:
            image = nib.load(
                str(candidate)
            )

            data = np.asarray(
                image.dataobj
            )

        except Exception:
            continue

        if looks_binary(data):
            return candidate

    found = "\n".join(
        f"  {p}"
        for p in sorted(work_dir.iterdir())
    )

    raise RuntimeError(
        "HD-BET completed but no binary mask could be identified.\n"
        f"Files produced:\n{found}"
    )


def run_hdbet_native_mask(
    input_img: nib.spatialimages.SpatialImage,
    profile: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run HD-BET on a temporary closest-canonical copy and transform the mask
    back to the exact native voxel ordering.
    """

    if shutil.which("hd-bet") is None:
        raise RuntimeError(
            "hd-bet executable not found in PATH."
        )

    if len(input_img.shape) != 3:
        raise ValueError(
            f"HD-BET input must be 3D, got {input_img.shape}."
        )

    native_orientation = io_orientation(
        input_img.affine
    )

    canonical_img = nib.as_closest_canonical(
        input_img
    )

    canonical_orientation = io_orientation(
        canonical_img.affine
    )

    to_native = ornt_transform(
        canonical_orientation,
        native_orientation,
    )

    with tempfile.TemporaryDirectory(
        prefix="apf_rlx05a_"
    ) as tmp:
        work_dir = Path(tmp)

        canonical_input = (
            work_dir
            / "dual_echo1_canonical.nii.gz"
        )

        requested_output = (
            work_dir
            / "hdbet_output.nii.gz"
        )

        nib.save(
            canonical_img,
            str(canonical_input),
        )

        command = hd_bet_command(
            input_path=canonical_input,
            output_path=requested_output,
            profile=profile,
        )

        print(
            "  HD-BET command: "
            + " ".join(command)
        )

        start = time.perf_counter()

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        runtime_s = (
            time.perf_counter()
            - start
        )

        if result.stdout:
            print(result.stdout.rstrip())

        if result.returncode != 0:
            raise RuntimeError(
                "HD-BET failed with exit code "
                f"{result.returncode}."
            )

        mask_path = find_hdbet_mask(
            work_dir=work_dir,
            requested_output=requested_output,
        )

        mask_img = nib.load(
            str(mask_path)
        )

        mask_canonical = (
            np.asarray(
                mask_img.dataobj
            )
            > 0.5
        )

        if (
            mask_canonical.shape
            != canonical_img.shape
        ):
            raise ValueError(
                "HD-BET mask geometry does not match "
                "the canonical temporary input."
            )

        mask_native = apply_orientation(
            mask_canonical,
            to_native,
        )

        if (
            mask_native.shape
            != input_img.shape
        ):
            raise ValueError(
                "Reoriented HD-BET mask does not match "
                "the original native shape:\n"
                f"  mask: {mask_native.shape}\n"
                f"  input: {input_img.shape}"
            )

        metadata = {
            "command": command,
            "runtime_seconds": runtime_s,
            "native_orientation":
                native_orientation.tolist(),
            "canonical_orientation":
                canonical_orientation.tolist(),
            "temporary_canonical_shape":
                list(canonical_img.shape),
        }

    return (
        mask_native.astype(bool),
        metadata,
    )


# -------------------------------------------------------------------------
# Mask post-processing
# -------------------------------------------------------------------------


def erode_mask_one_voxel(
    mask: np.ndarray,
) -> np.ndarray:
    """
    One voxel 3D erosion using 6-connected nearest neighbours.
    """

    structure = generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    return binary_erosion(
        mask,
        structure=structure,
        iterations=1,
        border_value=0,
    )


# -------------------------------------------------------------------------
# QC
# -------------------------------------------------------------------------


def make_qc_png(
    case: str,
    image: np.ndarray,
    mask: np.ndarray,
    mask_ero1: np.ndarray,
    path: Path,
) -> None:
    z_scores = np.sum(
        mask,
        axis=(0, 1),
    )

    z = (
        int(np.argmax(z_scores))
        if np.any(z_scores)
        else image.shape[2] // 2
    )

    brain_values = image[
        mask
        & np.isfinite(image)
    ]

    if brain_values.size:
        vmin = float(
            np.percentile(
                brain_values,
                1,
            )
        )

        vmax = float(
            np.percentile(
                brain_values,
                99,
            )
        )
    else:
        vmin = 0.0
        vmax = 1.0

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    # Original echo 1.
    axes[0, 0].imshow(
        np.rot90(
            image[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0, 0].set_title(
        "Dual-TSE echo 1"
    )

    # Mask.
    axes[0, 1].imshow(
        np.rot90(
            mask[:, :, z]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 1].set_title(
        "HD-BET brain mask"
    )

    # Eroded mask.
    axes[0, 2].imshow(
        np.rot90(
            mask_ero1[:, :, z]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 2].set_title(
        "Brain mask ero1"
    )

    # Overlay original mask.
    axes[1, 0].imshow(
        np.rot90(
            image[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1, 0].contour(
        np.rot90(
            mask[:, :, z]
        ),
        levels=[0.5],
        linewidths=0.8,
    )

    axes[1, 0].set_title(
        "Mask boundary"
    )

    # Overlay eroded mask.
    axes[1, 1].imshow(
        np.rot90(
            image[:, :, z]
        ),
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1, 1].contour(
        np.rot90(
            mask_ero1[:, :, z]
        ),
        levels=[0.5],
        linewidths=0.8,
    )

    axes[1, 1].set_title(
        "Ero1 boundary"
    )

    # Difference between mask and erosion.
    removed = (
        mask
        & ~mask_ero1
    )

    axes[1, 2].imshow(
        np.rot90(
            removed[:, :, z]
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1, 2].set_title(
        "Voxels removed by ero1"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{TAG} dual-TSE native brain-mask QC\n"
        f"{case}, z={z}"
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# -------------------------------------------------------------------------
# Case processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    profile: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    start_iso = now_iso()

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    selection = load_dual_echo1(
        paths,
        case,
    )

    input_path = selection[
        "image_path"
    ]

    echo1 = selection[
        "echo1"
    ]

    dual = selection[
        "dual_tse"
    ]

    print(
        f"  Profile: {profile}"
    )

    print(
        f"  Echo 1: {input_path}"
    )

    print(
        f"  TE1: "
        f"{float(echo1['echo_time_s']) * 1000:.3f} ms"
    )

    image = nib.load(
        str(input_path)
    )

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected 3D input, got {image.shape}."
        )

    voxel_size = tuple(
        float(x)
        for x in image.header.get_zooms()[:3]
    )

    print(
        f"  Native shape: {image.shape}"
    )

    print(
        "  Native voxel size: "
        f"{voxel_size[0]:.3f} x "
        f"{voxel_size[1]:.3f} x "
        f"{voxel_size[2]:.3f} mm"
    )

    out_dir = rlx05a_dir(
        paths,
        case,
    )

    metadata_path = (
        out_dir
        / f"rlx05a_{case}_metadata.json"
    )

    if (
        metadata_path.exists()
        and not overwrite
        and not dry_run
    ):
        raise FileExistsError(
            f"rlx05a outputs already exist:\n"
            f"  {metadata_path}\n"
            "Use --overwrite to replace them."
        )

    if dry_run:
        return {
            "case": case,
            "success": True,
            "dry_run": True,
            "profile": profile,
            "input": str(input_path),
            "shape": list(image.shape),
            "voxel_size_mm": list(voxel_size),
        }

    input_data = np.asarray(
        image.dataobj,
        dtype=np.float32,
    )

    mask, hdbet_metadata = run_hdbet_native_mask(
        input_img=image,
        profile=profile,
    )

    mask_ero1 = erode_mask_one_voxel(
        mask
    )

    brain = np.zeros_like(
        input_data,
        dtype=np.float32,
    )

    brain[mask] = input_data[
        mask
    ]

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = f"rlx05a_{case}"

    mask_path = (
        out_dir
        / f"{stem}_brain_mask.nii.gz"
    )

    mask_ero1_path = (
        out_dir
        / f"{stem}_brain_mask_ero1.nii.gz"
    )

    brain_path = (
        out_dir
        / f"{stem}_brain.nii.gz"
    )

    qc_path = (
        out_dir
        / f"{stem}_qc.png"
    )

    summary_path = (
        out_dir
        / f"{stem}_summary.csv"
    )

    save_nifti(
        mask.astype(np.uint8),
        image,
        mask_path,
        np.uint8,
    )

    save_nifti(
        mask_ero1.astype(np.uint8),
        image,
        mask_ero1_path,
        np.uint8,
    )

    save_nifti(
        brain,
        image,
        brain_path,
        np.float32,
    )

    make_qc_png(
        case=case,
        image=input_data,
        mask=mask,
        mask_ero1=mask_ero1,
        path=qc_path,
    )

    n_mask = int(
        np.count_nonzero(mask)
    )

    n_ero1 = int(
        np.count_nonzero(mask_ero1)
    )

    voxel_volume_mm3 = float(
        np.prod(voxel_size)
    )

    mask_volume_ml = (
        n_mask
        * voxel_volume_mm3
        / 1000.0
    )

    ero1_volume_ml = (
        n_ero1
        * voxel_volume_mm3
        / 1000.0
    )

    summary = {
        "case": case,
        "profile": profile,
        "shape_x": image.shape[0],
        "shape_y": image.shape[1],
        "shape_z": image.shape[2],
        "voxel_x_mm": voxel_size[0],
        "voxel_y_mm": voxel_size[1],
        "voxel_z_mm": voxel_size[2],
        "brain_mask_voxels": n_mask,
        "brain_mask_volume_ml": mask_volume_ml,
        "brain_mask_ero1_voxels": n_ero1,
        "brain_mask_ero1_volume_ml": ero1_volume_ml,
        "ero1_retained_fraction": (
            float(n_ero1 / n_mask)
            if n_mask > 0
            else None
        ),
        "hdbet_runtime_seconds":
            hdbet_metadata["runtime_seconds"],
    }

    write_csv(
        summary_path,
        [summary],
    )

    end_iso = now_iso()

    metadata = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "created": end_iso,

        "execution": {
            "start": start_iso,
            "end": end_iso,
            "profile": profile,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "command": " ".join(sys.argv),
            "hdbet":
                hdbet_metadata,
        },

        "upstream": {
            "rlx01_selected_inputs":
                str(selection["metadata_path"]),

            "acquisition_id":
                dual.get("acquisition_id"),

            "echo1_image":
                str(input_path),

            "echo1_TE_ms":
                float(
                    echo1["echo_time_s"]
                ) * 1000.0,

            "series_number":
                dual.get("series_number"),

            "series_description":
                dual.get("series_description"),

            "protocol_name":
                dual.get("protocol_name"),

            "native_shape":
                list(image.shape),

            "native_voxel_size_mm":
                list(voxel_size),
        },

        "method": {
            "brain_extraction":
                "HD-BET",

            "input":
                "dual-TSE echo 1",

            "native_output_space":
                True,

            "temporary_orientation_handling":
                (
                    "Input temporarily reoriented to closest canonical "
                    "orientation for HD-BET. Binary mask transformed back "
                    "to the exact original voxel ordering. Final outputs "
                    "use the original native affine/header."
                ),

            "erosion":
                (
                    "One-voxel 3D binary erosion with "
                    "6-connected neighbourhood."
                ),

            "default_downstream_mask":
                "brain_mask",

            "ero1_role":
                (
                    "Additional conservative mask retained for QC and "
                    "sensitivity analysis; not selected as primary yet."
                ),
        },

        "summary":
            summary,

        "outputs": {
            "brain_mask":
                str(mask_path),

            "brain_mask_ero1":
                str(mask_ero1_path),

            "brain":
                str(brain_path),

            "qc_png":
                str(qc_path),

            "summary_csv":
                str(summary_path),
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    print(
        f"  brain mask voxels: {n_mask}"
    )

    print(
        f"  brain mask volume: "
        f"{mask_volume_ml:.1f} ml"
    )

    print(
        f"  ero1 retained: "
        f"{100 * n_ero1 / n_mask:.2f}%"
        if n_mask
        else "  ero1 retained: n/a"
    )

    print(
        f"  HD-BET runtime: "
        f"{hdbet_metadata['runtime_seconds']:.1f} s"
    )

    print(
        f"  DONE: {case}"
    )

    return metadata


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF rlx05a: native-space HD-BET masking "
            "for high-resolution dual-echo TSE."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--profile",
        choices=[
            PROFILE_LAPTOP,
            PROFILE_WORKSTATION,
        ],
        default=DEFAULT_PROFILE,
        help=(
            "Execution profile. "
            "'laptop' = CPU + no TTA; "
            "'workstation' = CUDA + TTA. "
            f"Default: {DEFAULT_PROFILE}."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rlx05a outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs without running HD-BET.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


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

    cohort_rows: List[
        Dict[str, Any]
    ] = []

    failures = 0

    for case in cases:
        try:
            metadata = process_case(
                case=case,
                paths=paths,
                profile=args.profile,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                cohort_rows.append(
                    {
                        "case": case,
                        "success": True,
                        "dry_run": True,
                        "profile":
                            metadata["profile"],
                        "input":
                            metadata["input"],
                    }
                )

            else:
                cohort_rows.append(
                    {
                        "success": True,
                        **metadata["summary"],
                    }
                )

        except Exception as exc:
            failures += 1

            print(
                f"ERROR in {case}: {exc}",
                file=sys.stderr,
            )

            cohort_rows.append(
                {
                    "case": case,
                    "success": False,
                    "error": str(exc),
                }
            )

            if len(cases) == 1:
                raise

    if not args.dry_run:
        out = summary_dir(
            paths
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            out / "rlx05a_cohort_summary.csv",
            cohort_rows,
        )

        write_json(
            out / "rlx05a_cohort_metadata.json",
            {
                "script": SCRIPT,
                "tag": TAG,
                "version": VERSION,
                "created": now_iso(),
                "command": " ".join(sys.argv),
                "n_cases_requested":
                    len(cases),
                "n_failures":
                    failures,
                "rows":
                    cohort_rows,
            },
        )

    if failures:
        raise SystemExit(
            f"rlx05a completed with "
            f"{failures} failed case(s)."
        )


if __name__ == "__main__":
    main()