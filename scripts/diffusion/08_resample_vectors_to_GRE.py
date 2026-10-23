#!/usr/bin/env python3
"""
dwi08_resample_vectors_to_GRE.py

APF diffusion module 08:
Spatially resample already reoriented DTI eigenvectors from native DWI grid to
GRE grid, then recompute orientation maps in GRE space.

Conceptual contract
-------------------
dwi05: tensor fitting, native DWI grid, native DWI vector orientation
dwi06: scalar maps/mask registered to GRE grid
dwi07: V1/V2/V3 reoriented into GRE/world coordinate system, still native DWI grid
dwi08: V1/V2/V3 spatially resampled to GRE grid, vector orientation remains GRE/world

This script intentionally does not perform vesselness/orientation analysis. That belongs
in dwi09 or later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SCRIPT_NAME = "08_resample_vectors_to_GRE.py"
SCRIPT_TAG = "dwi08"
EPS = 1e-8


# ----------------------------- generic helpers -----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"    $ {printable}")
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), check=True)


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def fsl_available() -> bool:
    return shutil.which("flirt") is not None and shutil.which("fslmaths") is not None


# ----------------------------- path discovery ------------------------------

def repo_root_from_cwd() -> Path:
    # APF is normally launched from repository root via ./apf.
    return Path.cwd().resolve()


def derivatives_case_dir(root: Path, case: str) -> Path:
    return root / "derivatives" / case


def shell_sort_key(shell: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in shell if ch.isdigit())
    return (int(digits) if digits else 999999, shell)


def find_shell_dirs_dwi07(case_deriv: Path) -> Dict[str, Path]:
    """Find per-shell dwi07 output directories.

    Expected/accepted examples:
      derivatives/<case>/DWI/dwi07_orientation_native/b1000/
      derivatives/<case>/DWI/dwi07_*/b2000/
      derivatives/<case>/DWI/**/dwi07*/**/b3000/
    """
    dwi_root = case_deriv / "DWI"
    candidates: Dict[str, Path] = {}
    for p in dwi_root.rglob("*"):
        if not p.is_dir():
            continue
        shell = p.name
        if not shell.lower().startswith("b"):
            continue
        parent_text = str(p.parent).lower()
        if "dwi07" not in parent_text:
            continue
        if find_vector_file(p, "V1", required=False) is not None:
            candidates[shell] = p
    return dict(sorted(candidates.items(), key=lambda kv: shell_sort_key(kv[0])))


def find_vector_file(shell_dir: Path, vec: str, *, required: bool = True) -> Optional[Path]:
    patterns = [
        f"*{vec}*GREorient*.nii.gz",
        f"*{vec}*GRE_orient*.nii.gz",
        f"*{vec}*oriented*.nii.gz",
        f"*{vec}*native*.nii.gz",
        f"*{vec}*.nii.gz",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(shell_dir.glob(pat))
    # Prefer files not already GREgrid.
    hits = [h for h in hits if "GREgrid" not in h.name and "rgb" not in h.name.lower()]
    hits = sorted(set(hits), key=lambda p: (len(p.name), p.name))
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(f"Could not find {vec} file in {shell_dir}")
    return None


def find_dwi07_metadata(shell_dir: Path) -> Optional[Path]:
    hits = sorted(shell_dir.glob("*metadata*.json")) + sorted(shell_dir.glob("*.json"))
    return hits[0] if hits else None


def find_transform(case_deriv: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return require_file(Path(explicit), "DWI/b0 to GRE transform matrix")
    dwi_root = case_deriv / "DWI"
    patterns = [
        "**/*b0*to*GRE*.mat",
        "**/*b0_to_GRE*.mat",
        "**/*dwi*to*GRE*.mat",
        "**/*DWI*to*GRE*.mat",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(dwi_root.glob(pat))
    hits = sorted(set(hits), key=lambda p: ("dwi06" not in str(p).lower(), len(str(p)), str(p)))
    if not hits:
        raise FileNotFoundError(f"Could not locate b0_to_GRE.mat under {dwi_root}")
    return hits[0]


def find_gre_reference(case_deriv: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return require_file(Path(explicit), "GRE reference image")
    patterns = [
        "GRE/**/*e7*.nii.gz",
        "GRE/**/*E7*.nii.gz",
        "**/GRE/**/*e7*.nii.gz",
        "**/*GRE*e7*.nii.gz",
        "**/*gre*e7*.nii.gz",
        "**/*GRE*ref*.nii.gz",
        "**/*reference*.nii.gz",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(case_deriv.glob(pat))
    # Exclude masks/maps if possible.
    hits = [h for h in set(hits) if not any(x in h.name.lower() for x in ["mask", "vessel", "r2", "field", "rgb"])]
    hits = sorted(hits, key=lambda p: ("e7" not in p.name.lower(), len(str(p)), str(p)))
    if not hits:
        raise FileNotFoundError(f"Could not locate GRE reference under {case_deriv}; pass --gre-ref")
    return hits[0]


def find_gre_brain_mask(case_deriv: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return require_file(Path(explicit), "GRE brain mask")
    patterns = [
        "GRE/**/*mask_ero1*.nii.gz",
        "GRE/**/*brain*mask*.nii.gz",
        "**/*GRE*mask_ero1*.nii.gz",
        "**/*brain*mask*.nii.gz",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(case_deriv.glob(pat))
    hits = sorted(set(hits), key=lambda p: ("ero1" not in p.name.lower(), len(str(p)), str(p)))
    return hits[0] if hits else None


def find_gre_dwi_mask(case_deriv: Path, shell: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return require_file(Path(explicit), "GRE-space DWI mask")
    dwi_root = case_deriv / "DWI"
    patterns = [
        f"**/*{shell}*mask*GRE*.nii.gz",
        f"**/*{shell}*GRE*mask*.nii.gz",
        "**/*dwi06*mask*GRE*.nii.gz",
        "**/*DWI*mask*GRE*.nii.gz",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(dwi_root.glob(pat))
    hits = [h for h in set(hits) if "rgb" not in h.name.lower()]
    hits = sorted(hits, key=lambda p: (shell not in p.name, len(str(p)), str(p)))
    return hits[0] if hits else None


def find_native_dwi_mask(shell_dir: Path) -> Optional[Path]:
    patterns = ["*mask*.nii.gz", "../*mask*.nii.gz", "../../*mask*.nii.gz"]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(shell_dir.glob(pat))
    hits = [h.resolve() for h in set(hits) if "gre" not in h.name.lower() and "rgb" not in h.name.lower()]
    hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
    return hits[0] if hits else None


# --------------------------- image/vector helpers --------------------------

def split_vector_components(vec_path: Path, tmpdir: Path) -> Tuple[Path, Path, Path]:
    img = nib.load(str(vec_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError(f"Expected 4D vector image with 3 components: {vec_path}, got shape {data.shape}")
    outs = []
    base = strip_nii_suffix(vec_path.name)
    for i, comp in enumerate(["x", "y", "z"]):
        out = tmpdir / f"{base}_{comp}.nii.gz"
        nib.save(nib.Nifti1Image(data[..., i].astype(np.float32), img.affine, img.header), str(out))
        outs.append(out)
    return outs[0], outs[1], outs[2]


def merge_components(x: Path, y: Path, z: Path, reference_img: nib.Nifti1Image, out_path: Path) -> np.ndarray:
    dx = nib.load(str(x)).get_fdata(dtype=np.float32)
    dy = nib.load(str(y)).get_fdata(dtype=np.float32)
    dz = nib.load(str(z)).get_fdata(dtype=np.float32)
    data = np.stack([dx, dy, dz], axis=-1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = reference_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, reference_img.affine, hdr), str(out_path))
    return data


def save_scalar(data: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path, nan_ok: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.float32)
    if not nan_ok:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    hdr = reference_img.header.copy()
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(arr, reference_img.affine, hdr), str(out_path))


def save_vector(data: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = reference_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), reference_img.affine, hdr), str(out_path))


def load_mask(path: Optional[Path], shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    data = nib.load(str(path)).get_fdata()
    if data.shape[:3] != shape:
        raise ValueError(f"Mask shape mismatch for {path}: {data.shape} vs reference {shape}")
    return data > 0.5


def normalize_vectors(v: np.ndarray, valid_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    norm = np.linalg.norm(v, axis=-1)
    valid = np.isfinite(norm) & (norm > EPS)
    if valid_mask is not None:
        valid &= valid_mask
    out = np.zeros_like(v, dtype=np.float32)
    out[valid] = v[valid] / norm[valid, None]
    qc = {
        "input_nonzero_voxels": int(np.count_nonzero(norm > EPS)),
        "valid_voxels_after_masking": int(np.count_nonzero(valid)),
        "norm_min_before": float(np.nanmin(norm)) if np.isfinite(norm).any() else None,
        "norm_median_before": float(np.nanmedian(norm[np.isfinite(norm)])) if np.isfinite(norm).any() else None,
        "norm_max_before": float(np.nanmax(norm)) if np.isfinite(norm).any() else None,
    }
    return out, valid, qc


def extract_b0_direction(meta: Optional[Dict[str, Any]], allow_default: bool) -> Tuple[np.ndarray, str, List[str]]:
    warnings: List[str] = []
    candidates: List[Any] = []
    if meta:
        # Flexible key search for likely dwi07 metadata variants.
        for key in [
            "b0_direction_GRE_world",
            "B0_direction_GRE_world",
            "b0_unit_vector_GRE_world",
            "B0_unit_vector_GRE_world",
            "b0_direction",
            "B0_direction",
        ]:
            if key in meta:
                candidates.append(meta[key])
        for parent in ["orientation", "B0", "b0", "concept", "parameters"]:
            obj = meta.get(parent) if isinstance(meta.get(parent), dict) else None
            if obj:
                for key in [
                    "b0_direction_GRE_world",
                    "B0_direction_GRE_world",
                    "b0_unit_vector_GRE_world",
                    "B0_unit_vector_GRE_world",
                    "direction_GRE_world",
                    "unit_vector_GRE_world",
                    "direction",
                ]:
                    if key in obj:
                        candidates.append(obj[key])
    for c in candidates:
        try:
            arr = np.asarray(c, dtype=float).reshape(3)
            n = float(np.linalg.norm(arr))
            if n > EPS:
                return (arr / n).astype(float), "dwi07_metadata", warnings
        except Exception:
            continue
    if allow_default:
        warnings.append("B0 direction not found in dwi07 metadata; defaulted to [0, 0, 1]. Verify this before thesis analysis.")
        return np.array([0.0, 0.0, 1.0], dtype=float), "default_[0,0,1]", warnings
    raise ValueError(
        "Could not find B0 direction in dwi07 metadata. "
        "Add b0_direction_GRE_world to dwi07 metadata or rerun with --allow-default-b0."
    )


def compute_orientation_maps(v1: np.ndarray, valid: np.ndarray, b0: np.ndarray) -> Dict[str, np.ndarray]:
    dot = np.sum(v1 * b0.reshape(1, 1, 1, 3), axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    cos_abs = np.abs(dot)
    angle = np.full(v1.shape[:3], np.nan, dtype=np.float32)
    cos_theta = np.full(v1.shape[:3], np.nan, dtype=np.float32)
    cos2 = np.full(v1.shape[:3], np.nan, dtype=np.float32)
    incl = np.full(v1.shape[:3], np.nan, dtype=np.float32)
    az = np.full(v1.shape[:3], np.nan, dtype=np.float32)

    angle[valid] = np.degrees(np.arccos(cos_abs[valid])).astype(np.float32)  # acute angle to B0
    cos_theta[valid] = cos_abs[valid].astype(np.float32)
    cos2[valid] = (cos_abs[valid] ** 2).astype(np.float32)
    # Signed spherical maps in GRE/world component system. Sign ambiguity remains.
    z = np.clip(v1[..., 2], -1.0, 1.0)
    incl[valid] = np.degrees(np.arccos(z[valid])).astype(np.float32)
    az[valid] = np.degrees(np.arctan2(v1[..., 1][valid], v1[..., 0][valid])).astype(np.float32)
    rgb = np.abs(v1).astype(np.float32)
    rgb[~valid] = 0.0
    return {
        "angle_to_B0_deg": angle,
        "cos_theta": cos_theta,
        "cos2_theta": cos2,
        "inclination_deg": incl,
        "azimuth_deg": az,
        "V1_rgb_abs": rgb,
    }


def header_validation(out_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> Dict[str, Any]:
    out_hdr = out_img.header
    ref_hdr = ref_img.header
    out_shape3 = tuple(out_img.shape[:3])
    ref_shape3 = tuple(ref_img.shape[:3])
    out_zooms = tuple(float(x) for x in out_hdr.get_zooms()[:3])
    ref_zooms = tuple(float(x) for x in ref_hdr.get_zooms()[:3])
    qform_match = np.allclose(out_img.get_qform(), ref_img.get_qform(), atol=1e-5, equal_nan=True)
    sform_match = np.allclose(out_img.get_sform(), ref_img.get_sform(), atol=1e-5, equal_nan=True)
    affine_match = np.allclose(out_img.affine, ref_img.affine, atol=1e-5, equal_nan=True)
    dims_match = out_shape3 == ref_shape3
    zooms_match = np.allclose(out_zooms, ref_zooms, atol=1e-5)
    return {
        "matches_reference": bool(dims_match and zooms_match and qform_match and sform_match and affine_match),
        "dimensions_match": bool(dims_match),
        "voxel_size_match": bool(zooms_match),
        "qform_match": bool(qform_match),
        "sform_match": bool(sform_match),
        "affine_match": bool(affine_match),
        "output_dimensions": list(out_shape3),
        "reference_dimensions": list(ref_shape3),
        "output_voxel_size": list(out_zooms),
        "reference_voxel_size": list(ref_zooms),
    }


def make_qc_png(
    out_path: Path,
    gre_ref_path: Path,
    v1_rgb: np.ndarray,
    valid: np.ndarray,
    angle: np.ndarray,
    cos2: np.ndarray,
    coverage_percent: Optional[float],
) -> None:
    if plt is None:
        print("    WARNING: matplotlib unavailable; skipping QC PNG")
        return
    gre = nib.load(str(gre_ref_path)).get_fdata(dtype=np.float32)
    # Use central valid slice if possible.
    z_candidates = np.where(valid.sum(axis=(0, 1)) > 0)[0]
    z = int(z_candidates[len(z_candidates) // 2]) if len(z_candidates) else gre.shape[2] // 2
    bg = gre[:, :, z]
    bg_lo, bg_hi = np.nanpercentile(bg, [1, 99]) if np.isfinite(bg).any() else (0, 1)
    if bg_hi <= bg_lo:
        bg_hi = bg_lo + 1

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.rot90(bg), cmap="gray", vmin=bg_lo, vmax=bg_hi)
    ax1.set_title(f"GRE ref + valid DWI mask, z={z}")
    ax1.contour(np.rot90(valid[:, :, z].astype(float)), levels=[0.5], linewidths=0.8)
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    rgb_slice = np.rot90(v1_rgb[:, :, z, :])
    ax2.imshow(np.clip(rgb_slice, 0, 1))
    ax2.set_title("abs(V1) RGB, GRE grid")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    vals = angle[np.isfinite(angle)]
    ax3.hist(vals, bins=60)
    ax3.set_title("angle to B0 [deg]")
    ax3.set_xlabel("deg")
    ax3.set_ylabel("voxels")

    ax4 = fig.add_subplot(2, 2, 4)
    vals2 = cos2[np.isfinite(cos2)]
    ax4.hist(vals2, bins=60)
    ax4.set_title("cos²(theta)")
    ax4.set_xlabel("cos²")
    ax4.set_ylabel("voxels")
    title = "dwi08 orientation QC"
    if coverage_percent is not None:
        title += f" | DWI/GRE brain coverage {coverage_percent:.1f}%"
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------ core process -------------------------------

def process_shell(
    *,
    root: Path,
    case: str,
    shell: str,
    shell_dir: Path,
    out_base: Path,
    transform_mat: Path,
    gre_ref: Path,
    gre_brain_mask: Optional[Path],
    explicit_gre_dwi_mask: Optional[str],
    allow_default_b0: bool,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    start = time.time()
    start_iso = now_iso()
    print(f"  Shell: {shell}")

    if not fsl_available() and not dry_run:
        raise RuntimeError("FSL flirt/fslmaths not found on PATH. Activate APF/FSL environment first.")

    ref_img = nib.load(str(gre_ref))
    out_dir = out_base / shell
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{SCRIPT_TAG}_{case}_{shell}"

    outputs = {
        "V1_GREgrid": out_dir / f"{prefix}_V1_GREgrid.nii.gz",
        "V2_GREgrid": out_dir / f"{prefix}_V2_GREgrid.nii.gz",
        "V3_GREgrid": out_dir / f"{prefix}_V3_GREgrid.nii.gz",
        "angle_to_B0_deg_GREgrid": out_dir / f"{prefix}_angle_to_B0_deg_GREgrid.nii.gz",
        "cos_theta_GREgrid": out_dir / f"{prefix}_cos_theta_GREgrid.nii.gz",
        "cos2_theta_GREgrid": out_dir / f"{prefix}_cos2_theta_GREgrid.nii.gz",
        "inclination_deg_GREgrid": out_dir / f"{prefix}_inclination_deg_GREgrid.nii.gz",
        "azimuth_deg_GREgrid": out_dir / f"{prefix}_azimuth_deg_GREgrid.nii.gz",
        "V1_rgb_abs_GREgrid": out_dir / f"{prefix}_V1_rgb_abs_GREgrid.nii.gz",
        "orientation_qc_png": out_dir / f"{prefix}_orientation_qc.png",
        "metadata_json": out_dir / f"{prefix}_metadata.json",
    }
    if any(p.exists() for p in outputs.values()) and not overwrite:
        raise FileExistsError(f"Outputs already exist in {out_dir}; rerun with --overwrite")

    v_paths = {vec: find_vector_file(shell_dir, vec) for vec in ["V1", "V2", "V3"]}
    meta_path = find_dwi07_metadata(shell_dir)
    dwi07_meta = load_json(meta_path) if meta_path else None
    b0, b0_source, warnings = extract_b0_direction(dwi07_meta, allow_default_b0)

    gre_dwi_mask = find_gre_dwi_mask(root / "derivatives" / case, shell, explicit_gre_dwi_mask)
    native_dwi_mask = None
    mask_source = "none"

    with tempfile.TemporaryDirectory(prefix=f"{SCRIPT_TAG}_{case}_{shell}_") as td:
        tmpdir = Path(td)
        # If dwi06 mask not found, transform a native mask if discoverable.
        if gre_dwi_mask is None:
            native_dwi_mask = find_native_dwi_mask(shell_dir)
            if native_dwi_mask is not None:
                gre_dwi_mask = out_dir / f"{prefix}_DWI_mask_GREgrid.nii.gz"
                run([
                    "flirt", "-in", native_dwi_mask, "-ref", gre_ref,
                    "-applyxfm", "-init", transform_mat,
                    "-interp", "nearestneighbour", "-out", gre_dwi_mask,
                ], dry_run=dry_run)
                mask_source = "created_from_native_dwi_mask_in_dwi08"
            else:
                warnings.append("No GRE-space or native DWI mask found; valid mask derived only from vector norm.")
        else:
            mask_source = "dwi06_GRE_space_DWI_mask"

        resampled_vectors: Dict[str, np.ndarray] = {}
        vector_norm_qc: Dict[str, Any] = {}
        for vec, vec_path in v_paths.items():
            comp_in = split_vector_components(vec_path, tmpdir)
            comp_out = []
            for comp_path, comp_name in zip(comp_in, ["x", "y", "z"]):
                out_comp = tmpdir / f"{prefix}_{vec}_{comp_name}_GREgrid.nii.gz"
                run([
                    "flirt", "-in", comp_path, "-ref", gre_ref,
                    "-applyxfm", "-init", transform_mat,
                    "-interp", "trilinear", "-out", out_comp,
                ], dry_run=dry_run)
                comp_out.append(out_comp)
            if dry_run:
                continue
            raw_vec = merge_components(comp_out[0], comp_out[1], comp_out[2], ref_img, tmpdir / f"{prefix}_{vec}_raw_GREgrid.nii.gz")
            mask_arr = load_mask(gre_dwi_mask, ref_img.shape[:3]) if gre_dwi_mask is not None and gre_dwi_mask.exists() else None
            normed, valid_vec, qc = normalize_vectors(raw_vec, mask_arr)
            save_vector(normed, ref_img, outputs[f"{vec}_GREgrid"])
            resampled_vectors[vec] = normed
            vector_norm_qc[vec] = qc

        if dry_run:
            return {"case": case, "shell": shell, "dry_run": True}

    v1 = resampled_vectors["V1"]
    mask_arr = load_mask(gre_dwi_mask, ref_img.shape[:3]) if gre_dwi_mask is not None and gre_dwi_mask.exists() else None
    v1_norm = np.linalg.norm(v1, axis=-1)
    valid = np.isfinite(v1_norm) & (v1_norm > EPS)
    if mask_arr is not None:
        valid &= mask_arr

    maps = compute_orientation_maps(v1, valid, b0)
    save_scalar(maps["angle_to_B0_deg"], ref_img, outputs["angle_to_B0_deg_GREgrid"], nan_ok=True)
    save_scalar(maps["cos_theta"], ref_img, outputs["cos_theta_GREgrid"], nan_ok=True)
    save_scalar(maps["cos2_theta"], ref_img, outputs["cos2_theta_GREgrid"], nan_ok=True)
    save_scalar(maps["inclination_deg"], ref_img, outputs["inclination_deg_GREgrid"], nan_ok=True)
    save_scalar(maps["azimuth_deg"], ref_img, outputs["azimuth_deg_GREgrid"], nan_ok=True)
    save_vector(maps["V1_rgb_abs"], ref_img, outputs["V1_rgb_abs_GREgrid"])

    # Header validation against GRE reference.
    hv = header_validation(nib.load(str(outputs["V1_GREgrid"])), ref_img)

    # Coverage validation: DWI mask coverage inside GRE brain mask if available.
    gre_brain = load_mask(gre_brain_mask, ref_img.shape[:3]) if gre_brain_mask is not None else None
    valid_voxels = int(np.count_nonzero(valid))
    brain_voxels = int(np.count_nonzero(gre_brain)) if gre_brain is not None else None
    overlap_voxels = int(np.count_nonzero(valid & gre_brain)) if gre_brain is not None else None
    overlap_percent = (100.0 * overlap_voxels / brain_voxels) if brain_voxels else None
    coverage = {
        "source": mask_source,
        "mask_path": str(gre_dwi_mask) if gre_dwi_mask is not None else None,
        "voxels": valid_voxels,
        "gre_brain_mask_path": str(gre_brain_mask) if gre_brain_mask is not None else None,
        "gre_brain_voxels": brain_voxels,
        "brain_overlap_voxels": overlap_voxels,
        "brain_overlap_percent": round(overlap_percent, 3) if overlap_percent is not None else None,
    }

    make_qc_png(
        outputs["orientation_qc_png"], gre_ref, maps["V1_rgb_abs"], valid,
        maps["angle_to_B0_deg"], maps["cos2_theta"], overlap_percent,
    )

    end_iso = now_iso()
    runtime = time.time() - start
    cmd = " ".join(sys.argv)
    metadata = {
        "script": SCRIPT_NAME,
        "tag": SCRIPT_TAG,
        "case": case,
        "shell": shell,
        "created": end_iso,
        "concept": {
            "vector_orientation": "GRE/world coordinate system from dwi07",
            "spatial_grid": "GRE reference grid from dwi06",
            "script_role": "spatial resampling of already reoriented tensor eigenvectors",
            "sign_ambiguity_note": "DTI eigenvector signs are ambiguous; cos²(theta) is sign-invariant, abs RGB is used for display, signed azimuth/inclination should be interpreted cautiously.",
        },
        "inputs": {
            "dwi07_shell_dir": str(shell_dir),
            "V1_native_grid_GRE_oriented": str(v_paths["V1"]),
            "V2_native_grid_GRE_oriented": str(v_paths["V2"]),
            "V3_native_grid_GRE_oriented": str(v_paths["V3"]),
            "dwi07_metadata": str(meta_path) if meta_path else None,
            "gre_reference_image": str(gre_ref),
            "transform_matrix": str(transform_mat),
            "gre_brain_mask": str(gre_brain_mask) if gre_brain_mask else None,
        },
        "processing": {
            "component_resampling": {
                "tool": "FSL flirt -applyxfm",
                "interpolation": "trilinear",
                "transform_direction": "native DWI/b0 grid to GRE reference grid",
            },
            "mask_resampling": {
                "tool": "FSL flirt -applyxfm",
                "interpolation": "nearestneighbour",
                "used_when": "only if GRE-space DWI mask from dwi06 is unavailable and a native DWI mask is found",
            },
            "renormalized": True,
            "invalid_voxels": {
                "angle_and_scalar_maps": "NaN outside valid DWI vector mask or near-zero norm",
                "RGB": "0 outside valid DWI vector mask or near-zero norm",
            },
            "B0_direction": {
                "unit_vector_GRE_world": [float(x) for x in b0],
                "source": b0_source,
            },
        },
        "header_validation": hv,
        "dwi_mask": coverage,
        "transform": {
            "reference_image": str(gre_ref),
            "matrix": str(transform_mat),
            "interpolation": "trilinear for vector components; nearestneighbour for masks",
            "renormalized": True,
        },
        "qc": {
            "vector_norms": vector_norm_qc,
            "orientation": {
                "valid_voxels": valid_voxels,
                "angle_deg_median": float(np.nanmedian(maps["angle_to_B0_deg"])),
                "angle_deg_p05": float(np.nanpercentile(maps["angle_to_B0_deg"], 5)),
                "angle_deg_p95": float(np.nanpercentile(maps["angle_to_B0_deg"], 95)),
                "cos2_median": float(np.nanmedian(maps["cos2_theta"])),
                "cos2_p05": float(np.nanpercentile(maps["cos2_theta"], 5)),
                "cos2_p95": float(np.nanpercentile(maps["cos2_theta"], 95)),
            },
            "orientation_qc_png": str(outputs["orientation_qc_png"]),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "warnings": warnings,
        "execution": {
            "start": start_iso,
            "end": end_iso,
            "runtime_seconds": round(runtime, 3),
            "host": platform.node(),
            "thread_count": os.cpu_count(),
            "processing_tool": "FSL flirt + nibabel/numpy",
            "tool_version": get_fsl_version(),
            "executed_command": cmd,
        },
    }
    write_json(outputs["metadata_json"], metadata)
    print(f"    Wrote: {out_dir}")
    if not hv["matches_reference"]:
        print("    WARNING: header validation does not fully match GRE reference")
    if overlap_percent is not None:
        print(f"    DWI/GRE brain overlap: {overlap_percent:.1f}%")
    return metadata


def get_fsl_version() -> Optional[str]:
    try:
        out = subprocess.check_output(["flirt", "-version"], stderr=subprocess.STDOUT, text=True).strip()
        return out
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="dwi08: resample dwi07 GRE-oriented eigenvectors to GRE grid and recompute orientation maps."
    )
    p.add_argument("case", help="Case ID, e.g. ANISO_00_000_InS")
    p.add_argument("--shell", action="append", help="Shell to process, e.g. b1000. Can be repeated.")
    p.add_argument("--all-shells", action="store_true", help="Process all shells found in dwi07 outputs.")
    p.add_argument("--gre-ref", help="Explicit GRE reference image. Overrides auto-discovery.")
    p.add_argument("--transform", help="Explicit b0_to_GRE.mat. Overrides auto-discovery.")
    p.add_argument("--gre-brain-mask", help="Explicit GRE brain mask for coverage QC.")
    p.add_argument("--gre-dwi-mask", help="Explicit GRE-space DWI mask. Overrides auto-discovery.")
    p.add_argument("--allow-default-b0", action="store_true", help="If dwi07 metadata lacks B0 direction, default to [0,0,1] with warning.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--dry-run", action="store_true", help="Print planned operations without running FSL or writing final maps.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root_from_cwd()
    case_deriv = derivatives_case_dir(root, args.case)
    require_file(case_deriv, "case derivatives directory")

    print("=" * 72)
    print(f"{SCRIPT_TAG}: {args.case}")
    print("=" * 72)

    shell_dirs = find_shell_dirs_dwi07(case_deriv)
    if not shell_dirs:
        raise FileNotFoundError(f"No dwi07 shell directories found under {case_deriv / 'DWI'}")

    if args.all_shells:
        shells = list(shell_dirs.keys())
    elif args.shell:
        shells = args.shell
    else:
        raise SystemExit("Specify --all-shells or at least one --shell bXXXX")

    missing = [s for s in shells if s not in shell_dirs]
    if missing:
        raise FileNotFoundError(f"Requested shells not found in dwi07 outputs: {missing}. Available: {list(shell_dirs)}")

    transform_mat = find_transform(case_deriv, args.transform)
    gre_ref = find_gre_reference(case_deriv, args.gre_ref)
    gre_brain_mask = find_gre_brain_mask(case_deriv, args.gre_brain_mask)
    out_base = case_deriv / "DWI" / "dwi08_orientation_GREgrid"

    print(f"  GRE reference: {gre_ref}")
    print(f"  Transform:     {transform_mat}")
    print(f"  GRE brain mask:{gre_brain_mask if gre_brain_mask else ' not found; coverage percent unavailable'}")
    print(f"  Shells:        {', '.join(shells)}")

    rows: List[Dict[str, Any]] = []
    for shell in shells:
        meta = process_shell(
            root=root,
            case=args.case,
            shell=shell,
            shell_dir=shell_dirs[shell],
            out_base=out_base,
            transform_mat=transform_mat,
            gre_ref=gre_ref,
            gre_brain_mask=gre_brain_mask,
            explicit_gre_dwi_mask=args.gre_dwi_mask,
            allow_default_b0=args.allow_default_b0,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            rows.append({
                "case": args.case,
                "shell": shell,
                "valid_voxels": meta["dwi_mask"]["voxels"],
                "brain_overlap_percent": meta["dwi_mask"]["brain_overlap_percent"],
                "header_matches_reference": meta["header_validation"]["matches_reference"],
                "angle_deg_median": meta["qc"]["orientation"]["angle_deg_median"],
                "cos2_median": meta["qc"]["orientation"]["cos2_median"],
                "metadata_json": meta["outputs"]["metadata_json"],
            })

    if rows:
        summary_csv = out_base / f"{SCRIPT_TAG}_{args.case}_summary.csv"
        save_csv(summary_csv, rows)
        print(f"  Summary: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
