#!/usr/bin/env python3
"""
APF GRE05b v0.3.0
V-SHARP background-field removal.

Consumes
--------
- GRE04b WLS total field [Hz]
- validated GRE05a Mask 3 (hole-filled BFR mask)

Methodological implementation
-----------------------------
This version follows a variable-kernel SHARP structure:

1. Variable-kernel convolution:
   - in the deep interior use the largest solid-sphere SMV kernel;
   - toward the Mask-3 boundary, use progressively smaller solid spheres;
   - radii decrease from 9 mm to 1 mm in 1-mm steps;
   - all radii, including 1 mm, use the same solid-sphere delta-SMV operator.

2. Assemble one spatially variable SHARP right-hand side over the retained
   V-SHARP support.

3. Perform one Fourier-domain deconvolution using the SHARP transfer function
   corresponding to the largest, 9-mm solid sphere.

4. Regularize the inverse explicitly by truncated singular-value decomposition
   (TSVD): invert only Fourier coefficients for which the magnitude of the
   9-mm delta-SMV transfer function satisfies |H| >= 0.50.

No separate post-hoc radial high-pass filter is applied in v0.3.0.

Development history
-------------------
GRE05b v0.1.x used per-radius deconvolution and did not pass validation.

GRE05b v0.2.0 introduced the variable-kernel / single-deconvolution
architecture but used an effectively untruncated SHARP inverse
(kernel_eps = 1e-12) followed by an independent radial high-pass filter.
Controlled comparison against STI Suite V-SHARP showed systematic
field-interior disagreement.

v0.3.0 replaces that regularization strategy with explicit SHARP-domain TSVD.
A cohort sensitivity analysis showed minimum STI-reference RMSE at
|H| = 0.50 in all 10 cases.

Outputs
-------
- V-SHARP local field [Hz]
- estimated background field [Hz]
- V-SHARP support mask
- assigned sphere-radius map [mm]
- variable-kernel SHARP right-hand side [Hz]
- reconstruction residual [Hz]
- QC JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


SCRIPT_VERSION = "0.3.0"
MODULE_NAME = "gre05b"
METHOD_NAME = "vsharp_variable_kernel_single_deconvolution_tsvd"
OUTPUT_SUBDIR = Path("GRE") / "05b_vsharp_apf_v030"

DEFAULT_MAX_RADIUS_MM = 9.0
DEFAULT_MIN_SPHERE_RADIUS_MM = 1.0
DEFAULT_RADIUS_STEP_MM = 1.0
DEFAULT_TSVD_THRESHOLD = 0.50


@dataclass(frozen=True)
class Inputs:
    case_id: str
    total_field_hz: Path
    mask3_bfr: Path


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None
    input_mask_voxels: int | None = None
    output_mask_voxels: int | None = None
    retained_fraction: float | None = None
    local_sd_hz: float | None = None
    local_min_hz: float | None = None
    local_max_hz: float | None = None
    output_dir: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        root = cli_root.expanduser().resolve()
    elif os.environ.get("APF_ROOT"):
        root = Path(os.environ["APF_ROOT"]).expanduser().resolve()
    else:
        root = Path.cwd().resolve()
    if not root.exists():
        raise FileNotFoundError(f"APF root does not exist: {root}")
    return root


def list_cases(root: Path) -> list[str]:
    d = root / "derivatives"
    if not d.exists():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and p.name.startswith("ANISO_") and p.name.endswith("_InS")
    )


def unique_match(paths: Sequence[Path], description: str) -> Path:
    unique = sorted({p.resolve() for p in paths if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"No {description} found")
    if len(unique) > 1:
        raise RuntimeError(
            f"Ambiguous {description}; found {len(unique)} files:\n  "
            + "\n  ".join(str(p) for p in unique)
        )
    return unique[0]


def discover_inputs(root: Path, case_id: str) -> Inputs:
    case_dir = root / "derivatives" / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case derivatives directory not found: {case_dir}")

    field = (
        case_dir / "GRE" / "04b_field_wls" /
        f"{case_id}_GRE04b_total_field_hz.nii.gz"
    )
    if not field.is_file():
        field = unique_match(
            list(case_dir.rglob("*GRE04b_total_field_hz.nii*")),
            "GRE04b total-field Hz map",
        )

    preferred = [
        case_dir / "GRE" / "05a_bfr_masks" /
        f"{case_id}_GRE05_mask3_bfr_holefilled.nii.gz",
        case_dir / "GRE" / "05_vsharp" /
        f"{case_id}_GRE05_mask3_bfr_holefilled.nii.gz",
    ]
    existing = [p for p in preferred if p.is_file()]
    if existing:
        mask3 = existing[0]
    else:
        mask3 = unique_match(
            list(case_dir.rglob("*GRE05_mask3_bfr_holefilled.nii*")),
            "GRE05a Mask 3 BFR mask",
        )

    return Inputs(case_id, field.resolve(), mask3.resolve())


def load_img(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {data.shape}: {path}")
    return img, data


def validate_geometry(ref: nib.Nifti1Image, other: nib.Nifti1Image, label: str) -> None:
    if tuple(ref.shape[:3]) != tuple(other.shape[:3]):
        raise ValueError(
            f"{label} shape mismatch: {other.shape[:3]} vs {ref.shape[:3]}"
        )
    if not np.allclose(ref.affine, other.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{label} affine mismatch")


def save_nifti(data, ref: nib.Nifti1Image, path: Path, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref.header.copy()
    hdr.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=dtype), ref.affine, hdr),
        str(path),
    )


def build_sphere_radii(max_r: float, min_r: float, step: float) -> list[float]:
    if max_r < min_r or min_r <= 0 or step <= 0:
        raise ValueError("Require max_radius >= min_radius > 0 and step > 0")
    radii = []
    r = float(max_r)
    while r >= min_r - 1e-9:
        radii.append(round(r, 10))
        r -= step
    if not np.isclose(radii[-1], min_r):
        radii.append(float(min_r))
    return sorted(set(radii), reverse=True)


def solid_sphere_smv(radius_mm: float, vox_mm: Sequence[float]) -> np.ndarray:
    vox = np.asarray(vox_mm[:3], dtype=float)
    if np.any(vox <= 0):
        raise ValueError(f"Invalid voxel sizes: {vox}")

    half = np.ceil(radius_mm / vox).astype(int)
    x = np.arange(-half[0], half[0] + 1) * vox[0]
    y = np.arange(-half[1], half[1] + 1) * vox[1]
    z = np.arange(-half[2], half[2] + 1) * vox[2]
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    sphere = (gx*gx + gy*gy + gz*gz) <= radius_mm**2 + 1e-12

    smv = sphere.astype(np.float64)
    if smv.sum() <= 1:
        raise ValueError(
            f"Radius {radius_mm:g} mm produces <=1 kernel voxel at {vox}"
        )
    smv /= smv.sum()
    return smv


def sharp_kernel_from_smv(smv: np.ndarray) -> np.ndarray:
    h = -smv.astype(np.float64, copy=True)
    c = tuple(s // 2 for s in h.shape)
    h[c] += 1.0
    return h



def valid_centers(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    return ndimage.binary_erosion(
        mask,
        structure=footprint.astype(bool),
        border_value=0,
    )


def convolve_kernel_valid(
    field: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear convolution, retaining values only where the kernel footprint lies
    fully inside the BFR mask.
    """
    footprint = np.abs(kernel) > 0
    valid = valid_centers(mask, footprint)
    masked = np.where(mask, field, 0.0).astype(np.float64)
    filtered = fftconvolve(masked, kernel, mode="same")
    filtered[~valid] = 0.0
    return filtered, valid


def fft_kernel_to_shape(kernel: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """
    Embed a centered compact kernel in the full grid, then move its origin to
    FFT index zero.
    """
    shape = tuple(int(v) for v in shape)
    if any(k > s for k, s in zip(kernel.shape, shape)):
        raise ValueError(f"Kernel {kernel.shape} larger than image {shape}")

    arr = np.zeros(shape, dtype=np.float64)
    starts = [
        s // 2 - k // 2
        for s, k in zip(shape, kernel.shape)
    ]
    sl = tuple(slice(st, st + k) for st, k in zip(starts, kernel.shape))
    arr[sl] = kernel
    arr = np.fft.ifftshift(arr)
    return np.fft.fftn(arr)



def variable_kernel_rhs(
    field_hz: np.ndarray,
    mask3: np.ndarray,
    vox_mm: Sequence[float],
    sphere_radii_mm: Sequence[float],
):
    """
    Assemble the V-SHARP variable-kernel convolution result.

    Each voxel receives the response of the largest solid-sphere
    delta-SMV kernel whose footprint lies fully inside the BFR mask.
    The radius sequence extends to the configured minimum sphere radius;
    v0.3.0 uses 1 mm by default and therefore requires no separate
    Laplacian minimum operator.
    """
    rhs = np.zeros(field_hz.shape, dtype=np.float64)
    radius_map = np.zeros(field_hz.shape, dtype=np.float32)
    assigned = np.zeros(field_hz.shape, dtype=bool)
    steps = []

    max_sharp = None

    for i, r in enumerate(sphere_radii_mm):
        smv = solid_sphere_smv(r, vox_mm)
        h = sharp_kernel_from_smv(smv)
        filtered, valid = convolve_kernel_valid(field_hz, mask3, h)

        shell = valid & ~assigned
        rhs[shell] = filtered[shell]
        radius_map[shell] = float(r)
        assigned[shell] = True

        if i == 0:
            max_sharp = h

        steps.append({
            "operator": "solid_sphere_delta_minus_smv",
            "radius_mm": float(r),
            "kernel_shape": list(h.shape),
            "kernel_support_voxels": int(np.count_nonzero(smv)),
            "valid_centers_voxels": int(np.count_nonzero(valid)),
            "newly_assigned_voxels": int(np.count_nonzero(shell)),
        })

    if max_sharp is None:
        raise RuntimeError("No maximum sphere kernel generated")

    rhs[~assigned] = 0.0

    return rhs, assigned, radius_map, max_sharp, steps


def single_deconvolution(
    rhs: np.ndarray,
    support: np.ndarray,
    max_sharp_kernel: np.ndarray,
    tsvd_threshold: float,
) -> tuple[np.ndarray, dict]:
    """
    V-SHARP deconvolution using the largest-sphere SHARP kernel.

    Truncated singular-value decomposition (TSVD) is implemented in
    Fourier space by inverting the delta-SMV transfer function only
    where its magnitude is at least tsvd_threshold.
    """
    h_hat = fft_kernel_to_shape(max_sharp_kernel, rhs.shape)
    rhs_hat = np.fft.fftn(rhs)

    abs_h = np.abs(h_hat)
    invert = abs_h >= tsvd_threshold

    local_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    local_hat[invert] = rhs_hat[invert] / h_hat[invert]

    local = np.real(np.fft.ifftn(local_hat))
    local[~support] = 0.0

    n_total = int(rhs.size)
    n_inverted = int(np.count_nonzero(invert))
    n_truncated = n_total - n_inverted

    qc = {
        "tsvd_threshold": float(tsvd_threshold),
        "n_fourier_coefficients": n_total,
        "n_inverted_coefficients": n_inverted,
        "n_truncated_coefficients": n_truncated,
        "truncated_fraction": float(n_truncated / n_total),
        "min_inverted_abs_transfer": (
            float(np.min(abs_h[invert])) if np.any(invert) else None
        ),
        "max_abs_transfer": float(np.max(abs_h)),
    }

    return local.astype(np.float32), qc


def vsharp(
    field_hz: np.ndarray,
    mask3: np.ndarray,
    vox_mm: Sequence[float],
    sphere_radii_mm: Sequence[float],
    tsvd_threshold: float,
):
    work_mask = mask3 & np.isfinite(field_hz)
    field = np.where(work_mask, field_hz, 0.0).astype(np.float32)

    (
        rhs,
        support,
        radius_map,
        max_sharp_kernel,
        steps,
    ) = variable_kernel_rhs(
        field_hz=field,
        mask3=work_mask,
        vox_mm=vox_mm,
        sphere_radii_mm=sphere_radii_mm,
    )

    local, deconv_qc = single_deconvolution(
        rhs=rhs,
        support=support,
        max_sharp_kernel=max_sharp_kernel,
        tsvd_threshold=tsvd_threshold,
    )

    background = np.zeros_like(local)
    background[support] = field_hz[support] - local[support]

    residual = np.zeros_like(local)
    residual[support] = (
        field_hz[support] - local[support] - background[support]
    )

    qc = {
        "working_mask_voxels": int(np.count_nonzero(work_mask)),
        "support_voxels": int(np.count_nonzero(support)),
        "unassigned_mask3_voxels": int(
            np.count_nonzero(work_mask & ~support)
        ),
        "variable_kernel_steps": steps,
        "single_deconvolution_tsvd": deconv_qc,
    }

    return (
        local,
        background,
        support,
        radius_map,
        residual,
        rhs.astype(np.float32),
        qc,
    )


def stats(data: np.ndarray, mask: np.ndarray) -> dict:
    v = np.asarray(data[mask], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v)),
        "p001": float(np.percentile(v, 0.1)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "p999": float(np.percentile(v, 99.9)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    d = root / "derivatives" / case_id / OUTPUT_SUBDIR
    p = f"{case_id}_GRE05b"
    return {
        "dir": d,
        "local": d / f"{p}_local_field_hz.nii.gz",
        "background": d / f"{p}_background_field_hz.nii.gz",
        "mask": d / f"{p}_vsharp_mask.nii.gz",
        "radius": d / f"{p}_assigned_radius_mm.nii.gz",
        "variable_rhs": d / f"{p}_variable_kernel_rhs_hz.nii.gz",
        "residual": d / f"{p}_reconstruction_residual_hz.nii.gz",
        "qc": d / f"{p}_qc.json",
    }


def process_case(root: Path, case_id: str, args) -> CaseResult:
    t0 = time.perf_counter()
    out = output_paths(root, case_id)

    try:
        inp = discover_inputs(root, case_id)

        if args.dry_run:
            print(f"\n[{case_id}]")
            for k, v in asdict(inp).items():
                if k != "case_id":
                    print(f"  {k}: {v}")
            return CaseResult(
                case_id, "DRYRUN", "Input discovery successful",
                output_dir=str(out["dir"]),
            )

        if out["qc"].exists() and not args.force:
            return CaseResult(
                case_id, "SKIPPED",
                "GRE05b outputs already exist; use --force",
                output_dir=str(out["dir"]),
            )

        field_img, field = load_img(inp.total_field_hz)
        mask_img, mask_data = load_img(inp.mask3_bfr)
        validate_geometry(field_img, mask_img, "GRE05a Mask 3")

        mask3 = mask_data > 0
        if not np.any(mask3):
            raise ValueError("Mask 3 is empty")

        vox_mm = tuple(float(v) for v in field_img.header.get_zooms()[:3])
        radii = build_sphere_radii(
            args.max_radius_mm,
            args.min_sphere_radius_mm,
            args.radius_step_mm,
        )

        (
            local,
            background,
            support,
            radius_map,
            residual,
            rhs,
            detail,
        ) = vsharp(
            field_hz=field,
            mask3=mask3,
            vox_mm=vox_mm,
            sphere_radii_mm=radii,
            tsvd_threshold=args.tsvd_threshold,
        )

        n_in = int(np.count_nonzero(mask3))
        n_out = int(np.count_nonzero(support))
        retained = n_out / n_in

        out["dir"].mkdir(parents=True, exist_ok=True)
        save_nifti(local, field_img, out["local"], np.float32)
        save_nifti(background, field_img, out["background"], np.float32)
        save_nifti(support, field_img, out["mask"], np.uint8)
        save_nifti(radius_map, field_img, out["radius"], np.float32)
        save_nifti(rhs, field_img, out["variable_rhs"], np.float32)
        save_nifti(residual, field_img, out["residual"], np.float32)

        total_stats = stats(field, support)
        local_stats = stats(local, support)
        bg_stats = stats(background, support)
        resid_stats = stats(residual, support)

        qc = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "method": METHOD_NAME,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": time.perf_counter() - t0,
            "prototype_warning": (
                "GRE05b v0.1.x used per-radius deconvolution and failed QC; "
                "GRE05b v0.2.0 used an effectively untruncated SHARP inverse "
                "followed by a post-hoc radial high-pass and showed systematic "
                "disagreement with the external reference implementation. Do not use "
                "v0.1.x or v0.2.0 outputs "
                "downstream."
            ),
            "inputs": {
                "total_field_hz": str(inp.total_field_hz),
                "mask3_bfr": str(inp.mask3_bfr),
            },
            "configuration": {
                "sphere_type": "solid",
                "sphere_radius_units": "mm",
                "sphere_radii_mm": [float(r) for r in radii],
                "max_radius_mm": args.max_radius_mm,
                "min_sphere_radius_mm": args.min_sphere_radius_mm,
                "radius_step_mm": args.radius_step_mm,
                "minimum_operator": (
                    "solid-sphere delta-SMV at minimum radius"
                ),
                "deconvolution": (
                    "single largest-sphere delta-SMV Fourier inversion "
                    "with explicit TSVD"
                ),
                "tsvd_threshold": args.tsvd_threshold,
                "voxel_sizes_mm": list(vox_mm),
            },
            "mask_qc": {
                "mask3_voxels": n_in,
                "vsharp_support_voxels": n_out,
                "retained_fraction": retained,
                "removed_fraction": 1.0 - retained,
            },
            "field_stats_hz": {
                "total": total_stats,
                "local": local_stats,
                "background": bg_stats,
                "identity_residual": resid_stats,
            },
            "identity_check": {
                "equation": "total = local + background within support",
                "note": "bookkeeping check only; background is defined by subtraction",
            },
            "vsharp_details": detail,
            "outputs": {
                k: str(v) for k, v in out.items() if k != "dir"
            },
        }
        out["qc"].write_text(json.dumps(qc, indent=2), encoding="utf-8")

        return CaseResult(
            case_id=case_id,
            status="OK",
            message="V-SHARP v0.3.0 completed",
            runtime_s=time.perf_counter() - t0,
            input_mask_voxels=n_in,
            output_mask_voxels=n_out,
            retained_fraction=retained,
            local_sd_hz=local_stats.get("sd"),
            local_min_hz=local_stats.get("min"),
            local_max_hz=local_stats.get("max"),
            output_dir=str(out["dir"]),
        )

    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - t0,
            output_dir=str(out["dir"]),
        )


def build_parser():
    p = argparse.ArgumentParser(
        description="APF GRE05b v0.3.0: V-SHARP background-field removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--all-gre-ready", action="store_true")
    p.add_argument(
        "--max-radius-mm", type=float, default=DEFAULT_MAX_RADIUS_MM
    )
    p.add_argument(
        "--min-sphere-radius-mm",
        type=float,
        default=DEFAULT_MIN_SPHERE_RADIUS_MM,
    )
    p.add_argument(
        "--radius-step-mm", type=float, default=DEFAULT_RADIUS_STEP_MM
    )
    p.add_argument(
        "--tsvd-threshold",
        type=float,
        default=DEFAULT_TSVD_THRESHOLD,
        help=(
            "Minimum |delta-SMV transfer function| retained during "
            "largest-kernel Fourier inversion"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    if not (0.0 < args.tsvd_threshold < 1.0):
        raise SystemExit("--tsvd-threshold must satisfy 0 < threshold < 1")

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit("Select --case CASE_ID or use --all-gre-ready")

    radii = build_sphere_radii(
        args.max_radius_mm,
        args.min_sphere_radius_mm,
        args.radius_step_mm,
    )

    print(f"APF GRE05b v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Method: {METHOD_NAME}")
    print(f"Cases: {len(cases)}")
    print("Sphere radii [mm]: " + ", ".join(f"{r:g}" for r in radii))
    print("Minimum operator: solid-sphere delta-SMV")
    print("Deconvolution: single largest-sphere SHARP kernel with TSVD")
    print(f"TSVD threshold |H|: {args.tsvd_threshold:g}")

    results = []
    for case in cases:
        result = process_case(root, case, args)
        results.append(result)
        print(f"[{result.status}] {case}: {result.message}")
        if result.retained_fraction is not None:
            print(
                f"  V-SHARP support: "
                f"{100.0 * result.retained_fraction:.3f}% of Mask 3"
            )
        if result.local_sd_hz is not None:
            print(
                f"  local field: SD={result.local_sd_hz:.3f} Hz, "
                f"min={result.local_min_hz:.3f} Hz, "
                f"max={result.local_max_hz:.3f} Hz"
            )

    n_ok = sum(r.status == "OK" for r in results)
    n_failed = sum(r.status == "FAILED" for r in results)
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
