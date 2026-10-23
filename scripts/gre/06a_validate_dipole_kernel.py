#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

MODULE = "gre06a"
VERSION = "0.1.0"
OUT_SUBDIR = Path("GRE") / "06a_dipole_kernel_validation"
AFFINE_ATOL = 1e-6
B0_ATOL = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic QSM dipole-kernel/B0 validation; no inversion.")
    p.add_argument("case")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--sphere-radius-mm", type=float, default=12.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def unit(v, name):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"Invalid {name}: {v}")
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError(f"Zero-norm {name}")
    return v / n, n


def dipole_kernel(shape, zooms, b0_voxel):
    b0, _ = unit(b0_voxel, "B0 voxel vector")
    nx, ny, nz = shape
    dx, dy, dz = zooms
    kx = np.fft.fftfreq(nx, d=dx)[:, None, None]
    ky = np.fft.fftfreq(ny, d=dy)[None, :, None]
    kz = np.fft.fftfreq(nz, d=dz)[None, None, :]
    k2 = kx*kx + ky*ky + kz*kz
    kb = b0[0]*kx + b0[1]*ky + b0[2]*kz
    D = np.zeros(shape, dtype=np.float64)
    m = k2 > 0
    D[m] = 1.0/3.0 - (kb[m]**2)/k2[m]
    D[0, 0, 0] = 0.0
    return D


def sphere(shape, zooms, radius_mm):
    c = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    x = (np.arange(shape[0]) - c[0]) * zooms[0]
    y = (np.arange(shape[1]) - c[1]) * zooms[1]
    z = (np.arange(shape[2]) - c[2]) * zooms[2]
    r2 = x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2
    return (r2 <= radius_mm**2).astype(np.float64)


def forward(chi, D):
    out = np.fft.ifftn(D * np.fft.fftn(chi))
    return out.real, float(np.max(np.abs(out.imag)))


def corr(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / den) if den else float("nan")


def save_like(data, ref, path):
    hdr = ref.header.copy(); hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine, hdr), str(path))


def main():
    a = parse_args()
    root = a.root.resolve(); case = a.case
    gre05 = root / "derivatives" / case / "GRE" / "05b_vsharp_apf_v030" / f"{case}_GRE05b_local_field_hz.nii.gz"
    b0json = root / "derivatives" / case / "GRE" / "phase_preparation_gre02" / f"gre02_{case}_b0_direction.json"
    if not gre05.is_file(): raise FileNotFoundError(gre05)
    if not b0json.is_file(): raise FileNotFoundError(b0json)

    img = nib.load(str(gre05))
    if len(img.shape) != 3: raise ValueError(f"Expected 3D GRE05b image, got {img.shape}")
    shape = tuple(map(int, img.shape))
    zooms = tuple(map(float, img.header.get_zooms()[:3]))

    meta = json.loads(b0json.read_text())
    stored_aff = np.asarray(meta["affine"], dtype=float)
    affdiff = float(np.max(np.abs(stored_aff - img.affine)))
    if not np.allclose(stored_aff, img.affine, rtol=0, atol=AFFINE_ATOL):
        raise ValueError(f"GRE02/GRE05b affine mismatch; max abs diff={affdiff:.3g}")

    b0_world, _ = unit(meta["world_b0_direction"], "world B0")
    b0_voxel, b0norm = unit(meta["voxel_b0_direction"], "voxel B0")
    if abs(b0norm - 1.0) > B0_ATOL:
        raise ValueError(f"Stored voxel B0 is not normalized: {b0norm:.12f}")

    A = np.asarray(img.affine[:3, :3], dtype=float)
    R = A / np.linalg.norm(A, axis=0, keepdims=True)
    b0_recalc, _ = unit(R.T @ b0_world, "recomputed voxel B0")
    b0diff = float(np.max(np.abs(b0_recalc - b0_voxel)))
    if not np.allclose(b0_recalc, b0_voxel, rtol=0, atol=B0_ATOL):
        raise ValueError(f"Stored/recomputed voxel B0 mismatch; max abs diff={b0diff:.3g}")

    naive = np.array([0.0, 0.0, 1.0])
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(b0_voxel, naive)), 0, 1))))

    D = dipole_kernel(shape, zooms, b0_voxel)
    D_naive = dipole_kernel(shape, zooms, naive)
    kmin, kmax = float(D.min()), float(D.max())
    range_ok = kmin >= -2/3 - 1e-12 and kmax <= 1/3 + 1e-12 and D[0,0,0] == 0

    chi = sphere(shape, zooms, a.sphere_radius_mm)
    fld, imag = forward(chi, D)
    fld_naive, imag_naive = forward(chi, D_naive)
    diff = fld - fld_naive
    mean = float(fld.mean()); sd = float(fld.std())
    rmse = float(np.sqrt(np.mean(diff**2)))
    nrmse = rmse/sd if sd else float("nan")
    r = corr(fld, fld_naive)

    outdir = root / "derivatives" / case / OUT_SUBDIR
    outdir.mkdir(parents=True, exist_ok=True)
    outs = {
        "sphere": outdir / f"{case}_GRE06a_sphere_chi_ppm.nii.gz",
        "field_true": outdir / f"{case}_GRE06a_field_true_b0_ppm.nii.gz",
        "field_naive": outdir / f"{case}_GRE06a_field_naive_voxel_z_ppm.nii.gz",
        "difference": outdir / f"{case}_GRE06a_true_minus_naive_field_ppm.nii.gz",
        "qc": outdir / f"{case}_GRE06a_qc.json",
    }
    if not a.overwrite:
        existing = [str(p) for p in outs.values() if p.exists()]
        if existing: raise FileExistsError("Outputs exist; use --overwrite:\n" + "\n".join(existing))

    save_like(chi, img, outs["sphere"])
    save_like(fld, img, outs["field_true"])
    save_like(fld_naive, img, outs["field_naive"])
    save_like(diff, img, outs["difference"])

    overall = bool(range_ok and abs(mean) < 1e-12 and max(imag, imag_naive) < 1e-10)
    qc = {
        "script": Path(__file__).name, "module": MODULE, "version": VERSION,
        "created": datetime.now().isoformat(timespec="seconds"), "case": case,
        "purpose": "Synthetic pre-inversion validation of QSM dipole-kernel physics and B0 geometry.",
        "geometry": {"shape": list(shape), "voxel_size_mm": list(zooms), "axcodes": list(nib.aff2axcodes(img.affine)), "gre02_gre05b_affine_match": True, "affine_max_abs_diff": affdiff},
        "b0": {"world": b0_world.tolist(), "voxel": b0_voxel.tolist(), "recomputed_voxel": b0_recalc.tolist(), "stored_vs_recomputed_max_abs_diff": b0diff, "naive_voxel_z": naive.tolist(), "angle_to_naive_voxel_z_deg": angle},
        "physics": {"forward_model": "b_ppm = IFFT(D * FFT(chi_ppm))", "dipole_kernel": "D=1/3-(k.B0)^2/|k|^2; D(0)=0", "k_units": "cycles/mm"},
        "phantom": {"susceptibility_ppm": 1.0, "radius_mm": a.sphere_radius_mm, "voxel_count": int(chi.sum())},
        "kernel_qc": {"min": kmin, "max": kmax, "dc": float(D[0,0,0]), "theoretical_range_ok": bool(range_ok)},
        "field_qc": {"mean_ppm": mean, "sd_ppm": sd, "max_imaginary_true": imag, "max_imaginary_naive": imag_naive},
        "true_vs_naive_voxel_z": {"pearson_r": r, "rmse_ppm": rmse, "rmse_over_true_field_sd": nrmse, "max_abs_difference_ppm": float(np.max(np.abs(diff)))},
        "overall_pass": overall,
    }
    outs["qc"].write_text(json.dumps(qc, indent=2) + "\n")

    print(f"\n{MODULE} synthetic dipole validation")
    print("="*72)
    print(f"case:                       {case}")
    print(f"shape / voxel mm:           {shape} / {zooms}")
    print(f"B0 voxel:                   {b0_voxel}")
    print(f"angle to naive voxel +z:    {angle:.4f} deg")
    print(f"D(k) min/max:               {kmin:.9f} / {kmax:.9f}")
    print(f"D(0):                       {D[0,0,0]:.9g}")
    print(f"true-field mean / SD ppm:   {mean:.3e} / {sd:.3e}")
    print(f"max imaginary residual:     {imag:.3e}")
    print("True B0 versus WRONG voxel [0,0,1]:")
    print(f"  Pearson r:                {r:.8f}")
    print(f"  RMSE ppm:                 {rmse:.6e}")
    print(f"  RMSE / true-field SD:     {nrmse:.6f}")
    print(f"  max abs difference ppm:   {np.max(np.abs(diff)):.6e}")
    print(f"overall validation:         {'PASS' if overall else 'FAIL'}")
    print(f"outputs:                    {outdir.relative_to(root)}\n")


if __name__ == "__main__":
    main()
