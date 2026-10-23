#!/usr/bin/env python3
"""GRE06b: linear isotropic-TV QSM using case-specific B0 geometry.

Objective:
    min_chi 0.5 || M (D chi - b_ppm) ||_2^2 + lambda_tv * TV_iso(chi)
    subject to chi = 0 outside GRE05b support.

Solver: Chambolle-Pock / PDHG.
"""
from __future__ import annotations

import argparse, json, math
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

MODULE = "gre06b"
VERSION = "0.1.0"
GAMMA_HZ_PER_T = 42.57747892e6
AFFINE_ATOL = 1e-6
B0_ATOL = 1e-6


def argspec():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--lambda-tv", required=True, type=float)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--min-iterations", type=int, default=50)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--relative-update-tol", type=float, default=1e-4)
    p.add_argument("--power-iterations", type=int, default=20)
    p.add_argument("--step-safety", type=float, default=0.95)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def unit(v, name):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector")
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError(f"{name} has zero norm")
    return v / n, n


def device_from_arg(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(s)


def resolve_inputs(root, case):
    base = root / "derivatives" / case / "GRE"
    field = base / "05b_vsharp_apf_v030" / f"{case}_GRE05b_local_field_hz.nii.gz"
    candidates = [
        base / "05b_vsharp_apf_v030" / f"{case}_GRE05b_vsharp_mask.nii.gz",
        base / "05b_vsharp_apf_v030" / f"{case}_GRE05b_mask.nii.gz",
    ]
    mask = next((p for p in candidates if p.is_file()), None)
    b0j = base / "phase_preparation_gre02" / f"gre02_{case}_b0_direction.json"
    source = sorted((root / "nifti" / case / "GRE" / "mag").glob(f"{case}_GRE*_e1.json"))
    if not field.is_file(): raise FileNotFoundError(field)
    if mask is None: raise FileNotFoundError("GRE05b support mask not found")
    if not b0j.is_file(): raise FileNotFoundError(b0j)
    if not source: raise FileNotFoundError("Original GRE magnitude e1 JSON not found")
    return field, mask, b0j, source[0]


def validate_geometry(field_img, mask_img, meta):
    if field_img.shape != mask_img.shape:
        raise ValueError("Field/mask shape mismatch")
    if not np.allclose(field_img.affine, mask_img.affine, rtol=0, atol=AFFINE_ATOL):
        raise ValueError("Field/mask affine mismatch")
    stored_aff = np.asarray(meta.get("affine"), dtype=np.float64)
    if stored_aff.shape != (4,4): raise ValueError("Invalid GRE02 stored affine")
    adiff = float(np.max(np.abs(stored_aff - field_img.affine)))
    if not np.allclose(stored_aff, field_img.affine, rtol=0, atol=AFFINE_ATOL):
        raise ValueError(f"GRE02/GRE05b affine mismatch: {adiff:g}")
    b0, b0norm = unit(meta.get("voxel_b0_direction"), "voxel_b0_direction")
    if abs(b0norm - 1.0) > B0_ATOL:
        raise ValueError(f"Stored B0 not normalized: {b0norm}")
    bw, _ = unit(meta.get("world_b0_direction"), "world_b0_direction")
    A = field_img.affine[:3,:3].astype(np.float64)
    R = A / np.linalg.norm(A, axis=0, keepdims=True)
    brec, _ = unit(R.T @ bw, "recomputed B0")
    bdiff = float(np.max(np.abs(brec - b0)))
    if not np.allclose(brec, b0, rtol=0, atol=B0_ATOL):
        raise ValueError(f"Stored/recomputed B0 mismatch: {bdiff:g}")
    return b0, adiff, bdiff, brec


def dipole_kernel(shape, zooms, b0, dev):
    nx, ny, nz = shape
    dx, dy, dz = zooms
    kx = torch.fft.fftfreq(nx, d=dx, device=dev, dtype=torch.float32)[:,None,None]
    ky = torch.fft.fftfreq(ny, d=dy, device=dev, dtype=torch.float32)[None,:,None]
    kz = torch.fft.fftfreq(nz, d=dz, device=dev, dtype=torch.float32)[None,None,:]
    b = torch.as_tensor(b0, device=dev, dtype=torch.float32)
    k2 = kx*kx + ky*ky + kz*kz
    kb = b[0]*kx + b[1]*ky + b[2]*kz
    D = torch.zeros(shape, device=dev, dtype=torch.float32)
    ok = k2 > 0
    D[ok] = 1/3 - (kb[ok]*kb[ok])/k2[ok]
    D[0,0,0] = 0
    return D


def Df(x, D):
    return torch.fft.ifftn(D * torch.fft.fftn(x)).real


def edge_masks(mask):
    mx = torch.zeros_like(mask); my = torch.zeros_like(mask); mz = torch.zeros_like(mask)
    mx[:-1] = mask[:-1] & mask[1:]
    my[:,:-1] = mask[:,:-1] & mask[:,1:]
    mz[:,:,:-1] = mask[:,:,:-1] & mask[:,:,1:]
    return mx,my,mz


def grad(x, zooms, em):
    dx,dy,dz = zooms; mx,my,mz = em
    gx=torch.zeros_like(x); gy=torch.zeros_like(x); gz=torch.zeros_like(x)
    gx[:-1]=(x[1:]-x[:-1])/dx
    gy[:,:-1]=(x[:,1:]-x[:,:-1])/dy
    gz[:,:,:-1]=(x[:,:,1:]-x[:,:,:-1])/dz
    return gx*mx, gy*my, gz*mz


def gradT(px,py,pz,zooms,em):
    dx,dy,dz=zooms; mx,my,mz=em
    px=px*mx; py=py*my; pz=pz*mz
    out=torch.zeros_like(px)
    out[:-1]-=px[:-1]/dx; out[1:]+=px[:-1]/dx
    out[:,:-1]-=py[:,:-1]/dy; out[:,1:]+=py[:,:-1]/dy
    out[:,:,:-1]-=pz[:,:,:-1]/dz; out[:,:,1:]+=pz[:,:,:-1]/dz
    return out


def Kt(yd, px,py,pz, D, mf, zooms, em):
    return Df(mf*yd, D) + gradT(px,py,pz,zooms,em)


@torch.no_grad()
def estimate_norm(shape,D,mf,zooms,em,niter):
    x=torch.randn(shape,device=D.device,dtype=torch.float32)*mf
    x/=torch.linalg.vector_norm(x)
    eig=None
    for _ in range(niter):
        yd=mf*Df(x,D)
        gx,gy,gz=grad(x,zooms,em)
        z=Kt(yd,gx,gy,gz,D,mf,zooms,em)*mf
        eig=torch.linalg.vector_norm(z)
        x=z/eig
    return math.sqrt(float(eig))


@torch.no_grad()
def metrics(x,b,D,mf,nvox,zooms,em,lam):
    pred=Df(x,D); resid=mf*(pred-b)
    l2=float(torch.linalg.vector_norm(resid))
    rmse=float(torch.sqrt(torch.sum(resid*resid)/nvox))
    gx,gy,gz=grad(x,zooms,em)
    tv=float(torch.sum(torch.sqrt(gx*gx+gy*gy+gz*gz)))
    vals=x[mf.bool()]
    return {
        "objective":0.5*l2*l2+lam*tv,
        "data_term":0.5*l2*l2,
        "residual_l2":l2,
        "residual_rmse_ppm":rmse,
        "tv_norm":tv,
        "chi_min_ppm":float(vals.min()),
        "chi_max_ppm":float(vals.max()),
        "chi_median_ppm":float(vals.median()),
        "chi_mean_ppm":float(vals.mean()),
        "chi_sd_ppm":float(vals.std(unbiased=False)),
    }, pred, resid


def save_nii(a, ref, p):
    hdr=ref.header.copy(); hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(a.astype(np.float32),ref.affine,hdr),str(p))


def tag(v):
    return f"{v:.8g}".replace("-","m").replace("+","").replace(".","p")


def main():
    a=argspec()
    if a.lambda_tv<=0: raise ValueError("--lambda-tv must be >0")
    root=a.root.resolve(); case=a.case; dev=device_from_arg(a.device)
    field_p,mask_p,b0_p,src_p=resolve_inputs(root,case)
    fi=nib.load(str(field_p)); mi=nib.load(str(mask_p))
    b0meta=load_json(b0_p); src=load_json(src_p)
    B0=float(src["MagneticFieldStrength"])
    b0,adiff,bdiff,brec=validate_geometry(fi,mi,b0meta)
    shape=tuple(map(int,fi.shape)); zooms=tuple(map(float,fi.header.get_zooms()[:3]))
    field=np.asarray(fi.dataobj,dtype=np.float32)
    mask=np.asarray(mi.dataobj)>0
    if not np.all(np.isfinite(field[mask])): raise ValueError("Non-finite field values inside mask")
    nvox=int(mask.sum()); hz_per_ppm=GAMMA_HZ_PER_T*B0/1e6
    bppm=field/hz_per_ppm; bppm[~mask]=0
    b=torch.from_numpy(bppm).to(dev); support=torch.from_numpy(mask).to(dev); mf=support.float()
    D=dipole_kernel(shape,zooms,b0,dev); em=edge_masks(support)
    opn=estimate_norm(shape,D,mf,zooms,em,a.power_iterations)
    tau=sigma=a.step_safety/opn

    x=torch.zeros(shape,device=dev); xp=torch.zeros_like(x); xb=torch.zeros_like(x)
    yd=torch.zeros_like(x); px=torch.zeros_like(x); py=torch.zeros_like(x); pz=torch.zeros_like(x)
    history=[]; converged=False; rel=float("inf")

    print(f"\n{MODULE} linear isotropic-TV QSM\n"+"="*76)
    print(f"case:                {case}")
    print(f"device:              {dev}")
    if dev.type=="cuda": print(f"GPU:                 {torch.cuda.get_device_name(dev)}")
    print(f"shape / voxel mm:    {shape} / {zooms}")
    print(f"support voxels:      {nvox:,}")
    print(f"B0 voxel:            {b0}")
    print(f"lambda_tv:           {a.lambda_tv:.8g}")
    print(f"estimated ||K||:     {opn:.6f}")

    for it in range(1,a.max_iterations+1):
        qd=yd + sigma*(mf*Df(xb,D))
        yd=(qd-sigma*b)/(1+sigma); yd*=mf
        gx,gy,gz=grad(xb,zooms,em)
        px+=sigma*gx; py+=sigma*gy; pz+=sigma*gz
        pn=torch.sqrt(px*px+py*py+pz*pz)
        sc=torch.clamp(pn/a.lambda_tv,min=1.0)
        px/=sc; py/=sc; pz/=sc
        xp.copy_(x)
        x-=tau*Kt(yd,px,py,pz,D,mf,zooms,em); x*=mf
        xb=x+a.theta*(x-xp)
        if it==1 or it%a.check_every==0 or it==a.max_iterations:
            denom=torch.linalg.vector_norm(x); numer=torch.linalg.vector_norm(x-xp)
            rel=float(numer/denom) if float(denom)>0 else float("inf")
            m,_,_=metrics(x,b,D,mf,nvox,zooms,em,a.lambda_tv)
            m.update(iteration=it,relative_update=rel); history.append(m)
            print(f"iter {it:4d} obj={m['objective']:.6e} RMSE={m['residual_rmse_ppm']:.6e} ppm TV={m['tv_norm']:.6e} relupd={rel:.3e}")
            if it>=a.min_iterations and rel<=a.relative_update_tol:
                converged=True; break

    m,pred,resid=metrics(x,b,D,mf,nvox,zooms,em,a.lambda_tv)
    out=root/"derivatives"/case/"GRE"/"06b_qsm_tv"/f"lambda_{tag(a.lambda_tv)}"
    out.mkdir(parents=True,exist_ok=True)
    files={
        "chi":out/f"{case}_GRE06b_qsm_tv_chi_ppm.nii.gz",
        "forward":out/f"{case}_GRE06b_qsm_tv_forward_field_ppm.nii.gz",
        "residual":out/f"{case}_GRE06b_qsm_tv_residual_field_ppm.nii.gz",
        "mask":out/f"{case}_GRE06b_qsm_tv_mask.nii.gz",
        "qc":out/f"{case}_GRE06b_qsm_tv_qc.json",
    }
    if not a.overwrite:
        exists=[p for p in files.values() if p.exists()]
        if exists: raise FileExistsError("Outputs exist; use --overwrite:\n  "+"\n  ".join(map(str,exists)))
    save_nii(x.detach().cpu().numpy(),fi,files["chi"])
    save_nii(pred.detach().cpu().numpy(),fi,files["forward"])
    save_nii(resid.detach().cpu().numpy(),fi,files["residual"])
    save_nii(mask.astype(np.float32),fi,files["mask"])
    qc={
        "script":Path(__file__).name,"module":MODULE,"version":VERSION,"created":datetime.now().isoformat(timespec="seconds"),"case":case,
        "method":"linear isotropic-TV QSM solved with Chambolle-Pock / PDHG",
        "objective":"0.5*||M(D chi-b_ppm)||_2^2 + lambda_tv*TV_iso(chi), chi=0 outside GRE05b support",
        "inputs":{"field":str(field_p.relative_to(root)),"mask":str(mask_p.relative_to(root)),"b0_json":str(b0_p.relative_to(root)),"source_json":str(src_p.relative_to(root))},
        "units":{"input_field":"Hz","normalized_field":"ppm","susceptibility":"ppm","gamma_hz_per_t":GAMMA_HZ_PER_T,"b0_t":B0,"hz_per_ppm":hz_per_ppm},
        "geometry":{"shape":list(shape),"voxel_size_mm":list(zooms),"axcodes":list(nib.aff2axcodes(fi.affine)),"voxel_b0_direction":b0.tolist(),"recomputed_voxel_b0_direction":brec.tolist(),"affine_max_abs_diff":adiff,"stored_vs_recomputed_b0_max_abs_diff":bdiff},
        "mask":{"support_voxels":nvox,"support_fraction_of_array":nvox/float(np.prod(shape)),"tv_boundary_policy":"exclude finite-difference edges crossing GRE05b mask boundary","outside_support_chi":0.0},
        "solver":{"lambda_tv":a.lambda_tv,"iterations_completed":history[-1]["iteration"],"max_iterations":a.max_iterations,"min_iterations":a.min_iterations,"check_every":a.check_every,"relative_update_tolerance":a.relative_update_tol,"final_relative_update":rel,"converged":converged,"theta":a.theta,"step_safety":a.step_safety,"power_iterations":a.power_iterations,"estimated_operator_norm":opn,"tau":tau,"sigma":sigma,"device":str(dev),"gpu_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else None},
        "final_metrics":m,"history":history,"referencing":"none; raw inversion output",
        "outputs":{k:str(v.relative_to(root)) for k,v in files.items() if k!="qc"},
    }
    with open(files["qc"],"w",encoding="utf-8") as f: json.dump(qc,f,indent=2); f.write("\n")
    print("\n"+"="*76)
    print(f"iterations:           {history[-1]['iteration']}")
    print(f"converged:            {converged}")
    print(f"final rel update:     {rel:.3e}")
    print(f"residual RMSE:        {m['residual_rmse_ppm']:.6e} ppm")
    print(f"TV norm:              {m['tv_norm']:.6e}")
    print(f"chi min/median/max:   {m['chi_min_ppm']:.6f} / {m['chi_median_ppm']:.6f} / {m['chi_max_ppm']:.6f} ppm")
    print(f"outputs:              {out.relative_to(root)}\n")

if __name__=="__main__": main()
