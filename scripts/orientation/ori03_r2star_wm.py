#!/usr/bin/env python3
"""ORI03 v0.2.2: GRE-space R2* vs fibre angle; case and cohort.

Requires independently generated, GRE-registered FSL vecreg V1 fields.
Scanner B0 assumed parallel to scanner-world +z. FSL scaled-voxel axis
convention is used for the three V1 components (x flipped when affine det>0).
No input images are altered, and missing V1 is NEVER silently substituted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

VERSION = '0.2.2'
SELECTIONS = ('FA03', 'WM08', 'WM09', 'WM09_FA03')
EDGES = np.arange(0., 91., 5.)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def paths_for(root: Path, case: str) -> dict[str, Path]:
    base = root / 'derivatives' / case
    r = base / 'relaxometry/r2star/r2star02_loglinear'
    return {
        'r2star': r / f'r2star02_{case}_r2star_s-1.nii.gz',
        'fit_mask': r / f'r2star02_{case}_fit_mask.nii.gz',
        'wm08': base / 'mp2rage/mp2rage06/wm_mask_p08_gre.nii.gz',
        'wm09': base / 'mp2rage/mp2rage06/wm_mask_p09_gre.nii.gz',
        'fa': base / f'DWI/registration/dwi06/b1000/dwi06_{case}_b1000_FA_in_GRE.nii.gz',
        'v1': base / f'DWI/orientation_validation/ori01_{case}_b1000_V1_vecreg.nii.gz',
    }


def get_cases(a: argparse.Namespace, root: Path) -> list[str]:
    if a.case:
        return [a.case]
    # Use APF's authoritative active in-situ case registry and selection interface.
    infra = root / 'scripts/infrastructure'
    if str(infra) not in sys.path:
        sys.path.insert(0, str(infra))
    from apf_paths import get_paths
    from apf_case_selection import resolve_cases
    registry = get_paths().ins_registry
    opts = argparse.Namespace(case=None, cases=None, all=True,
        all_dwi_ready=False, all_gre_ready=False, all_mp2rage_ready=False)
    cases = resolve_cases(opts, registry)
    if not cases:
        raise RuntimeError('APF registry returned no cases')
    return list(cases)


def check_geometry(images: dict[str, nib.spatialimages.SpatialImage]):
    ref = images['r2star']
    shape = ref.shape
    if len(shape) != 3:
        raise ValueError(f'R2* expected 3D, got {shape}')
    for key, im in images.items():
        expected = shape+(3,) if key == 'v1' else shape
        if im.shape != expected:
            raise ValueError(f'{key} has shape {im.shape}; expected {expected}')
        if not np.allclose(im.affine, ref.affine, atol=1e-3, rtol=0):
            raise ValueError(f'{key} affine does not match R2* (tolerance 1e-3 mm)')


def b0_from_affine(affine: np.ndarray):
    """Scanner +z to FSL scaled-voxel components; agrees with ORI01 pilot.

    The affine columns carry voxel-axis physical directions; normalize by
    voxel spacing to remove scaling before solving. FSL's scaled-voxel
    convention flips component x when det(affine[:3,:3]) is positive.
    """
    linear = affine[:3, :3].astype(float)
    det = float(np.linalg.det(linear))
    if not np.isfinite(det) or abs(det) < 1e-9:
        raise ValueError('Degenerate GRE affine')
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales <= 0):
        raise ValueError('Invalid voxel sizes')
    dirs = linear / scales[None, :]
    # Coordinate transform for scanner-world +z, in physical voxel axes.
    b_vox = np.linalg.solve(dirs, np.array([0., 0., 1.]))
    b_vox /= np.linalg.norm(b_vox)
    b_fsl = b_vox.copy()
    if det > 0:
        b_fsl[0] *= -1.
    return b_vox, b_fsl, det, scales


def image_data(im):
    return np.asarray(im.dataobj)


def numeric(value):
    return '' if not np.isfinite(value) else float(value)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def run_case(root: Path, case: str, force: bool, fa_threshold: float):
    paths = paths_for(root, case)
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing authoritative inputs:\n  ' + '\n  '.join(missing))
    images = {key: nib.load(str(p)) for key,p in paths.items()}
    check_geometry(images)
    r2 = image_data(images['r2star']).astype(np.float64)
    fit = image_data(images['fit_mask'])
    wm08 = image_data(images['wm08'])
    wm09 = image_data(images['wm09'])
    fa = image_data(images['fa']).astype(np.float64)
    v1 = image_data(images['v1']).astype(np.float64)
    for key, m in (('fit_mask',fit), ('wm08',wm08), ('wm09',wm09)):
        if not np.all(np.isfinite(m)) or not np.all(np.isin(np.unique(m), (0,1))):
            raise ValueError(f'{key} not a finite binary mask')
    if np.any((wm09>0)&(wm08==0)):
        raise ValueError('WM09 must be contained in WM08')
    b_vox,b_fsl,det,scales = b0_from_affine(images['r2star'].affine)
    norm = np.linalg.norm(v1, axis=-1)
    vr = np.isfinite(norm)&(norm>0.5)&np.all(np.isfinite(v1), axis=-1)
    rr = (fit>0)&np.isfinite(r2)&(r2>0)
    valid = rr & vr
    fsel = np.isfinite(fa)&(fa>=fa_threshold)&(fa<=1.)
    selections = {
        'FA03': valid & fsel,
        'WM08': valid & (wm08>0),
        'WM09': valid & (wm09>0),
        'WM09_FA03': valid & (wm09>0) & fsel,
    }
    dot = np.einsum('...i,i->...', v1, b_fsl)
    cosine = np.zeros(r2.shape, dtype=float)
    np.divide(np.abs(dot), norm, out=cosine, where=vr)
    theta = np.degrees(np.arccos(np.clip(cosine,0.,1.)))
    out = root/'derivatives'/case/'orientation/ori03'
    names = ('ori03_bins.csv','ori03_summary.csv','ori03_r2star_theta.png','ori03_provenance.json', 'ori03_anatomical_qc.png')
    existing = [n for n in names if (out/n).exists()]
    if existing and not force:
        raise FileExistsError(f'{case}: existing ORI03 outputs {existing}; use --force to replace after backup')
    # No changes to output directory until all analysis calculations succeed.
    bin_rows, sum_rows, pdata = [], [], {}
    for label in SELECTIONS:
        sel=selections[label]
        means=np.full(18,np.nan)
        counts=np.zeros(18,dtype=int)
        temp=[]
        for k in range(18):
            sub=sel & (theta>=EDGES[k]) & (theta<EDGES[k+1])
            vals=r2[sub]
            counts[k]=int(len(vals))
            if len(vals): means[k]=np.mean(vals)
            temp.append(dict(case=case,selection=label,bin_index=k+1,
                angle_lo_deg=EDGES[k],angle_hi_deg=EDGES[k+1],
                angle_center_deg=EDGES[k]+2.5,legacy_x_deg=EDGES[k+1],
                n=int(len(vals)),mean_r2star_s_1=numeric(means[k]),
                median_r2star_s_1=numeric(np.median(vals)) if len(vals) else '',
                sd_r2star_s_1=numeric(np.std(vals,ddof=1)) if len(vals)>1 else '',
                sem_r2star_s_1=numeric(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else ''))
        populated=int(np.isfinite(means).sum())
        mean_bins=float(np.mean(means)) if populated==18 else np.nan
        normalized=means/mean_bins if np.isfinite(mean_bins) and mean_bins!=0 else np.full(18,np.nan)
        finite=means[np.isfinite(means)]
        ai=(float((max(finite)-min(finite))/(max(finite)+min(finite)))
            if populated==18 and (max(finite)+min(finite))!=0 else np.nan)
        delta=float(means[-1]-means[0]) if np.isfinite(means[0]) and np.isfinite(means[-1]) else np.nan
        for k,row in enumerate(temp):
            row['normalized_legacy']=numeric(normalized[k])
        bin_rows.extend(temp)
        sum_rows.append(dict(case=case,selection=label,n_selected=int(np.count_nonzero(sel)),
            n_bins_populated=populated,mean_of_bin_means_s_1=numeric(mean_bins),
            voxelwise_mean_r2star_s_1=numeric(np.mean(r2[sel])) if np.any(sel) else '',
            delta_85_90_minus_0_5_s_1=numeric(delta),historical_anisotropy_index=numeric(ai)))
        pdata[label]=(means,normalized,counts)
        print(f'{case} {label:10s} N={np.count_nonzero(sel):8d} bins={populated:2d}/18 delta={delta:+.4f} s^-1 AI={ai:.5f}')
    # Build figure and QC before writing anything.
    fig, axes=plt.subplots(3,1,figsize=(10,11),sharex=True,layout='constrained')
    x=EDGES[:-1]+2.5
    for label,(means,normvals,counts) in pdata.items():
        axes[0].plot(x,means,marker='o',label=label)
        axes[1].plot(x,normvals,marker='o',label=label)
        axes[2].plot(x,counts,marker='o',label=label)
    axes[0].set_ylabel('Mean R2* (s$^{-1}$)')
    axes[1].set_ylabel('Mean / equal-bin mean')
    axes[2].set_ylabel('Voxel count per bin')
    axes[2].set_xlabel('Acute fibre angle to B0 (degrees), 5-degree bin centres')
    axes[0].legend(ncol=2)
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle(f'ORI03: {case} (GRE space; b1000 V1)')
    # QC: candidate WM boundary over GRE-space R2*; operator must inspect and accept.
    figqc, axq=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    cx,cy,cz=(np.array(r2.shape)//2).tolist()
    views=[(np.rot90(r2[:,:,cz]),np.rot90(wm08[:,:,cz]),'Axial'),
           (np.rot90(r2[cx,:,:]),np.rot90(wm08[cx,:,:]),'Sagittal'),
           (np.rot90(r2[:,cy,:]),np.rot90(wm08[:,cy,:]),'Coronal')]
    finite_vals=r2[rr]
    lo,hi=np.percentile(finite_vals,[5,95]) if finite_vals.size else (0,50)
    for ax,(bg,wm,title) in zip(axq,views):
        ax.imshow(bg,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        ax.contour(wm,levels=[0.5],colors=['red'],linewidths=0.75)
        ax.set_title(title+' / WM P≥0.8');ax.axis('off')
    figqc.suptitle(f'ORI03 anatomical overlay QC candidate — {case} (NOT reviewed)')
    prov={
        'module':'ORI03','version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
        'case':case,'analysis_space':'GRE/native R2STAR02',
        'b0_assumption':'Scanner B0 parallel to scanner-world +z; requires acquisition/header confirmation',
        'b0_component_convention':'FSL scaled-voxel axes for vecreg V1; flip x when det GRE affine >0',
        'b0_gre_voxel_axis_unit_vector':b_vox.tolist(),
        'b0_v1_fsl_scaled_voxel_unit_vector':b_fsl.tolist(),
        'gre_affine_determinant':det,'gre_voxel_scales_mm':scales.tolist(),
        'frame_validation':'A single-case ORI01 check validated the candidate vector against registered GRE affine, discrepancy 0.0 degrees at reported precision; other cases derive independently using same documented convention',
        'vector_method':'FSL vecreg b1000 output; 3-component vector branch uses dyadic eigenvector extraction, affine F * V, normalization (vecreg.cc)',
        'vector_registration_accuracy':'not established by coordinate check; formal anatomical QC pending',
        'angle_formula':'degrees(acos(abs(dot(V1,B0))/(norm(V1)*norm(B0))))',
        'bin_edges_degrees':EDGES.tolist(),'last_bin_excludes_exact_90':True,
        'normalization':'each bin mean / unweighted mean of all 18 bin means; requires all bins populated',
        'fa_threshold':fa_threshold,'minimum_v1_norm_strict_gt':0.5,
        'r2_positive_and_production_fit_mask':True,
        'selections':{'FA03':'FA>=0.3; no anatomical WM','WM08':'WM P>=0.8; no FA cutoff',
                      'WM09':'WM P>=0.9; no FA cutoff','WM09_FA03':'WM P>=0.9 and FA>=0.3'},
        'support_counts':{'valid_r2':int(np.count_nonzero(rr)),'valid_v1':int(np.count_nonzero(vr)),
                          'common':int(np.count_nonzero(valid))},
        'files':{key:{'relative_path':str(path.relative_to(root)), 'sha256':sha256(path)} for key,path in paths.items()},
        'anatomical_qc_status':'pending manual review','inputs_modified':False,
    }
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'ori03_bins.csv',bin_rows)
    write_csv(out/'ori03_summary.csv',sum_rows)
    fig.savefig(out/'ori03_r2star_theta.png',dpi=180)
    figqc.savefig(out/'ori03_anatomical_qc.png',dpi=180)
    plt.close(fig);plt.close(figqc)
    (out/'ori03_provenance.json').write_text(json.dumps(prov,indent=2)+'\n',encoding='utf-8')
    return bin_rows, sum_rows


def load_existing_case(root: Path, case: str, threshold: float):
    """Read a completed v0.2.x case without overwriting or recomputing it."""
    out = root/'derivatives'/case/'orientation/ori03'
    names = ('ori03_bins.csv','ori03_summary.csv','ori03_r2star_theta.png',
             'ori03_provenance.json','ori03_anatomical_qc.png')
    if not any((out/n).exists() for n in names):
        return None
    missing = [n for n in names if not (out/n).is_file() or (out/n).stat().st_size == 0]
    if missing:
        raise RuntimeError(f'{case}: partial ORI03 outputs; missing/empty {missing}')
    with (out/'ori03_provenance.json').open(encoding='utf-8') as f:
        prov = json.load(f)
    if prov.get('module') != 'ORI03' or prov.get('case') != case:
        raise RuntimeError(f'{case}: provenance module/case mismatch')
    if prov.get('version') not in ('0.2.0','0.2.1','0.2.2'):
        raise RuntimeError(f'{case}: unexpected ORI03 provenance version {prov.get("version")}')
    if abs(float(prov.get('fa_threshold', -99)) - threshold) > 1e-12:
        raise RuntimeError(f'{case}: existing FA threshold differs from requested {threshold}')
    if prov.get('b0_component_convention') != 'FSL scaled-voxel axes for vecreg V1; flip x when det GRE affine >0':
        raise RuntimeError(f'{case}: B0 convention mismatch')
    for key, desc in prov.get('files', {}).items():
        path = root / desc['relative_path']
        if not path.is_file() or sha256(path) != desc['sha256']:
            raise RuntimeError(f'{case}: input {key} differs from recorded provenance')
    if set(prov.get('files', {})) != set(paths_for(root,case)):
        raise RuntimeError(f'{case}: provenance lacks expected input files')
    with (out/'ori03_bins.csv').open(newline='',encoding='utf-8') as f:
        bins = list(csv.DictReader(f))
    with (out/'ori03_summary.csv').open(newline='',encoding='utf-8') as f:
        summary = list(csv.DictReader(f))
    if len(bins) != 72 or len(summary) != 4:
        raise RuntimeError(f'{case}: expected 72 bin rows/4 summary rows, got {len(bins)}/{len(summary)}')
    expected = {(label,k) for label in SELECTIONS for k in range(1,19)}
    found = {(r['selection'],int(r['bin_index'])) for r in bins}
    if found != expected or any(r['case'] != case for r in bins):
        raise RuntimeError(f'{case}: invalid bin identities')
    if {r['selection'] for r in summary} != set(SELECTIONS) or any(r['case'] != case for r in summary):
        raise RuntimeError(f'{case}: invalid summary identities')
    print(f'{case}: REUSE validated existing ORI03 outputs (no overwrite)')
    return bins, summary


def aggregate_curve_rows(all_bins):
    """Aggregate 18 bin means per selection with equal case weighting.

    Coerce bin indices from freshly calculated (int) and reloaded CSV (str)
    rows to the same type, and require unique case/bin identities.
    """
    seen = set()
    for r in all_bins:
        identity = (str(r['case']), str(r['selection']), int(r['bin_index']))
        if identity in seen:
            raise ValueError(f'Duplicate ORI03 case/bin identity: {identity}')
        seen.add(identity)
    curves = []
    for label in SELECTIONS:
        for k in range(1, 19):
            rows = [r for r in all_bins
                    if r['selection'] == label and int(r['bin_index']) == k]
            values = np.array([float(r['mean_r2star_s_1']) for r in rows
                               if r['mean_r2star_s_1'] != ''], dtype=float)
            normalized = np.array([float(r['normalized_legacy']) for r in rows
                                   if r['normalized_legacy'] != ''], dtype=float)
            if not (np.isfinite(values).all() and np.isfinite(normalized).all()):
                raise ValueError(f'Non-finite cohort data for {label} bin {k}')
            curves.append(dict(
                selection=label, bin_index=k, angle_center_deg=(k-.5)*5,
                n_cases=int(len(values)),
                mean_r2star_s_1=numeric(np.mean(values)) if len(values) else '',
                sd_between_cases_s_1=numeric(np.std(values,ddof=1)) if len(values)>1 else '',
                sem_between_cases_s_1=numeric(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else '',
                n_normalized_cases=int(len(normalized)),
                mean_normalized_legacy=numeric(np.mean(normalized)) if len(normalized) else '',
                sd_normalized_between_cases=numeric(np.std(normalized,ddof=1)) if len(normalized)>1 else ''
            ))
    return curves


def cohort(root: Path, cases: list[str], force: bool, threshold: float, strict: bool):
    all_bins=[]; all_sum=[]; errors=[]
    for case in cases:
        try:
            existing = None if force else load_existing_case(root,case,threshold)
            bins,summary = existing if existing is not None else run_case(root,case,force,threshold)
            all_bins.extend(bins);all_sum.extend(summary)
        except Exception as e:
            errors.append({'case':case,'error':f'{type(e).__name__}: {e}'})
            print(f'[FAILED] {case}: {e}',file=sys.stderr)
            if strict: raise
    if not all_bins:
        raise RuntimeError('No successful ORI03 cases; no cohort outputs created')
    sdir=root/'derivatives/summary/orientation/ori03'
    output_names=('ori03_cohort_bins.csv','ori03_cohort_cases.csv','ori03_cohort_curve.csv','ori03_cohort_provenance.json')
    if not force and any((sdir/n).exists() for n in output_names):
        raise FileExistsError('Cohort outputs already exist; use --force to replace')
    curves=aggregate_curve_rows(all_bins)
    sdir.mkdir(parents=True,exist_ok=True)
    write_csv(sdir/output_names[0],all_bins)
    write_csv(sdir/output_names[1],all_sum)
    write_csv(sdir/output_names[2],curves)
    (sdir/output_names[3]).write_text(json.dumps(dict(module='ORI03',version=VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),requested_cases=cases,
        successful_cases=sorted({r['case'] for r in all_sum}),failures=errors,
        cohort_statistic='equal-case arithmetic mean of per-case bin means; SD/SEM across cases',
        aggregation_fix='v0.2.2: robust int conversion of CSV-reloaded bin_index; duplicate identity check',
        anatomical_qc='pending manual acceptance per case',
        policy='Partial cohorts must not be interpreted as complete 10-case analysis'),indent=2)+'\n')
    print(f'Wrote cohort summary to {sdir} ({len(set(r["case"] for r in all_sum))}/{len(cases)} cases successful)')
    if errors: print('WARNING: INCOMPLETE COHORT, inspect provenance failures',file=sys.stderr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    grp=p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--case',help='Single in-situ case ID')
    grp.add_argument('--all',action='store_true',help='All active in-situ cases via APF registry')
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--fa-threshold',type=float,default=0.3)
    p.add_argument('--force',action='store_true',help='Overwrite existing ORI03 outputs deliberately')
    p.add_argument('--preflight',action='store_true',help='List missing inputs and exit WITHOUT creating outputs')
    p.add_argument('--allow-partial',action='store_true',help='For --all only, record incomplete cohort rather than abort after failure')
    a=p.parse_args()
    if not (0<=a.fa_threshold<=1):p.error('FA threshold must be between 0 and 1')
    root=a.root.expanduser().resolve()
    cases=get_cases(a,root)
    if a.preflight:
        any_missing=False
        for c in cases:
            missing=[key for key,path in paths_for(root,c).items() if not path.is_file()]
            print(f'{c}: '+('READY' if not missing else 'MISSING '+', '.join(missing)))
            any_missing |= bool(missing)
        sys.exit(2 if any_missing else 0)
    if a.all:
        missing_by_case={c:[key for key,path in paths_for(root,c).items() if not path.is_file()] for c in cases}
        missing_by_case={c:keys for c,keys in missing_by_case.items() if keys}
        if missing_by_case and not a.allow_partial:
            for c,keys in missing_by_case.items():
                print(f'{c}: missing {", ".join(keys)}',file=sys.stderr)
            raise RuntimeError('Full cohort blocked: generate missing inputs first, or explicitly pass --allow-partial')
        sdir=root/'derivatives/summary/orientation/ori03'
        if not a.force and any((sdir/n).exists() for n in ('ori03_cohort_bins.csv','ori03_cohort_cases.csv','ori03_cohort_curve.csv','ori03_cohort_provenance.json')):
            raise FileExistsError('Cohort summary already exists; use --force to replace')
        cohort(root,cases,a.force,a.fa_threshold,strict=not a.allow_partial)
    else:
        run_case(root,cases[0],a.force,a.fa_threshold)

if __name__=='__main__': main()
