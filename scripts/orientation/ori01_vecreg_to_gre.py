#!/usr/bin/env python3
"""ORI01 v0.1.0: reproducible FSL vecreg b1000 V1->GRE, with QC/provenance.

Existing outputs are NEVER changed by default. Validation records are added
only after verification. To regenerate an existing V1, use --force explicitly.
Uses APF's authoritative in-situ registry for --all.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

VERSION = '0.1.0'


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def paths_for(root: Path, case: str) -> dict[str, Path]:
    base = root / 'derivatives' / case
    outdir = base / 'DWI/orientation_validation'
    return {
        'native_v1': base / f'DWI/dtifit/b1000/dwi05_{case}_multi_b_N30_b1000_wls_V1.nii.gz',
        'matrix': base / f'DWI/registration/dwi06/dwi06_{case}_b0_to_GRE.mat',
        'gre_reference': base / f'GRE/mag_masked/s12_{case}_GRE_e7_masked_by_s11b_e1_hdbet_cpu_notta.nii.gz',
        'v1_gre': outdir / f'ori01_{case}_b1000_V1_vecreg.nii.gz',
        'provenance': outdir / f'ori01_{case}_b1000_vecreg_provenance.json',
    }


def resolve(root: Path, a: argparse.Namespace) -> list[str]:
    if a.case:
        return [a.case]
    infra = root / 'scripts/infrastructure'
    if str(infra) not in sys.path:
        sys.path.insert(0, str(infra))
    from apf_paths import get_paths
    from apf_case_selection import resolve_cases
    opts = argparse.Namespace(case=None, cases=None, all=True,
        all_dwi_ready=False, all_gre_ready=False, all_mp2rage_ready=False)
    return list(resolve_cases(opts, get_paths().ins_registry))


def read_matrix(path: Path) -> np.ndarray:
    mat = np.loadtxt(path)
    if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
        raise ValueError(f'Invalid FLIRT matrix: {path}')
    if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Last matrix row is not homogeneous [0,0,0,1]')
    if abs(float(np.linalg.det(mat[:3, :3]))) < 1e-8:
        raise ValueError('FLIRT matrix is singular')
    return mat


def validate(p: dict[str, Path], mat: np.ndarray) -> dict:
    native = nib.load(str(p['native_v1']))
    ref = nib.load(str(p['gre_reference']))
    registered = nib.load(str(p['v1_gre']))
    if len(native.shape) != 4 or native.shape[3] != 3:
        raise ValueError(f'Native V1 must have 3 components: {native.shape}')
    if len(ref.shape) != 3:
        raise ValueError(f'GRE reference must be 3D: {ref.shape}')
    if registered.shape != ref.shape + (3,):
        raise ValueError(f'Registered V1 shape {registered.shape}, expected {ref.shape + (3,)}')
    affine_diff = float(np.max(np.abs(registered.affine - ref.affine)))
    if affine_diff > 1e-3:
        raise ValueError(f'Vecreg output does not match GRE affine (max abs diff {affine_diff:.6g})')
    v = np.asarray(registered.dataobj, dtype=np.float32)
    finite_components = np.all(np.isfinite(v), axis=-1)
    norm = np.linalg.norm(v.astype(np.float64), axis=-1)
    valid = finite_components & (norm > 0.5)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError('Registered V1 contains no valid nonzero vectors')
    q = np.percentile(norm[valid], [0, 1, 50, 99, 100])
    deviation = np.abs(norm[valid] - 1.)
    # >0.5 support must be essentially unit length in vecreg output.
    if float(np.percentile(deviation, 99)) > 0.02:
        raise ValueError('More than 1% of valid registered V1 vectors deviate from unit norm by >0.02')
    return {
        'native_shape': list(native.shape),
        'gre_shape': list(ref.shape),
        'registered_shape': list(registered.shape),
        'affine_max_abs_difference_mm': affine_diff,
        'registered_v1_valid_norm_gt_0p5_count': n_valid,
        'registered_v1_valid_fraction_of_grid': n_valid / int(np.prod(ref.shape)),
        'registered_v1_norm_percentiles_0_1_50_99_100': [float(x) for x in q],
        'registered_v1_norm_abs_error_p99': float(np.percentile(deviation, 99)),
        'flirt_linear_matrix_determinant': float(np.linalg.det(mat[:3, :3])),
        'geometry_qc': 'passed',
        'fibre_direction_anatomical_qc': 'pending manual review',
    }


def run_case(root: Path, case: str, a: argparse.Namespace) -> dict:
    p = paths_for(root, case)
    missing = [k for k in ('native_v1', 'matrix', 'gre_reference') if not p[k].is_file()]
    if missing:
        raise FileNotFoundError(f'{case}: missing {missing}')
    mat = read_matrix(p['matrix'])
    exists = p['v1_gre'].is_file()
    if a.preflight:
        status = 'EXISTING' if exists else 'GENERATE'
        print(f'{case}: {status} (native/matrix/GRE ready)')
        return {'case': case, 'status': status}
    if exists and a.force:
        raise RuntimeError('Safe policy: --force cannot overwrite an existing V1. Move/archive it manually before regenerating.')
    if not exists:
        vecreg = shutil.which('vecreg')
        if not vecreg:
            raise FileNotFoundError('vecreg not found on PATH')
        p['v1_gre'].parent.mkdir(parents=True, exist_ok=True)
        # Temporary distinct path prevents incomplete authoritative output.
        temporary = p['v1_gre'].with_name(p['v1_gre'].name.replace('.nii.gz', '.staging.nii.gz'))
        if temporary.exists():
            raise FileExistsError(f'Staging file exists, inspect/remove manually: {temporary}')
        cmd = [vecreg, '-i', str(p['native_v1']), '-r', str(p['gre_reference']),
               '-t', str(p['matrix']), '-o', str(temporary)]
        print(f'{case}: generating via FSL vecreg')
        result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
        if result.returncode != 0 or not temporary.is_file():
            raise RuntimeError(f'vecreg failed {result.returncode}: {result.stdout}\n{result.stderr}\n'
                               f'If staging file exists, inspect manually: {temporary}')
        check_p = dict(p, v1_gre=temporary)
        try:
            validate(check_p, mat)
        except Exception:
            raise RuntimeError(f'Generated staging file failed QC: {temporary}; preserved for inspection')
        temporary.rename(p['v1_gre'])
        status = 'generated'
    else:
        status = 'validated_existing'
        print(f'{case}: validating existing V1 without modification')

    metrics = validate(p, mat)
    provenance = {
        'module': 'ORI01', 'version': VERSION, 'case': case,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_space': 'native GRE reference grid',
        'status': status, 'output_image_was_overwritten': False,
        'source': 'FSL vecreg -i native_V1 -r GRE_reference -t DWI06_b0_to_GRE -o output',
        'vector_algorithm': 'FSL vecreg 3-component path: interpolate dyadic tensor, extract principal eigenvector, apply affine 3x3 F and normalize',
        'registration_method': 'DWI06 affine matrix; no additional registration optimized here',
        'coordinate_convention': 'FLIRT/FSL scaled-voxel components; B0 conversion belongs to downstream ORI03',
        'formal_anatomical_qc': 'pending manual review',
        'qc': metrics,
        'inputs': {key: {'relative_path': str(p[key].relative_to(root)),
                          'sha256': file_sha256(p[key])}
                   for key in ('native_v1', 'matrix', 'gre_reference')},
        'output': {'relative_path': str(p['v1_gre'].relative_to(root)),
                   'sha256': file_sha256(p['v1_gre'])},
    }
    # Existing provenance is not rewritten by default.
    if p['provenance'].exists() and not a.replace_provenance:
        print(f'{case}: existing provenance retained (pass --replace-provenance to refresh)')
    else:
        p['provenance'].parent.mkdir(parents=True, exist_ok=True)
        p['provenance'].write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return {'case': case, 'status': status, **metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', help='one active in-situ case identifier')
    group.add_argument('--all', action='store_true', help='APF active in-situ registry')
    parser.add_argument('--preflight', action='store_true', help='check dependencies only, no writes')
    parser.add_argument('--replace-provenance', action='store_true', help='refresh provenance JSON, not V1')
    parser.add_argument('--force', action='store_true', help='safety guard: existing V1 is NEVER overwritten')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cases = resolve(root, args)
    results = []
    failures = 0
    for case in cases:
        try:
            results.append(run_case(root, case, args))
        except Exception as e:
            print(f'{case}: ERROR {type(e).__name__}: {e}', file=sys.stderr)
            failures += 1
    if not args.preflight and results:
        path = root / 'derivatives/summary/orientation/ori01'
        path.mkdir(parents=True, exist_ok=True)
        summary = path / 'ori01_qc_summary.csv'
        if summary.exists() and not args.replace_provenance:
            print(f'Existing cohort summary retained: {summary}')
        else:
            with summary.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0]))
                writer.writeheader(); writer.writerows(results)
            print(f'QC summary written: {summary}')
    print(f'ORI01 completed: {len(results)}/{len(cases)}; failures={failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
