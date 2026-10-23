#!/usr/bin/env python3
"""MP2RAGE05: audit native-header alignment against GRE, without optimization.

Outputs are preliminary until independent visual QC. No legacy pilot writes.
Run with ./apf scripts/t1/27_mp2rage_to_gre_registration.py CASE|--all-mp2rage-ready
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from apf_paths import get_paths
from apf_case_selection import add_case_selection_args, resolve_cases, print_selected_cases

VERSION = '0.1.0'
MODULE = 'mp2rage'
STAGE = 'mp2rage05'


def discover(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(p for p in directory.glob(pattern) if p.is_file())
    if len(matches) != 1:
        raise RuntimeError(f'{label}: expected exactly one match in {directory} for {pattern}; found {len(matches)}: {matches}')
    return matches[0]


def sources(paths, case):
    # Discovery is confined to known legacy *stages*, not fixed complete filenames.
    seg = paths.case_derivatives_dir(case) / 'tissue_segmentation' / 'MP2RAGE_s26'
    gre = paths.case_derivatives_dir(case) / 'GRE' / 'mag_masked'
    moving = discover(seg, '*MP2RAGE_UNI_hdbet.nii.gz', 'brain-extracted UNI')
    fixed = discover(gre, '*GRE_e7_masked_by_*.nii.gz', 'GRE reference')
    return moving, fixed


def process(paths, case):
    moving_path, ref_path = sources(paths, case)
    moving, ref = nib.load(str(moving_path)), nib.load(str(ref_path))
    if len(moving.shape) != 3 or len(ref.shape) != 3:
        raise ValueError('Expected 3D MP2RAGE and GRE images')
    if not np.isfinite(moving.affine).all() or not np.isfinite(ref.affine).all():
        raise ValueError('Invalid affine')
    if min(abs(np.linalg.det(x.affine[:3, :3])) for x in (moving, ref)) < 1e-8:
        raise ValueError('Singular affine')

    outdir = paths.case_stage_dir(case, MODULE, STAGE)
    target = outdir / 'uni_header_in_gre.nii.gz'
    manifest = outdir / 'provenance.json'
    if target.exists() or manifest.exists():
        if not (target.is_file() and manifest.is_file()):
            raise RuntimeError('Incomplete existing stage; inspect manually')
        meta = json.loads(manifest.read_text())
        if meta.get('moving') != str(moving_path) or meta.get('reference') != str(ref_path):
            raise RuntimeError('Existing stage inputs differ; do not silently reuse')
        current = nib.load(str(target))
        if current.shape != ref.shape or not np.allclose(current.affine, ref.affine, atol=1e-5):
            raise RuntimeError('Existing output differs from GRE geometry')
        return 'existing', str(target)

    result = resample_from_to(moving, (ref.shape, ref.affine), order=1)
    vals = np.asarray(result.dataobj)
    if not np.isfinite(vals).all() or np.count_nonzero(vals) == 0:
        raise RuntimeError('Empty/nonfinite registered MP2RAGE')
    outdir.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(target))
    meta = {
        'module': 'MP2RAGE05', 'version': VERSION, 'case': case,
        'method': 'native NIfTI affine / physical-world coordinates; no fitted transform',
        'moving': str(moving_path), 'reference': str(ref_path),
        'output': str(target), 'interpolation': 'trilinear',
        'registration_qc': 'PENDING_VISUAL',
        'moving_affine': moving.affine.tolist(), 'reference_affine': ref.affine.tolist(),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(meta, indent=2) + '\n')
    return 'created', str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_selection_args(parser)
    args = parser.parse_args()
    paths = get_paths()
    cases = resolve_cases(args, paths.ins_registry)
    print_selected_cases(cases)
    rows = []
    for case in cases:
        try:
            status, out = process(paths, case)
            row = {'case': case, 'status': status, 'output': out, 'error': ''}
        except Exception as exc:
            row = {'case': case, 'status': 'failed', 'output': '', 'error': str(exc)}
        rows.append(row)
        print(row, flush=True)
    summary = paths.summary_module_dir(MODULE) / f'{STAGE}_summary.csv'
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Summary:', summary)
    if any(x['status'] == 'failed' for x in rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
