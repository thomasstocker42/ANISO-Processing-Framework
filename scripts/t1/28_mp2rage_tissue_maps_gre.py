#!/usr/bin/env python3
"""MP2RAGE06: resample native FAST PVE classes into GRE header geometry.

Output is preliminary until segmentation + registration visual QC is approved.
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
STAGE = 'mp2rage06'
TISSUES = {'CSF': 0, 'GM': 1, 'WM': 2}


def process(paths, case):
    base = paths.case_derivatives_dir(case)
    input_dir = base / 'tissue_segmentation' / 'MP2RAGE_s26'
    reg_meta = paths.case_stage_dir(case, MODULE, 'mp2rage05') / 'provenance.json'
    if not reg_meta.is_file():
        raise FileNotFoundError('MP2RAGE05 provenance absent; run MP2RAGE05 first')
    reg = json.loads(reg_meta.read_text())
    if reg.get('case') != case or reg.get('method') != 'native NIfTI affine / physical-world coordinates; no fitted transform':
        raise RuntimeError('Unexpected MP2RAGE05 provenance/method')
    ref_path = Path(reg['reference'])
    gre = nib.load(str(ref_path))
    outdir = paths.case_stage_dir(case, MODULE, STAGE)
    inputs = {}
    for tissue, index in TISSUES.items():
        candidates = list(input_dir.glob(f'*MP2RAGE_UNI_FAST_pve_{index}.nii.gz'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one {tissue} probability input, got {candidates}')
        inputs[tissue] = candidates[0]

    outputs = {t: outdir / f'{t.lower()}_prob_gre.nii.gz' for t in TISSUES}
    outputs.update({
        'wm08': outdir / 'wm_mask_p08_gre.nii.gz',
        'wm09': outdir / 'wm_mask_p09_gre.nii.gz',
    })
    manifest = outdir / 'provenance.json'
    existing = [p.exists() for p in [*outputs.values(), manifest]]
    if any(existing):
        if not all(existing):
            raise RuntimeError('Partial outputs exist; inspect before rerun')
        meta = json.loads(manifest.read_text())
        if (meta.get('inputs') != {k: str(v) for k,v in inputs.items()}
                or meta.get('reference') != str(ref_path)):
            raise RuntimeError('Existing provenance does not match current inputs')
        for p in outputs.values():
            image = nib.load(str(p))
            if image.shape != gre.shape or not np.allclose(image.affine, gre.affine, atol=1e-5):
                raise RuntimeError(f'Existing output geometry mismatch: {p}')
        wm = np.asarray(nib.load(str(outputs['WM'])).dataobj)
        return 'existing', int(np.count_nonzero(wm >= 0.8)), int(np.count_nonzero(wm >= 0.9))

    probabilities = {}
    for tissue, path in inputs.items():
        src = nib.load(str(path))
        if len(src.shape) != 3:
            raise ValueError(f'Non-3D {tissue} probability map')
        x = resample_from_to(src, (gre.shape, gre.affine), order=1)
        arr = np.asarray(x.dataobj, dtype=np.float32)
        if not np.isfinite(arr).all() or arr.min() < -0.01 or arr.max() > 1.01:
            raise ValueError(f'Invalid {tissue} probability values')
        probabilities[tissue] = np.clip(arr, 0, 1)
    wm = probabilities['WM']
    if not (np.any(wm >= 0.8) and np.any(wm >= 0.9)):
        raise RuntimeError('Empty thresholded WM mask')
    outdir.mkdir(parents=True, exist_ok=True)

    def save(data, target):
        hdr = gre.header.copy()
        hdr.set_data_dtype(data.dtype)
        nib.save(nib.Nifti1Image(data, gre.affine, hdr), str(target))

    for tissue, arr in probabilities.items():
        save(arr, outputs[tissue])
    save((wm >= 0.8).astype(np.uint8), outputs['wm08'])
    save((wm >= 0.9).astype(np.uint8), outputs['wm09'])
    meta = {
        'module': 'MP2RAGE06', 'version': VERSION, 'case': case,
        'reference': str(ref_path), 'inputs': {k: str(v) for k,v in inputs.items()},
        'outputs': {k: str(v) for k,v in outputs.items()},
        'interpolation': 'trilinear', 'thresholds': [0.8,0.9],
        'wm08_voxels': int(np.count_nonzero(wm >= 0.8)),
        'wm09_voxels': int(np.count_nonzero(wm >= 0.9)),
        'registration_qc': 'PENDING_VISUAL', 'segmentation_qc': 'PENDING_VISUAL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest.write_text(json.dumps(meta, indent=2) + '\n')
    return 'created', meta['wm08_voxels'], meta['wm09_voxels']


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
            status, n08, n09 = process(paths, case)
            row = {'case': case, 'status': status, 'wm08_voxels': n08,
                   'wm09_voxels': n09, 'error': ''}
        except Exception as exc:
            row = {'case': case, 'status': 'failed', 'wm08_voxels': '',
                   'wm09_voxels': '', 'error': str(exc)}
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
