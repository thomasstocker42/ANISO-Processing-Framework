# CHI03b — MEDI susceptibility source separation

## Purpose

CHI03b performs the production susceptibility source-separation reconstruction
using the SNU-LIST `chi_sep_MEDI` implementation.

## Inputs

Frozen CHI02 solver package:

- local field in ppm;
- raw R2-prime in s^-1;
- 12-echo magnitude series;
- common solver mask;
- acquisition parameters.

Additional inputs:

- GRE02 phase for `Preprocessing4Phase` noise weighting;
- R2STAR02 R2-star map for CSF-mask extraction;
- GRE06c STI-iLSQR QSM supplied through `option_data.qsm`.

## Solver-boundary R2-prime policy

The upstream R2-prime map remains unchanged.

At the MEDI solver boundary only:

1. non-finite R2-prime values are set to zero;
2. negative R2-prime values are set to zero;
3. the production solver mask is applied.

This preserves the quantitative upstream derivative while satisfying the
non-negative relaxation-rate assumption of the source-separation model.

## Reconstruction parameters

Production parameters:

- SNU-LIST version: 1.2.2
- Dr: 137 Hz/ppm
- lambda: 1
- lambda_CSF: 1
- case-specific B0 direction
- case-specific voxel dimensions
- phase-noise weighting from `Preprocessing4Phase`

## CSF reference

The default MEDI `extract_CSF` routine applies a 10-mm SMV erosion before
low-R2-star CSF identification. In this postmortem cohort, the default erosion
produced an empty CSF mask in 8/10 cases.

The optional erosion was therefore disabled while retaining the standard
R2-star threshold of 5 s^-1:

`extract_CSF(r2star, mask, voxel_size, 0, 5)`

This yielded non-empty CSF masks in all ten cases.

Visual QC subsequently showed that the masks were small and, in some cases,
not unequivocally specific to ventricular CSF. The influence of this
uncertainty was therefore evaluated separately in CHI03c.

## Outputs

Per case:

- chi-para in ppm;
- chi-dia in ppm;
- chi-total in ppm;
- CSF reference mask;
- reconstruction QC JSON;
- execution log;
- provenance manifest.

## Status

Production reconstruction complete in 10/10 in-situ cases.

The preceding chi-sep-iLSQR branch was retained as a diagnostic/rejected
implementation because both the QSM-CI and official SNU-LIST implementations
failed during the NNLS stage after successful LSQR initialization and
streaking-artifact estimation.
