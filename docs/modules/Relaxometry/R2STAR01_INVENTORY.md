# R2STAR01 — GRE R2* input inventory

**Status:** Finalized
**Implementation:** R2STAR01 v0.1.0

## Purpose

R2STAR01 audits the inputs required for voxel-wise R2* reconstruction from
the 12-echo GRE magnitude acquisition.

The module is validation-only. It does not generate quantitative R2* maps.

## Script

`scripts/relaxometry/r2star/01_inventory.py`

## Required inputs

For each in-situ case:

- 12 GRE magnitude echoes;
- corresponding JSON sidecars;
- valid `EchoNumber`;
- valid `EchoTime`;
- finalized GRE02b non-eroded brain mask.

Magnitude input pattern:

`nifti/<case>/GRE/mag/<case>_GRE_12echo_*_e*.nii.gz`

Brain-mask input:

`derivatives/<case>/GRE/02b_brain_mask/<case>_GRE02b_brain_mask.nii.gz`

Echo times are obtained from acquisition metadata and are not hard-coded by
the reconstruction module.

The current acquisition contains 12 echoes at:

`7.01, 11.04, 15.59, 19.66, 23.73, 27.80, 31.87, 35.94, 40.01, 44.08, 48.15, 52.21 ms`

## Validation checks

R2STAR01 verifies:

- exactly one magnitude NIfTI for each expected echo;
- presence and readability of each JSON sidecar;
- agreement between filename echo number and JSON `EchoNumber`;
- finite positive `EchoTime`;
- strictly increasing echo times;
- identical image dimensions across echoes;
- affine consistency across echoes;
- presence of the finalized GRE02b brain mask;
- shape and affine consistency between the GRE images and the brain mask.

Affine comparison uses an absolute numerical tolerance of `1e-5`.

## Case selection

Cases are discovered through the APF case registry with a CSV fallback.

The finalized thesis cohort consists of 10 in-situ cases.

## Outputs

Cohort inventory outputs are written to:

`derivatives/summary/relaxometry/r2star/r2star01_inventory/`

Files:

- `r2star01_inventory.csv`
- `r2star01_inventory.json`

## Production decision

R2STAR01 is finalized.

All quantitative R2* reconstruction is performed downstream by R2STAR02 only
after the required input and geometry checks have passed.
