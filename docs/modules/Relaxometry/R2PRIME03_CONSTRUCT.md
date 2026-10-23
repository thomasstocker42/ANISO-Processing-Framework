# R2PRIME03 — R2-prime construction and QC

**Status:** Finalized

**Production implementation:** R2PRIME03 v0.2.0

## Purpose

R2PRIME03 constructs the transverse relaxation-rate difference required for
subsequent susceptibility source separation:

\[
R_2' = R_2^* - R_{2,\mathrm{app}}^{GRE}.
\]

The module combines the finalized GRE-derived R2star map from R2STAR02 with
the high-resolution dual-echo R2app map registered to GRE space by R2PRIME02.

R2PRIME03 produces the raw derived R2-prime map and quantifies its spatial
support and value distribution. It does not apply any non-negativity
constraint or clipping.

## Prerequisites

- R2STAR02 — standardized GRE R2star reconstruction
- R2PRIME02 — high-resolution dual-echo R2app registered to GRE space
- GRE02b — GRE brain mask for spatial coverage QC

## Input quantities

### R2star

The R2star input is the finalized R2STAR02 log-linear reconstruction:

`derivatives/<case>/relaxometry/r2star/r2star02_loglinear/`

with:

- `r2star02_<case>_r2star_s-1.nii.gz`
- `r2star02_<case>_fit_mask.nii.gz`

The R2STAR02 fit mask defines voxels satisfying the finalized GRE
mono-exponential fitting criteria.

### Registered R2app

The R2app input is the finalized R2PRIME02 GRE-space map:

`derivatives/<case>/relaxometry/r2prime/r2prime02_register/`

with:

- `r2prime02_<case>_R2app_in_GRE_s-1.nii.gz`
- `r2prime02_<case>_valid_mask_in_GRE.nii.gz`

R2PRIME02 provides a mask-normalized trilinearly resampled high-resolution
dual-echo R2app map in the same 1-mm GRE grid as R2STAR02.

## Common support

R2PRIME03 defines the valid common support as

\[
M_{R_2'}
=
M_{R_2^*}
\cap
M_{R_{2,\mathrm{app}}}
\cap
F(R_2^*)
\cap
F(R_{2,\mathrm{app}}),
\]

where:

- \(M_{R_2^*}\) is the R2STAR02 final fit mask;
- \(M_{R_{2,\mathrm{app}}}\) is the R2PRIME02 final valid mask;
- \(F(\cdot)\) denotes finite-valued voxels.

No V-SHARP, QSM, or chi-separation-specific mask is included at this stage.
R2PRIME03 therefore remains a relaxometry-derived preprocessing module.
Any additional intersection with susceptibility-processing support is deferred
to the downstream chi-separation module.

## R2-prime calculation

Within the common support,

\[
R_2'
=
R_2^*
-
R_{2,\mathrm{app}}^{GRE}.
\]

Units are s^-1.

Outside the common support, the R2-prime map is stored as NaN.

## Negative-value policy

R2PRIME03 deliberately retains negative R2-prime values.

No operation such as

\[
R_2' \leftarrow \max(R_2',0)
\]

is applied.

The rationale is to preserve the directly derived quantitative result and to
allow the prevalence and spatial organization of negative values to be
evaluated explicitly.

Whether a downstream chi-separation implementation requires a non-negative
R2-prime input is treated as a separate algorithm-specific preprocessing
decision and is not imposed by R2PRIME03.

## Quantitative QC

For each case, R2PRIME03 reports:

- common-support voxel count;
- common support as a fraction of the GRE02b brain mask;
- median R2star;
- median R2app;
- minimum, 1st, 5th, median, mean, 95th, 99th percentile and maximum R2-prime;
- number and fraction of negative R2-prime voxels;
- number and fraction of exactly zero R2-prime voxels.

A separate binary negative-R2-prime mask is retained for spatial QC.

## Visual QC

Each case produces both PNG and PDF versions of the same six-panel QC figure.

The figure contains:

1. R2star map;
2. GRE-space R2app map;
3. raw R2-prime map;
4. common valid support;
5. negative-R2-prime voxels overlaid on the support;
6. R2-prime value distribution.

The PDF is retained as a high-quality thesis/publication-ready output,
whereas the PNG supports rapid visual QC and browsing.

## Outputs

Per case:

`derivatives/<case>/relaxometry/r2prime/r2prime03_construct/`

including:

- `r2prime03_<case>_common_mask.nii.gz`
- `r2prime03_<case>_R2prime_raw_s-1.nii.gz`
- `r2prime03_<case>_negative_mask.nii.gz`
- `r2prime03_<case>_zero_mask.nii.gz`
- `r2prime03_<case>_qc.png`
- `r2prime03_<case>_qc.pdf`
- `r2prime03_<case>_metadata.json`
- `r2prime03_<case>_summary.csv`

Cohort outputs:

`derivatives/summary/relaxometry/r2prime/r2prime03_construct/`

including:

- `r2prime03_cohort_summary.csv`
- `r2prime03_cohort_metadata.json`

## Downstream contract

R2PRIME03 provides two primary outputs for susceptibility source separation:

1. raw GRE-space R2-prime map;
2. corresponding common relaxometry support.

The downstream chi-separation module must explicitly define:

- the intersection between R2-prime and susceptibility/local-field support;
- whether negative R2-prime values are retained, excluded, or constrained;
- any calibration or scaling required by the selected chi-separation method.

These decisions are intentionally not encoded in R2PRIME03.
