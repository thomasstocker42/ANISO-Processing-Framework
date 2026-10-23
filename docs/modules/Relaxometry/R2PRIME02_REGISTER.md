# R2PRIME02 — TSE-to-GRE registration and R2app resampling

**Status:** Finalized

**Production implementation:** R2PRIME02 v0.1.0

**Validation implementation:** R2PRIME02b v0.1.0

## Purpose

R2PRIME02 brings the high-resolution dual-echo apparent transverse
relaxation-rate map into the native GRE geometry required for downstream
R2-prime construction and susceptibility source separation.

The production target quantity for the subsequent module is

\[
R_2' = R_2^* - R_{2,\mathrm{app}}^{GRE}.
\]

R2PRIME02 performs only spatial registration and quantitative resampling.
It does not calculate R2-prime.

## Prerequisites

- R2PRIME01 — authoritative input inventory and geometry validation
- RLX01 — authoritative dual-TSE acquisition selection
- RLX05a — native dual-TSE brain mask
- RLX05 — high-resolution dual-echo R2app and fit-valid mask
- GRE02b — GRE echo-1 magnitude and GRE brain mask
- R2STAR02 — production R2star map and downstream GRE geometry

## Production registration contract

### Moving anatomical image

RLX01-selected dual-echo TSE echo 1.

This image remains in the native high-resolution dual-TSE geometry.

### Fixed anatomical image

GRE02b magnitude echo 1.

This image defines the native GRE target geometry and is spatially
identical to the downstream R2STAR02 R2star map.

### Registration method

Registration is performed with FSL FLIRT.

Production settings:

- direction: dual-TSE -> GRE
- transform: rigid
- degrees of freedom: 6
- cost function: normalized mutual information (`normmi`)
- registration interpolation: trilinear
- output geometry: GRE02b echo-1 grid

Rigid registration was selected because the acquisitions represent the
same postmortem in-situ brain and the objective is alignment rather than
allowing scale, shear, or nonlinear anatomical deformation.

Normalized mutual information was selected because the moving TSE and
fixed GRE images have different image contrast.

## Quantitative R2app resampling

The RLX05 R2app map contains non-finite values outside its valid fitting
support. Direct interpolation of such a map can propagate non-finite
values and bias values near support boundaries.

R2PRIME02 therefore uses mask-normalized trilinear interpolation.

Let

- \(R\) denote the native RLX05 R2app map,
- \(M\) denote the binary RLX05 fit-valid mask,
- \(T\) denote the rigid TSE-to-GRE transform.

The GRE-space R2app map is calculated as

\[
R_{2,\mathrm{app}}^{GRE}
=
\frac{T(MR)}{T(M)}.
\]

Implementation:

1. replace invalid R2app voxels by zero in the interpolation numerator;
2. transform the masked numerator \(MR\) using trilinear interpolation;
3. transform the native fit-valid mask using trilinear interpolation;
4. divide the transformed numerator by the transformed mask weight;
5. retain only voxels whose transformed support weight is at least 0.5;
6. assign NaN outside the accepted support.

The continuous transformed support-weight image is retained as an
auditable intermediate product.

## Support threshold

The production support threshold is

\[
T(M) \ge 0.5.
\]

A threshold of 0.5 corresponds to a majority-support criterion after
trilinear interpolation of the binary native validity mask.

This threshold avoids accepting target voxels supported only weakly by
valid native R2app data while avoiding unnecessary erosion associated
with substantially more conservative thresholds.

Cohort validation showed that the threshold preserved support volume
very closely across all 10 cases.

## Production outputs

Per case:

`derivatives/<case>/relaxometry/r2prime/r2prime02_register/`

including:

- `r2prime02_<case>_TSEe1_to_GREe1.mat`
- `r2prime02_<case>_TSEe1_in_GRE.nii.gz`
- `r2prime02_<case>_valid_weight_in_GRE.nii.gz`
- `r2prime02_<case>_valid_mask_in_GRE.nii.gz`
- `r2prime02_<case>_R2app_in_GRE_s-1.nii.gz`
- `r2prime02_<case>_registration_qc.png`
- `r2prime02_<case>_metadata.json`
- `r2prime02_<case>_summary.csv`

Cohort outputs:

`derivatives/summary/relaxometry/r2prime/r2prime02_register/`

including:

- `r2prime02_cohort_summary.csv`
- `r2prime02_cohort_metadata.json`

## Registration QC

R2PRIME02 generates a per-case registration QC image showing:

- GRE echo 1 with registered TSE echo 1 in axial, coronal, and sagittal
  views;
- GRE echo 1 with the final transformed R2app support.

The QC figure is used to identify gross registration failures, including
incorrect translation, incorrect rotation, left-right inversion, or
obvious support displacement.

## R2PRIME02b validation

R2PRIME02b provides a dedicated validation layer for the production
registration.

R2PRIME02b is validation-only. It consumes the frozen R2PRIME02
transform and production support outputs but does not modify or replace
the production GRE-space R2app image.

Its purpose is to distinguish reduced final GRE-space R2app coverage
from three potential sources:

1. incomplete anatomical overlap between the native dual-TSE acquisition
   and the GRE-defined brain;
2. voxels rejected by the RLX05 dual-echo fit-valid criteria;
3. support change introduced by TSE-to-GRE resampling.

## Coverage metrics

### TSE anatomical coverage in GRE

The RLX05a native TSE brain mask is transformed using the exact
R2PRIME02 rigid FLIRT matrix with trilinear interpolation and threshold
0.5.

The anatomical coverage metric is

\[
100
\frac{
|M_{\mathrm{TSE, GRE}} \cap M_{\mathrm{GRE}}|
}{
|M_{\mathrm{GRE}}|
}.
\]

This measures how much of the GRE-defined brain is anatomically covered
by the transformed TSE acquisition support.

### Native fit-valid coverage

The native fit-valid coverage is

\[
100
\frac{
|M_{\mathrm{fit}} \cap M_{\mathrm{TSE}}|
}{
|M_{\mathrm{TSE}}|
}.
\]

This measures the fraction of the native RLX05a brain mask retained by
the RLX05 dual-echo validity criteria.

### Final R2app coverage in GRE

The final GRE-space R2app coverage is

\[
100
\frac{
|M_{\mathrm{R2app,GRE}} \cap M_{\mathrm{GRE}}|
}{
|M_{\mathrm{GRE}}|
}.
\]

This is the usable fraction of the GRE-defined brain for downstream
R2-prime calculation.

### Support change during resampling

R2PRIME02b additionally reports

\[
100
\frac{
|M_{\mathrm{R2app,GRE}}| - |M_{\mathrm{fit}}|
}{
|M_{\mathrm{fit}}|
}.
\]

This quantifies whether spatial resampling itself materially expands or
erodes the valid support.

### R2app support outside GRE brain

The fraction of final R2app-support voxels outside the GRE02b brain mask
is also recorded as a spatial-overlap QC metric.

## R2PRIME02b outputs

Per case:

`derivatives/<case>/relaxometry/r2prime/r2prime02b_validation/`

including:

- `r2prime02b_<case>_TSE_brain_weight_in_GRE.nii.gz`
- `r2prime02b_<case>_TSE_brain_mask_in_GRE.nii.gz`
- `r2prime02b_<case>_coverage_qc.png`
- `r2prime02b_<case>_metadata.json`
- `r2prime02b_<case>_summary.csv`

Cohort outputs:

`derivatives/summary/relaxometry/r2prime/r2prime02b_validation/`

including:

- `r2prime02b_cohort_summary.csv`
- `r2prime02b_cohort_metadata.json`

## R2PRIME02b QC figure

The coverage-validation figure contains six coronal panels:

Native TSE geometry:

1. native TSE echo 1;
2. native TSE echo 1 with RLX05a brain mask;
3. native TSE echo 1 with RLX05 fit-valid mask.

GRE geometry:

4. GRE echo 1;
5. GRE echo 1 with transformed TSE brain mask;
6. GRE echo 1 with final R2app-valid mask.

The figure title reports:

- TSE anatomical coverage in GRE;
- native fit-valid coverage;
- final R2app coverage in GRE.

## Downstream contract

R2PRIME03 consumes:

- R2PRIME02 GRE-space R2app;
- R2PRIME02 final R2app-valid support;
- R2STAR02 R2star;
- an explicitly defined common valid support.

R2PRIME03 will construct

\[
R_2' = R_2^* - R_{2,\mathrm{app}}^{GRE}.
\]
