# R2STAR02 — log-linear GRE R2* reconstruction

**Status:** Finalized
**Production implementation:** R2STAR02 v0.1.0
**Fit method:** ordinary least squares in log-signal space

## Purpose

R2STAR02 reconstructs voxel-wise R2* maps from the 12-echo GRE magnitude
series.

The module replaces the legacy s19 magnitude-based R2* implementation while
preserving its scientific fitting model and adding APF-standardized input
handling, geometry checks, explicit fit masks, structured outputs and
provenance.

## Script

`scripts/relaxometry/r2star/02_fit_loglinear.py`

## Prerequisites

- R2STAR01 — GRE R2* input inventory;
- GRE02b — finalized non-eroded GRE brain mask;
- 12 GRE magnitude echoes with valid echo-time metadata.

## Signal model

The magnitude signal is modeled as

\[
S(TE)=S_0\exp(-R_2^*TE).
\]

Taking the natural logarithm yields

\[
\ln S(TE)=\ln S_0-R_2^*TE.
\]

For each voxel, ordinary least-squares linear regression is performed in
log-signal space,

\[
y=a+b\,TE,
\]

with

\[
R_2^*=-b
\]

and

\[
S_0=\exp(a).
\]

The fit is unweighted. No magnitude- or SNR-dependent weighting is applied.

## Echo-time handling

Echo times are read from the individual GRE JSON sidecars.

The current 12-echo acquisition uses:

`7.01, 11.04, 15.59, 19.66, 23.73, 27.80, 31.87, 35.94, 40.01, 44.08, 48.15, 52.21 ms`

Echo times are converted to seconds for estimation of R2* in `s^-1`.

## Processing support

The anatomical processing domain is the finalized GRE02b non-eroded brain
mask:

`derivatives/<case>/GRE/02b_brain_mask/<case>_GRE02b_brain_mask.nii.gz`

The GRE magnitude volumes and brain mask must have matching image dimensions
and compatible affines.

## Valid observations

For a given voxel, an echo is considered usable when its magnitude signal is:

- finite;
- strictly greater than zero;
- inside the GRE02b brain mask.

A minimum of 6 valid echoes is required before regression is attempted.

## Fit masks

Three processing domains are distinguished.

### Brain mask

The GRE02b non-eroded brain mask defines the anatomical support.

### Candidate fit mask

A voxel enters the candidate fit mask when at least 6 finite positive GRE
magnitude samples are available.

### Final fit mask

A candidate voxel is retained in the final fit mask only when:

- the regression denominator is numerically valid;
- the fitted parameters are finite;
- the fitted relaxation rate satisfies

\[
0<R_2^*\leq1000~\mathrm{s^{-1}}.
\]

Downstream analyses should use the explicit final fit mask rather than
interpreting zero-valued map voxels as valid measurements.

## Derived quantities

R2STAR02 produces:

- R2* in `s^-1`;
- R2* in `ms^-1`;
- T2* in ms;
- fitted S0;
- log-domain fit RMSE;
- number of valid echoes;
- candidate fit mask;
- final fit mask.

T2* is calculated as

\[
T_2^*=\frac{1000}{R_2^*}
\]

when R2* is expressed in `s^-1`.

## Residual metric

Fit residual is quantified in log-signal space as

\[
\mathrm{RMSE}_{\log S}
=
\sqrt{
\frac{1}{N}
\sum_i
\left[
\ln S_i-(a+bTE_i)
\right]^2
}.
\]

The residual map is retained as a quality-control measure.

No additional RMSE threshold is applied during production reconstruction.
Historical RMSE thresholds used in downstream vesselness analyses are not
part of R2STAR02.

## Per-case outputs

Production directory:

`derivatives/<case>/relaxometry/r2star/r2star02_loglinear/`

Outputs:

- `r2star02_<case>_r2star_s-1.nii.gz`
- `r2star02_<case>_r2star_ms-1.nii.gz`
- `r2star02_<case>_t2star_ms.nii.gz`
- `r2star02_<case>_s0.nii.gz`
- `r2star02_<case>_log_rmse.nii.gz`
- `r2star02_<case>_n_echo_used.nii.gz`
- `r2star02_<case>_candidate_fit_mask.nii.gz`
- `r2star02_<case>_fit_mask.nii.gz`
- `r2star02_<case>_qc.json`

## Cohort outputs

Cohort summaries are written to:

`derivatives/summary/relaxometry/r2star/r2star02_loglinear/`

Files:

- `r2star02_cohort_summary.csv`
- `r2star02_cohort_summary.json`

## Legacy implementation validation

The APF rewrite was compared voxelwise against the historical s19
implementation in one validation case.

The validation examined:

- image dimensions;
- affine agreement;
- reconstruction support;
- voxel-wise R2* differences;
- RMSE between implementations;
- Pearson correlation.

The result demonstrated numerical equivalence of the APF and legacy fitting
implementations.

## Production decision

R2STAR02 v0.1.0 is the finalized production R2* implementation.

The log-linear method is retained because:

- it reproduces the established legacy implementation;
- it is deterministic;
- it provides complete standardized processing of all 10 in-situ cases;
- no current evidence demonstrates a material downstream advantage from a
  more complex fitting model.

A direct magnitude-domain mono-exponential fit will be evaluated separately in
R2STAR03 as a validation analysis. It does not replace R2STAR02 unless that
comparison demonstrates a scientifically meaningful advantage.

## Downstream use

R2STAR02 provides the GRE-space R2* input required for later calculation of

\[
R_2'=R_2^*-R_2.
\]

The R2 data must first be brought into the appropriate GRE analysis space
before calculation of R2' for susceptibility source separation.
