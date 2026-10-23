# GRE06 — QSM reconstruction

**Status:** Finalized

**Production implementation:** GRE06b linear isotropic-TV QSM

**Validation implementations:** GRE06a dipole-kernel/B0 validation; GRE06c STI Suite v3 iLSQR external/reference validation

## Purpose

GRE06 reconstructs quantitative susceptibility maps from the local magnetic
field produced by GRE05b.

The production reconstruction uses linear dipole inversion with isotropic
total-variation regularization.

GRE06 is divided into:

- GRE06a — dipole-kernel and B0-direction validation;
- GRE06b — linear isotropic-TV QSM reconstruction;
- GRE06c — independent QSM reference/comparator using STI Suite v3 iLSQR.

GRE06c is validation-only and does not alter the frozen GRE06b production
parameter.

## Prerequisites

- GRE01 — inventory
- GRE02 — phase preparation and B0-direction provenance
- GRE02b — standardized GRE brain mask
- GRE03 — ROMEO phase unwrapping
- GRE04b — finalized parity-aware total-field estimation
- GRE05a — consensus BFR mask construction
- GRE05b — finalized V-SHARP local-field estimation

The direct GRE06b and GRE06c field input is the finalized GRE05b v0.3.0 local
field.

## GRE06a — dipole-kernel and B0-direction validation

Script:

`06a_validate_dipole_kernel.py`

Version:

`v0.1.0`

GRE06a verifies the geometry and subject-specific B0 direction required for
QSM dipole inversion.

### B0 convention

Scanner/world +Z is taken as the acquisition B0 direction and transformed into
NIfTI voxel coordinates using the image affine.

The resulting voxel-space direction is normalized before construction of the
dipole kernel.

GRE06 requires the case-specific GRE02 B0-direction provenance. No fallback
to voxel

`[0, 0, 1]`

is permitted.

The sign of the B0 vector does not affect the scalar dipole kernel because the
B0 projection enters quadratically.

### Dipole kernel

The scalar dipole kernel is

\[
D(\mathbf{k})
=
\frac{1}{3}
-
\frac{
(\hat{\mathbf{B}}_0\cdot\mathbf{k})^2
}{
|\mathbf{k}|^2
},
\]

with

\[
D(\mathbf{0})=0.
\]

Physical spatial-frequency coordinates are generated from the NIfTI voxel
dimensions.

GRE06a also verifies the forward model

\[
b(\mathbf{r})
=
\mathcal{F}^{-1}
\left[
D(\mathbf{k})
\mathcal{F}\{\chi\}
\right].
\]

GRE06a is validation-only and does not provide downstream production input.

## GRE06b — linear isotropic-TV QSM reconstruction

Script:

`06b_reconstruct_qsm_tv.py`

Production version:

`v0.1.0`

Parameter-sweep script:

`06b_sweep_qsm_tv.py`

Additional reproducibility scripts:

- `06b_run_cohort_lambda_sweep.py`
- `06b_analyze_lambda_selection.py`

The additional reproducibility scripts capture the finalized cohort sweep and
lambda-selection analysis used for GRE06b parameter validation.

### Inputs

For each case GRE06b requires:

- GRE05b local magnetic field [Hz];
- GRE05b V-SHARP support mask;
- GRE02 voxel-space B0 direction;
- NIfTI affine and voxel dimensions;
- acquisition magnetic-field strength.

The reconstruction support is the final GRE05b V-SHARP support, not the
larger GRE05a Mask 3.

### Field conversion

The GRE05b local field is converted from Hz to ppm using

\[
b_{\mathrm{ppm}}
=
\frac{
b_{\mathrm{Hz}}
}{
\gamma B_0
}
\times 10^6,
\]

where

\[
\gamma
=
42.57747892\times10^6~\mathrm{Hz/T}.
\]

The magnetic-field strength is obtained from acquisition metadata rather than
hard-coded.

Echo time is not used in GRE06b because GRE04 has already estimated field
frequency from the multi-echo phase evolution.

No additional sign inversion is applied in GRE06.

### Forward operator

The forward model is

\[
b(\mathbf r)
=
\mathcal F^{-1}
\left[
D(\mathbf k)\,
\mathcal F\{\chi\}
\right],
\]

with

\[
D(\mathbf{k})
=
\frac{1}{3}
-
\frac{
(\hat{\mathbf{B}}_0\cdot\mathbf{k})^2
}{
|\mathbf{k}|^2
},
\qquad
D(\mathbf{0})=0.
\]

The spatial-frequency grid is generated in physical coordinates from the
voxel dimensions using the discrete FFT frequency convention.

The case-specific GRE02 voxel-space B0 direction is mandatory.

Affine consistency is checked before reconstruction.

### Optimization problem

GRE06b solves

\[
\min_{\chi}
\frac{1}{2}
\left\|
M(D\chi-b_{\mathrm{ppm}})
\right\|_2^2
+
\lambda_{\mathrm{TV}}
TV_{\mathrm{iso}}(\chi),
\]

subject to

\[
\chi=0
\]

outside the reconstruction support \(M\).

The TV term is isotropic and uses physical voxel dimensions.

Spatial derivatives crossing from valid mask support into invalid voxels are
excluded from the TV penalty.

The inversion is solved using a primal-dual hybrid-gradient /
Chambolle--Pock scheme.

The numerical operator norm is estimated before optimization and used to set
stable primal/dual step sizes.

GRE06b supports both CPU and CUDA execution through PyTorch.

### Convergence

Production convergence criterion:

\[
\mathrm{relative\ update}<10^{-4}.
\]

Maximum iteration ceiling:

`2000`

The value 2000 is a maximum rather than a fixed iteration count.

Reconstruction terminates earlier when the stopping criterion is reached.

### TV regularization-parameter selection

The production TV parameter was determined from a controlled cohort-wide
sweep over all 10 in-situ cases.

Candidate values:

`1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.01, 0.03, 0.1`

All candidate reconstructions used the same numerical configuration and a
maximum of 2000 iterations.

The production parameter was selected from the L-curve relation between:

- residual-field RMSE;
- isotropic TV norm.

Because absolute residual and TV magnitudes differed between specimens,
L-curve axes were evaluated in log space and normalized independently within
each case.

The finalized knee diagnostic used the maximum perpendicular distance from
the chord connecting the endpoints of the predefined transition interval

\[
3\times10^{-4}
\le
\lambda_{\mathrm{TV}}
\le
10^{-2}.
\]

The per-case knee estimates were:

- \(\lambda_{\mathrm{TV}}=0.0020\): 1/10 cases;
- \(\lambda_{\mathrm{TV}}=0.0025\): 7/10 cases;
- \(\lambda_{\mathrm{TV}}=0.0030\): 2/10 cases.

The cohort median was

\[
\boxed{\lambda_{\mathrm{TV}}=0.0025}.
\]

Standardized visual comparison of susceptibility maps at
`0.002, 0.0025, 0.003`
used identical image locations and display ranges.

Corresponding residual-field maps were also inspected. Differences within this
narrow parameter range were minor and provided no systematic reason to
override the cohort L-curve result.

The final production parameter is therefore

\[
\boxed{\lambda_{\mathrm{TV}}=0.0025}.
\]

One common value is used for the complete cohort. Subject-specific
regularization tuning is not performed.

### Rejected knee diagnostic

A local three-point curvature estimator was investigated during method
development.

This diagnostic selected \(\lambda_{\mathrm{TV}}=0.01\) in all cases.

Inspection showed that the estimate was strongly influenced by the nonuniform
spacing of the sampled lambda values, particularly the large interval between
0.01 and 0.03 and the transition into the strongly regularized tail.

The local-curvature diagnostic was therefore rejected.

It is retained only as development provenance and is not part of the finalized
parameter-selection procedure.

## GRE06b outputs

Production directory:

`GRE/06b_qsm_tv/lambda_0p0025/`

For each case:

- `*_GRE06b_qsm_tv_chi_ppm.nii.gz`
- `*_GRE06b_qsm_tv_forward_field_ppm.nii.gz`
- `*_GRE06b_qsm_tv_residual_field_ppm.nii.gz`
- `*_GRE06b_qsm_tv_mask.nii.gz`
- `*_GRE06b_qsm_tv_qc.json`

The susceptibility map is stored in ppm.

No susceptibility referencing is performed during GRE06b inversion.

Referencing, if required, is a separate downstream analysis operation.

## GRE06c — STI Suite iLSQR external/reference validation

Reference runner:

`06c_run_sti_ilsqr_reference.py`

Version:

`v0.1.0`

Comparison script:

`06c_compare_qsm_implementations.py`

Version:

`v0.1.0`

GRE06c provides an independent QSM comparator using the published STI Suite v3
iLSQR implementation executed through a pinned QSM-CI Docker image.

GRE06c is validation-only. The GRE06b production parameter was frozen before
GRE06c comparison and was not retrospectively tuned against STI.

### GRE06c reference inputs

To isolate dipole-inversion differences, STI iLSQR uses the same finalized
GRE05b local field and GRE05b V-SHARP support as GRE06b.

For each case the reference runner uses:

- GRE05b local magnetic field [Hz];
- GRE05b V-SHARP support mask;
- GRE02 case-specific voxel-space B0 direction;
- acquisition magnetic-field strength;
- first-echo TE from the original GRE metadata;
- NIfTI voxel dimensions and affine geometry.

No fallback B0 direction, magnetic-field strength, or TE is permitted.

The GRE05b local field is converted from Hz to ppm using the same proton
gyromagnetic-ratio convention as GRE06b.

The STI Suite iLSQR configuration uses:

`padsize = 12`

The runner validates input and output geometry and writes complete per-case
provenance.

### GRE06c outputs

Per-case directory:

`GRE/06c_sti_ilsqr/`

For each case:

- `*_GRE06c_sti_ilsqr_chi_ppm.nii.gz`
- `*_GRE06c_sti_ilsqr_provenance.json`

Reference reconstruction completed successfully for all 10 in-situ cases.

The implementation-comparison outputs are stored in:

`derivatives/GRE06c_qsm_method_comparison/`

with:

- `gre06c_qsm_method_comparison_per_case.csv`
- `gre06c_qsm_method_comparison_summary.json`

### GRE06c comparison design

The finalized GRE06b QSM at \(\lambda_{\mathrm{TV}}=0.0025\) is compared
directly against STI iLSQR on the common GRE05b V-SHARP support.

No susceptibility referencing, rescaling, registration, or parameter tuning is
performed.

Primary comparison metrics are:

- Pearson correlation;
- ordinary-least-squares APF-on-STI slope and intercept;
- raw bias, MAE, and RMSE;
- APF and STI susceptibility standard deviation;
- STI/APF standard-deviation ratio;
- robust 1st and 99th percentiles.

Offset-corrected RMSE is reported as a secondary metric because dipole
inversion does not determine the absolute susceptibility zero.

### GRE06c validation outcome

Across the 10-case cohort, APF TV-QSM and STI iLSQR showed consistent spatial
correspondence, with median Pearson correlation

\[
r=0.7939
\]

and range

\[
0.7018\text{--}0.8345.
\]

APF-on-STI regression slopes were consistently below unity, with cohort median

\[
0.5859
\]

and range

\[
0.5181\text{--}0.6488.
\]

The STI maps therefore showed systematically larger susceptibility variation.
The median STI/APF susceptibility standard-deviation ratio was

\[
1.395
\]

with range

\[
1.115\text{--}1.495.
\]

Median raw RMSE was

\[
0.0757~\mathrm{ppm},
\]

and median offset-corrected RMSE was likewise

\[
0.0757~\mathrm{ppm}.
\]

The negligible effect of mean-offset correction indicates that the
inter-method difference was not primarily caused by susceptibility-zero
offsets.

Visual comparison confirmed broadly corresponding anatomical susceptibility
patterns. The case with the lowest inter-method correlation and
highest RMSE showed substantially increased susceptibility variation in both
reconstructions rather than an APF-specific reconstruction failure. STI iLSQR
showed stronger susceptibility extrema, whereas APF TV-QSM showed a reduced
dynamic range.

GRE06c therefore supports GRE06b as a spatially concordant production
reconstruction while demonstrating systematic method-dependent differences in
susceptibility amplitude.

## Final production decision

GRE06 is finalized.

GRE06b remains the production QSM implementation with:

- linear dipole forward model;
- case-specific GRE02 B0 direction;
- GRE05b V-SHARP reconstruction support;
- isotropic physical-coordinate TV;
- PDHG / Chambolle--Pock optimization;
- relative-update tolerance `1e-4`;
- maximum 2000 iterations;
- cohort-wide `lambda_TV = 0.0025`;
- no subject-specific lambda tuning;
- no susceptibility referencing during inversion.

The existing converged `lambda_0p0025` reconstructions from the finalized
cohort sweep are accepted as the GRE06b production outputs.

GRE06c is retained as an independent validation/reference implementation and
does not replace GRE06b.
