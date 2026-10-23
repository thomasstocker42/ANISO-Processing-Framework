# GRE05 — Background-field removal

**Status:** Finalized
**Production implementation:** GRE05b v0.3.0
**Validation implementation:** GRE05c v0.1.2

## Purpose

GRE05 removes background magnetic-field contributions from the multi-echo
GRE total-field estimate and produces the local magnetic field required for
subsequent quantitative susceptibility mapping (QSM).

The production background-field removal method is variable-kernel SHARP
(V-SHARP).

## Prerequisites

- GRE01 — inventory
- GRE02 — phase preparation
- GRE02b — standardized GRE brain mask
- GRE03 — ROMEO phase unwrapping
- GRE04b — magnitude-squared weighted least-squares total-field estimation

## Module structure

### GRE05a — consensus BFR mask construction

Script:

`05a_construct_consensus_qsm_masks.py`

Production version:

`v0.2.0`

GRE05a constructs the masks used for background-field removal.

The finalized GRE04b total-field model includes an empirical odd/even echo
phase nuisance term,

\[
\phi_e =
\beta_0 + \omega TE_e + \beta_{\mathrm{OE}} p_e,
\]

where \(p_e=+1\) for odd-numbered echoes and \(p_e=-1\) for even-numbered
echoes.

GRE05a uses a complex-signal reliability model that is explicitly consistent
with this finalized field model. Relative to the first echo, predicted phase
is

\[
\Delta\phi_{\mathrm{pred},e}
=
\omega(TE_e-TE_1)
+
\beta_{\mathrm{OE}}(p_e-p_1).
\]

Because \(p_1=+1\), the parity contribution is zero for odd echoes and
\(-2\beta_{\mathrm{OE}}\) for even echoes relative to echo 1.

The relative-residual reliability threshold remains fixed at

\[
\mathrm{relative\ residual} \le 0.30.
\]

This threshold was established before the parity-aware model update and was
not retuned afterward.

Mask definitions:

- Mask 1: non-eroded GRE02b whole-brain mask;
- Mask 2: reliable phase/field support based on the complex-signal residual;
- Mask 3: hole-filled intersection of Mask 1 and Mask 2.

The final BFR input is Mask 3.

Production input:

`GRE05_mask3_bfr_holefilled.nii.gz`

The parity-aware refresh substantially reduced the complex-signal residual and
increased reliable BFR support while leaving both the residual threshold and
mask-topology rules unchanged.

### GRE05b — V-SHARP background-field removal

Script:

`05b_remove_background_field_vsharp.py`

Production version:

`v0.3.0`

Method identifier:

`vsharp_variable_kernel_single_deconvolution_tsvd`

Output directory:

`GRE/05b_vsharp_apf_v030/`

#### Inputs

- GRE04b total field, Hz
- GRE05a Mask 3
- voxel dimensions from the GRE image header

#### Variable-radius filtering

Solid spherical-mean-value kernels are generated in physical coordinates.

The production radius sequence is:

`9, 8, 7, 6, 5, 4, 3, 2, 1 mm`

For each radius, a SHARP kernel is formed as

\[
H_R = \delta - S_R ,
\]

where \(S_R\) denotes the normalized solid-sphere averaging kernel of radius
\(R\).

The total field is convolved with each kernel. A voxel is assigned the
largest-radius result for which the complete spherical footprint lies inside
the BFR mask. Progressively smaller radii therefore recover support near the
mask boundary.

The 1-mm shell is implemented using the same solid-sphere delta-SMV operator
as all other radii. No separate Laplacian fallback is used in v0.3.0.

#### Single deconvolution

After construction of the spatially variable V-SHARP right-hand side, one
Fourier-domain deconvolution is performed using the transfer function of the
largest, 9-mm SHARP kernel.

The inverse is regularized by explicit truncated singular-value decomposition
(TSVD):

\[
\hat f_{\mathrm{local}}(\mathbf{k}) =
\begin{cases}
\hat b(\mathbf{k}) / H_9(\mathbf{k}),
& |H_9(\mathbf{k})| \ge \tau, \\
0,
& |H_9(\mathbf{k})| < \tau .
\end{cases}
\]

The production threshold is

\[
\tau = 0.50.
\]

The production threshold was selected during controlled implementation
validation against the external STI Suite V-SHARP reference. A cohort-level
threshold sweep demonstrated a clear error minimum at `|H| = 0.50`, with all
10 cases independently selecting 0.50 as the minimum-RMSE threshold.

This result was interpreted as evidence that explicit suppression of the
poorly conditioned SHARP transfer-function coefficients, rather than the
previous numerical-zero guard followed by an independent radial high-pass
filter, was the dominant correction required to reproduce the reference
implementation.

The threshold was therefore fixed at `|H| >= 0.50` before the later
parity-aware GRE04b/GRE05a upstream revision. Following that revision, GRE05b
was rerun without retuning this parameter and was revalidated against newly
generated STI Suite references using the corrected total-field maps and BFR
masks. External agreement remained high across the full cohort
(median Pearson r = 0.99545, median APF-on-STI slope = 0.99398, median
RMSE = 0.00305 ppm).

Accordingly, `|H| >= 0.50` is frozen for the production GRE05b v0.3.0
implementation. Further optimization against the STI reference was not
performed after the refreshed validation in order to avoid retrospective
parameter tuning to the validation target.

The value 0.50 is an empirically validated parameter of the present APF
implementation and acquisition/processing configuration; it should not be
interpreted as a universal V-SHARP constant.

No additional post-hoc radial high-pass filter is applied in v0.3.0.

#### Background field

Within the final V-SHARP support,

\[
f_{\mathrm{background}}
=
f_{\mathrm{total}} - f_{\mathrm{local}} .
\]

The stored reconstruction residual is therefore a bookkeeping identity check,
not an independent measure of BFR accuracy.

## GRE05b outputs

For each case:

- `*_GRE05b_local_field_hz.nii.gz`
- `*_GRE05b_background_field_hz.nii.gz`
- `*_GRE05b_vsharp_mask.nii.gz`
- `*_GRE05b_assigned_radius_mm.nii.gz`
- `*_GRE05b_variable_kernel_rhs_hz.nii.gz`
- `*_GRE05b_reconstruction_residual_hz.nii.gz`
- `*_GRE05b_qc.json`

The local field is stored in Hz.

## External STI Suite reference generation

Script:

`05b_run_sti_vsharp_reference.py`

Version:

`v0.2.1`

The STI Suite reference runner is validation-only. It uses:

- the finalized parity-aware GRE04b total-field map;
- GRE05a v0.2.0 Mask 3 from `GRE/05a_bfr_masks`;
- a pinned STI Suite V-SHARP container;
- `smvsize=9` as the primary implementation reference;
- `smvsize=12` as an additional sensitivity reference.

The STI Suite implementation and pinned container were unchanged during the
parity-aware refresh. The reference data were regenerated because the upstream
GRE04b total-field maps and GRE05a masks had changed.

## GRE05c — external implementation validation

Script:

`05c_compare_vsharp_implementations.py`

Version:

`v0.1.2`

GRE05c compares APF GRE05b v0.3.0 against the controlled STI Suite V-SHARP
reference generated with `smvsize=9`.

Because the STI output is stored in ppm whereas APF stores field in Hz, APF
fields are converted using

\[
1~\mathrm{ppm}
=
\gamma B_0 \times 10^{-6}
=
127.732435554~\mathrm{Hz}
\]

for \(B_0=3~\mathrm{T}\) and

\[
\gamma = 42.577478518~\mathrm{MHz/T}.
\]

Field-value comparisons are performed only on the intersection of APF and STI
support. Support overlap is quantified independently using Jaccard and
directional containment measures so that differences in effective mask erosion
are not conflated with differences in reconstructed local-field values.

Summary outputs:

- `derivatives/summary/GRE/gre05c_vsharp_implementation_comparison_qc.csv`
- `derivatives/summary/GRE/gre05c_vsharp_implementation_comparison_run.json`

With `--write-plots`, GRE05c v0.1.2 writes one standardized four-panel
validation figure per selected case as PNG and PDF. The figure contains:

- APF local field on common support;
- STI local field on common support;
- APF-minus-STI difference map;
- APF-versus-STI scatter plot with identity line and summary metrics.

The final refreshed 10-case external validation yielded:

- median Pearson r = 0.99545;
- median APF-on-STI slope = 0.99398;
- median bias = -0.000251 ppm;
- median RMSE = 0.00305 ppm;
- median MAE = 0.00209 ppm;
- median support Jaccard = 0.92824.

The largest discrepancy occurred in one case
(r = 0.99359, slope = 0.98542, RMSE = 0.00629 ppm), without evidence of a
general reconstruction failure.

## Implementation history

GRE05b v0.1.x used per-radius deconvolution and did not pass validation.

GRE05b v0.2.0 introduced the variable-kernel / single-deconvolution
architecture. Its Fourier inversion was effectively untruncated
(`kernel_eps = 1e-12`) and was followed by a separate radial high-pass filter.
Controlled comparison demonstrated systematic disagreement with the external
STI Suite implementation.

Diagnostic experiments showed that boundary effects, the minimum-radius
Laplacian approximation, additional APF support, FFT padding, and discrete
solid-sphere construction were not the principal cause of the discrepancy.

Explicit truncation of poorly conditioned SHARP transfer-function
coefficients produced the dominant improvement. A threshold sweep demonstrated
a clear minimum around `|H| = 0.50`; all 10 cohort cases independently selected
0.50 as the minimum-RMSE value relative to the controlled STI reference.

GRE05b v0.3.0 therefore replaced the numerical-zero guard plus post-hoc
high-pass with explicit SHARP-domain TSVD and standardized all radii,
including 1 mm, as solid-sphere delta-SMV operators.

## Downstream decision

Only GRE05b v0.3.0 outputs are approved for downstream QSM reconstruction.

GRE05b v0.1.x and v0.2.0 outputs are retained only for provenance and
method-development history.
