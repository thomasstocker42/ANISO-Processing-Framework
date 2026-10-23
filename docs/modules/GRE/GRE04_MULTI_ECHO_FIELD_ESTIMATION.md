# GRE04 — Multi-echo total-field estimation

## Status

**Implementation status:** finalized and cohort-validated
**Pipeline position:** GRE03 ROMEO unwrapping → **GRE04 multi-echo field estimation and validation** → GRE05 mask construction/background-field removal → GRE06 QSM

## Objective

GRE04 estimates the voxel-wise total magnetic field from the ROMEO-unwrapped multi-echo GRE phase series.

The finalized module contains four components:

- **GRE04a — parity-aware ordinary least squares (OLS):** equal-weight reference estimator;
- **GRE04b — parity-aware magnitude-squared weighted least squares (WLS):** production total-field estimator;
- **GRE04c — four-scenario model comparison:** factorial comparison of weighting and odd/even parity modelling;
- **GRE04d — odd/even echo diagnostic:** dedicated diagnostic used to establish and characterize the systematic parity effect.

GRE04 does not remove background fields and does not perform QSM. Its primary quantitative output is the **total frequency field**, containing both local tissue-field and background-field contributions.

## Scientific development

The initial GRE04a/GRE04b implementations used the two-parameter phase model

\[
\phi_i=\phi_0+\omega TE_i+\varepsilon_i.
\]

Inspection of GRE04b residuals revealed a systematic odd/even echo alternation. GRE04d demonstrated that this effect was present in all 10 thesis cases, with strict parity sign alternation and a coherent spatial pattern.

Because image data alone do not establish the physical readout-gradient polarity, the effect is described as a **systematic odd/even echo phase term consistent with alternating readout polarity / bipolar-gradient behaviour** rather than as definitively confirmed bipolar readout.

The production GRE04a and GRE04b estimators were therefore revised to a three-parameter parity-aware model.

## Final signal model

For each voxel,

\[
\phi_i
=
\phi_0
+
\omega TE_i
+
\beta_{\mathrm{OE}} s_i
+
\varepsilon_i,
\]

where

\[
s_i=
\begin{cases}
+1,& i=1,3,5,\ldots\\
-1,& i=2,4,6,\ldots
\end{cases}.
\]

Here:

- \(\phi_i\) is the ROMEO-unwrapped phase at echo \(i\), in radians;
- \(\phi_0\) is the fitted phase intercept, in radians;
- \(\omega\) is the angular-frequency offset, in rad/s;
- \(\beta_{\mathrm{OE}}\) is the odd/even phase coefficient, in radians;
- \(s_i\) is the parity regressor;
- \(\varepsilon_i\) is the residual phase error.

The frequency offset in hertz is

\[
\Delta f=\frac{\omega}{2\pi}.
\]

The relative total field in ppm is

\[
\Delta B_{\mathrm{ppm}}
=
\frac{\Delta f}{\bar{\gamma}B_0}\times10^6,
\]

with

\[
\bar{\gamma}=42.57747892\times10^6\ \mathrm{Hz/T}.
\]

For the thesis cohort, \(B_0=3\) T is supplied explicitly via `--b0-t`.

## Inputs

| Input | Source | Purpose |
|---|---|---|
| ROMEO-unwrapped 4D phase | GRE03 | Primary phase series |
| Effective ROMEO mask | GRE03 | Spatial fitting domain |
| Echo-time vector | GRE02 | Regressor \(TE_i\) |
| Multi-echo magnitude | GRE02 | Required for WLS weighting |
| Scanner field strength \(B_0\) | explicit CLI | Hz-to-ppm conversion |

Phase, magnitude, and mask geometry must match.

## Spatial analysis domain

GRE04 operates on the effective GRE03 mask associated with the ROMEO-unwrapped phase data.

Following the mask-policy revision, GRE03 and GRE04 use the **full GRE02b HD-BET brain mask** rather than the previously used one-voxel-eroded mask.

The processing chain is therefore

```text
GRE02b full HD-BET brain mask
        ↓
GRE03 ROMEO phase unwrapping
        ↓
GRE04a parity-aware OLS
GRE04b parity-aware WLS
        ↓
GRE04c four-scenario comparison
```

Early anatomical erosion is avoided because background-field removal introduces its own boundary loss downstream.

## GRE04a — parity-aware ordinary least squares

Script:

```text
scripts/gre/04a_estimate_total_field_ols.py
```

Current production version:

```text
v0.2.0
```

GRE04a minimizes the unweighted residual sum of squares

\[
\min_{\phi_0,\omega,\beta_{\mathrm{OE}}}
\sum_i
\left[
\phi_i-\phi_0-\omega TE_i-\beta_{\mathrm{OE}}s_i
\right]^2.
\]

All valid echoes contribute equally.

GRE04a is retained as the transparent equal-weight reference estimator.

## GRE04b — parity-aware magnitude-squared WLS

Script:

```text
scripts/gre/04b_estimate_total_field_wls.py
```

Current production version:

```text
v0.2.0
```

GRE04b minimizes

\[
\min_{\phi_0,\omega,\beta_{\mathrm{OE}}}
\sum_i
w_i
\left[
\phi_i-\phi_0-\omega TE_i-\beta_{\mathrm{OE}}s_i
\right]^2,
\]

with

\[
w_i=M_i^2.
\]

A small voxel-wise floor is applied to valid magnitude-squared weights:

\[
w_i^{\mathrm{used}}
=
\max\left(M_i^2,\ f_{\mathrm{floor}}\max_j M_j^2\right),
\]

with default

```text
weight_floor_fraction = 1e-6
```

for valid positive-magnitude echoes.

The floor prevents numerical collapse of very small weights while preserving magnitude-based relative weighting.

Rationale:

- at sufficiently high SNR, phase variance is approximately inversely related to squared magnitude;
- later echoes generally have lower magnitude and noisier phase;
- magnitude-squared weighting reduces the influence of low-SNR observations without discarding them.

## Common fitting policy

Both parity-aware estimators:

- use echo times in seconds internally;
- fit intercept, frequency slope, and odd/even coefficient simultaneously;
- require at least **four valid echoes**;
- require valid support from both odd and even echo groups;
- operate only inside the effective GRE03 mask;
- preserve input NIfTI geometry;
- assign zero outside the final valid-fit mask;
- record configuration and provenance in per-case JSON;
- write optional predicted-phase and residual 4D series with `--write-residuals`.

The minimum of four valid echoes provides at least one residual degree of freedom for the three-parameter model.

## Parameter uncertainty

### OLS

Classical OLS covariance is used:

\[
\mathrm{Cov}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1},
\qquad
\hat{\sigma}^2=\frac{\mathrm{SSE}}{n-3}.
\]

The frequency standard error is derived from the variance of the \(\omega\) coefficient and converted from rad/s to Hz.

### WLS

The WLS covariance model assumes

\[
\mathrm{Var}(\phi_i)\propto\frac{1}{w_i}.
\]

The residual scale is estimated from weighted SSE with \(n-3\) residual degrees of freedom.

Because magnitude-derived weights are relative reliability weights rather than known absolute inverse variances, the resulting WLS frequency standard error is interpreted comparatively rather than as an exact physical uncertainty.

## Output hierarchy

```text
derivatives/<CASE>/GRE/
├── 04a_field_ols/
├── 04b_field_wls/
├── 04c_field_fit_model_comparison/
└── 04d_odd_even_diagnostic/
```

Archived pre-parity implementations are retained separately:

```text
04a_field_ols_v010_legacy/
04b_field_wls_v010_legacy/
```

## Primary GRE04a/GRE04b outputs

Both estimators write:

```text
<CASE>_GRE04[a|b]_omega_rad_s.nii.gz
<CASE>_GRE04[a|b]_total_field_hz.nii.gz
<CASE>_GRE04[a|b]_total_field_ppm.nii.gz
<CASE>_GRE04[a|b]_phase_intercept_rad.nii.gz
<CASE>_GRE04[a|b]_beta_odd_even_rad.nii.gz
<CASE>_GRE04[a|b]_phase_rmse_rad.nii.gz
<CASE>_GRE04[a|b]_phase_r2.nii.gz
<CASE>_GRE04[a|b]_phase_maxabs_residual_rad.nii.gz
<CASE>_GRE04[a|b]_phase_linearity_error_rad.nii.gz
<CASE>_GRE04[a|b]_frequency_se_hz.nii.gz
<CASE>_GRE04[a|b]_valid_fit_mask.nii.gz
<CASE>_GRE04[a|b]_n_valid_echoes.nii.gz
<CASE>_GRE04[a|b]_qc.json
```

Optional:

```text
<CASE>_GRE04[a|b]_phase_predicted_rad_4d.nii.gz
<CASE>_GRE04[a|b]_phase_residuals_rad_4d.nii.gz
```

GRE04b additionally writes:

```text
<CASE>_GRE04b_phase_weighted_rmse_rad.nii.gz
<CASE>_GRE04b_phase_weighted_r2.nii.gz
```

## GRE04c — four-scenario model comparison

Script:

```text
scripts/gre/04c_compare_ols_wls.py
```

Current version:

```text
v0.2.0
```

GRE04c compares four finalized scenarios on a **single four-way common valid mask**:

| Label | Weighting | Parity |
|---|---|---|
| A | OLS | no |
| B | magnitude² WLS | no |
| C | OLS | yes |
| D | magnitude² WLS | yes |

The primary contrasts are:

\[
B-A
\]

weighting effect without parity,

\[
C-A
\]

parity effect under OLS,

\[
D-B
\]

parity effect under WLS,

and

\[
D-C
\]

weighting effect after parity correction.

The weighting-by-parity interaction is

\[
(D-C)-(B-A).
\]

GRE04c also compares the OLS and WLS parity coefficients and summarizes fit-quality metrics across the four models.

Cohort summary reporting is generated reproducibly by

```text
scripts/gre/04c_summarize_field_fit_model_comparison.py
```

from the canonical GRE04c cohort CSV.

## GRE04d — odd/even echo diagnostic

GRE04d is retained as a dedicated diagnostic/provenance module.

It established:

- strict odd/even residual sign alternation in all 10 cases;
- a spatially coherent odd/even residual component;
- measurable field bias when parity was omitted;
- consistent cohort-wide behaviour.

GRE04d is not the production field estimator. Its purpose is to justify and characterize the parity term used in GRE04a/GRE04b v0.2.0.

## Cohort outputs

```text
derivatives/summary/GRE/
├── gre04a_field_ols_qc.csv
├── gre04a_field_ols_run.json
├── gre04b_field_wls_qc.csv
├── gre04b_field_wls_run.json
├── gre04c_field_fit_model_comparison_qc.csv
├── gre04c_field_fit_model_comparison_run.json
├── gre04c_field_fit_model_decision_table.md
├── gre04c_field_fit_model_decision_summary.csv
├── gre04c_field_fit_model_decision_summary.json
├── gre04d_odd_even_echo_diagnostic_qc.csv
└── gre04d_odd_even_echo_diagnostic_run.json
```

## Final methodological decision

The cohort comparison showed that both parity modelling and echo weighting materially affect the estimated total field.

Parity-aware OLS has the lower unweighted residual RMSE, as expected because OLS directly minimizes unweighted SSE.

Magnitude-squared WLS remains the production estimator because:

- the GRE phase noise is heteroscedastic across echoes;
- magnitude² weighting provides a physically motivated relative precision model;
- parity-aware WLS markedly improves its weighted residual objective relative to the no-parity WLS model;
- its model-based frequency uncertainty is comparable to or slightly lower than parity-aware OLS at cohort level;
- the weighting effect remains substantial after parity correction.

Therefore:

> **GRE04b v0.2.0 parity-aware magnitude-squared WLS is the canonical downstream total-field source.**

GRE04a v0.2.0 is retained as the equal-weight comparator, while the archived no-parity estimators are retained for methodological provenance.

## Acceptance status

GRE04 is accepted as technically and scientifically validated for the 10-case thesis cohort.

Acceptance criteria met:

- all intended cases completed;
- full-mask spatial support was retained except for negligible method-dependent voxel loss;
- field maps are finite on the valid support;
- geometry is preserved;
- parity-aware fits markedly reduce residual structure;
- OLS/WLS and parity/no-parity effects were quantified on common support;
- the production estimator is explicitly selected and documented.

## References

Bilgic B, Costagli M, Chan K-S, et al. Recommended implementation of quantitative susceptibility mapping for clinical research in the brain: a consensus of the ISMRM Electro-Magnetic Tissue Properties Study Group. *Magnetic Resonance in Medicine*. 2024. doi:10.1002/mrm.30006.

Gilbert G, Savard G, Bard C, Beaudoin G. Quantitative comparison between a multiecho sequence and a single-echo sequence for susceptibility-weighted phase imaging. *Magnetic Resonance Imaging*. 2012;30:722–730.

Robinson SD, Bredies K, Khabipova D, et al. An illustrated comparison of processing methods for MR phase imaging and QSM: combining array coil signals and phase unwrapping. *NMR in Biomedicine*. 2017;30:e3601.

Dymerska B, Eckstein K, Bachrata B, et al. Phase unwrapping with a rapid open-source minimum spanning tree algorithm (ROMEO). *Magnetic Resonance in Medicine*. 2021;85:2294–2308. doi:10.1002/mrm.28563.