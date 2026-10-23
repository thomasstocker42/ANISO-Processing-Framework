# CHI02 — χ-separation solver-input preparation

**Status:** Finalized

**Production implementation:** CHI02 v0.1.0

## Purpose

CHI02 prepares the exact solver-input package required by the audited
QSM-CI `chi-sep-ilsqr` wrapper.

CHI02 consumes only inputs already validated by finalized CHI01.

No reconstruction is performed in CHI02.

## QSM-CI input contract

The cached QSM-CI wrapper consumes exactly:

- `localfield.nii.gz`
- `r2prime.nii.gz`
- `magnitude.nii.gz`
- `mask.nii.gz`
- `params.json`

The wrapper reconstructs its own QSM internally with STI Suite
`QSM_iLSQR`; therefore no external APF QSM map is supplied.

## Upstream inputs

CHI02 consumes:

### GRE05b

- finalized V-SHARP local field in Hz;
- finalized V-SHARP mask.

### R2PRIME03

- raw GRE-space R2-prime map;
- finalized R2PRIME03 common support.

### GRE acquisition

- complete 12-echo magnitude series;
- complete TE vector;
- MagneticFieldStrength.

### GRE02

- voxel-space B0 direction.

### CHI01

CHI01 metadata must report the case as READY.

## Production mask

The CHI02 production support is

\[
M_{\chi}
=
M_{\mathrm{VSHARP}}
\cap
M_{R_2'}
\cap
F(B_{\mathrm{local}})
\cap
F(R_2'),
\]

where \(F(\cdot)\) denotes finite-valued voxels.

This definition is identical to the prospective support evaluated by CHI01.

CHI02 verifies that the resulting voxel count exactly matches the finalized
CHI01 prospective-support count.

Magnitude data do not silently shrink the mask. Instead, CHI02 fails if
non-finite, negative or zero-RSS magnitude values occur inside the production
support.

## Local-field conversion

QSM-CI expects `localfield.nii.gz` in ppm.

CHI02 therefore converts the finalized GRE05b local field from Hz to ppm.

For the scientific description and thesis, the proton frequency gyromagnetic
ratio is represented as

\[
\gamma / 2\pi = 42.577478518\ \mathrm{MHz/T}.
\]

However, the audited QSM-CI `recon.m` wrapper internally uses the rounded
implementation constant

\[
42.5774\ \mathrm{MHz/T}.
\]

CHI02 deliberately uses this same wrapper constant only for the interface
conversion:

\[
B_{\mathrm{ppm}}
=
\frac{B_{\mathrm{Hz}}}
     {42.5774\times10^6\,B_0}
\times10^6.
\]

This ensures that the subsequent ppm-to-Hz conversion performed by QSM-CI is
numerically self-consistent with the wrapper implementation.

The wrapper-specific constant is therefore an execution-interface parameter,
not the physical constant adopted for the thesis.

CHI02 explicitly validates the numerical round trip

Hz → ppm → Hz

using the same QSM-CI constant.

## R2-prime policy

`r2prime.nii.gz` contains the raw R2PRIME03 values in s^-1.

Numerically,

\[
1\ \mathrm{s}^{-1} = 1\ \mathrm{Hz}.
\]

No clipping or other physical-value transformation is applied.

Rare negative R2-prime values are retained unchanged.

## Magnitude preparation

The 12 GRE magnitude echoes are stacked in echo order into

`magnitude.nii.gz`

with dimensions

`x × y × z × 12`.

The fourth NIfTI dimension is treated as an echo index, not a uniformly
sampled time axis.

Therefore its NIfTI zoom is explicitly set to 1.0.

The actual echo timing is represented exclusively by the full TE vector in
`params.json`.

## params.json

CHI02 writes:

- `B0` in tesla;
- `B0_dir` in voxel coordinates;
- `voxel_size` in mm;
- `TE` as the full 12-element vector in seconds.

For the present cohort:

- B0 = 3 T;
- TE1 = 7.01 ms;
- TE2 = 11.04 ms;
- TE2 - TE1 = 4.03 ms.

The QSM-CI wrapper uses `TE(2)-TE(1)` internally when reconstructing its
STI-Suite iLSQR QSM.

## Output layout

Per case:

`derivatives/<CASE>/chi_separation/chi02_prepare/`

Solver input package:

`derivatives/<CASE>/chi_separation/chi02_prepare/qsm_ci_input/`

containing:

- `localfield.nii.gz`
- `r2prime.nii.gz`
- `magnitude.nii.gz`
- `mask.nii.gz`
- `params.json`

Additional provenance:

- `chi02_<CASE>_metadata.json`
- `chi02_<CASE>_summary.csv`

Cohort outputs:

`derivatives/summary/chi_separation/chi02_prepare/`

including:

- `chi02_cohort_summary.csv`
- `chi02_cohort_metadata.json`

## Downstream contract

CHI03 consumes the CHI02 `qsm_ci_input` directory without further scientific
input preparation and executes the QSM-CI `chi-sep-ilsqr` reconstruction.

Runner/container availability is resolved in CHI03 and is not assumed by
CHI02.
