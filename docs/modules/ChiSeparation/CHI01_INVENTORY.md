# CHI01 — χ-separation input inventory and compatibility checks

**Status:** Finalized

**Production implementation:** CHI01 v0.1.1

## Purpose

CHI01 verifies that all frozen upstream products required for the
QSM-CI `chi-sep-ilsqr` implementation are present, geometrically compatible
and accompanied by valid acquisition metadata.

The module is strictly read-only with respect to scientific image content.

CHI01 does not:

- convert local-field units;
- resample any image;
- stack magnitude echoes;
- construct a production mask;
- modify R2-prime values;
- clip negative R2-prime voxels;
- run chi-separation.

Those steps are deferred to downstream modules.

## QSM-CI target implementation

The target reconstruction method is:

- method slug: `chi-sep-ilsqr`;
- QSM-CI record: `21697749`;
- engine: SNU-LIST chi-separation toolbox;
- internal QSM reconstruction: STI Suite `QSM_iLSQR`.

The pinned image recorded by the QSM-CI method manifest is:

`ghcr.io/astewartau/qsm-ci/chi-sep-ilsqr@sha256:d07d167420d04a26403c20d42d47269ecd95ff354529521dddd282449e6bf35a`

The cached QSM-CI `recon.m` wrapper shows that the method consumes:

- local field in ppm;
- R2-prime in Hz, numerically equivalent to s^-1;
- GRE magnitude data;
- a binary mask;
- acquisition parameters including B0, voxel-space B0 direction,
  voxel size and TE vector.

QSM is reconstructed internally using STI Suite `QSM_iLSQR`, rather than
accepting an external APF QSM map.

## Required upstream inputs

### GRE05b

From:

`derivatives/<CASE>/GRE/05b_vsharp_apf_v030/`

CHI01 requires:

- `<CASE>_GRE05b_local_field_hz.nii.gz`
- `<CASE>_GRE05b_vsharp_mask.nii.gz`

The local field remains in Hz in CHI01. Conversion to ppm is deferred to CHI02.

### R2PRIME03

From:

`derivatives/<CASE>/relaxometry/r2prime/r2prime03_construct/`

CHI01 requires:

- `r2prime03_<CASE>_R2prime_raw_s-1.nii.gz`
- `r2prime03_<CASE>_common_mask.nii.gz`
- `r2prime03_<CASE>_metadata.json`

The raw R2-prime map is preserved without clipping.

### GRE magnitude series

CHI01 identifies exactly one complete GRE 12-echo magnitude acquisition.

The acquisition series index is not hard-coded. Both forms such as

- `GRE_12echo_10_e1...e12`
- `GRE_12echo_11_e1...e12`

are accepted, provided that exactly one complete 12-echo series is present.

For every echo, the corresponding JSON sidecar is required.

### GRE02 B0 direction

CHI01 reads:

`derivatives/<CASE>/GRE/phase_preparation_gre02/gre02_<CASE>_b0_direction.json`

The `voxel_b0_direction` field must:

- contain exactly three values;
- be finite;
- have unit norm within tolerance.

## Acquisition metadata validation

For the 12 GRE echoes, CHI01 verifies:

- exactly 12 echoes;
- echo numbers 1 through 12;
- strictly increasing EchoTime values;
- valid positive MagneticFieldStrength values;
- identical field strength across echoes.

The full TE vector is retained in metadata.

The acquisition used in the current cohort has:

- TE1 = 7.01 ms;
- TE2 = 11.04 ms;
- TE2 - TE1 = 4.03 ms;
- echo-spacing range = 4.03–4.55 ms;
- B0 = 3 T.

The QSM-CI `recon.m` wrapper explicitly computes its internal echo spacing as

`TE(2) - TE(1)`

for STI-Suite QSM reconstruction. Therefore CHI01 records the exact full TE
vector rather than replacing it with a nominal constant spacing.

## Geometry validation

The GRE05b local field defines the reference GRE geometry.

CHI01 verifies matching shape and affine for:

- GRE05b V-SHARP mask;
- R2PRIME03 raw R2-prime;
- R2PRIME03 common support;
- all 12 GRE magnitude echoes.

The default affine tolerance is 1e-5.

## Finite-value validation

CHI01 requires:

- finite local-field values inside V-SHARP support;
- finite R2-prime values inside R2PRIME03 support.

## Prospective common support

For readiness evaluation only, CHI01 computes in memory:

\[
M_{\mathrm{prospective}}
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

CHI01 does not write this mask.

The finalized production chi-separation support will be constructed explicitly
by CHI02.

## Negative R2-prime policy

CHI01 does not alter negative R2-prime values.

For each case it records:

- fraction of negative R2-prime voxels within R2PRIME03 support;
- fraction of negative R2-prime voxels within the prospective
  chi-separation support.

No APF-side clipping is applied.

## Outputs

Per case:

`derivatives/<CASE>/chi_separation/chi01_inventory/`

including:

- `chi01_<CASE>_metadata.json`
- `chi01_<CASE>_summary.csv`

Cohort outputs:

`derivatives/summary/chi_separation/chi01_inventory/`

including:

- `chi01_cohort_summary.csv`
- `chi01_cohort_metadata.json`

## Downstream contract

CHI02 consumes CHI01-validated upstream products and prepares the exact
solver-ready input package required by QSM-CI `chi-sep-ilsqr`.

CHI02 will be responsible for:

- local-field Hz-to-ppm conversion;
- production common-mask construction;
- 4D magnitude stacking;
- staging raw R2-prime;
- construction of `params.json`.
