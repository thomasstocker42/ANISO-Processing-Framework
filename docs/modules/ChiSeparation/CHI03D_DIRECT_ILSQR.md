# CHI03d — Direct iLSQR susceptibility source separation

## Purpose

Generate paramagnetic, diamagnetic-magnitude and total
susceptibility maps with the SNU-LIST direct chi_sep_iLSQR solver.

## Production implementation

MATLAB runner:

`scripts/chiseparation/chi03d_ilsqr_direct_runner.m`

The runner calls SNU-LIST `chi_sep_iLSQR` directly.

Inputs:
- CHI02 combined GRE magnitude
- CHI02 local magnetic field (Hz)
- CHI02 R2-prime map
- CHI02 common binary solver mask
- GRE06c STI-iLSQR QSM (auxiliary input)
- Multi-echo GRE magnitude and phase for noise estimation

Solver parameters:
- Dr = 137 Hz/ppm
- Acquisition-specific central frequency (Hz)
- Acquisition-specific B0 direction
- Acquisition-specific voxel dimensions (mm)

Noise information is generated through:

`Preprocessing4Phase(mag4, phase4)`

Its `N_std` output is supplied in `option_data`.

## R2-prime solver-boundary processing

The original quantitative R2-prime derivative is preserved.

The solver input is prepared by replacing non-finite values
with zero, clipping negative values to zero, and applying
the CHI02 reconstruction mask.

Counts of non-finite and negative original values are
retained in the case-specific QC metadata.

## Solver outputs

- chi_para: paramagnetic susceptibility (ppm)
- chi_dia: diamagnetic susceptibility magnitude (ppm)
- chi_tot: chi_para minus chi_dia (ppm)

Each component is stored as a compressed float32 NIfTI
in the original reconstruction geometry.

## Quality control

The runner verifies:
- Finite solver inputs where required
- Finite susceptibility components inside the mask
- Algebraic consistency of chi_tot = chi_para - chi_dia

Algebraic RMSE must not exceed 1e-8 ppm.

Per-case QC JSON additionally records:
- Execution time and solver mask volume
- Input R2-prime validity counts
- Component distribution statistics
- Correlation and RMSE against auxiliary QSM
- Paramagnetic versus diamagnetic component correlation
- Completion status

## Related comparison

`scripts/chiseparation/chi03e_compare_medi_ilsqr.m`

This MATLAB analysis compares CHI03b MEDI against CHI03d
direct iLSQR.

The later CHI05 Python module produces the finalized
ten-case comparison figures, tables and CSF sensitivity
analysis used in the thesis.

## Reproduction

Run from MATLAB with the required SNU-LIST functions
available on the MATLAB search path:

    chi03d_ilsqr_direct_runner('ANISO_00_000_InS')

Replace the case identifier as appropriate.

Reproduction requires the CHI02 and GRE06c derivatives
and the original compatible SNU-LIST implementation.

## Status

Production MATLAB source recovered from the workstation
and synchronized to the laptop repository.

Existing reconstruction results passed the recorded
case-level QC checks. Fresh full-cohort regeneration has
not been performed during this documentation close-out.
