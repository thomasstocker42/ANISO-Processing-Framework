# CHI03c — CSF-Regularization Sensitivity

## Purpose

CHI03c evaluates the sensitivity of the production MEDI susceptibility
source-separation reconstruction to CSF regularization.

The production CHI03b reconstruction uses:

- `lambda = 1`
- `lambda_CSF = 1`
- `Dr = 137 Hz/ppm`

CHI03c repeats selected reconstructions with:

- `lambda = 1`
- `lambda_CSF = 0`
- `Dr = 137 Hz/ppm`

All other inputs and solver settings are retained unchanged.

## Motivation

The default MEDI `extract_CSF` procedure applies a 10-mm SMV erosion.
In the postmortem cohort this eliminated the CSF reference in 8/10 cases.

The optional erosion was therefore disabled while retaining the standard
R2* < 5 s^-1 criterion. This produced non-empty masks in all cases, but
visual QC showed that some masks were small and not unequivocally specific
to ventricular CSF.

CHI03c tests whether uncertainty in this CSF support materially affects the
reconstructed susceptibility components.

## Initial pilot cases

Three representative cases were initially selected:

- Pilot case A — visually plausible CSF support
- Pilot case B — questionable CSF-mask specificity
- Pilot case C — reconstruction/QSM comparison outlier

## Experimental design

CHI03c is a one-variable sensitivity experiment.

Changed parameter:

`lambda_CSF: 1 -> 0`

Unchanged:

- CHI02 inputs
- R2-prime solver-boundary clipping
- GRE02 phase noise weighting
- R2STAR02 input
- GRE06c STI-iLSQR QSM
- MEDI solver implementation
- `lambda = 1`
- `Dr = 137 Hz/ppm`
- CSF-mask extraction itself

## Interpretation

Differences between CHI03b and CHI03c therefore quantify the dependence of
the MEDI reconstruction on the CSF regularization term.

## Cohort extension and final status

Following the initial three-case pilot, CHI03c was extended
to all ten postmortem in-situ datasets.

The final comparison used production MEDI with
lambda_CSF = 1 versus the sensitivity reconstruction with
lambda_CSF = 0, retaining the other solver inputs.

All ten cases were included in the CHI05 quantitative
comparison. See `CHI05_RESULTS.md` in the APF Results
ChiSeparation directory.

Status: completed for 10/10 datasets.
