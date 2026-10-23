# CHI04 — reconstruction quality control

## Purpose

CHI04 performs quantitative and visual QC of the finalized CHI03b MEDI
susceptibility source-separation outputs.

CHI04 does not modify reconstruction outputs.

## Quantitative QC

Per case, CHI04 evaluates:

- required-output completeness;
- finite values within the solver mask;
- chi-total algebraic consistency;
- agreement of chi-total with GRE06c STI-iLSQR QSM;
- chi-para versus chi-dia correlation;
- CSF-mask size;
- negative raw R2-prime count;
- source-component distribution summaries.

## Visual QC

Standardized orthogonal montages display:

- chi-para;
- chi-dia;
- chi-total;
- GRE06c QSM;
- raw R2-prime;
- chi-total with the CSF reference overlay.

Chi-total and QSM are shown on a common symmetric susceptibility scale to
avoid hiding amplitude differences by independent windowing.

## Interpretation

QSM agreement is used as a descriptive comparator and not as an automatic
pass/fail criterion because GRE06c QSM is not ground truth and source
separation solves a different inverse problem.

No case is automatically excluded by CHI04.

## Status

Quantitative and visual QC completed in 10/10 cases.

Final manual visual-review classification:

- PASS: 9/10 cases
- PASS_WITH_FLAG: 1/10 cases
- excluded: 0/10 cases

One case was retained as PASS_WITH_FLAG because it was the principal
quantitative chi-total versus GRE06c QSM comparison outlier, while remaining
visually acceptable and without evidence of gross reconstruction failure.
