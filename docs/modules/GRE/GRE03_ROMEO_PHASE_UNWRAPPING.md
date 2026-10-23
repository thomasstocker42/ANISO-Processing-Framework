# GRE03 — ROMEO multi-echo phase unwrapping

CAVE:
output directories in script might still be:
derivatives/<CASE>/gre/03_romeo_unwrap/ and
derivatives/summary/
change to:
derivatives/<CASE>/GRE/
derivatives/summary/GRE/

## Status

**Implementation status:** validated cohort implementation (`v0.1.0`)
**Pipeline position:** GRE02 phase preparation → **GRE03 ROMEO unwrapping** → GRE04 field estimation

## Objective

GRE03 performs spatially and temporally informed phase unwrapping of the prepared multi-echo GRE phase data using ROMEO. The module replaces the legacy exploratory procedure in which `numpy.unwrap()` was applied only along the echo dimension.

GRE03 is deliberately restricted to **phase unwrapping**. It does not:

- estimate the definitive total-field map;
- remove background fields;
- perform dipole inversion;
- generate QSM;
- perform χ-separation.

These operations remain separate downstream APF modules.

## Scientific rationale

The legacy workflow applied temporal unwrapping independently within each voxel. That approach could correct echo-to-echo \(2\pi\) discontinuities but could not resolve spatial phase wraps between neighboring voxels.

ROMEO is a path-following 3D/4D phase-unwrapping method. It constructs an unwrapping path using phase coherence and, when supplied, magnitude and multi-echo information. For the APF GRE data, ROMEO is therefore run on the complete 4D multi-echo phase series with the corresponding 4D magnitude series and echo times.

The purpose of GRE03 is to generate spatially and temporally continuous phase data suitable for voxel-wise multi-echo field estimation during GRE04.

## Inputs

GRE03 consumes the validated GRE02 outputs:

| Input | Requirement |
|---|---|
| `*_phase_rad_4d.nii.gz` | 4D wrapped phase, already scaled to radians |
| `*_magnitude_4d.nii.gz` | 4D magnitude with identical dimensions and affine |
| `*echo_times*` | One strictly increasing echo time per 4D phase volume |
| `*_mask_ero1.nii.gz` | Preferred standardized eroded GRE analysis mask |

Echo times are passed to ROMEO in **milliseconds**.

## ROMEO configuration

Default APF call:

```bash
romeo \
  -p <phase_rad_4d.nii.gz> \
  -m <magnitude_4d.nii.gz> \
  -t '[TE1,TE2,...,TEn]' \
  -k <GRE_mask_ero1.nii.gz> \
  --mask-unwrapped \
  --no-phase-rescale \
  -o <phase_unwrapped_romeo_rad_4d.nii.gz>
```

### Parameter decisions

#### Multi-echo 4D unwrapping

Echoes are supplied together as a 4D series. This allows ROMEO to use the multi-echo information rather than unwrapping each echo independently.

#### Magnitude weighting

The corresponding magnitude series is supplied because magnitude information can improve the reliability of the unwrapping path.

#### Explicit GRE analysis mask

GRE03 preferentially uses the standardized **s11b HD-BET eroded GRE analysis mask**, which was accepted during the earlier GRE preprocessing workflow.

This mask is preferred because it:

- maintains consistency with the established GRE preprocessing pipeline;
- excludes low-SNR background voxels;
- provides reproducible spatial support across downstream GRE and QSM modules.

Some recently onboarded cases do not yet contain an s11b analysis mask. For these cases, GRE03 may optionally fall back to ROMEO's internally generated `robustmask` when the user explicitly supplies:

```bash
--allow-romeo-robustmask
```

During the validated cohort run, the s11b HD-BET eroded GRE mask was used whenever available. ROMEO `robustmask` was used for cases without an existing standardized GRE mask.

The `robustmask` fallback is accepted for phase unwrapping, but it is not automatically treated as the definitive downstream QSM analysis mask. A standardized GRE/QSM mask remains preferable for quantitative background-field removal, inversion, and biological analyses.


### Brain-mask discovery policy

GRE03 searches for masks in the following order:

1. GRE02b APF brain_mask_ero1
2. Legacy s11b brain_mask_ero1
3. ROMEO robustmask (only when explicitly enabled)

This policy standardizes all downstream GRE analyses while preserving
backward compatibility with historical datasets.


#### No phase rescaling

GRE02 has already converted the Siemens phase values into radians in approximately \([-\pi,\pi]\). ROMEO normally rescales phase input automatically; therefore GRE03 passes:

```bash
--no-phase-rescale
```

This prevents a second and scientifically incorrect phase rescaling.

#### No B0 calculation in GRE03

GRE03 does **not** pass `--compute-B0`. Field estimation is retained as a distinct downstream module so that:

- unwrapping and field fitting can be validated separately;
- the fitting model and weighting remain explicit;
- ROMEO output can be compared with the legacy temporal-unwrapping estimate;
- the APF pipeline preserves modular scientific provenance.

#### No phase-offset correction by default

GRE03 v0.1.0 does not activate ROMEO/MCPC-3D-S phase-offset correction. The current input is already coil-combined scanner phase, and the appropriate offset model must be evaluated from the data rather than enabled implicitly.

Likewise, `--individual-unwrapping` is not the default because it discards ROMEO's multi-echo temporal information. It remains a diagnostic alternative if QC demonstrates failure of template-based multi-echo unwrapping.

## Outputs

Per case:

```text
derivatives/<CASE>/gre/03_romeo_unwrap/
├── <CASE>_GRE_phase_unwrapped_romeo_rad_4d.nii.gz
├── <CASE>_GRE_romeo_mask.nii.gz
├── <CASE>_GRE_romeo_command.txt
├── <CASE>_GRE_romeo_stdout.log
├── <CASE>_GRE_romeo_stderr.log
└── <CASE>_GRE_romeo_qc.json
```

Cohort summary:

```text
derivatives/summary/
├── gre03_ROMEO_unwrap_qc.csv
└── gre03_ROMEO_unwrap_run.json
```

## Automated validation and QC

Before execution, GRE03 verifies:

- phase is 4D;
- magnitude and phase shapes match;
- magnitude and phase affines match;
- the number of echo times matches the fourth dimension;
- echo times are finite, positive, and strictly increasing;
- mask dimensions and affine match the phase data;
- first-echo phase values are compatible with radian-scaled input.

After execution, GRE03 verifies:

- ROMEO returned exit code zero;
- the expected output exists;
- output shape matches input shape;
- finite-value fraction inside the effective mask;
- zero-value fraction inside the effective mask;
- masked phase minimum, maximum, median, and percentile range.

The exact command line, ROMEO executable/version, inputs, outputs, runtime, and QC values are written to JSON.

## Cohort validation

GRE03 was executed and validated on the complete cohort of ten GRE-ready in-situ cases.

### Cohort summary

| Metric | Result |
|---|---|
| Cases processed | 10 |
| Successful | 10 |
| Failed | 0 |
| Echoes per case | 12 |
| Echo-time range | 7.01–52.21 ms |
| Runtime | 36.9–65.2 s per case |
| Finite voxel fraction inside the effective mask | 1.0 for every case |
| Mask strategy | s11b HD-BET eroded GRE mask when available; ROMEO `robustmask` otherwise |

All ten cases completed successfully with ROMEO v4.7.1. No execution failures, missing outputs, shape mismatches, or non-finite values inside the effective masks were identified.

The zero-valued fractions inside the effective masks were approximately \(5.0 \times 10^{-5}\) to \(7.1 \times 10^{-5}\). These very small fractions did not indicate a general unwrapping failure but remain recorded for reproducibility and future case-level QC comparisons.

## Execution

Open the script:

```bash
code scripts/gre/03_unwrap_phase_romeo.py
```

Check ROMEO:

```bash
command -v romeo
romeo --version
```

Dry-run one case:

```bash
cd ~/ANISO_processing

python scripts/gre/03_unwrap_phase_romeo.py \
  --case ANISO_00_000_InS \
  --dry-run
```

Run one case:

```bash
python scripts/gre/03_unwrap_phase_romeo.py \
  --case ANISO_00_000_InS \
  --allow-romeo-robustmask
```

Re-run one case:

```bash
python scripts/gre/03_unwrap_phase_romeo.py \
  --case ANISO_00_000_InS \
  --allow-romeo-robustmask \
  --force
```

Run the GRE-ready cohort after single-case validation:

```bash
python scripts/gre/03_unwrap_phase_romeo.py \
  --all-gre-ready \
  --allow-romeo-robustmask
```

For network-mounted data, add:

```bash
--no-mmap
```

## Single-case validation checklist

For `ANISO_00_000_InS`:

1. Confirm the script discovers the intended GRE02 phase, magnitude, TE, and mask files.
2. Inspect `*_romeo_stdout.log` and `*_romeo_stderr.log`.
3. Confirm output dimensions match the input 4D phase.
4. Inspect wrapped versus unwrapped phase for several echoes.
5. Inspect phase differences between adjacent echoes.
6. Check for:
   - residual sharp \(2\pi\) boundaries;
   - unwrapping streaks;
   - disconnected regions;
   - unexpectedly large global offsets;
   - failures near vessels or tissue boundaries;
   - excessive zeros inside the analysis mask.
7. Compare the later GRE04 frequency estimate against the legacy s22 temporal-only estimate as a QC comparator, not as the reference solution.

## Visual validation

Representative outputs were inspected in FSLeyes by directly comparing the wrapped GRE02 phase data with the ROMEO-unwrapped GRE03 phase data at matched slices and echoes.

### Spatial phase continuity

The wrapped GRE02 phase images contained the expected abrupt transitions at the \(-\pi\) and \(+\pi\) boundaries. After ROMEO unwrapping:

- the visible phase-wrap discontinuities disappeared;
- large-scale phase gradients became spatially continuous;
- anatomical structures remained preserved;
- small venous structures remained visible;
- no obvious isolated unwrapping islands were observed;
- no obvious streaking, checkerboard patterns, or spatial branch errors were observed.

The larger numerical range of the unwrapped phase, including values substantially outside \([-\pi,\pi]\), is expected because ROMEO restores the appropriate multiples of \(2\pi\).

### Echo-wise phase evolution

The unwrapped 4D phase data were inspected while scrolling through all twelve GRE echoes.

The following behavior was observed:

- phase changed gradually and smoothly from echo to echo;
- no abrupt temporal jumps were visible;
- regions with a positive phase slope generally accumulated increasingly positive phase;
- regions with a negative phase slope generally accumulated increasingly negative phase;
- a spatially varying phase offset remained present.

The expected voxel-wise phase model is

\[
\phi(TE)=\phi_0+\omega TE,
\]

where:

- \(\phi_0\) is the phase intercept or offset;
- \(\omega\) is the angular-frequency offset.

The sign of the phase value at a particular echo does not by itself determine the sign of the frequency offset. A voxel may have negative phase values but a positive slope, depending on its intercept. The quantity relevant to field estimation is therefore the phase slope with respect to echo time, not the absolute phase value.

GRE04 will estimate this slope voxel-wise and will provide quantitative residual and fit-quality measures that serve as an additional validation of GRE03.

### Cerebellar phase appearance

The cerebellum appeared generally darker than the cerebral hemispheres in the inspected unwrapped phase images.

This appearance is physically plausible at the GRE03 stage because the data still represent total accumulated phase. The cerebellum and posterior fossa are located near the skull base, petrous temporal bones, mastoid air cells, and other strong susceptibility interfaces. These structures can produce substantial large-scale background-field perturbations.

The observed cerebellar intensity difference should therefore not be interpreted directly as tissue susceptibility or pathology. It may reflect a combination of:

- the local frequency slope;
- the phase intercept;
- large-scale background fields;
- air–bone–tissue susceptibility interfaces;
- scanner- and reconstruction-dependent phase offsets.

The multi-echo slope will be estimated during GRE04. Large-scale background-field contributions will subsequently be removed during GRE06 using V-SHARP.

### Visual QC conclusion

The inspected data showed the expected transition from wrapped to spatially continuous phase and smooth echo-wise phase evolution without obvious unwrapping failure. These observations support acceptance of GRE03 for downstream quantitative field estimation.

## Planned QC figures

The initial implementation records numerical QC. A dedicated visualization step should subsequently generate:

- wrapped and unwrapped phase mosaics for representative echoes;
- difference images between consecutive unwrapped echoes;
- spatial phase-gradient maps;
- voxel-wise linear-fit residual maps after GRE04;
- comparison with the legacy s22 frequency map.

## Methodological record

### Accepted implementation

- ROMEO v4.7.1.
- Multi-echo 4D phase unwrapping.
- GRE02 phase input already scaled to radians.
- Corresponding 4D magnitude input.
- Explicit echo-time vector in milliseconds.
- `--no-phase-rescale`.
- Preferred standardized s11b HD-BET eroded GRE mask.
- Optional ROMEO `robustmask` fallback for cases without an existing standardized GRE mask.
- Mask applied to the unwrapped output.
- No direct ROMEO B0 computation.
- No phase-offset correction by default.
- No individual-echo unwrapping by default.

### Legacy comparator

The exploratory legacy workflow applied temporal-only unwrapping:

```python
numpy.unwrap(..., axis=echo)
```

This method could correct echo-to-echo \(2\pi\) discontinuities independently within each voxel, but it could not resolve spatial phase wraps between neighboring voxels.

The legacy result is retained only as a historical and downstream QC comparator. It is not considered sufficient as the definitive phase-unwrapping method for the final APF QSM pipeline.

## Module acceptance

GRE03 was accepted after satisfying the following implementation and validation criteria.

### Automated criteria

- Successful execution for all ten GRE-ready cases.
- Correct handling of all twelve echoes.
- Correct echo-time range of 7.01–52.21 ms.
- Output dimensions matching the GRE02 phase input.
- Finite voxel fraction of 1.0 inside the effective mask for every case.
- Complete command-line, version, runtime, input, output, and QC provenance.
- No cohort-level execution failures.

### Visual criteria

- Removal of visible \(\pm\pi\) phase wraps.
- Spatially continuous unwrapped phase.
- Smooth phase evolution across all twelve echoes.
- No obvious temporal jumps.
- No obvious isolated unwrapping islands or streaking artifacts.
- Preservation of anatomical and venous structures.
- Physically plausible large-scale posterior-fossa and cerebellar phase variation.

## Analysis mask
GRE03 uses the **full GRE02b HD-BET brain mask** as the canonical spatial
domain for ROMEO phase unwrapping.

The previous workflow used the one-voxel-eroded GRE02b mask
(`*_GRE02b_brain_mask_ero1.nii.gz`). This policy was revised after explicit
single-case testing and subsequent cohort validation.

The current mask-discovery preference is:

1. full APF GRE02b HD-BET mask:
   `*_GRE02b_brain_mask.nii.gz`;
2. legacy s11b one-voxel-eroded HD-BET mask, retained only as a
   backward-compatible fallback;
3. ROMEO robust masking only when explicitly enabled.

The rationale is to avoid discarding potentially usable peripheral brain
voxels before phase quality has been evaluated. Brain-boundary erosion is
therefore no longer imposed at GRE03. Instead, QSM-specific mask refinement
and erosion are deferred to GRE05 immediately before background-field
removal.

ROMEO is called with `--mask-unwrapped`, so the written
`*_GRE_romeo_mask.nii.gz` represents the effective spatial domain used for
the unwrapped phase output.

### Cohort validation

GRE03 v0.2.2 was rerun for all 10 thesis cases using the full GRE02b mask.
For every case, the resulting ROMEO mask was voxelwise identical to the
corresponding full GRE02b brain mask.

The previous eroded-mask GRE03 outputs were retained in
`03_romeo_unwrap_ero1_reference/` for methodological comparison and
provenance.

- **Current script version:** GRE03 v0.2.2
- **Canonical mask:** full GRE02b HD-BET brain mask
- **Cohort status:** completed and validated for 10/10 cases

GRE03 is therefore accepted as the validated phase-unwrapping stage of the APF GRE/QSM pipeline.

The validated output is the input to GRE04, where voxel-wise multi-echo phase fitting will estimate the total frequency field and provide quantitative fit residuals for further validation.

## References

Dymerska B, Eckstein K, Bachrata B, et al. Phase unwrapping with a rapid opensource minimum spanning tree algorithm (ROMEO). *Magnetic Resonance in Medicine*. 2021;85:2294–2308. doi:10.1002/mrm.28563.

ROMEO software repository and command-line documentation: `korbinian90/ROMEO.jl`.
