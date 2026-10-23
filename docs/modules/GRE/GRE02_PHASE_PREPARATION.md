# GRE02 – Phase Preparation

**Module:** GRE02
**Status:** Implemented
**APF Version:** v0.1
**Related ADRs:** ADR-017, ADR-023, ADR-024, ADR-027

---

# 1. Purpose

GRE02 prepares all quantitative multi-echo GRE data required for subsequent susceptibility mapping and χ-separation.

The module converts vendor-specific phase representations into physically meaningful phase values, prepares the corresponding magnitude images, extracts acquisition metadata, and generates standardized outputs for downstream processing.

The primary objectives are to:

- standardize multi-echo phase data,
- prepare multi-echo magnitude data,
- extract echo times,
- determine the static magnetic field direction,
- establish reproducible metadata,
- provide standardized inputs for GRE03 and subsequent modules.

GRE02 deliberately performs **no quantitative estimation**. It prepares the dataset while preserving the original image information.

---

# 2. Scientific Background

Quantitative susceptibility mapping (QSM) relies on the linear evolution of MRI phase with echo time:

\[
\phi(TE)=\phi_0+\omega \cdot TE
\]

where

- \(\phi\) is the measured phase,
- \(\phi_0\) is the phase offset,
- \(\omega\) is the local frequency offset.

Accurate estimation of the local frequency requires:

- correctly scaled phase values,
- accurate echo times,
- knowledge of the main magnetic field orientation,
- reliable anatomical masking.

Vendor-specific phase encodings therefore must be converted into standardized physical units before any quantitative processing is performed.

---

# 3. Inputs

GRE02 expects the outputs of GRE01.

Primary inputs include:

- multi-echo GRE phase images
- multi-echo GRE magnitude images
- acquisition JSON sidecars
- DICOM-derived metadata
- NIfTI image geometry

The module automatically detects Siemens integer phase encoding and converts it into radians.

---

# 4. Outputs

For every case GRE02 generates

```
GRE/phase_preparation_gre02/
```

including

```
phase_rad_4d.nii.gz
magnitude_4d.nii.gz

echo_times_ms.txt
echo_times_s.txt

b0_direction.txt

phase_ranges.json
```

These outputs constitute the standardized GRE preparation stage for all subsequent modules.

---

# 5. Processing Steps

## 5.1 Magnitude preparation

The individual magnitude echoes are assembled into a single four-dimensional volume while preserving:

- image geometry,
- voxel spacing,
- acquisition ordering.

No filtering or intensity normalization is performed.

---

## 5.2 Phase preparation

Vendor-specific integer phase values are converted into physical phase units.

For Siemens acquisitions this corresponds to

```
[-4096,4095]
```

being mapped to

```
[-π,+π]
```

The prepared phase images are stored as floating-point radians.

No phase unwrapping is performed in GRE02.

---

## 5.3 Echo-time extraction

Echo times are extracted from the acquisition metadata and stored in both

- milliseconds
- seconds

to avoid repeated unit conversions in downstream modules.

These values define the independent variable used during total-field estimation.

---

## 5.4 B0 direction extraction

The orientation of the main magnetic field is extracted from the image affine.

The resulting normalized B0 vector is stored for later use by

- dipole kernel generation,
- susceptibility inversion,
- χ-separation.

---

# 6. Quality Control

GRE02 performs automated quality control including

- phase range verification,
- magnitude statistics,
- echo-time consistency,
- phase scaling validation,
- B0 direction extraction.

Module summaries are written to

```
derivatives/summary/GRE/
```

allowing rapid cohort-wide inspection.

---

# 7. GRE02b – Brain Mask Generation

## 7.1 Motivation

Reliable quantitative susceptibility processing requires a standardized anatomical analysis domain.

Earlier exploratory processing occasionally relied on ROMEO's internally generated robust mask when no anatomical mask was available.

Although suitable for phase unwrapping, the ROMEO robust mask is not guaranteed to represent brain tissue exclusively and may include extracranial structures.

GRE02b establishes a reproducible anatomical mask before any quantitative processing begins.

---

## 7.2 HD-BET

GRE02b performs brain extraction using HD-BET.

The module

1. extracts the first GRE magnitude echo,
2. performs HD-BET brain extraction,
3. converts the output into a standardized binary mask,
4. creates a one-voxel-eroded mask,
5. records complete provenance and quality-control information.

Outputs are stored in

```
GRE/02b_brain_mask/
```

---

## 7.3 Why the first GRE echo?

The first echo provides

- highest signal-to-noise ratio,
- minimal T2* decay,
- reduced susceptibility-induced signal loss,
- optimal anatomical contrast for brain extraction.

Later echoes exhibit increasing signal attenuation and susceptibility dropout, particularly near air–bone interfaces, making them less suitable for robust masking.

---

## 7.4 Why the one-voxel-eroded mask?

The eroded mask is used as the canonical GRE analysis mask for all downstream processing.

Reasons include:

### Boundary robustness

Brain boundary voxels frequently contain partial-volume mixtures of

- brain tissue,
- CSF,
- dura,
- skull,
- scalp.

Removing a single voxel substantially reduces these boundary effects.

### Improved quantitative processing

The eroded mask improves robustness during

- phase unwrapping,
- total-field estimation,
- background-field removal,
- QSM inversion,
- χ-separation.

### Consistent analysis domain

Using one standardized anatomical mask ensures that all downstream GRE modules operate on exactly the same voxel support.

Observed cohort quality control demonstrated highly consistent erosion behaviour across all ten cases, with approximately 96% of brain-mask voxels retained after one-voxel erosion.

---

## 7.5 Outputs

GRE02b generates

```
GRE/02b_brain_mask/
```

containing

```
<CASE>_GRE02b_magnitude_e1.nii.gz

<CASE>_GRE02b_brain_extracted_e1.nii.gz

<CASE>_GRE02b_brain_mask.nii.gz

<CASE>_GRE02b_brain_mask_ero1.nii.gz

<CASE>_GRE02b_qc.json
```

Cohort summaries are written to

```
derivatives/summary/GRE/
```

---

# 8. Downstream Dependencies

The standardized GRE02 outputs are consumed by

- GRE03 — ROMEO phase unwrapping
- GRE04a — Ordinary least-squares total-field estimation
- GRE04b — Magnitude-weighted least-squares total-field estimation
- GRE05 — Background-field removal (V-SHARP)
- GRE06 — QSM inversion
- GRE07 — χ-separation

The GRE02b one-voxel-eroded mask is defined as the canonical anatomical analysis mask throughout the GRE processing pipeline (see ADR-027).

---

# 9. References

1. Dymerska B, Eckstein K, Bachrata B, et al. *Phase Unwrapping with ROMEO*. Magnetic Resonance in Medicine.

2. Isensee F, et al. *HD-BET: Automated brain extraction of multi-sequence MRI using artificial neural networks*. Human Brain Mapping. 2019.

3. Wang Y, Liu T. *Quantitative Susceptibility Mapping (QSM): Decoding MRI Data for a Tissue Magnetic Biomarker*. Magnetic Resonance in Medicine.

4. Schweser F, et al. *Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase*. NeuroImage.

5. APF ADR-027 — Standardized GRE Analysis Mask.