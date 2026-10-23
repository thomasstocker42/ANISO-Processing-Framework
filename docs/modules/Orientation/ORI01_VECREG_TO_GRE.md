# ORI01 — Principal diffusion eigenvector registration to GRE

**Version:** 0.1.0; **status:** cohort V1 fields generated from terminal commands on 2026-10-15, requiring formalization and independent QC. This is an upstream module for ORI03.

## Purpose

Produce or validate the b1000 WLS principal diffusion eigenvector (DTIFIT V1) in the common GRE analysis geometry. Explicitly preserve undirected fibre-axis information. No R2*, MP2RAGE or DWI preprocessing results are changed. The same registered V1 is used for R2*, QSM, χpara and χdia orientation analysis.

## Inputs per active in-situ case

- `derivatives/<case>/DWI/dtifit/b1000/dwi05_<case>_multi_b_N30_b1000_wls_V1.nii.gz`
- `derivatives/<case>/DWI/registration/dwi06/dwi06_<case>_b0_to_GRE.mat`
- `derivatives/<case>/GRE/mag_masked/s12_<case>_GRE_e7_masked_by_s11b_e1_hdbet_cpu_notta.nii.gz`

The affine and V1 must describe the same native DWI grid. Review DWI06 registration before accepting this derivative. The GRE reference is authoritative (256×256×144, 1 mm isotropic for the pilot).

## Method

```bash
vecreg -i "$native" -r "$ref" -t "$mat" -o "$out"
```

FSL `vecreg.cc` (installed source inspected 2026-10-15) 3-component affine branch builds/interpolates the six components of a dyadic tensor (`vvᵀ`), recovers its principal eigenvector, applies the **full upper-left 3×3 FLIRT linear transform F** to that vector, and normalizes it. Do **not** claim it uses only the polar rotation: the source calculates R but applies F in this vector branch. This differs from scalar FLIRT interpolation used in Lennart's supplied `Registrierung_etc_lb_HDBET.sh`, where component reorientation is not explicit. This is a stronger method for undirected fibre axes but does not prove DWI06 registration accuracy.

Vectors are interpreted as FSL scaled-voxel components; GRE-space B0 is derived downstream in ORI03 from the GRE affine and assumed scanner-world +z. In one validation case, independently derived GRE-FSL B0 `(0.12572902,0.23432712,0.96399326)` matched the candidate vector with reported angular difference 0.0°. Synthetic constant x/y/z field experiments using the same matrix tested the `vecreg` command but do not prove absolute physical DWI registration.

## Outputs

- `derivatives/<case>/DWI/orientation_validation/ori01_<case>_b1000_V1_vecreg.nii.gz`
- `derivatives/<case>/DWI/orientation_validation/ori01_<case>_b1000_vecreg_provenance.json`
- `derivatives/summary/orientation/ori01/ori01_qc_summary.csv`

The existing `orientation_validation` directory is retained for backward compatibility with ORI03 v0.2.0; migrating to another path would require a separate ADR.

## Reproducibility and safety

`./apf scripts/orientation/ori01_vecreg_to_gre.py --all --preflight` checks dependencies without writing. `--all` uses `apf_paths.get_paths().ins_registry` and `apf_case_selection.resolve_cases`, analogous to ORI03. When the V1 output exists, the module **does not run `vecreg` or rewrite V1**, but validates geometry/norm and records input/output SHA-256 in JSON provenance. When missing, it generates the image via `vecreg` in a unique staging filename, validates it and then renames it into place. Existing provenance is retained unless `--replace-provenance` is supplied. `--force` deliberately refuses to overwrite V1; explicit manual archive of the existing image is necessary for replacement. There is no silent fallback to DWI08 FLIRT-only V1.

## Automatic QC

Checks that native V1 has a three-component 4D volume; DWI06 matrix is finite, homogeneous and nonsingular; reference is 3D; output has GRE target shape with three components; output/reference affines agree within 0.001 mm; significant nonzero valid V1 support exists; and the 99th percentile of unit-norm error for valid vectors is ≤0.02. It writes valid voxel counts, support fraction, norm percentiles and affine-difference metrics.

**Limitations:** Unit-norm QC is not an assessment of registration accuracy, crossing fibres, DTI quality or image anatomy. Formal anatomical overlays against GRE and FA are a manual acceptance gate; do not claim these have passed merely because the module succeeds. The +z B0 scanner assumption requires acquisition confirmation. Source derivatives from console execution are traceable by hashes and command, but the historical console execution environment was not automatically recorded when created; subsequent reproduction must record FSL software version.

## Execution, after nine terminal `vecreg` outputs already exist

```bash
cd ~/ANISO_processing
./apf -m py_compile scripts/orientation/ori01_vecreg_to_gre.py
./apf scripts/orientation/ori01_vecreg_to_gre.py --all --preflight
./apf scripts/orientation/ori01_vecreg_to_gre.py --all
./apf scripts/orientation/ori03_r2star_wm.py --all --preflight
```

Before executing ORI03 cohort, separately prove ORI03 v0.2.0 reproduces the original 077 ORI03 pilot and complete/record anatomical overlay QC. ORI02a/b remain exploratory terminal-based sensitivity analyses, not silently relabelled as a finalized APF module.
