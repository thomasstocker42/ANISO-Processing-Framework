# ANISO Processing Framework (APF)

**Research-software companion to the MSc thesis _Magnetic Susceptibility Source Separation in Postmortem In-Situ Brain MRI_ (2026).**

> **Source-available research code for methodological review.** This code documents the implementations used in the thesis; it is not a claim of independently verified end-to-end execution on a clean installation.

## Purpose and scope

The ANISO Processing Framework (APF) contains the scripts used to prepare and analyze multi-modal, postmortem in-situ brain MRI for the thesis. Its goals are to document the actual processing implementations, scientifically relevant parameter choices, and provenance of MRI relaxometry, GRE phase processing, local-field estimation, QSM, susceptibility source separation, diffusion-based fibre orientation, and MP2RAGE-based tissue processing.

This repository is a **methods and code release**, not a release of the original postmortem MRI dataset or a general-purpose, turnkey processing service. Some later specimen-specific group analyses have not yet been converted into standardized APF modules and are deliberately omitted.

## Processing overview

```text
Input MRI (multi-echo GRE, TSE, DWI, MP2RAGE)
  |-- GRE01–GRE05: phase/magnitude -> phase unwrapping -> local field
  |-- GRE06: dipole validation, TV-QSM, external STI comparison
  |-- R2STAR01–03: multi-echo GRE R2*
  |-- RLX01–07: multi-/dual-echo TSE relaxometry and EPG sensitivity
  |-- R2PRIME01–03: register R2app to GRE; R2' = R2* - R2app
  |-- CHI01–04: validate inputs -> prepare solver inputs ->
  |              external MEDI / direct-iLSQR reconstruction -> QC
  |-- DWI/MP2RAGE/ORI01/ORI03: supporting registration and orientation
  `-- Sato vesselness and some cohort-level analyses: methods described,
      implementations not part of this initial code release
```

The χ-separation methods are separate from the in-house APF scripts and rely on independently obtained third-party software. In particular, the MATLAB wrappers reference SNU-LIST χ-separation Toolbox and MEDI. Do not assume these upstream packages are redistributed here.

## Software requirements and verification status

The following package versions were **observed in the authors' `flair` environment during the release audit on 23 October 2026**. They are a reference environment snapshot, **not** a claim that this is an exhaustive dependency list or a tested clean-install lockfile.

| Component | Observed version |
| --- | --- |
| Python | 3.10.19 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| NiBabel | 5.3.3 |
| Matplotlib | 3.10.8 |
| Pandas | 2.3.3 |
| scikit-image | 0.25.2 |
| SimpleITK | 2.5.5 |
| PyTorch | 2.12.0 |
| Docker Engine | 29.4.0 |

A later static import audit identified `numpy`, `scipy`, `nibabel`, `matplotlib`, and `torch` as external direct Python imports across the 65 staged Python scripts (all five available in the inspected `flair` environment). Dynamic imports and external executables were not covered by that import scan. The audit confirmed that executable paths were available for ROMEO, FSL (`fslmaths`, TOPUP, EDDY and DTIFIT), and HD-BET. Their versions and end-to-end compatibility have **not yet been independently verified in the staged release**. FSL FAST and registration tools are required by the corresponding stages. Some QSM validation/reference runners use externally supplied, versioned QSM-CI/STI containers.

**MATLAB and χ-separation:** The Python CHI03b script is an orchestration and preflight wrapper; it does not implement MEDI in Python. The MATLAB runners call external SNU-LIST χ-separation Toolbox (v1.2.2 as recorded by the code) and MEDI functions; these toolboxes are **not included**. The author's implementation uses a separately provisioned, authenticated MATLAB R2026a container with the following mount convention:

- `/project` — writable APF project directory containing user-provided MRI inputs and derivatives.
- `/toolbox` — separately obtained SNU-LIST χ-separation toolbox, read-only.
- `/medi` — separately obtained MEDI toolbox, read-only.

The CHI03b implementation uses GRE06c STI-iLSQR QSM as an **input**, in addition to CHI02-prepared data. Thus GRE06c's external reference solver is necessary to reproduce that particular implemented CHI03b route. The required MATLAB image `matlab-r2026a-chisep:latest` was **not available in the Docker environment inspected on 23 October 2026**, although the Docker engine itself was running. A host `matlab` executable was also not on `PATH`. Accordingly, **MATLAB-based χ-separation execution has not been reproduced in that environment as part of this public-release audit**. Docker availability by itself does not imply MATLAB licensing or authentication.

Third-party software must be obtained and used under its own license; this repository does not redistribute any MATLAB, SNU-LIST, MEDI, STI or ROMEO package. The private `flair_*` environment files are not included as authoritative installation instructions.

## Expected local layout

```text
ANISO-Processing-Framework/
  apf
  scripts/
    infrastructure/
    gre/
    relaxometry/
    diffusion/
    t1/
    orientation/
    chiseparation/
  config/
    InS_cases.example.csv
  raw_dicom/             # user-supplied; never commit
  nifti/                 # user-supplied / locally generated; never commit
  derivatives/           # generated outputs; never commit
```

APF uses a case registry and project-relative derivatives. The required registry columns include `case`, `status`, `type`, `date_added`, and `raw_root`. Copy the **synthetic example registry** to the location required by `apf_registry.py` and adapt it privately to your own de-identified dataset; do not commit actual study registries or metadata. The example entry is a format demonstration only and contains no MRI inputs. Case IDs are file and directory keys, not proof of de-identification.

## Inspecting and executing an APF module

After obtaining this release, install the direct Python requirements in a suitable Python 3.10 environment. For a basic Python-only installation (not a tested environment lockfile):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy nibabel matplotlib torch
```

FSL, ROMEO, HD-BET, MATLAB, MEDI, SNU-LIST and the referenced container images are installed/configured separately where needed. Make sure the relevant external executables can be found on `PATH`. To set up your own case registry privately (the provided example contains no imaging data):

```bash
cp config/InS_cases.example.csv config/InS_cases.csv
# Edit config/InS_cases.csv for your own authorized, de-identified data
```

From the repository root, after installing the software requirements:

```bash
export APF_ROOT="$(pwd)"
./apf scripts/gre/01_inventory.py --help
./apf scripts/relaxometry/r2star/02_fit_loglinear.py --help
./apf scripts/chiseparation/01_inventory.py --help
```

The helper `./apf` runs the requested Python script with the APF infrastructure directory on `PYTHONPATH`. The APF project root is discovered relative to the installed code unless `APF_ROOT` is set explicitly. Individual modules may use a positional case identifier, `--all`, `--all-gre-ready`, or other module-specific selection flags; inspect each script's `--help` before running it.

**Example, only once a case has been correctly registered and all prerequisites exist:**

```bash
./apf scripts/gre/01_inventory.py YOUR_CASE_ID
./apf scripts/gre/02_prepare_phase.py YOUR_CASE_ID
./apf scripts/relaxometry/r2star/02_fit_loglinear.py YOUR_CASE_ID
./apf scripts/chiseparation/01_inventory.py YOUR_CASE_ID
```

These are **illustrative stage invocations**, not a complete executable pipeline. Upstream steps (including brain masking, unwrapping, field estimation, relaxometry registration and suitable QSM reconstruction) must be completed in the documented order. The CHI input inventory will reject missing or incompatible inputs rather than silently generate them.

Where the source uses a container mount convention such as `/project`, `/toolbox`, or `/medi`, configure the corresponding paths or mounts explicitly. Paths to the authors' original workstation are not required for the intended release.

## Reproducing the thesis methods

Consult the curated `docs/modules/` pages for the scientific definitions, input-output contracts, algorithm settings and QC decisions. Not all thesis figures can be regenerated without the original postmortem imaging data or later, unreleased cohort-analysis scripts. The presence of an analysis figure in this repository does not imply its generation script is included.

## Data availability and confidentiality

No source DICOM/NIfTI images, private specimen registries, or original MRI processing derivatives are distributed as part of the code release. A supplementary figure that was already intentionally published in the public repository is retained; its case labels should not be confused with a release of the underlying imaging data. You must obtain appropriate institutional authorization and ensure de-identification before running the code on any research or clinical data. Some processed data and software dependencies cannot be redistributed under institutional restrictions or third-party licenses.

## Code availability, copyright and citation

The ANISO Processing Framework is made publicly available primarily to
support transparency and methodological review of the accompanying MSc
thesis. The repository documents processing implementations, parameter
choices, and selected validation procedures.

Copyright (c) 2026 Thomas R. Stocker. All rights reserved.

No general open-source license is granted at this stage. Public access
does not constitute permission for unrestricted copying, modification,
redistribution, or commercial reuse. GitHub's applicable platform terms
continue to apply. Requests for additional permissions may be directed
to the author.

Software citation metadata is provided in `CITATION.cff`. Relevant
original scientific method publications should also be cited.

Third-party software, external toolboxes, MRI data, and separately
published supplementary material remain subject to their respective
rights and licensing conditions.

## Release status

APF v0.1 is a source-available research-code and methods release
intended primarily for methodological inspection.

Release-preparation checks included syntax parsing of 65 Python files,
eight selected command-line help tests, dependency inspection, and
an automated audit of the 91-file release candidate with zero
automated blockers.

The absence of a general open-source license is intentional.
Automated screening does not guarantee the absence of sensitive
information. Independent clean-machine end-to-end execution,
external MATLAB reconstruction, and complete third-party licensing
verification have not been established. This release is not presented
as a turnkey processing pipeline.
