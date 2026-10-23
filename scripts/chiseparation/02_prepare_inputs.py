#!/usr/bin/env python3
"""
APF CHI02: prepare exact QSM-CI chi-sep-ilsqr solver inputs.

Version 0.1.0

CHI02 consumes the inputs validated by finalized CHI01 and creates a
self-contained solver-input package matching the audited QSM-CI wrapper:

    localfield.nii.gz   local field [ppm]
    r2prime.nii.gz      raw R2-prime [s^-1 == Hz]
    magnitude.nii.gz    4D GRE magnitude [x,y,z,echo]
    mask.nii.gz         production chi-separation support
    params.json         B0, B0_dir, voxel_size, TE

Scientific policy
-----------------
- No resampling.
- No R2-prime clipping.
- No QSM input is supplied; chi-sep-ilsqr reconstructs QSM internally.
- Production support is the finalized susceptibility/relaxometry intersection:
      V-SHARP mask
      ∩ R2PRIME03 common support
      ∩ finite local field
      ∩ finite R2-prime
- Magnitude data must be finite, non-negative, and non-zero in that support.
- The complete measured TE vector is retained in seconds.
- Hz -> ppm uses 42.5774e6 Hz/T exactly, matching the audited recon.m wrapper.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from apf_paths import APFPaths, get_paths
from apf_case_selection import (
    add_case_selection_args,
    print_selected_cases,
    resolve_cases,
)


SCRIPT = "02_prepare_inputs.py"
SCRIPT_ID = "CHI02"
TAG = "chi02"
VERSION = "0.1.0"

EXPECTED_ECHOES = tuple(range(1, 13))
AFFINE_ATOL = 1e-5

# IMPORTANT:
# Use the exact constant from cached QSM-CI recon.m so that
# CHI02 Hz -> ppm followed by recon.m ppm -> Hz is self-consistent.
QSM_CI_GAMMA_HZ_PER_T = 42.5774e6

QSM_CI_METHOD = "chi-sep-ilsqr"
QSM_CI_RECORD_ID = "21697749"
QSM_CI_IMAGE = (
    "ghcr.io/astewartau/qsm-ci/"
    "chi-sep-ilsqr@sha256:"
    "d07d167420d04a26403c20d42d47269ecd95ff354529521dddd282449e6bf35a"
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def case_paths(
    paths: APFPaths,
    case: str,
) -> dict[str, Path]:
    gre = paths.derivatives_dir / case / "GRE"

    return {
        "local_field_hz": (
            gre
            / "05b_vsharp_apf_v030"
            / f"{case}_GRE05b_local_field_hz.nii.gz"
        ),
        "vsharp_mask": (
            gre
            / "05b_vsharp_apf_v030"
            / f"{case}_GRE05b_vsharp_mask.nii.gz"
        ),
        "b0_json": (
            gre
            / "phase_preparation_gre02"
            / f"gre02_{case}_b0_direction.json"
        ),
        "r2prime": (
            paths.derivatives_dir
            / case
            / "relaxometry"
            / "r2prime"
            / "r2prime03_construct"
            / f"r2prime03_{case}_R2prime_raw_s-1.nii.gz"
        ),
        "r2prime_mask": (
            paths.derivatives_dir
            / case
            / "relaxometry"
            / "r2prime"
            / "r2prime03_construct"
            / f"r2prime03_{case}_common_mask.nii.gz"
        ),
        "mag_dir": (
            paths.root
            / "nifti"
            / case
            / "GRE"
            / "mag"
        ),
        "chi01_metadata": (
            paths.derivatives_dir
            / case
            / "chi_separation"
            / "chi01_inventory"
            / f"chi01_{case}_metadata.json"
        ),
    }


def output_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "chi_separation"
        / "chi02_prepare"
    )


def cohort_output_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "chi_separation"
        / "chi02_prepare"
    )


def geometry_equal(
    reference: nib.spatialimages.SpatialImage,
    other: nib.spatialimages.SpatialImage,
) -> bool:
    return (
        reference.shape == other.shape
        and np.allclose(
            reference.affine,
            other.affine,
            atol=AFFINE_ATOL,
            rtol=0,
        )
    )


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: Any,
) -> None:
    hdr = reference.header.copy()
    hdr.set_data_dtype(dtype)
    hdr.set_slope_inter(1.0, 0.0)

    img = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        hdr,
    )

    # A stacked GRE echo dimension is not a uniformly sampled time axis.
    # Echo timing is represented explicitly by the full TE vector in
    # params.json. Therefore do not inherit an unrelated pixdim[4] from
    # the source 3D GRE header.
    if data.ndim == 4:
        img.header.set_zooms(
            (*reference.header.get_zooms()[:3], 1.0)
        )

    qcode = int(reference.header["qform_code"])
    scode = int(reference.header["sform_code"])

    if qcode > 0:
        img.set_qform(
            reference.get_qform(),
            code=qcode,
        )

    if scode > 0:
        img.set_sform(
            reference.get_sform(),
            code=scode,
        )

    nib.save(
        img,
        str(path),
    )


def read_b0_direction(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())

    if "voxel_b0_direction" not in data:
        raise RuntimeError(
            f"Missing voxel_b0_direction in {path}"
        )

    b0 = np.asarray(
        data["voxel_b0_direction"],
        dtype=np.float64,
    )

    if b0.shape != (3,):
        raise RuntimeError(
            f"Invalid B0 direction shape: {b0.shape}"
        )

    if not np.all(np.isfinite(b0)):
        raise RuntimeError(
            "B0 direction contains non-finite values"
        )

    norm = float(np.linalg.norm(b0))

    if not np.isclose(
        norm,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"B0 direction is not normalized: {norm}"
        )

    return b0


def discover_echo_series(
    mag_dir: Path,
    case: str,
) -> tuple[list[Path], list[Path], str]:
    if not mag_dir.is_dir():
        raise FileNotFoundError(
            f"Missing GRE magnitude directory: {mag_dir}"
        )

    pattern = re.compile(
        rf"^(?P<stem>{re.escape(case)}_GRE_12echo_\d+)"
        rf"_e(?P<echo>\d+)\.nii\.gz$"
    )

    series: dict[str, dict[int, Path]] = {}

    for p in mag_dir.glob(
        f"{case}_GRE_12echo_*_e*.nii.gz"
    ):
        m = pattern.match(p.name)

        if not m:
            continue

        stem = m.group("stem")
        echo = int(m.group("echo"))

        series.setdefault(
            stem,
            {},
        )[echo] = p

    complete = {
        stem: echoes
        for stem, echoes in series.items()
        if tuple(sorted(echoes)) == EXPECTED_ECHOES
    }

    if len(complete) != 1:
        discovered = {
            stem: tuple(sorted(echoes))
            for stem, echoes in series.items()
        }

        raise RuntimeError(
            "Expected exactly one complete 12-echo GRE magnitude "
            f"series; found {len(complete)}. "
            f"Discovered: {discovered}"
        )

    stem, echoes = next(
        iter(complete.items())
    )

    niftis = [
        echoes[e]
        for e in EXPECTED_ECHOES
    ]

    jsons: list[Path] = []

    for nifti in niftis:
        js = nifti.with_suffix("").with_suffix(".json")

        require_file(
            js,
            f"JSON sidecar for {nifti.name}",
        )

        jsons.append(js)

    return niftis, jsons, stem


def read_acquisition(
    json_paths: list[Path],
) -> tuple[list[float], float]:
    te_s: list[float] = []
    b0_values: list[float] = []

    for path in json_paths:
        data = json.loads(
            path.read_text()
        )

        if "EchoTime" not in data:
            raise RuntimeError(
                f"Missing EchoTime in {path}"
            )

        if "MagneticFieldStrength" not in data:
            raise RuntimeError(
                f"Missing MagneticFieldStrength in {path}"
            )

        te = float(
            data["EchoTime"]
        )

        b0 = float(
            data["MagneticFieldStrength"]
        )

        if not np.isfinite(te) or te <= 0:
            raise RuntimeError(
                f"Invalid EchoTime in {path}: {te}"
            )

        if not np.isfinite(b0) or b0 <= 0:
            raise RuntimeError(
                f"Invalid B0 in {path}: {b0}"
            )

        te_s.append(te)
        b0_values.append(b0)

    te = np.asarray(
        te_s,
        dtype=np.float64,
    )

    if not np.all(
        np.diff(te) > 0
    ):
        raise RuntimeError(
            f"Echo times not strictly increasing: {te_s}"
        )

    b0_array = np.asarray(
        b0_values,
        dtype=np.float64,
    )

    if not np.allclose(
        b0_array,
        b0_array[0],
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            f"Inconsistent B0 across echoes: {b0_values}"
        )

    return te_s, float(b0_array[0])


def validate_chi01(
    path: Path,
) -> dict[str, Any]:
    require_file(
        path,
        "CHI01 metadata",
    )

    data = json.loads(
        path.read_text()
    )

    if data.get("status") != "READY":
        raise RuntimeError(
            f"CHI01 is not READY: {path}"
        )

    readiness = data.get(
        "readiness",
        {},
    )

    if not readiness.get(
        "ready",
        False,
    ):
        raise RuntimeError(
            f"CHI01 readiness flag is false: {path}"
        )

    return data


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def process_case(
    paths: APFPaths,
    case: str,
    overwrite: bool,
) -> dict[str, Any]:
    print("=" * 72)
    print(f"{SCRIPT_ID}: {case}")
    print("=" * 72)

    inp = case_paths(
        paths,
        case,
    )

    for key in (
        "local_field_hz",
        "vsharp_mask",
        "b0_json",
        "r2prime",
        "r2prime_mask",
        "chi01_metadata",
    ):
        require_file(
            inp[key],
            key,
        )

    chi01 = validate_chi01(
        inp["chi01_metadata"]
    )

    mag_paths, json_paths, series_stem = discover_echo_series(
        inp["mag_dir"],
        case,
    )

    te_s, b0_t = read_acquisition(
        json_paths
    )

    b0_dir = read_b0_direction(
        inp["b0_json"]
    )

    field_img = nib.load(
        str(inp["local_field_hz"])
    )

    vsharp_img = nib.load(
        str(inp["vsharp_mask"])
    )

    r2prime_img = nib.load(
        str(inp["r2prime"])
    )

    r2mask_img = nib.load(
        str(inp["r2prime_mask"])
    )

    for label, image in (
        ("V-SHARP mask", vsharp_img),
        ("R2prime", r2prime_img),
        ("R2prime mask", r2mask_img),
    ):
        if not geometry_equal(
            field_img,
            image,
        ):
            raise RuntimeError(
                f"Geometry mismatch: {label}"
            )

    magnitude_images: list[nib.spatialimages.SpatialImage] = []
    magnitude_arrays: list[np.ndarray] = []

    for index, path in enumerate(
        mag_paths,
        start=1,
    ):
        image = nib.load(
            str(path)
        )

        if not geometry_equal(
            field_img,
            image,
        ):
            raise RuntimeError(
                f"Geometry mismatch: magnitude echo {index}: {path}"
            )

        arr = np.asarray(
            image.dataobj,
            dtype=np.float32,
        )

        magnitude_images.append(
            image
        )

        magnitude_arrays.append(
            arr
        )

    field_hz = np.asarray(
        field_img.dataobj,
        dtype=np.float32,
    )

    vsharp = (
        np.asarray(
            vsharp_img.dataobj
        )
        > 0
    )

    r2prime = np.asarray(
        r2prime_img.dataobj,
        dtype=np.float32,
    )

    r2mask = (
        np.asarray(
            r2mask_img.dataobj
        )
        > 0
    )

    candidate_mask = (
        vsharp
        & r2mask
        & np.isfinite(field_hz)
        & np.isfinite(r2prime)
    )

    n_mask = int(
        np.count_nonzero(candidate_mask)
    )

    if n_mask == 0:
        raise RuntimeError(
            "Production chi-separation support is empty"
        )

    mag4 = np.stack(
        magnitude_arrays,
        axis=3,
    ).astype(
        np.float32,
        copy=False,
    )

    mag_finite = np.all(
        np.isfinite(mag4),
        axis=3,
    )

    bad_mag_finite = int(
        np.count_nonzero(
            candidate_mask
            & ~mag_finite
        )
    )

    if bad_mag_finite:
        raise RuntimeError(
            f"{bad_mag_finite} production-support voxels "
            "contain non-finite magnitude values"
        )

    negative_mag = int(
        np.count_nonzero(
            candidate_mask[..., None]
            & (mag4 < 0)
        )
    )

    if negative_mag:
        raise RuntimeError(
            f"{negative_mag} negative magnitude samples "
            "inside production support"
        )

    mag_rss = np.sqrt(
        np.sum(
            mag4 * mag4,
            axis=3,
            dtype=np.float64,
        )
    )

    zero_mag_rss = int(
        np.count_nonzero(
            candidate_mask
            & (mag_rss <= 0)
        )
    )

    if zero_mag_rss:
        raise RuntimeError(
            f"{zero_mag_rss} production-support voxels "
            "have zero RSS magnitude"
        )

    chi01_expected = (
        chi01
        .get("support", {})
        .get("prospective_common_voxels")
    )

    if (
        chi01_expected is not None
        and int(chi01_expected) != n_mask
    ):
        raise RuntimeError(
            "CHI02 support differs from finalized CHI01 "
            f"prospective support: {n_mask} vs {chi01_expected}"
        )

    field_ppm = (
        field_hz.astype(np.float64)
        / (QSM_CI_GAMMA_HZ_PER_T * b0_t)
        * 1e6
    ).astype(
        np.float32
    )

    # Verify numerical round-trip against the exact recon.m constant.
    field_roundtrip_hz = (
        field_ppm.astype(np.float64)
        * 1e-6
        * QSM_CI_GAMMA_HZ_PER_T
        * b0_t
    )

    roundtrip_diff = (
        field_roundtrip_hz
        - field_hz.astype(np.float64)
    )

    roundtrip_rmse_hz = float(
        np.sqrt(
            np.mean(
                np.square(
                    roundtrip_diff[candidate_mask]
                )
            )
        )
    )

    roundtrip_maxabs_hz = float(
        np.max(
            np.abs(
                roundtrip_diff[candidate_mask]
            )
        )
    )

    n_negative = int(
        np.count_nonzero(
            candidate_mask
            & (r2prime < 0)
        )
    )

    negative_pct = (
        100.0
        * n_negative
        / n_mask
    )

    vox = tuple(
        float(v)
        for v in field_img.header.get_zooms()[:3]
    )

    out = output_dir(
        paths,
        case,
    )

    solver = (
        out
        / "qsm_ci_input"
    )

    metadata_path = (
        out
        / f"chi02_{case}_metadata.json"
    )

    summary_path = (
        out
        / f"chi02_{case}_summary.csv"
    )

    if out.exists():
        existing_solver_outputs = [
            solver / "localfield.nii.gz",
            solver / "r2prime.nii.gz",
            solver / "magnitude.nii.gz",
            solver / "mask.nii.gz",
            solver / "params.json",
            metadata_path,
            summary_path,
        ]

        if (
            any(p.exists() for p in existing_solver_outputs)
            and not overwrite
        ):
            raise FileExistsError(
                f"CHI02 outputs already exist for {case}. "
                "Use --overwrite."
            )

        if overwrite and solver.exists():
            shutil.rmtree(
                solver
            )

    solver.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_nifti(
        field_ppm,
        field_img,
        solver / "localfield.nii.gz",
        np.float32,
    )

    # R2prime is staged without clipping or physical-value transformation.
    save_nifti(
        r2prime,
        r2prime_img,
        solver / "r2prime.nii.gz",
        np.float32,
    )

    save_nifti(
        mag4,
        magnitude_images[0],
        solver / "magnitude.nii.gz",
        np.float32,
    )

    save_nifti(
        candidate_mask.astype(np.uint8),
        field_img,
        solver / "mask.nii.gz",
        np.uint8,
    )

    params = {
        "B0":
            float(b0_t),
        "B0_dir":
            [
                float(v)
                for v in b0_dir
            ],
        "voxel_size":
            [
                float(v)
                for v in vox
            ],
        "TE":
            [
                float(v)
                for v in te_s
            ],
    }

    (
        solver
        / "params.json"
    ).write_text(
        json.dumps(
            params,
            indent=2,
        )
        + "\n"
    )

    # Re-open saved products and validate exact solver package.
    lf_out = nib.load(
        str(solver / "localfield.nii.gz")
    )

    rp_out = nib.load(
        str(solver / "r2prime.nii.gz")
    )

    mag_out = nib.load(
        str(solver / "magnitude.nii.gz")
    )

    mask_out = nib.load(
        str(solver / "mask.nii.gz")
    )

    if lf_out.shape != field_img.shape:
        raise RuntimeError(
            "Saved localfield shape mismatch"
        )

    if rp_out.shape != field_img.shape:
        raise RuntimeError(
            "Saved r2prime shape mismatch"
        )

    if mask_out.shape != field_img.shape:
        raise RuntimeError(
            "Saved mask shape mismatch"
        )

    expected_mag_shape = (
        *field_img.shape,
        len(EXPECTED_ECHOES),
    )

    if mag_out.shape != expected_mag_shape:
        raise RuntimeError(
            "Saved magnitude shape mismatch: "
            f"{mag_out.shape} vs {expected_mag_shape}"
        )

    for label, image in (
        ("localfield output", lf_out),
        ("r2prime output", rp_out),
        ("mask output", mask_out),
    ):
        if not np.allclose(
            field_img.affine,
            image.affine,
            atol=AFFINE_ATOL,
            rtol=0,
        ):
            raise RuntimeError(
                f"Saved affine mismatch: {label}"
            )

    if not np.allclose(
        field_img.affine,
        mag_out.affine,
        atol=AFFINE_ATOL,
        rtol=0,
    ):
        raise RuntimeError(
            "Saved magnitude affine mismatch"
        )

    saved_mask = (
        np.asarray(
            mask_out.dataobj
        )
        > 0
    )

    if not np.array_equal(
        saved_mask,
        candidate_mask,
    ):
        raise RuntimeError(
            "Saved production mask differs from in-memory mask"
        )

    row = {
        "case":
            case,
        "status":
            "READY",
        "series":
            series_stem,
        "n_echoes":
            len(mag_paths),
        "B0_T":
            b0_t,
        "dTE12_ms":
            (te_s[1] - te_s[0]) * 1000.0,
        "mask_voxels":
            n_mask,
        "negative_R2prime_voxels":
            n_negative,
        "negative_R2prime_pct":
            negative_pct,
        "localfield_ppm_min_mask":
            float(
                np.min(
                    field_ppm[candidate_mask]
                )
            ),
        "localfield_ppm_max_mask":
            float(
                np.max(
                    field_ppm[candidate_mask]
                )
            ),
        "localfield_ppm_sd_mask":
            float(
                np.std(
                    field_ppm[candidate_mask]
                )
            ),
        "hz_ppm_hz_roundtrip_rmse":
            roundtrip_rmse_hz,
        "hz_ppm_hz_roundtrip_maxabs":
            roundtrip_maxabs_hz,
        "magnitude_shape":
            "x".join(
                str(v)
                for v in mag4.shape
            ),
    }

    metadata = {
        "script":
            SCRIPT,
        "script_id":
            SCRIPT_ID,
        "tag":
            TAG,
        "version":
            VERSION,
        "created":
            now_iso(),
        "command":
            " ".join(sys.argv),
        "case":
            case,
        "status":
            "READY",
        "upstream": {
            "CHI01_metadata":
                str(inp["chi01_metadata"]),
            "CHI01_version":
                chi01.get("version"),
            "GRE05b_local_field_hz":
                str(inp["local_field_hz"]),
            "GRE05b_vsharp_mask":
                str(inp["vsharp_mask"]),
            "R2PRIME03_raw":
                str(inp["r2prime"]),
            "R2PRIME03_mask":
                str(inp["r2prime_mask"]),
            "GRE02_b0_direction":
                str(inp["b0_json"]),
            "GRE_magnitude_series":
                series_stem,
            "GRE_magnitude_echoes":
                [
                    str(p)
                    for p in mag_paths
                ],
            "GRE_json_sidecars":
                [
                    str(p)
                    for p in json_paths
                ],
        },
        "qsm_ci_target": {
            "method":
                QSM_CI_METHOD,
            "record_id":
                QSM_CI_RECORD_ID,
            "pinned_image":
                QSM_CI_IMAGE,
            "image_availability":
                (
                    "Not assumed by CHI02. Cached algorithm.yml "
                    "states the GHCR image was not built at the "
                    "time of that method record; execution is "
                    "resolved in CHI03."
                ),
            "input_contract": [
                "localfield.nii.gz",
                "r2prime.nii.gz",
                "magnitude.nii.gz",
                "mask.nii.gz",
                "params.json",
            ],
        },
        "scientific_policy": {
            "localfield_conversion":
                (
                    "Hz -> ppm using gamma = 42.5774e6 Hz/T "
                    "to exactly match recon.m"
                ),
            "gamma_hz_per_t":
                QSM_CI_GAMMA_HZ_PER_T,
            "r2prime":
                (
                    "Raw R2PRIME03 values retained without clipping; "
                    "s^-1 is numerically equivalent to Hz."
                ),
            "mask_definition":
                (
                    "GRE05b V-SHARP mask ∩ R2PRIME03 support "
                    "∩ finite local field ∩ finite R2prime"
                ),
            "qsm":
                (
                    "No external QSM input. QSM-CI reconstructs "
                    "STI-Suite QSM_iLSQR internally."
                ),
        },
        "params":
            params,
        "support": {
            "voxels":
                n_mask,
            "negative_R2prime_voxels":
                n_negative,
            "negative_R2prime_pct":
                negative_pct,
            "nonfinite_magnitude_voxels":
                bad_mag_finite,
            "zero_RSS_magnitude_voxels":
                zero_mag_rss,
        },
        "conversion_qc": {
            "localfield_ppm_min_mask":
                row["localfield_ppm_min_mask"],
            "localfield_ppm_max_mask":
                row["localfield_ppm_max_mask"],
            "localfield_ppm_sd_mask":
                row["localfield_ppm_sd_mask"],
            "hz_to_ppm_to_hz_roundtrip_rmse_hz":
                roundtrip_rmse_hz,
            "hz_to_ppm_to_hz_roundtrip_maxabs_hz":
                roundtrip_maxabs_hz,
        },
        "outputs": {
            "solver_input_directory":
                str(solver),
            "localfield":
                str(solver / "localfield.nii.gz"),
            "r2prime":
                str(solver / "r2prime.nii.gz"),
            "magnitude":
                str(solver / "magnitude.nii.gz"),
            "mask":
                str(solver / "mask.nii.gz"),
            "params":
                str(solver / "params.json"),
        },
        "execution": {
            "host":
                platform.node(),
            "platform":
                platform.platform(),
            "python":
                sys.version.split()[0],
            "numpy":
                np.__version__,
            "nibabel":
                nib.__version__,
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    write_csv(
        summary_path,
        [row],
    )

    print("  status                  : READY")
    print(f"  GRE series              : {series_stem}")
    print(f"  echoes                  : {len(mag_paths)}")
    print(f"  B0                      : {b0_t:.6f} T")
    print(
        "  B0 dir voxel            : "
        + " ".join(
            f"{v:.6f}"
            for v in b0_dir
        )
    )
    print(
        f"  TE1 / TE2               : "
        f"{te_s[0]*1e3:.3f} / {te_s[1]*1e3:.3f} ms"
    )
    print(
        f"  dTE12                    : "
        f"{(te_s[1]-te_s[0])*1e3:.3f} ms"
    )
    print(
        f"  production mask         : "
        f"{n_mask:,} voxels"
    )
    print(
        f"  negative R2prime        : "
        f"{n_negative:,} ({negative_pct:.3f}%)"
    )
    print(
        f"  magnitude stack         : "
        f"{mag4.shape}"
    )
    print(
        f"  field ppm range         : "
        f"{row['localfield_ppm_min_mask']:.6f} .. "
        f"{row['localfield_ppm_max_mask']:.6f}"
    )
    print(
        f"  Hz→ppm→Hz RMSE          : "
        f"{roundtrip_rmse_hz:.3e} Hz"
    )
    print(
        f"  Hz→ppm→Hz max abs       : "
        f"{roundtrip_maxabs_hz:.3e} Hz"
    )
    print(
        f"  solver input            : "
        f"{solver}"
    )

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF CHI02: prepare exact QSM-CI "
            "chi-sep-ilsqr solver inputs."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CHI02 outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )

    print_selected_cases(
        cases
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for case in cases:
        try:
            rows.append(
                process_case(
                    paths=paths,
                    case=case,
                    overwrite=args.overwrite,
                )
            )

        except Exception as exc:
            print(
                f"ERROR: {case}: {exc}",
                file=sys.stderr,
            )

            failures.append(
                {
                    "case":
                        case,
                    "error":
                        str(exc),
                }
            )

    out = cohort_output_dir(
        paths
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_csv = (
        out
        / "chi02_cohort_summary.csv"
    )

    cohort_json = (
        out
        / "chi02_cohort_metadata.json"
    )

    write_csv(
        cohort_csv,
        rows,
    )

    cohort_json.write_text(
        json.dumps(
            {
                "script":
                    SCRIPT,
                "script_id":
                    SCRIPT_ID,
                "tag":
                    TAG,
                "version":
                    VERSION,
                "created":
                    now_iso(),
                "command":
                    " ".join(sys.argv),
                "qsm_ci_method":
                    QSM_CI_METHOD,
                "qsm_ci_record_id":
                    QSM_CI_RECORD_ID,
                "gamma_hz_per_t":
                    QSM_CI_GAMMA_HZ_PER_T,
                "n_cases_requested":
                    len(cases),
                "n_ready":
                    len(rows),
                "n_failures":
                    len(failures),
                "rows":
                    rows,
                "failures":
                    failures,
            },
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 72)
    print("CHI02 cohort summary")
    print("=" * 72)
    print(f"Cases requested : {len(cases)}")
    print(f"READY           : {len(rows)}")
    print(f"Failures        : {len(failures)}")
    print(f"Cohort CSV      : {cohort_csv}")
    print(f"Cohort metadata : {cohort_json}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
