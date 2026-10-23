#!/usr/bin/env python3
"""
APF CHI01: chi-separation input inventory and compatibility validation.

Version 0.1.0

Purpose
-------
Inventory and validate the frozen upstream inputs required for QSM-CI
chi-sep-ilsqr without modifying, resampling, converting, stacking, clipping,
or otherwise preparing them.

Required upstream inputs
------------------------
GRE05b:
    - finalized V-SHARP local field [Hz]
    - finalized V-SHARP mask

R2PRIME03:
    - raw GRE-space R2-prime [s^-1 == Hz]
    - common relaxometry support

GRE acquisition:
    - 12 magnitude echoes
    - 12 JSON sidecars with EchoTime and MagneticFieldStrength

GRE02:
    - voxel-space B0 direction

CHI01 checks
------------
- required-file presence;
- complete 12-echo GRE magnitude/JSON series;
- strictly increasing echo times;
- acquisition B0 consistency;
- valid normalized voxel-space B0 direction;
- common GRE geometry across quantitative maps, masks and magnitude echoes;
- finite-value support sanity;
- R2-prime negative-value prevalence;
- prospective common support available for chi-separation.

No production chi-separation input is written here. Input preparation is
deferred to CHI02.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
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


SCRIPT = "01_inventory.py"
SCRIPT_ID = "CHI01"
TAG = "chi01"
VERSION = "0.1.1"

EXPECTED_ECHOES = tuple(range(1, 13))
AFFINE_ATOL_DEFAULT = 1e-5

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


def geometry_equal(
    reference: nib.spatialimages.SpatialImage,
    other: nib.spatialimages.SpatialImage,
    atol: float,
) -> bool:
    return (
        reference.shape == other.shape
        and np.allclose(
            reference.affine,
            other.affine,
            atol=atol,
            rtol=0,
        )
    )


def voxel_size_mm(
    image: nib.spatialimages.SpatialImage,
) -> tuple[float, float, float]:
    zooms = image.header.get_zooms()[:3]
    return tuple(float(v) for v in zooms)


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
            f"Invalid voxel_b0_direction shape: {b0.shape}"
        )

    if not np.all(np.isfinite(b0)):
        raise RuntimeError(
            "voxel_b0_direction contains non-finite values"
        )

    norm = float(np.linalg.norm(b0))

    if not np.isclose(
        norm,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"B0 direction is not normalized: norm={norm}"
        )

    return b0


def input_paths(
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
        "r2prime_metadata": (
            paths.derivatives_dir
            / case
            / "relaxometry"
            / "r2prime"
            / "r2prime03_construct"
            / f"r2prime03_{case}_metadata.json"
        ),
        "mag_dir": (
            paths.root
            / "nifti"
            / case
            / "GRE"
            / "mag"
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
        / "chi01_inventory"
    )


def cohort_output_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "chi_separation"
        / "chi01_inventory"
    )


def discover_echo_series(
    mag_dir: Path,
    case: str,
) -> tuple[list[Path], list[Path]]:
    if not mag_dir.is_dir():
        raise FileNotFoundError(
            f"Missing GRE magnitude directory: {mag_dir}"
        )

    pattern = re.compile(
        rf"^(?P<stem>{re.escape(case)}_GRE_12echo_\d+)_e(?P<echo>\d+)\.nii\.gz$"
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

    complete: dict[str, dict[int, Path]] = {
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
            "Expected exactly one complete 12-echo GRE magnitude series; "
            f"found {len(complete)} complete series. "
            f"Discovered: {discovered}"
        )

    stem, echo_nifti = next(
        iter(complete.items())
    )

    niftis = [
        echo_nifti[e]
        for e in EXPECTED_ECHOES
    ]

    jsons: list[Path] = []

    for nifti in niftis:
        js = nifti.with_suffix("").with_suffix(".json")

        require_file(
            js,
            f"GRE JSON sidecar for {nifti.name}",
        )

        jsons.append(js)

    return niftis, jsons


def read_acquisition_series(
    json_paths: list[Path],
) -> tuple[list[float], float]:
    te_s: list[float] = []
    b0_values: list[float] = []

    for path in json_paths:
        data = json.loads(path.read_text())

        te = data.get("EchoTime")
        b0 = data.get("MagneticFieldStrength")

        if te is None:
            raise RuntimeError(
                f"Missing EchoTime in {path}"
            )

        if b0 is None:
            raise RuntimeError(
                f"Missing MagneticFieldStrength in {path}"
            )

        te = float(te)
        b0 = float(b0)

        if not np.isfinite(te) or te <= 0:
            raise RuntimeError(
                f"Invalid EchoTime in {path}: {te}"
            )

        if not np.isfinite(b0) or b0 <= 0:
            raise RuntimeError(
                f"Invalid MagneticFieldStrength in {path}: {b0}"
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
            f"Echo times are not strictly increasing: {te_s}"
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
            f"Inconsistent field strength across echoes: {b0_values}"
        )

    return te_s, float(b0_array[0])


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

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
    affine_atol: float,
    overwrite: bool,
) -> dict[str, Any]:
    print("=" * 72)
    print(f"{SCRIPT_ID}: {case}")
    print("=" * 72)

    inp = input_paths(
        paths,
        case,
    )

    for key in (
        "local_field_hz",
        "vsharp_mask",
        "b0_json",
        "r2prime",
        "r2prime_mask",
        "r2prime_metadata",
    ):
        require_file(
            inp[key],
            key,
        )

    magnitude_paths, json_paths = discover_echo_series(
        inp["mag_dir"],
        case,
    )

    te_s, b0_t = read_acquisition_series(
        json_paths
    )

    b0_dir = read_b0_direction(
        inp["b0_json"]
    )

    field_img = nib.load(
        str(inp["local_field_hz"])
    )

    vsharp_mask_img = nib.load(
        str(inp["vsharp_mask"])
    )

    r2prime_img = nib.load(
        str(inp["r2prime"])
    )

    r2prime_mask_img = nib.load(
        str(inp["r2prime_mask"])
    )

    geometry_checks: dict[str, bool] = {
        "vsharp_mask":
            geometry_equal(
                field_img,
                vsharp_mask_img,
                affine_atol,
            ),
        "r2prime":
            geometry_equal(
                field_img,
                r2prime_img,
                affine_atol,
            ),
        "r2prime_mask":
            geometry_equal(
                field_img,
                r2prime_mask_img,
                affine_atol,
            ),
    }

    magnitude_geometry: list[bool] = []

    for p in magnitude_paths:
        img = nib.load(str(p))
        magnitude_geometry.append(
            geometry_equal(
                field_img,
                img,
                affine_atol,
            )
        )

    geometry_checks[
        "all_magnitude_echoes"
    ] = all(magnitude_geometry)

    if not all(
        geometry_checks.values()
    ):
        failed = [
            k
            for k, ok in geometry_checks.items()
            if not ok
        ]
        raise RuntimeError(
            "GRE geometry mismatch: "
            + ", ".join(failed)
        )

    field = np.asarray(
        field_img.dataobj,
        dtype=np.float32,
    )

    vsharp_mask = (
        np.asarray(
            vsharp_mask_img.dataobj
        )
        > 0
    )

    r2prime = np.asarray(
        r2prime_img.dataobj,
        dtype=np.float32,
    )

    r2prime_mask = (
        np.asarray(
            r2prime_mask_img.dataobj
        )
        > 0
    )

    if not np.isfinite(
        field[vsharp_mask]
    ).all():
        raise RuntimeError(
            "Non-finite GRE05b local-field values "
            "inside V-SHARP support."
        )

    if not np.isfinite(
        r2prime[r2prime_mask]
    ).all():
        raise RuntimeError(
            "Non-finite R2prime values "
            "inside R2PRIME03 support."
        )

    prospective_common = (
        vsharp_mask
        & r2prime_mask
        & np.isfinite(field)
        & np.isfinite(r2prime)
    )

    n_vsharp = int(
        np.count_nonzero(vsharp_mask)
    )

    n_r2prime = int(
        np.count_nonzero(r2prime_mask)
    )

    n_common = int(
        np.count_nonzero(prospective_common)
    )

    if n_common == 0:
        raise RuntimeError(
            "Prospective chi-separation support is empty."
        )

    negative = (
        r2prime_mask
        & (r2prime < 0)
    )

    negative_common = (
        prospective_common
        & (r2prime < 0)
    )

    n_negative = int(
        np.count_nonzero(negative)
    )

    n_negative_common = int(
        np.count_nonzero(negative_common)
    )

    negative_fraction_r2prime_pct = (
        100.0
        * n_negative
        / n_r2prime
    )

    negative_fraction_common_pct = (
        100.0
        * n_negative_common
        / n_common
    )

    common_fraction_vsharp_pct = (
        100.0
        * n_common
        / n_vsharp
    )

    common_fraction_r2prime_pct = (
        100.0
        * n_common
        / n_r2prime
    )

    te = np.asarray(
        te_s,
        dtype=np.float64,
    )

    dte_ms = (
        np.diff(te)
        * 1000.0
    )

    vox = voxel_size_mm(
        field_img
    )

    out = output_dir(
        paths,
        case,
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = f"chi01_{case}"

    metadata_path = (
        out
        / f"{prefix}_metadata.json"
    )

    summary_path = (
        out
        / f"{prefix}_summary.csv"
    )

    if (
        metadata_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Output exists: {metadata_path}\n"
            "Use --overwrite to replace."
        )

    row = {
        "case":
            case,
        "status":
            "READY",
        "n_echoes":
            len(magnitude_paths),
        "B0_T":
            b0_t,
        "TE1_ms":
            te_s[0] * 1000.0,
        "TE2_ms":
            te_s[1] * 1000.0,
        "dTE12_ms":
            dte_ms[0],
        "dTE_min_ms":
            float(np.min(dte_ms)),
        "dTE_max_ms":
            float(np.max(dte_ms)),
        "voxel_x_mm":
            vox[0],
        "voxel_y_mm":
            vox[1],
        "voxel_z_mm":
            vox[2],
        "vsharp_voxels":
            n_vsharp,
        "r2prime_voxels":
            n_r2prime,
        "prospective_common_voxels":
            n_common,
        "common_fraction_of_vsharp_pct":
            common_fraction_vsharp_pct,
        "common_fraction_of_r2prime_pct":
            common_fraction_r2prime_pct,
        "negative_R2prime_fraction_pct":
            negative_fraction_r2prime_pct,
        "negative_R2prime_common_fraction_pct":
            negative_fraction_common_pct,
        "geometry_all_match":
            all(geometry_checks.values()),
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
        "purpose": (
            "Inventory and compatibility validation "
            "for QSM-CI chi-sep-ilsqr."
        ),
        "qsm_ci_target": {
            "method":
                QSM_CI_METHOD,
            "record_id":
                QSM_CI_RECORD_ID,
            "pinned_image":
                QSM_CI_IMAGE,
            "wrapper_contract": {
                "localfield":
                    "ppm; CHI02 conversion required from GRE05b Hz",
                "r2prime":
                    "Hz == s^-1; raw R2PRIME03 values retained",
                "magnitude":
                    "12-echo GRE magnitude; CHI02 will stack to 4D",
                "mask":
                    "CHI02 common support to be constructed",
                "params":
                    "B0, B0_dir, voxel_size, full TE vector",
            },
        },
        "inputs": {
            "local_field_hz":
                str(inp["local_field_hz"]),
            "vsharp_mask":
                str(inp["vsharp_mask"]),
            "r2prime_raw":
                str(inp["r2prime"]),
            "r2prime_common_mask":
                str(inp["r2prime_mask"]),
            "r2prime_metadata":
                str(inp["r2prime_metadata"]),
            "b0_direction_json":
                str(inp["b0_json"]),
            "magnitude_echoes":
                [str(p) for p in magnitude_paths],
            "json_sidecars":
                [str(p) for p in json_paths],
        },
        "acquisition": {
            "B0_T":
                b0_t,
            "B0_dir_voxel":
                [float(v) for v in b0_dir],
            "voxel_size_mm":
                list(vox),
            "TE_s":
                [float(v) for v in te_s],
            "TE_ms":
                [
                    float(v * 1000.0)
                    for v in te_s
                ],
            "dTE_ms":
                [
                    float(v)
                    for v in dte_ms
                ],
            "dTE12_ms":
                float(dte_ms[0]),
            "dTE_min_ms":
                float(np.min(dte_ms)),
            "dTE_max_ms":
                float(np.max(dte_ms)),
            "note": (
                "QSM-CI chi-sep-ilsqr recon.m uses "
                "TE(2)-TE(1) internally for STI-Suite "
                "QSM_iLSQR tissue-phase reconstruction."
            ),
        },
        "geometry": {
            "reference":
                str(inp["local_field_hz"]),
            "shape":
                list(field_img.shape),
            "voxel_size_mm":
                list(vox),
            "affine_atol":
                affine_atol,
            "checks":
                geometry_checks,
            "all_match":
                all(geometry_checks.values()),
        },
        "support": {
            "vsharp_voxels":
                n_vsharp,
            "r2prime_voxels":
                n_r2prime,
            "prospective_common_voxels":
                n_common,
            "common_fraction_of_vsharp_pct":
                common_fraction_vsharp_pct,
            "common_fraction_of_r2prime_pct":
                common_fraction_r2prime_pct,
            "note": (
                "Prospective common support is evaluated "
                "for readiness only. CHI01 writes no mask; "
                "CHI02 will construct the production mask."
            ),
        },
        "r2prime_qc": {
            "negative_voxels_in_r2prime_support":
                n_negative,
            "negative_fraction_of_r2prime_support_pct":
                negative_fraction_r2prime_pct,
            "negative_voxels_in_prospective_common_support":
                n_negative_common,
            "negative_fraction_of_prospective_common_support_pct":
                negative_fraction_common_pct,
            "negative_policy":
                (
                    "No clipping in CHI01. Raw R2PRIME03 "
                    "values are preserved for downstream use."
                ),
        },
        "readiness": {
            "ready":
                True,
            "reason":
                (
                    "All required chi-sep-ilsqr inputs are present, "
                    "metadata are valid, and GRE geometry is compatible."
                ),
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

    print(f"  status                  : READY")
    print(f"  echoes                  : {len(magnitude_paths)}")
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
        f"{dte_ms[0]:.3f} ms"
    )
    print(
        f"  dTE range               : "
        f"{np.min(dte_ms):.3f}–{np.max(dte_ms):.3f} ms"
    )
    print(
        "  voxel size              : "
        + " ".join(
            f"{v:.3f}"
            for v in vox
        )
        + " mm"
    )
    print(
        f"  V-SHARP support         : "
        f"{n_vsharp:,}"
    )
    print(
        f"  R2prime support         : "
        f"{n_r2prime:,}"
    )
    print(
        f"  prospective common      : "
        f"{n_common:,}"
    )
    print(
        f"  common / V-SHARP        : "
        f"{common_fraction_vsharp_pct:.3f}%"
    )
    print(
        f"  common / R2prime        : "
        f"{common_fraction_r2prime_pct:.3f}%"
    )
    print(
        f"  negative R2prime        : "
        f"{negative_fraction_r2prime_pct:.3f}% "
        f"(R2prime support)"
    )
    print(
        f"  negative in common      : "
        f"{negative_fraction_common_pct:.3f}%"
    )
    print(
        f"  geometry                : PASS"
    )

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF CHI01: inventory and compatibility validation "
            "for chi-separation inputs."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--affine-atol",
        type=float,
        default=AFFINE_ATOL_DEFAULT,
        help=(
            "Absolute affine tolerance for geometry checks "
            "(default: 1e-5)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CHI01 outputs.",
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
            row = process_case(
                paths=paths,
                case=case,
                affine_atol=args.affine_atol,
                overwrite=args.overwrite,
            )
            rows.append(row)

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
        / "chi01_cohort_summary.csv"
    )

    cohort_json = (
        out
        / "chi01_cohort_metadata.json"
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
    print("CHI01 cohort summary")
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
