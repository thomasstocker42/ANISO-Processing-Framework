#!/usr/bin/env python3
"""
APF R2PRIME01: R2-prime input inventory and geometry validation.

Purpose
-------
R2PRIME01 freezes the authoritative inputs required for subsequent
R2-prime construction for chi-separation.

Authoritative inputs
--------------------
1. RLX01 selected high-resolution dual-echo TSE
       echo 1 = moving anatomical image for registration
       echo 2 = provenance / R2app source

2. RLX05 high-resolution dual-echo apparent R2
       rlx05_<case>_R2app_s-1.nii.gz

3. GRE02b first-echo magnitude
       <case>_GRE02b_magnitude_e1.nii.gz

   This is the authoritative fixed registration target.

4. R2STAR02 production R2*
       r2star02_<case>_r2star_s-1.nii.gz

Geometry contract
-----------------
- RLX01 dual-TSE echoes must have identical native geometry.
- RLX01 dual-TSE echo 1 and RLX05 R2app must have identical geometry.
- GRE02b magnitude echo 1 and R2STAR02 R2* must have identical geometry.
- TSE and GRE geometry are expected to differ before registration.

Future R2-prime construction
----------------------------
R2PRIME02:
    rigid 6-DOF dual-TSE -> GRE registration

R2PRIME03:
    R2' = R2* - R2app_GRE

R2PRIME01 performs no registration, interpolation, resampling, or
quantitative modification.

Run examples
------------
    ./apf scripts/relaxometry/r2prime/01_inventory.py ANISO_00_000_InS

    ./apf scripts/relaxometry/r2prime/01_inventory.py --all-ready
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT = "01_inventory.py"
TAG = "r2prime01"
VERSION = "0.2.0"

DEFAULT_AFFINE_ATOL = 1e-5


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "R2PRIME01 requires nibabel."
    ) from exc


try:
    from apf_paths import APFPaths, get_paths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. "
        "Run through ./apf."
    ) from exc


# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------------
# APF paths
# -------------------------------------------------------------------------


def rlx01_selected_inputs_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx01_inventory"
        / f"rlx01_{case}_selected_inputs.json"
    )


def rlx05_r2app_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "rlx05"
        / f"rlx05_{case}_R2app_s-1.nii.gz"
    )


def gre02b_magnitude_e1_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "GRE"
        / "02b_brain_mask"
        / f"{case}_GRE02b_magnitude_e1.nii.gz"
    )


def r2star02_path(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2star"
        / "r2star02_loglinear"
        / f"r2star02_{case}_r2star_s-1.nii.gz"
    )


def r2prime01_dir(
    paths: APFPaths,
    case: str,
) -> Path:
    return (
        paths.derivatives_dir
        / case
        / "relaxometry"
        / "r2prime"
        / "r2prime01_inventory"
    )


def summary_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "r2prime"
        / "r2prime01_inventory"
    )


# -------------------------------------------------------------------------
# NIfTI utilities
# -------------------------------------------------------------------------


def load_nifti(
    path: Path,
    label: str,
) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n"
            f"  {path}"
        )

    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {label}:\n"
            f"  {path}"
        ) from exc

    if len(img.shape) < 3:
        raise ValueError(
            f"{label} is not at least 3D:\n"
            f"  {path}\n"
            f"  shape={img.shape}"
        )

    return img


def nifti_geometry(
    img: nib.Nifti1Image,
) -> Dict[str, Any]:
    return {
        "shape": [
            int(x)
            for x in img.shape[:3]
        ],
        "voxel_size_mm": [
            float(x)
            for x in img.header.get_zooms()[:3]
        ],
        "orientation": list(
            nib.aff2axcodes(img.affine)
        ),
        "affine": img.affine.tolist(),
    }


def geometry_equal(
    a: nib.Nifti1Image,
    b: nib.Nifti1Image,
    affine_atol: float,
) -> bool:
    return bool(
        tuple(a.shape[:3])
        == tuple(b.shape[:3])
        and np.allclose(
            a.affine,
            b.affine,
            atol=affine_atol,
            rtol=0.0,
        )
    )


def quantitative_qc(
    img: nib.Nifti1Image,
) -> Dict[str, Any]:
    data = np.asarray(
        img.dataobj,
        dtype=np.float32,
    )

    finite = np.isfinite(data)

    n_voxels = int(data.size)
    n_finite = int(np.count_nonzero(finite))
    n_nonfinite = int(n_voxels - n_finite)

    finite_values = data[finite]

    return {
        "n_voxels":
            n_voxels,
        "n_finite":
            n_finite,
        "n_nonfinite":
            n_nonfinite,
        "finite_fraction":
            float(n_finite / n_voxels)
            if n_voxels > 0
            else 0.0,
        "has_finite_data":
            bool(n_finite > 0),
        "finite_min":
            float(np.min(finite_values))
            if n_finite > 0
            else None,
        "finite_max":
            float(np.max(finite_values))
            if n_finite > 0
            else None,
    }


# -------------------------------------------------------------------------
# RLX01 input contract
# -------------------------------------------------------------------------


def load_rlx01_dual_tse(
    paths: APFPaths,
    case: str,
) -> Dict[str, Any]:
    metadata_path = (
        rlx01_selected_inputs_path(
            paths,
            case,
        )
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Required RLX01 selected-input "
            "metadata not found:\n"
            f"  {metadata_path}\n"
            "Run RLX01 first."
        )

    metadata = json.loads(
        metadata_path.read_text()
    )

    dual = metadata.get("dual_tse")

    if not isinstance(
        dual,
        dict,
    ):
        raise ValueError(
            "RLX01 metadata has no "
            f"dual_tse object for {case}."
        )

    if not dual.get(
        "ready",
        False,
    ):
        raise RuntimeError(
            f"dual_tse is not READY for {case}: "
            f"{dual.get('readiness_reason', 'unknown reason')}"
        )

    if not dual.get(
        "selected",
        False,
    ):
        raise RuntimeError(
            f"dual_tse is not selected for {case}."
        )

    if dual.get("n_echoes") != 2:
        raise ValueError(
            "Expected exactly two dual-TSE "
            f"echoes, got {dual.get('n_echoes')}."
        )

    echoes = dual.get("echoes")

    if (
        not isinstance(echoes, list)
        or len(echoes) != 2
    ):
        raise ValueError(
            "dual_tse.echoes must contain "
            "exactly two entries."
        )

    try:
        echoes = sorted(
            echoes,
            key=lambda x:
                int(x["echo_number"]),
        )
    except Exception as exc:
        raise ValueError(
            "Could not order dual-TSE echoes "
            "by echo_number."
        ) from exc

    resolved: List[Dict[str, Any]] = []

    for echo in echoes:
        raw_image = echo.get("image")
        raw_te = echo.get("echo_time_s")
        echo_number = echo.get(
            "echo_number"
        )

        if not isinstance(
            raw_image,
            str,
        ):
            raise ValueError(
                "Missing image path for "
                f"echo {echo_number}."
            )

        try:
            te_s = float(raw_te)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Invalid echo_time_s for "
                f"echo {echo_number}: "
                f"{raw_te}"
            ) from exc

        image_path = Path(raw_image)

        if not image_path.is_absolute():
            image_path = (
                paths.root
                / image_path
            )

        if not image_path.exists():
            raise FileNotFoundError(
                "Missing dual-TSE echo "
                f"{echo_number}:\n"
                f"  {image_path}"
            )

        resolved.append({
            "echo_number":
                int(echo_number),
            "echo_time_s":
                te_s,
            "echo_time_ms":
                te_s * 1000.0,
            "image":
                str(image_path),
        })

    delta_te_s = (
        resolved[1]["echo_time_s"]
        - resolved[0]["echo_time_s"]
    )

    if delta_te_s <= 0:
        raise ValueError(
            "Dual-TSE echo times are not "
            "strictly increasing."
        )

    return {
        "metadata_path":
            str(metadata_path),
        "ready":
            True,
        "selected":
            True,
        "n_echoes":
            2,
        "echoes":
            resolved,
        "delta_te_s":
            delta_te_s,
        "delta_te_ms":
            delta_te_s * 1000.0,
    }


# -------------------------------------------------------------------------
# Case processing
# -------------------------------------------------------------------------


def process_case(
    case: str,
    paths: APFPaths,
    affine_atol: float,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:

    dual = load_rlx01_dual_tse(
        paths,
        case,
    )

    tse_e1_path = Path(
        dual["echoes"][0]["image"]
    )
    tse_e2_path = Path(
        dual["echoes"][1]["image"]
    )

    r2app_path = rlx05_r2app_path(
        paths,
        case,
    )

    gre_e1_path = (
        gre02b_magnitude_e1_path(
            paths,
            case,
        )
    )

    r2star_path = r2star02_path(
        paths,
        case,
    )

    out_dir = r2prime01_dir(
        paths,
        case,
    )

    out_json = (
        out_dir
        / f"{TAG}_{case}_metadata.json"
    )

    out_csv = (
        out_dir
        / f"{TAG}_{case}_summary.csv"
    )

    if (
        not overwrite
        and not dry_run
        and (
            out_json.exists()
            or out_csv.exists()
        )
    ):
        raise FileExistsError(
            "R2PRIME01 output already "
            f"exists for {case}. "
            "Use --overwrite."
        )

    tse_e1 = load_nifti(
        tse_e1_path,
        "RLX01 dual-TSE echo 1",
    )

    tse_e2 = load_nifti(
        tse_e2_path,
        "RLX01 dual-TSE echo 2",
    )

    r2app = load_nifti(
        r2app_path,
        "RLX05 R2app",
    )

    gre_e1 = load_nifti(
        gre_e1_path,
        "GRE02b magnitude echo 1",
    )

    r2star = load_nifti(
        r2star_path,
        "R2STAR02 R2*",
    )

    tse_echo_geometry_match = (
        geometry_equal(
            tse_e1,
            tse_e2,
            affine_atol,
        )
    )

    tse_r2app_geometry_match = (
        geometry_equal(
            tse_e1,
            r2app,
            affine_atol,
        )
    )

    gre_r2star_geometry_match = (
        geometry_equal(
            gre_e1,
            r2star,
            affine_atol,
        )
    )

    tse_gre_geometry_identical = (
        geometry_equal(
            tse_e1,
            gre_e1,
            affine_atol,
        )
    )

    r2app_qc = quantitative_qc(
        r2app
    )

    r2star_qc = quantitative_qc(
        r2star
    )

    checks = {
        "dual_tse_ready":
            True,
        "dual_tse_selected":
            True,
        "dual_tse_two_echoes":
            True,
        "dual_tse_echo_geometry_match":
            tse_echo_geometry_match,
        "tse_e1_r2app_geometry_match":
            tse_r2app_geometry_match,
        "gre_e1_r2star_geometry_match":
            gre_r2star_geometry_match,
        "r2app_has_finite_data":
            r2app_qc["has_finite_data"],
        "r2star_has_finite_data":
            r2star_qc["has_finite_data"],
        "tse_gre_geometry_identical":
            tse_gre_geometry_identical,
    }

    required_checks = [
        checks[
            "dual_tse_ready"
        ],
        checks[
            "dual_tse_selected"
        ],
        checks[
            "dual_tse_two_echoes"
        ],
        checks[
            "dual_tse_echo_geometry_match"
        ],
        checks[
            "tse_e1_r2app_geometry_match"
        ],
        checks[
            "gre_e1_r2star_geometry_match"
        ],
        checks[
            "r2app_has_finite_data"
        ],
        checks[
            "r2star_has_finite_data"
        ],
    ]

    ready = bool(
        all(required_checks)
    )

    metadata: Dict[str, Any] = {
        "script":
            SCRIPT,
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
            "READY"
            if ready
            else "FAIL",

        "purpose": (
            "Freeze authoritative "
            "R2-prime input provenance "
            "and native geometry before "
            "dual-TSE to GRE registration."
        ),

        "production_contract": {
            "moving_registration_image":
                "RLX01 selected dual-TSE echo 1",

            "quantitative_moving_image":
                "RLX05 R2app [s^-1]",

            "fixed_registration_image":
                "GRE02b magnitude echo 1",

            "quantitative_fixed_image":
                "R2STAR02 R2* [s^-1]",

            "registration_direction":
                "dual-TSE -> GRE",

            "planned_transform":
                "rigid 6-DOF",

            "planned_registration_metric":
                "normalized mutual information",

            "planned_r2app_interpolation":
                "linear",

            "planned_r2prime_equation":
                "R2prime = R2star - R2app_GRE",
        },

        "inputs": {
            "rlx01_dual_tse":
                dual,

            "rlx05_r2app":
                str(r2app_path),

            "gre02b_magnitude_e1":
                str(gre_e1_path),

            "r2star02_r2star":
                str(r2star_path),
        },

        "units": {
            "rlx05_r2app":
                "s^-1",

            "r2star02_r2star":
                "s^-1",

            "planned_r2prime":
                "s^-1",
        },

        "geometry": {
            "dual_tse_echo1":
                nifti_geometry(
                    tse_e1
                ),

            "dual_tse_echo2":
                nifti_geometry(
                    tse_e2
                ),

            "rlx05_r2app":
                nifti_geometry(
                    r2app
                ),

            "gre02b_magnitude_e1":
                nifti_geometry(
                    gre_e1
                ),

            "r2star02_r2star":
                nifti_geometry(
                    r2star
                ),
        },

        "data_qc": {
            "rlx05_r2app":
                r2app_qc,

            "r2star02_r2star":
                r2star_qc,
        },

        "checks":
            checks,

        "interpretation": {
            "native_tse_and_r2app_match":
                tse_r2app_geometry_match,

            "native_gre_and_r2star_match":
                gre_r2star_geometry_match,

            "registration_required":
                not tse_gre_geometry_identical,
        },
    }

    summary: Dict[str, Any] = {
        "case":
            case,

        "status":
            "READY"
            if ready
            else "FAIL",

        "TE1_ms":
            dual["echoes"][0][
                "echo_time_ms"
            ],

        "TE2_ms":
            dual["echoes"][1][
                "echo_time_ms"
            ],

        "deltaTE_ms":
            dual["delta_te_ms"],

        "TSE_shape":
            "x".join(
                str(x)
                for x in
                tse_e1.shape[:3]
            ),

        "R2app_shape":
            "x".join(
                str(x)
                for x in
                r2app.shape[:3]
            ),

        "GRE_shape":
            "x".join(
                str(x)
                for x in
                gre_e1.shape[:3]
            ),

        "R2star_shape":
            "x".join(
                str(x)
                for x in
                r2star.shape[:3]
            ),

        "TSE_echo_geometry":
            (
                "OK"
                if tse_echo_geometry_match
                else "FAIL"
            ),

        "TSE_R2app_geometry":
            (
                "OK"
                if tse_r2app_geometry_match
                else "FAIL"
            ),

        "GRE_R2star_geometry":
            (
                "OK"
                if gre_r2star_geometry_match
                else "FAIL"
            ),

        "TSE_GRE_identical":
            (
                "YES"
                if tse_gre_geometry_identical
                else "NO"
            ),

        "registration_required":
            not tse_gre_geometry_identical,

        "R2app_finite_fraction":
            r2app_qc["finite_fraction"],

        "R2star_finite_fraction":
            r2star_qc["finite_fraction"],

        "registration_target":
            "GRE02b magnitude echo 1",
    }

    print()
    print(f"[{case}]")
    print(
        f"  status            : "
        f"{summary['status']}"
    )
    print(
        f"  dual-TSE TE       : "
        f"{summary['TE1_ms']:.3f}, "
        f"{summary['TE2_ms']:.3f} ms"
    )
    print(
        f"  delta TE          : "
        f"{summary['deltaTE_ms']:.3f} ms"
    )
    print(
        f"  TSE echo geom     : "
        f"{summary['TSE_echo_geometry']}"
    )
    print(
        f"  TSE -> R2app geom : "
        f"{summary['TSE_R2app_geometry']}"
    )
    print(
        f"  GRE -> R2* geom   : "
        f"{summary['GRE_R2star_geometry']}"
    )
    print(
        f"  TSE = GRE         : "
        f"{summary['TSE_GRE_identical']}"
    )
    print(
        f"  registration req. : "
        f"{summary['registration_required']}"
    )
    print(
        f"  fixed target      : "
        f"{summary['registration_target']}"
    )

    if dry_run:
        print(
            "  dry-run           : "
            "no outputs written"
        )
        return summary

    write_json(
        out_json,
        metadata,
    )

    write_csv(
        out_csv,
        [summary],
    )

    return summary


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF R2PRIME01: freeze "
            "R2-prime input provenance "
            "and native image geometry."
        )
    )

    add_case_selection_args(
        parser
    )

    parser.add_argument(
        "--affine-atol",
        type=float,
        default=DEFAULT_AFFINE_ATOL,
        help=(
            "Absolute tolerance used when "
            "comparing NIfTI affines. "
            f"Default: {DEFAULT_AFFINE_ATOL:g}."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing "
            "R2PRIME01 outputs."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and geometry "
            "without writing outputs."
        ),
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )

    print(
        f"APF R2PRIME01 "
        f"inventory v{VERSION}"
    )
    print(
        f"Root: {paths.root}"
    )
    print(
        "Fixed registration target: "
        "GRE02b magnitude echo 1"
    )
    print(
        "Registration direction: "
        "dual-TSE -> GRE"
    )

    print_selected_cases(
        cases
    )

    cohort_rows: List[
        Dict[str, Any]
    ] = []

    failures = 0

    for case in cases:
        try:
            row = process_case(
                case=case,
                paths=paths,
                affine_atol=
                    args.affine_atol,
                overwrite=
                    args.overwrite,
                dry_run=
                    args.dry_run,
            )

            cohort_rows.append(
                row
            )

            if (
                row["status"]
                != "READY"
            ):
                failures += 1

        except Exception as exc:
            failures += 1

            print(
                f"ERROR in {case}: "
                f"{exc}",
                file=sys.stderr,
            )

            cohort_rows.append({
                "case":
                    case,
                "status":
                    "FAIL",
                "error":
                    str(exc),
            })

            if len(cases) == 1:
                raise

    if not args.dry_run:
        out = summary_dir(
            paths
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            out
            / "r2prime01_cohort_summary.csv",
            cohort_rows,
        )

        write_json(
            out
            / "r2prime01_cohort_metadata.json",
            {
                "script":
                    SCRIPT,

                "tag":
                    TAG,

                "version":
                    VERSION,

                "created":
                    now_iso(),

                "command":
                    " ".join(
                        sys.argv
                    ),

                "n_cases_requested":
                    len(cases),

                "n_ready":
                    int(
                        sum(
                            row.get(
                                "status"
                            )
                            == "READY"
                            for row
                            in cohort_rows
                        )
                    ),

                "n_failures":
                    failures,

                "fixed_registration_target":
                    "GRE02b magnitude echo 1",

                "registration_direction":
                    "dual-TSE -> GRE",

                "rows":
                    cohort_rows,
            },
        )

    print()
    print("=" * 68)
    print(
        "R2PRIME01 cohort summary"
    )
    print("=" * 68)

    print(
        f"Cases : {len(cases)}"
    )

    print(
        "READY : "
        f"{sum(row.get('status') == 'READY' for row in cohort_rows)}"
    )

    print(
        f"FAIL  : {failures}"
    )

    if args.dry_run:
        print(
            "Mode  : dry-run"
        )

    if failures:
        raise SystemExit(
            "R2PRIME01 completed with "
            f"{failures} failed case(s)."
        )


if __name__ == "__main__":
    main()
