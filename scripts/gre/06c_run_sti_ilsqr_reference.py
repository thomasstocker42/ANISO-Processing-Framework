#!/usr/bin/env python3
"""
APF GRE06c external QSM reference runner: STI Suite iLSQR.

Version 0.1.0

Purpose
-------
Run the published STI Suite v3 iLSQR dipole inversion on the exact finalized
GRE05b APF local field and GRE05b V-SHARP support used by GRE06b.

This module is validation-only.

The runner:
1. discovers GRE06-ready cases,
2. reads the finalized GRE05b local field in Hz,
3. converts the field to ppm using acquisition B0 and the GRE06 proton
   gyromagnetic-ratio convention,
4. reads the case-specific GRE02 voxel-space B0 direction,
5. stages the exact input contract expected by the pinned QSM-CI STI iLSQR
   container,
6. runs the published STI Suite v3 iLSQR implementation,
7. stores the resulting susceptibility map and complete provenance.

GRE06c intentionally uses the APF GRE05b local field rather than an STI
V-SHARP field so that the comparison with GRE06b isolates dipole inversion.

No APF GRE06b parameter is tuned from the GRE06c result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "0.1.0"

METHOD_SLUG = "matlab-sti-ilsqr"
QSM_CI_RECORD_ID = "21549373"

STI_DOCKER_IMAGE = (
    "ghcr.io/astewartau/qsm-ci/"
    "matlab-sti-ilsqr@sha256:"
    "f12eb85c0d1137f38ea5104a5df5b24d3d5d27661279680d3ec59c8f33108151"
)

GAMMA_HZ_PER_T = 42.57747892e6

DEFAULT_RUNNER = "docker"


def resolve_root(root: str | None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    return Path.cwd().resolve()


def case_paths(root: Path, case: str) -> dict[str, Path]:
    gre = root / "derivatives" / case / "GRE"

    return {
        "field_hz": (
            gre
            / "05b_vsharp_apf_v030"
            / f"{case}_GRE05b_local_field_hz.nii.gz"
        ),
        "mask": (
            gre
            / "05b_vsharp_apf_v030"
            / f"{case}_GRE05b_vsharp_mask.nii.gz"
        ),
        "b0_json": (
            gre
            / "phase_preparation_gre02"
            / f"gre02_{case}_b0_direction.json"
        ),
        "out_dir": gre / "06c_sti_ilsqr",
    }


def list_cases(root: Path) -> list[str]:
    derivatives = root / "derivatives"
    cases: list[str] = []

    for case_dir in sorted(derivatives.glob("ANISO_*_InS")):
        case = case_dir.name
        p = case_paths(root, case)

        if (
            p["field_hz"].exists()
            and p["mask"].exists()
            and p["b0_json"].exists()
        ):
            cases.append(case)

    return cases


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required command not found on PATH: {name}")
    return path


def command_output(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def read_b0_direction(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())

    if "voxel_b0_direction" not in data:
        raise RuntimeError(
            f"Missing voxel_b0_direction in {path}"
        )

    b0 = np.asarray(data["voxel_b0_direction"], dtype=np.float64)

    if b0.shape != (3,):
        raise RuntimeError(
            f"Invalid voxel_b0_direction shape in {path}: {b0.shape}"
        )

    if not np.all(np.isfinite(b0)):
        raise RuntimeError(
            f"Non-finite voxel_b0_direction in {path}"
        )

    norm = float(np.linalg.norm(b0))

    if not np.isclose(norm, 1.0, atol=1e-5):
        raise RuntimeError(
            f"B0 direction is not normalized in {path}: norm={norm}"
        )

    return b0


def read_acquisition_metadata(
    root: Path,
    case: str,
) -> tuple[float, float, Path]:
    """
    Read field strength and first-echo TE from the original GRE JSON sidecar.

    No hard-coded B0 or TE fallback is permitted.
    """
    mag_dir = root / "nifti" / case / "GRE" / "mag"

    candidates = sorted(mag_dir.glob(f"{case}_GRE*_e1.json"))

    if not candidates:
        raise RuntimeError(
            f"No first-echo GRE JSON found in {mag_dir}"
        )

    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        b0_value = data.get("MagneticFieldStrength")
        te_value = data.get("EchoTime")

        if b0_value is None or te_value is None:
            continue

        b0_t = float(b0_value)
        te_s = float(te_value)

        if not np.isfinite(b0_t) or b0_t <= 0:
            continue

        if not np.isfinite(te_s) or te_s <= 0:
            continue

        return b0_t, te_s, path

    raise RuntimeError(
        f"No valid MagneticFieldStrength + EchoTime found for {case}"
    )


def validate_geometry(
    field_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
) -> None:
    if field_img.shape != mask_img.shape:
        raise RuntimeError(
            "Field/mask shape mismatch: "
            f"{field_img.shape} vs {mask_img.shape}"
        )

    if not np.allclose(
        field_img.affine,
        mask_img.affine,
        atol=1e-5,
        rtol=0,
    ):
        raise RuntimeError("Field/mask affine mismatch")


def prepare_ppm_input(
    field_hz_path: Path,
    ppm_path: Path,
    b0_t: float,
) -> dict[str, float]:
    img = nib.load(str(field_hz_path))
    field_hz = img.get_fdata(dtype=np.float32)

    field_ppm = (
        field_hz
        / (GAMMA_HZ_PER_T * b0_t)
        * 1e6
    )

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)

    nib.save(
        nib.Nifti1Image(
            field_ppm.astype(np.float32),
            img.affine,
            hdr,
        ),
        str(ppm_path),
    )

    return {
        "hz_min": float(np.nanmin(field_hz)),
        "hz_max": float(np.nanmax(field_hz)),
        "ppm_min": float(np.nanmin(field_ppm)),
        "ppm_max": float(np.nanmax(field_ppm)),
        "ppm_sd_global": float(np.nanstd(field_ppm)),
    }


def run_sti(
    root: Path,
    case: str,
    runner: str,
    dry_run: bool,
    force: bool,
    qsm_ci_version: str,
) -> bool:
    p = case_paths(root, case)

    field_hz = p["field_hz"]
    mask_path = p["mask"]
    b0_json = p["b0_json"]
    out_dir = p["out_dir"]

    out_path = (
        out_dir
        / f"{case}_GRE06c_sti_ilsqr_chi_ppm.nii.gz"
    )

    provenance_path = (
        out_dir
        / f"{case}_GRE06c_sti_ilsqr_provenance.json"
    )

    print(f"\n[{case}]")
    print(f"  GRE05b field : {field_hz}")
    print(f"  GRE05b mask  : {mask_path}")
    print(f"  GRE02 B0 JSON: {b0_json}")
    print(f"  STI output   : {out_path}")
    print(f"  runner       : {runner}")

    for label, path in [
        ("GRE05b local field", field_hz),
        ("GRE05b mask", mask_path),
        ("GRE02 B0 JSON", b0_json),
    ]:
        if not path.exists():
            print(f"  [FAILED] missing {label}: {path}")
            return False

    try:
        b0_dir = read_b0_direction(b0_json)
        b0_t, te_s, acquisition_metadata_path = read_acquisition_metadata(
            root,
            case,
        )

        field_img = nib.load(str(field_hz))
        mask_img = nib.load(str(mask_path))

        validate_geometry(field_img, mask_img)

        voxel_size = np.asarray(
            field_img.header.get_zooms()[:3],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
            raise RuntimeError(
                f"Invalid voxel size: {voxel_size.tolist()}"
            )

    except Exception as exc:
        print(f"  [FAILED] input validation: {exc}")
        return False

    print(
        "  B0 voxel dir : "
        + " ".join(f"{v:+.9f}" for v in b0_dir)
    )
    print(f"  B0 strength  : {b0_t:.6f} T")
    print(f"  TE1          : {te_s * 1e3:.6f} ms")
    print(
        "  voxel size   : "
        + " ".join(f"{v:.6f}" for v in voxel_size)
        + " mm"
    )

    if out_path.exists() and not force:
        print("  [SKIP] output already exists; use --force to rerun")
        return True

    if dry_run:
        print(f"  [DRYRUN] image={STI_DOCKER_IMAGE}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    cmd: list[str] = []
    result: subprocess.CompletedProcess | None = None
    ppm_stats: dict[str, float] = {}

    with tempfile.TemporaryDirectory(
        prefix=f"apf-sti-ilsqr-{case}-"
    ) as tmp:
        tmp_root = Path(tmp)
        docker_input = tmp_root / "input"
        docker_output = tmp_root / "output"

        docker_input.mkdir()
        docker_output.mkdir()

        ppm_path = docker_input / "localfield.nii.gz"

        ppm_stats = prepare_ppm_input(
            field_hz_path=field_hz,
            ppm_path=ppm_path,
            b0_t=b0_t,
        )

        shutil.copy2(
            mask_path,
            docker_input / "mask.nii.gz",
        )

        params = {
            "B0_dir": [
                float(v) for v in b0_dir
            ],
            "voxel_size": [
                float(v) for v in voxel_size
            ],
            "B0": float(b0_t),
            "TE": [float(te_s)],
        }

        (docker_input / "params.json").write_text(
            json.dumps(params, indent=2) + "\n"
        )

        config = {
            "padsize": 12,
        }

        (docker_input / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )

        cmd = [
            "docker",
            "run",
            "--rm",
            "-e",
            "MCR_CACHE_ROOT=/tmp/mcr_cache",
            "-v",
            f"{docker_input}:/input:ro",
            "-v",
            f"{docker_output}:/output",
            STI_DOCKER_IMAGE,
            "/opt/qsm-ci/recon",
            "/input",
            "/output",
        ]

        print("  running pinned STI Suite iLSQR container...")
        print("  command: " + " ".join(cmd))

        result = subprocess.run(cmd)

        container_output = docker_output / "chimap.nii.gz"

        if result.returncode == 0 and container_output.exists():
            shutil.copy2(container_output, out_path)

    finished = datetime.now(timezone.utc)

    assert result is not None

    provenance = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "case": case,
        "module": "GRE06c",
        "role": "validation_only",
        "method": METHOD_SLUG,
        "reference_implementation": "STI Suite v3 iLSQR",
        "qsm_ci_record_id": QSM_CI_RECORD_ID,
        "qsm_ci_version": qsm_ci_version,
        "execution": "direct_docker",
        "container_image": STI_DOCKER_IMAGE,
        "runner": runner,
        "gamma_hz_per_t": GAMMA_HZ_PER_T,
        "b0_t": float(b0_t),
        "te_s": float(te_s),
        "acquisition_metadata_json": str(acquisition_metadata_path),
        "sti_ilsqr_padsize": 12,
        "voxel_b0_direction": [
            float(v) for v in b0_dir
        ],
        "voxel_size_mm": [
            float(v) for v in voxel_size
        ],
        "input_local_field_hz": str(field_hz),
        "input_mask": str(mask_path),
        "input_b0_json": str(b0_json),
        "output_chi_ppm": str(out_path),
        "ppm_conversion_stats": ppm_stats,
        "command": cmd,
        "note": (
            "Validation-only STI Suite iLSQR reconstruction. "
            "The input local field and support are the finalized APF "
            "GRE05b products also used by GRE06b, so the comparison "
            "isolates dipole inversion. Case-specific GRE02 voxel-space "
            "B0 direction is supplied explicitly; voxel +z fallback is "
            "not permitted."
        ),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "return_code": int(result.returncode),
        "status": "OK" if result.returncode == 0 else "FAILED",
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    if result.returncode != 0:
        print(f"  [FAILED] Docker returned {result.returncode}")
        return False

    if not out_path.exists():
        print(
            "  [FAILED] container returned 0 but "
            "chimap.nii.gz is missing"
        )
        return False

    out_img = nib.load(str(out_path))

    if out_img.shape != field_img.shape:
        print(
            "  [FAILED] output shape mismatch: "
            f"{out_img.shape} vs {field_img.shape}"
        )
        return False

    if not np.allclose(
        out_img.affine,
        field_img.affine,
        atol=1e-5,
        rtol=0,
    ):
        print("  [FAILED] output affine mismatch")
        return False

    chi = out_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32) > 0

    values = chi[mask & np.isfinite(chi)]

    if values.size == 0:
        print("  [FAILED] no finite susceptibility voxels in mask")
        return False

    print(
        "  chi masked   : "
        f"min={float(np.min(values)):.6f}, "
        f"max={float(np.max(values)):.6f}, "
        f"mean={float(np.mean(values)):.6f}, "
        f"sd={float(np.std(values)):.6f} ppm"
    )
    print("  [OK]")

    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "APF GRE06c: run STI Suite v3 iLSQR external "
            "QSM reference using the pinned QSM-CI Docker image"
        )
    )

    p.add_argument("--root")

    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID; repeat for multiple cases",
    )

    p.add_argument(
        "--all-ready",
        action="store_true",
        help=(
            "Run all cases with finalized GRE05b field/mask "
            "and GRE02 B0 provenance"
        ),
    )

    p.add_argument(
        "--runner",
        choices=["docker"],
        default=DEFAULT_RUNNER,
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")

    return p


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    require_command("docker")
    require_command("qsm-ci")

    cases = list(dict.fromkeys(args.case))

    if args.all_ready:
        cases = list_cases(root)

    if not cases:
        raise SystemExit(
            "Select --case CASE_ID or use --all-ready"
        )

    qsm_ci_version = command_output(
        ["qsm-ci", "--version"]
    )

    print(f"APF GRE06c STI iLSQR runner v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Method: {METHOD_SLUG}")
    print(f"Reference: STI Suite v3 iLSQR")
    print(f"QSM-CI: {qsm_ci_version}")
    print(f"QSM-CI record: {QSM_CI_RECORD_ID}")
    print(f"Container: {STI_DOCKER_IMAGE}")
    print("Execution: direct pinned Docker image")
    print(f"Cases: {len(cases)}")
    print(
        "Field conversion: "
        f"Hz / (gamma * B0) * 1e6, "
        f"gamma={GAMMA_HZ_PER_T:.8e} Hz/T"
    )
    print("Mask: GRE05b final V-SHARP support")
    print("B0 direction: case-specific GRE02 voxel vector")

    total = 0
    ok = 0

    for case in cases:
        total += 1

        if run_sti(
            root=root,
            case=case,
            runner=args.runner,
            dry_run=args.dry_run,
            force=args.force,
            qsm_ci_version=qsm_ci_version,
        ):
            ok += 1

    failed = total - ok

    print("\n============================================================")
    print(f"GRE06c STI iLSQR complete: OK={ok}, FAILED={failed}")
    print("============================================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
