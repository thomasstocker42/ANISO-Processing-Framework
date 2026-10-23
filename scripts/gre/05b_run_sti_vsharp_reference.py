#!/usr/bin/env python3
"""
APF GRE05b external reference runner: STI Suite V-SHARP.

Version 0.2.1:
- uses finalized parity-aware GRE04b total-field input;
- uses GRE05a v0.2.0 Mask 3 from GRE/05a_bfr_masks;
- STI Suite reconstruction method, container, and smvsize handling unchanged.

Purpose
-------
Run the published STI Suite V-SHARP implementation on the same upstream
GRE04b WLS total field and GRE05 Mask 3 used for APF GRE05b validation.

QSM-CI 0.3.0 discovers the matlab-sti-vsharp reference implementation,
but its installed registry does not expose the method's `smvsize`
parameter to `--set`, despite Zenodo record 21549381 declaring it and
recon.m accepting config.json.

Therefore this runner executes the exact QSM-CI pinned Docker image
directly and supplies config.json explicitly.

The script:
1. discovers GRE-ready cases,
2. converts GRE04b total field from Hz to ppm,
3. stages the exact input contract expected by recon.m,
4. runs the pinned STI Suite V-SHARP container,
5. supports selected smvsize values,
6. records provenance for every reconstruction.

This script does NOT implement V-SHARP itself.
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


SCRIPT_VERSION = "0.2.1"

METHOD_SLUG = "matlab-sti-vsharp"
QSM_CI_RECORD_ID = "21549381"

STI_DOCKER_IMAGE = (
    "ghcr.io/astewartau/qsm-ci/"
    "matlab-sti-vsharp@sha256:"
    "fc9dfbadd1d8402872d9644dfb101959ea158ac6f2907456cbf81fb16b871a40"
)

B0_T = 3.0
GAMMA_MHZ_PER_T = 42.577478518
HZ_PER_PPM = GAMMA_MHZ_PER_T * B0_T

DEFAULT_SMV_SIZES = [9, 12]
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
            / "04b_field_wls"
            / f"{case}_GRE04b_total_field_hz.nii.gz"
        ),
        "mask3": (
            gre
            / "05a_bfr_masks"   ## "05_vsharp" vorher
            / f"{case}_GRE05_mask3_bfr_holefilled.nii.gz"
        ),
        "sti_input_dir": gre / "05b_vsharp_sti_input",
    }


def list_cases(root: Path) -> list[str]:
    """Discover cases having both GRE04b WLS field and GRE05 Mask 3."""
    derivatives = root / "derivatives"
    cases: list[str] = []

    for case_dir in sorted(derivatives.glob("ANISO_*_InS")):
        case = case_dir.name
        p = case_paths(root, case)

        if p["field_hz"].exists() and p["mask3"].exists():
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


def prepare_ppm_input(
    field_hz_path: Path,
    ppm_path: Path,
    force: bool,
) -> dict[str, float]:
    if ppm_path.exists() and not force:
        img = nib.load(str(ppm_path))
        ppm = img.get_fdata(dtype=np.float32)

        return {
            "ppm_min": float(np.nanmin(ppm)),
            "ppm_max": float(np.nanmax(ppm)),
        }

    img = nib.load(str(field_hz_path))
    field_hz = img.get_fdata(dtype=np.float32)

    field_ppm = field_hz / HZ_PER_PPM

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)

    ppm_path.parent.mkdir(parents=True, exist_ok=True)

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
    }


def run_sti(
    root: Path,
    case: str,
    smvsize: int,
    runner: str,
    dry_run: bool,
    force: bool,
    qsm_ci_version: str,
) -> bool:
    p = case_paths(root, case)

    field_hz = p["field_hz"]
    mask3 = p["mask3"]

    input_dir = p["sti_input_dir"]
    ppm_path = (
        input_dir
        / f"{case}_GRE04b_total_field_ppm.nii.gz"
    )

    out_dir = (
        root
        / "derivatives"
        / case
        / "GRE"
        / f"05b_vsharp_sti_smv{smvsize}"
    )

    out_path = (
        out_dir
        / f"{case}_GRE05b_sti_vsharp_smv{smvsize}"
        "_local_field_ppm.nii.gz"
    )

    provenance_path = (
        out_dir
        / f"{case}_GRE05b_sti_vsharp_smv{smvsize}"
        "_provenance.json"
    )

    print(f"\n[{case}]")
    print(f"  GRE04b field : {field_hz}")
    print(f"  Mask 3       : {mask3}")
    print(f"  STI ppm input: {ppm_path}")
    print(f"  STI output   : {out_path}")
    print(f"  smvsize      : {smvsize}")
    print(f"  runner       : {runner}")

    if not field_hz.exists():
        print("  [FAILED] missing GRE04b total field")
        return False

    if not mask3.exists():
        print("  [FAILED] missing GRE05 Mask 3")
        return False

    if dry_run:
        print(f"  [DRYRUN] image={STI_DOCKER_IMAGE}")
        print(f"           config.json: smvsize={smvsize}")
        return True

    ppm_stats = prepare_ppm_input(
        field_hz_path=field_hz,
        ppm_path=ppm_path,
        force=force,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("  [SKIP] output already exists; use --force to rerun")
        return True

    field_img = nib.load(str(ppm_path))
    voxel_size = [
        float(v) for v in field_img.header.get_zooms()[:3]
    ]

    started = datetime.now(timezone.utc)
    cmd: list[str] = []
    result: subprocess.CompletedProcess | None = None

    with tempfile.TemporaryDirectory(
        prefix=f"apf-sti-vsharp-{case}-smv{smvsize}-"
    ) as tmp:
        tmp_root = Path(tmp)
        docker_input = tmp_root / "input"
        docker_output = tmp_root / "output"

        docker_input.mkdir()
        docker_output.mkdir()

        # Exact filenames expected by the QSM-CI STI recon.m wrapper.
        shutil.copy2(
            ppm_path,
            docker_input / "totalfield.nii.gz",
        )
        shutil.copy2(
            mask3,
            docker_input / "mask.nii.gz",
        )

        params = {
            "voxel_size": voxel_size,
        }
        (docker_input / "params.json").write_text(
            json.dumps(params, indent=2) + "\n"
        )

        config = {
            "smvsize": int(smvsize),
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

        print("  running pinned STI Suite container...")
        print("  command: " + " ".join(cmd))

        result = subprocess.run(cmd)

        container_output = docker_output / "localfield.nii.gz"

        if result.returncode == 0 and container_output.exists():
            shutil.copy2(container_output, out_path)

    finished = datetime.now(timezone.utc)

    assert result is not None

    provenance = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "case": case,
        "method": METHOD_SLUG,
        "qsm_ci_record_id": QSM_CI_RECORD_ID,
        "qsm_ci_version": qsm_ci_version,
        "execution": "direct_docker",
        "container_image": STI_DOCKER_IMAGE,
        "smvsize": int(smvsize),
        "runner": "docker",
        "b0_t": B0_T,
        "gamma_mhz_per_t": GAMMA_MHZ_PER_T,
        "hz_per_ppm": HZ_PER_PPM,
        "voxel_size_mm": voxel_size,
        "input_total_field_hz": str(field_hz),
        "input_total_field_ppm": str(ppm_path),
        "input_mask": str(mask3),
        "output_local_field_ppm": str(out_path),
        "ppm_conversion_stats": ppm_stats,
        "command": cmd,
        "note": (
            "Direct execution of the QSM-CI pinned Docker image. "
            "QSM-CI 0.3.0 registry does not expose the method's "
            "smvsize parameter to --set, although Zenodo record "
            "21549381 recon.m/config.json supports it."
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
            "localfield.nii.gz is missing"
        )
        return False

    print("  [OK]")
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "APF GRE05b: run STI Suite V-SHARP reference "
            "using the QSM-CI pinned Docker image"
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
        "--all-gre-ready",
        action="store_true",
        help="Run all cases with GRE04b field + GRE05 Mask 3",
    )
    p.add_argument(
        "--smvsize",
        type=int,
        nargs="+",
        default=DEFAULT_SMV_SIZES,
        help="STI Suite smvsize values (default: 9 12)",
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

    if args.all_gre_ready:
        cases = list_cases(root)

    if not cases:
        raise SystemExit(
            "Select --case CASE_ID or use --all-gre-ready"
        )

    smv_sizes = list(dict.fromkeys(args.smvsize))

    if any(v <= 0 for v in smv_sizes):
        raise SystemExit("--smvsize values must be > 0")

    qsm_ci_version = command_output(["qsm-ci", "--version"])

    print(f"APF GRE05b STI reference runner v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Method: {METHOD_SLUG}")
    print(f"QSM-CI: {qsm_ci_version}")
    print(f"QSM-CI record: {QSM_CI_RECORD_ID}")
    print(f"Execution: direct pinned Docker image")
    print(f"Runner: {args.runner}")
    print(f"Cases: {len(cases)}")
    print(
        "STI smvsize values: "
        + ", ".join(str(v) for v in smv_sizes)
    )
    print(f"Field conversion: Hz / {HZ_PER_PPM:.9f} -> ppm")
    print("Mask: GRE05 Mask 3 (hole-filled BFR mask)")

    total = 0
    ok = 0

    for smvsize in smv_sizes:
        print(
            "\n"
            "============================================================"
        )
        print(f"STI Suite V-SHARP: smvsize={smvsize}")
        print(
            "============================================================"
        )

        for case in cases:
            total += 1
            if run_sti(
                root=root,
                case=case,
                smvsize=smvsize,
                runner=args.runner,
                dry_run=args.dry_run,
                force=args.force,
                qsm_ci_version=qsm_ci_version,
            ):
                ok += 1

    failed = total - ok

    print("\n============================================================")
    print(
        f"Completed: OK={ok}, FAILED={failed}, TOTAL={total}"
    )
    print("============================================================")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
