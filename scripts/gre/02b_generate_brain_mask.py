#!/usr/bin/env python3
"""APF GRE02b: generate a standardized GRE-native HD-BET brain mask."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from _field_fit_common import (
    case_search_roots,
    list_cases,
    now_iso,
    resolve_root,
    save_nifti,
    write_case_summary_json,
    write_cohort_csv,
)

SCRIPT_VERSION = "0.1.0"
MODULE_NAME = "gre02b"
METHOD_NAME = "hdbet_gre_first_echo"
OUTPUT_SUBDIR = Path("GRE") / "02b_brain_mask"


@dataclass(frozen=True)
class Inputs:
    case_id: str
    magnitude: Path
    source_kind: str


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None
    source_kind: str | None = None
    source_magnitude: str | None = None
    source_shape: str | None = None
    voxel_size_mm: str | None = None
    device: str | None = None
    threads: int | None = None
    mask_voxels: int | None = None
    eroded_mask_voxels: int | None = None
    erosion_retained_fraction: float | None = None
    brain_fraction_image: float | None = None
    mask_binary: bool | None = None
    output_dir: str | None = None


def unique_existing(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def discover_gre02_magnitude(roots: Sequence[Path]) -> Path:
    preferred: list[Path] = []
    fallback: list[Path] = []
    for root in roots:
        preferred.extend(root.glob("GRE/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz"))
        preferred.extend(root.glob("gre/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz"))
        fallback.extend(root.rglob("gre02_*_magnitude_4d.nii*"))

    candidates = unique_existing(preferred) or unique_existing(fallback)
    if not candidates:
        raise FileNotFoundError("No GRE02 prepared 4D magnitude image found")
    if len(candidates) > 1:
        joined = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Ambiguous GRE02 magnitude discovery; use --magnitude explicitly:\n"
            f"  {joined}"
        )
    return candidates[0]


def discover_legacy_e1(roots: Sequence[Path]) -> Path:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob("GRE/mag/*_e1.nii.gz"))
        candidates.extend(root.glob("gre/mag/*_e1.nii.gz"))
    candidates = unique_existing(candidates)
    if not candidates:
        raise FileNotFoundError("No legacy GRE first-echo magnitude image found")
    if len(candidates) > 1:
        joined = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Ambiguous legacy GRE e1 discovery; use --magnitude explicitly:\n"
            f"  {joined}"
        )
    return candidates[0]


def discover_inputs(
    root: Path,
    case_id: str,
    explicit_magnitude: Path | None,
    allow_legacy_e1: bool,
) -> Inputs:
    if explicit_magnitude is not None:
        path = explicit_magnitude.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit magnitude does not exist: {path}")
        return Inputs(case_id, path, "explicit")

    roots = case_search_roots(root, case_id)
    if not roots:
        raise FileNotFoundError(f"No case directories found for {case_id}")

    try:
        return Inputs(case_id, discover_gre02_magnitude(roots), "gre02_magnitude_4d")
    except FileNotFoundError:
        if not allow_legacy_e1:
            raise
    return Inputs(case_id, discover_legacy_e1(roots), "legacy_gre_e1")


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case_id / OUTPUT_SUBDIR
    prefix = f"{case_id}_GRE02b"
    return {
        "dir": outdir,
        "e1": outdir / f"{prefix}_magnitude_e1.nii.gz",
        "brain": outdir / f"{prefix}_brain_extracted_e1.nii.gz",
        "mask": outdir / f"{prefix}_brain_mask.nii.gz",
        "mask_ero1": outdir / f"{prefix}_brain_mask_ero1.nii.gz",
        "command": outdir / f"{prefix}_hdbet_command.txt",
        "qc": outdir / f"{prefix}_qc.json",
    }


def load_first_echo(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, tuple[int, ...]]:
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    if len(shape) == 3:
        data = np.asanyarray(img.dataobj, dtype=np.float32)
    elif len(shape) == 4 and shape[3] >= 1:
        data = np.asanyarray(img.dataobj[..., 0], dtype=np.float32)
    else:
        raise ValueError(f"Magnitude must be 3D or non-empty 4D, got {shape}")
    if not np.any(np.isfinite(data)):
        raise ValueError("Magnitude contains no finite voxels")
    return img, np.where(np.isfinite(data), data, 0.0).astype(np.float32), shape


def make_3d_reference(source: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image:
    header = source.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(data, source.affine, header)


def build_command(
    executable: str,
    input_path: Path,
    output_path: Path,
    device: str,
    enable_tta: bool,
) -> list[str]:
    command = [
        executable,
        "-i", str(input_path),
        "-o", str(output_path),
        "-device", device,
    ]
    if not enable_tta:
        command.append("--disable_tta")
    command.append("--save_bet_mask")
    return command


def find_hdbet_outputs(requested_output: Path) -> tuple[Path, Path]:
    directory = requested_output.parent
    name = requested_output.name
    stem = name[:-7] if name.endswith(".nii.gz") else requested_output.stem
    brain_candidates = unique_existing([
        requested_output,
        directory / f"{stem}.nii.gz",
        directory / f"{stem}.nii",
    ])
    mask_candidates = unique_existing([
        directory / f"{stem}_bet.nii.gz",
        directory / f"{stem}_bet.nii",
        directory / f"{stem}_mask.nii.gz",
        directory / f"{stem}_mask.nii",
    ])
    if len(brain_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify HD-BET brain output; found: "
            + ", ".join(str(x) for x in brain_candidates)
        )
    if len(mask_candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify HD-BET mask output; found: "
            + ", ".join(str(x) for x in mask_candidates)
        )
    return brain_candidates[0], mask_candidates[0]


def validate_geometry(reference: nib.Nifti1Image, other: nib.Nifti1Image, label: str) -> None:
    if tuple(reference.shape[:3]) != tuple(other.shape[:3]):
        raise ValueError(f"{label} shape mismatch: {other.shape} vs {reference.shape[:3]}")
    if not np.allclose(reference.affine, other.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{label} affine does not match GRE first echo")


def process_case(root: Path, case_id: str, args: argparse.Namespace) -> CaseResult:
    started = time.perf_counter()
    paths = output_paths(root, case_id)
    try:
        explicit_allowed = len(args.case) == 1
        inputs = discover_inputs(
            root,
            case_id,
            args.magnitude if explicit_allowed else None,
            args.allow_legacy_e1,
        )
        source_img, e1, source_shape = load_first_echo(inputs.magnitude)
        e1_img = make_3d_reference(source_img, e1)
        zooms = source_img.header.get_zooms()[:3]
        shape_text = "x".join(str(x) for x in source_shape)
        voxel_text = "x".join(f"{float(x):g}" for x in zooms)

        if paths["mask_ero1"].exists() and not args.force:
            return CaseResult(
                case_id, "SKIPPED_EXISTS",
                "Primary output exists; use --force to overwrite",
                source_kind=inputs.source_kind,
                source_magnitude=str(inputs.magnitude),
                source_shape=shape_text,
                voxel_size_mm=voxel_text,
                device=args.device,
                threads=args.threads,
                output_dir=str(paths["dir"]),
            )

        executable = shutil.which(args.hdbet_executable)
        if executable is None:
            raise FileNotFoundError(f"HD-BET executable not found: {args.hdbet_executable}")

        preview = build_command(executable, paths["e1"], paths["brain"], args.device, args.enable_tta)
        if args.dry_run:
            return CaseResult(
                case_id, "DRY_RUN",
                f"magnitude={inputs.magnitude}; source={inputs.source_kind}; command={' '.join(preview)}",
                source_kind=inputs.source_kind,
                source_magnitude=str(inputs.magnitude),
                source_shape=shape_text,
                voxel_size_mm=voxel_text,
                device=args.device,
                threads=args.threads,
                output_dir=str(paths["dir"]),
            )

        paths["dir"].mkdir(parents=True, exist_ok=True)
        if args.force:
            for key, path in paths.items():
                if key != "dir":
                    path.unlink(missing_ok=True)

        save_nifti(e1, e1_img, paths["e1"], np.float32)
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(args.threads)

        with tempfile.TemporaryDirectory(prefix=f"{case_id}_gre02b_", dir=str(paths["dir"])) as tmp:
            temp_output = Path(tmp) / "hdbet_brain.nii.gz"
            command = build_command(executable, paths["e1"], temp_output, args.device, args.enable_tta)
            paths["command"].write_text(" ".join(command) + "\n", encoding="utf-8")
            completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"HD-BET failed ({completed.returncode})\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            raw_brain_path, raw_mask_path = find_hdbet_outputs(temp_output)
            raw_brain_img = nib.load(str(raw_brain_path))
            raw_mask_img = nib.load(str(raw_mask_path))
            validate_geometry(e1_img, raw_brain_img, "HD-BET brain")
            validate_geometry(e1_img, raw_mask_img, "HD-BET mask")
            brain = np.asanyarray(raw_brain_img.dataobj, dtype=np.float32)
            raw_mask = np.asanyarray(raw_mask_img.dataobj)

        mask = np.isfinite(raw_mask) & (raw_mask > 0)
        if not np.any(mask):
            raise ValueError("HD-BET returned an empty mask")
        structure = generate_binary_structure(3, 1)
        mask_ero1 = binary_erosion(mask, structure=structure, iterations=1, border_value=0)
        if not np.any(mask_ero1):
            raise ValueError("One-voxel erosion produced an empty mask")

        brain = np.where(mask, brain, 0.0).astype(np.float32)
        mask_u8 = mask.astype(np.uint8)
        mask_ero1_u8 = mask_ero1.astype(np.uint8)
        save_nifti(brain, e1_img, paths["brain"], np.float32)
        save_nifti(mask_u8, e1_img, paths["mask"], np.uint8)
        save_nifti(mask_ero1_u8, e1_img, paths["mask_ero1"], np.uint8)

        mask_voxels = int(mask.sum())
        eroded_voxels = int(mask_ero1.sum())
        retained = float(eroded_voxels / mask_voxels)
        brain_fraction = float(mask_voxels / np.prod(mask.shape))
        runtime_s = time.perf_counter() - started

        provenance = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "method": METHOD_NAME,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": runtime_s,
            "inputs": {
                "source_magnitude": str(inputs.magnitude),
                "source_kind": inputs.source_kind,
            },
            "configuration": {
                "source_shape": list(source_shape),
                "selected_echo_index": 0,
                "device": args.device,
                "threads": args.threads,
                "tta_enabled": args.enable_tta,
                "hdbet_executable": executable,
                "erosion": {
                    "iterations": 1,
                    "connectivity": 1,
                    "implementation": "scipy.ndimage.binary_erosion",
                },
            },
            "outputs": {k: str(v) for k, v in paths.items() if k != "dir"},
            "qc": {
                "e1_shape": list(e1.shape),
                "voxel_size_mm": [float(x) for x in zooms],
                "mask_binary": True,
                "mask_voxels": mask_voxels,
                "eroded_mask_voxels": eroded_voxels,
                "erosion_retained_fraction": retained,
                "brain_fraction_image": brain_fraction,
                "brain_intensity": {
                    "minimum": float(np.min(brain[mask])),
                    "maximum": float(np.max(brain[mask])),
                    "mean": float(np.mean(brain[mask])),
                    "median": float(np.median(brain[mask])),
                },
            },
        }
        write_case_summary_json(paths["qc"], provenance)
        return CaseResult(
            case_id, "OK", "GRE first-echo HD-BET brain-mask generation completed",
            runtime_s=runtime_s,
            source_kind=inputs.source_kind,
            source_magnitude=str(inputs.magnitude),
            source_shape=shape_text,
            voxel_size_mm=voxel_text,
            device=args.device,
            threads=args.threads,
            mask_voxels=mask_voxels,
            eroded_mask_voxels=eroded_voxels,
            erosion_retained_fraction=retained,
            brain_fraction_image=brain_fraction,
            mask_binary=True,
            output_dir=str(paths["dir"]),
        )
    except Exception as exc:
        return CaseResult(
            case_id, "FAILED", f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - started,
            device=args.device,
            threads=args.threads,
            output_dir=str(paths["dir"]),
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="APF GRE02b: GRE first-echo HD-BET brain-mask generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, help="APF root; defaults to APF_ROOT or cwd")
    p.add_argument("--case", action="append", default=[], help="Case ID; repeat as needed")
    p.add_argument("--all-gre-ready", action="store_true", help="Attempt all discovered ANISO_*_InS cases")
    p.add_argument("--magnitude", type=Path, help="Explicit 3D e1 or 4D magnitude; single case only")
    p.add_argument("--allow-legacy-e1", action="store_true", help="Allow fallback to nifti/<CASE>/GRE/mag/*_e1.nii.gz")
    p.add_argument("--device", default="cpu", help="HD-BET device, e.g. cpu, cuda, cuda:0")
    p.add_argument("--threads", type=int, default=4, help="Numerical backend thread count")
    p.add_argument("--hdbet-executable", default="hd-bet", help="HD-BET executable name or path")
    p.add_argument("--enable-tta", action="store_true", help="Enable test-time augmentation")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true", help="Validate discovery and print intended command")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if not args.device.strip():
        raise SystemExit("--device must not be empty")

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit("Select at least one --case or use --all-gre-ready")
    if args.magnitude is not None and len(cases) != 1:
        raise SystemExit("--magnitude requires exactly one selected case")

    print(f"APF GRE02b v{SCRIPT_VERSION}")
    print(f"Root:    {root}")
    print(f"Method:  {METHOD_NAME}")
    print(f"Device:  {args.device}")
    print(f"Threads: {args.threads}")
    print(f"TTA:     {'enabled' if args.enable_tta else 'disabled'}")
    print(f"Cases:   {len(cases)}")

    started_utc = now_iso()
    results: list[CaseResult] = []
    for case_id in cases:
        result = process_case(root, case_id, args)
        results.append(result)
        print(f"[{result.status}] {case_id}: {result.message}")

    summary_dir = root / "derivatives" / "summary" / "GRE"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_cohort_csv(summary_dir / "gre02b_brain_mask_qc.csv", results)
    run_payload = {
        "module": MODULE_NAME,
        "script_version": SCRIPT_VERSION,
        "method": METHOD_NAME,
        "started_utc": started_utc,
        "finished_utc": now_iso(),
        "root": str(root),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": [asdict(x) for x in results],
    }
    (summary_dir / "gre02b_brain_mask_run.json").write_text(
        json.dumps(run_payload, indent=2) + "\n", encoding="utf-8"
    )
    n_failed = sum(x.status == "FAILED" for x in results)
    n_ok = sum(x.status == "OK" for x in results)
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
