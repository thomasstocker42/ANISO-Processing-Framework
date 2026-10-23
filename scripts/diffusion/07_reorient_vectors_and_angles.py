#!/usr/bin/env python3
"""
APF dwi07: Reorient DTI eigenvectors and compute fiber-orientation maps.

dwi07 = vector reorientation + orientation physics.

Inputs:
    dwi06 metadata
    dwi06 b0 -> GRE transform matrix
    dwi05 V1/V2/V3 per shell

Outputs:
    Reoriented V1/V2/V3 in GRE orientation
    angle_to_B0_deg
    cos_theta
    cos2_theta
    inclination_deg
    azimuth_deg
    RGB direction QC
    histograms

Important:
    DTI eigenvectors are axes, not arrows. Therefore v and -v are equivalent.
    Angle-to-B0 uses abs(dot(v, B0)).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SCRIPT = "07_reorient_vectors_and_angles.py"
TAG = "dwi07"
VERSION = "0.1.0"

VECTOR_FIELDS = ["V1", "V2", "V3"]
DEFAULT_SHELL = "b2000"

# APF convention for current workflow:
# B0 is assumed along GRE/image z-axis after dwi06 registration.
B0_VECTOR = np.array([0.0, 0.0, 1.0], dtype=float)


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("dwi07 requires nibabel.") from exc


try:
    from apf_paths import APFPaths
    from apf_registry import get_cases
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run via ./apf."
    ) from exc


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dwi05_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "dtifit"


def dwi06_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "registration" / "dwi06"


def dwi07_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "DWI" / "orientation" / "dwi07"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def find_dwi06_metadata(paths: APFPaths, case: str) -> Path:
    path = dwi06_dir(paths, case) / f"dwi06_{case}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dwi06 metadata: {path}")
    return path


def find_dtifit_shell_dirs(paths: APFPaths, case: str) -> List[Path]:
    base = dwi05_dir(paths, case)
    if not base.exists():
        raise FileNotFoundError(f"Missing dwi05 dtifit directory: {base}")

    shell_dirs = sorted(
        p for p in base.iterdir()
        if p.is_dir() and p.name.startswith("b")
    )

    if not shell_dirs:
        raise FileNotFoundError(f"No shell directories found in {base}")

    return shell_dirs


def find_shell_vector(shell_dir: Path, case: str, vector_name: str) -> Path:
    candidates = sorted(shell_dir.glob(f"dwi05_{case}_*_{vector_name}.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"Missing {vector_name} in {shell_dir}")
    return candidates[0]


def read_transform_matrix(metadata: Dict) -> np.ndarray:
    reg = metadata["registration"]

    if "transform_matrix" in reg and isinstance(reg["transform_matrix"], dict):
        mat = np.array(reg["transform_matrix"]["matrix"], dtype=float)
        return mat

    if "transform" in reg and isinstance(reg["transform"], dict):
        mat = np.array(reg["transform"]["matrix"], dtype=float)
        return mat

    if "transform_matrix" in reg and isinstance(reg["transform_matrix"], str):
        mat_path = Path(reg["transform_matrix"])
        return np.loadtxt(mat_path)

    raise KeyError("Could not find transform matrix in dwi06 metadata.")


def extract_rotation(matrix4: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Extract and orthogonalize the rotation component.

    For dwi06 default 6 DOF, the upper-left 3x3 should already be close to a
    pure rotation. SVD projection gives the nearest orthonormal rotation and
    guards against tiny numerical deviations.
    """
    linear = matrix4[:3, :3]
    u, s, vh = np.linalg.svd(linear)
    rotation = u @ vh

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    det_linear = float(np.linalg.det(linear))
    det_rotation = float(np.linalg.det(rotation))
    orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    scale_singular_values = [float(x) for x in s]

    trace = float(np.trace(rotation))
    cosang = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))

    metrics = {
        "determinant_linear_component": det_linear,
        "determinant_rotation": det_rotation,
        "orthogonality_error_frobenius": orth_error,
        "singular_values_linear_component": scale_singular_values,
        "rotation_angle_degrees": angle_deg,
    }

    return rotation, metrics


def normalize_vectors(vec: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vec, axis=-1)
    valid = np.isfinite(norms) & (norms > eps)

    out = np.zeros_like(vec, dtype=np.float32)
    out[valid] = vec[valid] / norms[valid, None]

    return out, valid


def reorient_vector_field(vec: np.ndarray, rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation to vector field.

    vec shape expected: X x Y x Z x 3.
    """
    if vec.ndim != 4 or vec.shape[-1] != 3:
        raise ValueError(f"Expected vector image with shape XxYxZx3, got {vec.shape}")

    rotated = np.einsum("ij,...j->...i", rotation, vec)
    normalized, valid = normalize_vectors(rotated)
    return normalized, valid


def compute_orientation_maps(v1: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute orientation products relative to B0.

    Because DTI eigenvectors are sign ambiguous, angle-to-B0 is axial:
        cos(theta) = abs(dot(v, B0))
    With B0 = z-axis, this is abs(Vz).
    """
    dot = np.einsum("...i,i->...", v1, B0_VECTOR)
    cos_theta = np.abs(dot)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    theta_deg = np.degrees(np.arccos(cos_theta))
    cos2_theta = cos_theta ** 2

    # Inclination is same as axial angle-to-B0 under this convention.
    inclination_deg = theta_deg.copy()

    # Azimuth in xy-plane. Sign ambiguity means azimuth is descriptive/QC,
    # not usually the primary scalar for axial fiber analyses.
    azimuth_deg = np.degrees(np.arctan2(v1[..., 1], v1[..., 0]))

    for arr in [cos_theta, theta_deg, cos2_theta, inclination_deg, azimuth_deg]:
        arr[~valid] = 0.0

    return {
        "angle_to_B0_deg": theta_deg.astype(np.float32),
        "cos_theta": cos_theta.astype(np.float32),
        "cos2_theta": cos2_theta.astype(np.float32),
        "inclination_deg": inclination_deg.astype(np.float32),
        "azimuth_deg": azimuth_deg.astype(np.float32),
    }


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header.copy())
    img.set_data_dtype(data.dtype)
    nib.save(img, str(out_path))


def make_rgb_direction_map(v1: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.abs(v1).astype(np.float32)
    rgb[~valid] = 0.0
    return rgb


def make_qc_png(
    angle_map: np.ndarray,
    cos2_map: np.ndarray,
    rgb_map: np.ndarray,
    valid: np.ndarray,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  QC PNG skipped: matplotlib unavailable.")
        return

    z = angle_map.shape[2] // 2
    y = angle_map.shape[1] // 2
    x = angle_map.shape[0] // 2

    valid_angle = angle_map[valid]
    valid_cos2 = cos2_map[valid]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(angle_map[:, :, z].T, origin="lower", vmin=0, vmax=90)
    axes[0, 0].set_title("Angle to B0 axial")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(angle_map[:, y, :].T, origin="lower", vmin=0, vmax=90)
    axes[0, 1].set_title("Angle to B0 coronal")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_map[:, :, z, :].transpose(1, 0, 2), origin="lower")
    axes[0, 2].set_title("RGB |V1| axial")
    axes[0, 2].axis("off")

    axes[1, 0].hist(valid_angle.ravel(), bins=90, range=(0, 90))
    axes[1, 0].set_title("Histogram θ")
    axes[1, 0].set_xlabel("degrees")
    axes[1, 0].set_ylabel("voxels")

    axes[1, 1].hist(valid_cos2.ravel(), bins=50, range=(0, 1))
    axes[1, 1].set_title("Histogram cos²θ")
    axes[1, 1].set_xlabel("cos²θ")
    axes[1, 1].set_ylabel("voxels")

    axes[1, 2].imshow(angle_map[x, :, :].T, origin="lower", vmin=0, vmax=90)
    axes[1, 2].set_title("Angle to B0 sagittal")
    axes[1, 2].axis("off")

    fig.suptitle("dwi07 orientation QC")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def summarize_array(data: np.ndarray, valid: np.ndarray) -> Dict:
    vals = data[valid]
    if vals.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
        }

    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
    }


def process_shell(
    case: str,
    shell_dir: Path,
    shell: str,
    rotation: np.ndarray,
    out_case_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    print(f"  Shell: {shell}")

    out_dir = out_case_dir / shell
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_inputs = {
        vec: find_shell_vector(shell_dir, case, vec)
        for vec in VECTOR_FIELDS
    }

    ref_img = nib.load(str(vector_inputs["V1"]))

    outputs = {}
    validation = {}

    reoriented = {}

    for vec_name, vec_path in vector_inputs.items():
        img = nib.load(str(vec_path))
        data = np.asanyarray(img.dataobj).astype(np.float64)

        vec_reoriented, valid = reorient_vector_field(data, rotation)
        reoriented[vec_name] = vec_reoriented

        out_path = out_dir / f"dwi07_{case}_{shell}_{vec_name}_reoriented_in_GRE.nii.gz"
        outputs[f"{vec_name}_reoriented"] = str(out_path)

        if overwrite or not out_path.exists():
            if not dry_run:
                save_nifti(vec_reoriented.astype(np.float32), img, out_path)

        norms = np.linalg.norm(vec_reoriented, axis=-1)
        validation[vec_name] = {
            "valid_voxels": int(valid.sum()),
            "norm_mean_valid": float(np.mean(norms[valid])) if valid.any() else None,
            "norm_std_valid": float(np.std(norms[valid])) if valid.any() else None,
            "norm_min_valid": float(np.min(norms[valid])) if valid.any() else None,
            "norm_max_valid": float(np.max(norms[valid])) if valid.any() else None,
        }

    # Use V1 for orientation products.
    v1 = reoriented["V1"]
    _, valid_v1 = normalize_vectors(v1)
    maps = compute_orientation_maps(v1, valid_v1)

    for name, data in maps.items():
        out_path = out_dir / f"dwi07_{case}_{shell}_{name}.nii.gz"
        outputs[name] = str(out_path)

        if overwrite or not out_path.exists():
            if not dry_run:
                save_nifti(data, ref_img, out_path)

    rgb = make_rgb_direction_map(v1, valid_v1)
    rgb_path = out_dir / f"dwi07_{case}_{shell}_V1_rgb_abs.nii.gz"
    outputs["V1_rgb_abs"] = str(rgb_path)

    if overwrite or not rgb_path.exists():
        if not dry_run:
            save_nifti(rgb.astype(np.float32), ref_img, rgb_path)

    qc_png = out_dir / f"dwi07_{case}_{shell}_orientation_qc.png"
    outputs["qc_png"] = str(qc_png)

    if not dry_run:
        make_qc_png(
            angle_map=maps["angle_to_B0_deg"],
            cos2_map=maps["cos2_theta"],
            rgb_map=rgb,
            valid=valid_v1,
            out_png=qc_png,
        )

    # Orthogonality QC between eigenvectors.
    v1 = reoriented["V1"]
    v2 = reoriented["V2"]
    v3 = reoriented["V3"]

    dot12 = np.sum(v1 * v2, axis=-1)
    dot13 = np.sum(v1 * v3, axis=-1)
    dot23 = np.sum(v2 * v3, axis=-1)

    valid_all = valid_v1 & np.isfinite(dot12) & np.isfinite(dot13) & np.isfinite(dot23)

    orth_qc = {
        "abs_dot_V1_V2_mean": float(np.mean(np.abs(dot12[valid_all]))) if valid_all.any() else None,
        "abs_dot_V1_V3_mean": float(np.mean(np.abs(dot13[valid_all]))) if valid_all.any() else None,
        "abs_dot_V2_V3_mean": float(np.mean(np.abs(dot23[valid_all]))) if valid_all.any() else None,
        "valid_voxels_all_vectors": int(valid_all.sum()),
    }

    summaries = {
        "angle_to_B0_deg": summarize_array(maps["angle_to_B0_deg"], valid_v1),
        "cos_theta": summarize_array(maps["cos_theta"], valid_v1),
        "cos2_theta": summarize_array(maps["cos2_theta"], valid_v1),
        "inclination_deg": summarize_array(maps["inclination_deg"], valid_v1),
        "azimuth_deg": summarize_array(maps["azimuth_deg"], valid_v1),
    }

    return {
        "shell": shell,
        "inputs": {k: str(v) for k, v in vector_inputs.items()},
        "outputs": outputs,
        "validation": {
            "vector_norm_summary": validation,
            "orthogonality": orth_qc,
        },
        "summaries": summaries,
    }


def process_case(
    case: str,
    paths: APFPaths,
    selected_shells: Optional[List[str]],
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    start = now_iso()

    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    out_case_dir = dwi07_dir(paths, case)
    out_case_dir.mkdir(parents=True, exist_ok=True)

    dwi06_metadata_path = find_dwi06_metadata(paths, case)
    dwi06_metadata = load_json(dwi06_metadata_path)

    matrix4 = read_transform_matrix(dwi06_metadata)
    rotation, rotation_metrics = extract_rotation(matrix4)

    shell_dirs_all = find_dtifit_shell_dirs(paths, case)
    shell_dirs = []

    for shell_dir in shell_dirs_all:
        if selected_shells is None or shell_dir.name in selected_shells:
            shell_dirs.append(shell_dir)

    if not shell_dirs:
        raise RuntimeError(f"No requested shells found for {case}: {selected_shells}")

    shell_results = {}

    for shell_dir in shell_dirs:
        result = process_shell(
            case=case,
            shell_dir=shell_dir,
            shell=shell_dir.name,
            rotation=rotation,
            out_case_dir=out_case_dir,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        shell_results[shell_dir.name] = result

    end = now_iso()

    metadata = {
        "script": SCRIPT,
        "tag": TAG,
        "version": VERSION,
        "case": case,
        "created": end,
        "execution": {
            "start": start,
            "end": end,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "command": " ".join(sys.argv),
            "dry_run": dry_run,
        },
        "upstream": {
            "dwi06_metadata": str(dwi06_metadata_path),
            "dwi06_registration_matrix": dwi06_metadata["registration"].get(
                "transform_matrix",
                dwi06_metadata["registration"].get("transform"),
            ),
        },
        "orientation_convention": {
            "space": "GRE/image space after dwi06 registration",
            "B0_vector": [float(x) for x in B0_VECTOR],
            "B0_assumption": "main magnetic field is represented by the image z-axis in GRE space",
            "eigenvector_sign_ambiguity": "v and -v are equivalent; angle_to_B0 uses abs(dot(v, B0))",
            "angle_to_B0_degrees": "theta = arccos(abs(dot(V1, B0)))",
            "cos_theta": "abs(dot(V1, B0))",
            "cos2_theta": "cos_theta squared",
            "inclination_degrees": "same as angle_to_B0 under current convention",
            "azimuth_degrees": "atan2(V1_y, V1_x), descriptive because eigenvectors are sign-ambiguous",
        },
        "reorientation": {
            "method": "apply nearest orthonormal rotation extracted from dwi06 FLIRT matrix via SVD projection",
            "full_matrix_4x4": matrix4.tolist(),
            "rotation_matrix_3x3": rotation.tolist(),
            "rotation_metrics": rotation_metrics,
        },
        "rotation_validation": {
            "determinant_linear_component": rotation_metrics["determinant_linear_component"],
            "determinant_rotation": rotation_metrics["determinant_rotation"],
            "orthogonality_error_frobenius": rotation_metrics["orthogonality_error_frobenius"],
            "rotation_angle_degrees": rotation_metrics["rotation_angle_degrees"],
            "singular_values_linear_component": rotation_metrics["singular_values_linear_component"],
        },
        "processed_shells": [p.name for p in shell_dirs],
        "shell_results": shell_results,
    }

    meta_path = out_case_dir / f"dwi07_{case}_metadata.json"
    qc_csv = out_case_dir / f"dwi07_{case}_qc_summary.csv"

    rows = []
    for shell, result in shell_results.items():
        summaries = result["summaries"]
        orth = result["validation"]["orthogonality"]

        rows.append(
            {
                "case": case,
                "shell": shell,
                "angle_mean": summaries["angle_to_B0_deg"]["mean"],
                "angle_median": summaries["angle_to_B0_deg"]["median"],
                "angle_std": summaries["angle_to_B0_deg"]["std"],
                "cos2_mean": summaries["cos2_theta"]["mean"],
                "cos2_median": summaries["cos2_theta"]["median"],
                "valid_voxels": summaries["angle_to_B0_deg"]["n"],
                "abs_dot_V1_V2_mean": orth["abs_dot_V1_V2_mean"],
                "abs_dot_V1_V3_mean": orth["abs_dot_V1_V3_mean"],
                "abs_dot_V2_V3_mean": orth["abs_dot_V2_V3_mean"],
            }
        )

    if not dry_run:
        write_json(meta_path, metadata)
        write_csv(qc_csv, rows)

    print(f"  DONE: {case}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF dwi07: reorient DTI eigenvectors and compute orientation maps."
    )

    parser.add_argument(
        "case",
        nargs="?",
        help="Case ID, e.g. ANISO_00_000_InS. Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all active InS cases from APF registry.",
    )
    parser.add_argument(
        "--shell",
        default=DEFAULT_SHELL,
        help=f"Single shell to process. Default: {DEFAULT_SHELL}.",
    )
    parser.add_argument(
        "--all-shells",
        action="store_true",
        help="Process all available dtifit shells.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dwi07 outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print workflow intent without writing outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = APFPaths(root=Path.cwd())

    if args.all:
        cases = get_cases(case_type="InS", active_only=True)
    elif args.case:
        cases = [args.case]
    else:
        raise SystemExit("Provide a case ID or use --all.")

    selected_shells = None if args.all_shells else [args.shell]

    all_rows = []

    for case in cases:
        try:
            meta = process_case(
                case=case,
                paths=paths,
                selected_shells=selected_shells,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            all_rows.append(
                {
                    "case": case,
                    "success": True,
                    "processed_shells": ";".join(meta["processed_shells"]),
                    "metadata": str(dwi07_dir(paths, case) / f"dwi07_{case}_metadata.json"),
                }
            )
        except Exception as exc:
            print(f"ERROR in {case}: {exc}", file=sys.stderr)
            all_rows.append(
                {
                    "case": case,
                    "success": False,
                    "processed_shells": "" if selected_shells is None else ";".join(selected_shells),
                    "error": str(exc),
                }
            )
            if not args.all:
                raise

    if args.all and not args.dry_run:
        summary_dir = paths.derivatives_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_csv(summary_dir / "dwi07_orientation_summary.csv", all_rows)


if __name__ == "__main__":
    main()