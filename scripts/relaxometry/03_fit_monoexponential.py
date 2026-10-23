#!/usr/bin/env python3
"""
APF rlx03: Native-space mono-exponential T2 fitting.

Scientific model
----------------
    S(TE) = S0 * exp(-R2 * TE)

Two fits are generated in one run:
    all12   : all available echoes from the authoritative rlx02 stack
    drop1   : first echo excluded

Primary upstream contract
-------------------------
rlx03 consumes the rlx02 metadata JSON and resolves:
    - the authoritative 4D 12-echo magnitude stack
    - echo times in seconds or milliseconds
    - the authoritative fitting mask, when provided

The metadata parser is deliberately defensive because older rlx02 revisions may
use slightly different key names. Ambiguous or inconsistent inputs cause a hard
failure rather than silent guessing.

Outputs per model
-----------------
    R2_s-1
    T2_ms
    S0
    RMSE
    NRMSE
    R2coef
    fit_valid_mask
    boundary_hit_mask
    failed_fit_mask

QC
--
    - per-model summary CSV
    - per-case summary CSV
    - metadata JSON
    - cohort summary CSV
    - compact QC PNG

Fitting method
--------------
A bounded mono-exponential least-squares fit is solved with variable projection:
for each candidate R2, the least-squares-optimal S0 is calculated analytically.
A coarse-to-fine one-dimensional R2 search is then performed in vectorized
voxel chunks. This is substantially faster than calling scipy.optimize once per
voxel and preserves the native acquisition grid.

Run through the APF launcher:
    ./apf scripts/relaxometry/03_fit_monoexponential.py ANISO_00_000_InS
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT = "03_fit_monoexponential.py"
TAG = "rlx03"
VERSION = "0.1.0"

MODEL_ALL = "all12"
MODEL_DROP1 = "drop1"

DEFAULT_R2_MIN = 0.5
DEFAULT_R2_MAX = 500.0
DEFAULT_S0_MIN = 0.0
DEFAULT_S0_MAX_FACTOR = 5.0
DEFAULT_MIN_SIGNAL = 1.0
DEFAULT_MIN_ECHOES = 4
DEFAULT_CHUNK_VOXELS = 50000
DEFAULT_COARSE_POINTS = 256
DEFAULT_REFINE_POINTS = 81
DEFAULT_BOUNDARY_RTOL = 1e-3
DEFAULT_NRMSE_EPS = 1e-8


try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("rlx03 requires nibabel.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("rlx03 requires matplotlib.") from exc

try:
    from apf_paths import APFPaths, get_paths
    from apf_case_selection import (
        add_case_selection_args,
        print_selected_cases,
        resolve_cases,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run via ./apf."
    ) from exc


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def rlx02_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "relaxometry" / "rlx02"


def rlx03_dir(paths: APFPaths, case: str) -> Path:
    return paths.derivatives_dir / case / "relaxometry" / "rlx03"


def summary_dir(paths: APFPaths) -> Path:
    return paths.derivatives_dir / "summary" / "relaxometry" / "rlx03"


def recursively_find_values(obj: Any, wanted_keys: Sequence[str]) -> List[Any]:
    wanted = {k.lower() for k in wanted_keys}
    found: List[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in wanted:
                    found.append(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return found


def flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, Path):
        out.append(str(value))
    elif isinstance(value, dict):
        for child in value.values():
            out.extend(flatten_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            out.extend(flatten_strings(child))
    return out


def resolve_path(raw: str, metadata_path: Path, project_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([
            project_root / candidate,
            metadata_path.parent / candidate,
        ])
    existing = first_existing(candidates)
    if existing is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Metadata path does not exist: {raw}\nTried:\n{tried}")
    return existing.resolve()


def find_rlx02_metadata(
    paths: APFPaths,
    case: str,
) -> Path:
    metadata_path = (
        paths.case_stage_dir(
            case=case,
            module="relaxometry",
            stage="rlx02_prepare_multi_echo_t2",
        )
        / f"rlx02_{case}_preparation.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Required rlx02 preparation metadata not found for {case}:\n"
            f"{metadata_path}\n"
            "Run rlx02 before rlx03."
        )

    return metadata_path


def load_metadata(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Metadata root must be a JSON object: {path}")
    return data


def resolve_rlx02_inputs(
    meta: Dict[str, Any],
    metadata_path: Path,
    project_root: Path,
    n_echoes_expected: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve authoritative rlx02 outputs using the explicit metadata schema.

    Required:
        outputs.series_4d
        outputs.fit_mask
        input.echo_times_s
    """

    outputs = meta.get("outputs")
    input_meta = meta.get("input")

    if not isinstance(outputs, dict):
        raise ValueError(
            f"rlx02 metadata lacks an 'outputs' object: {metadata_path}"
        )

    if not isinstance(input_meta, dict):
        raise ValueError(
            f"rlx02 metadata lacks an 'input' object: {metadata_path}"
        )

    raw_stack = outputs.get("series_4d")
    raw_mask = outputs.get("fit_mask")
    raw_tes = input_meta.get("echo_times_s")

    missing_keys = []

    if not isinstance(raw_stack, str) or not raw_stack.strip():
        missing_keys.append("outputs.series_4d")

    if not isinstance(raw_mask, str) or not raw_mask.strip():
        missing_keys.append("outputs.fit_mask")

    if not isinstance(raw_tes, list) or not raw_tes:
        missing_keys.append("input.echo_times_s")

    if missing_keys:
        formatted = "\n".join(f"  - {key}" for key in missing_keys)
        raise ValueError(
            "Incomplete rlx02 metadata contract:\n"
            f"{formatted}\n"
            f"Metadata: {metadata_path}"
        )

    stack_path = resolve_path(
        raw_stack,
        metadata_path,
        project_root,
    )

    mask_path = resolve_path(
        raw_mask,
        metadata_path,
        project_root,
    )

    try:
        tes = np.asarray(
            [float(value) for value in raw_tes],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "input.echo_times_s must contain numeric values."
        ) from exc

    if n_echoes_expected is not None:
        tes = normalize_echo_times(
            tes,
            n_echoes_expected,
        )

    return {
        "stack": stack_path,
        "fit_mask": mask_path,
        "echo_times_s": tes,
    }


def resolve_stack_from_metadata(
    meta: Dict[str, Any],
    metadata_path: Path,
    project_root: Path,
) -> Path:
    preferred_keys = [
        "authoritative_4d_stack",
        "magnitude_4d",
        "echo_stack",
        "stack_4d",
        "t2fit_4d",
        "se_12echo_4d",
        "output_4d",
        "merged_4d",
    ]
    values = recursively_find_values(meta, preferred_keys)
    raw_candidates: List[str] = []
    for value in values:
        raw_candidates.extend(flatten_strings(value))

    # Secondary scan: paths containing strong semantic tokens.
    for raw in flatten_strings(meta):
        low = raw.lower()
        if (
            (low.endswith(".nii") or low.endswith(".nii.gz"))
            and any(token in low for token in ["12echo", "12_echo", "t2fit", "echo_stack", "merged"])
            and not any(token in low for token in ["mask", "rmse", "r2", "t2map"])
        ):
            raw_candidates.append(raw)

    resolved: List[Path] = []
    for raw in raw_candidates:
        try:
            p = resolve_path(raw, metadata_path, project_root)
        except FileNotFoundError:
            continue
        if p.suffix == ".nii" or p.name.endswith(".nii.gz"):
            resolved.append(p)

    unique = list(dict.fromkeys(resolved))
    valid: List[Path] = []
    for path in unique:
        try:
            shape = nib.load(str(path)).shape
        except Exception:
            continue
        if len(shape) == 4 and shape[3] >= 4:
            valid.append(path)

    if len(valid) != 1:
        formatted = "\n".join(f"  {p}" for p in valid) or "  <none>"
        raise RuntimeError(
            "Could not uniquely resolve the authoritative rlx02 4D echo stack.\n"
            f"Valid candidates:\n{formatted}\n"
            "Add an explicit metadata key such as outputs.authoritative_4d_stack."
        )
    return valid[0]


def normalize_echo_times(values: Sequence[float], n_echoes: int) -> np.ndarray:
    tes = np.asarray(values, dtype=np.float64).reshape(-1)
    if tes.size != n_echoes:
        raise ValueError(
            f"Echo-time count ({tes.size}) does not match 4D stack ({n_echoes})."
        )
    if not np.all(np.isfinite(tes)) or np.any(tes <= 0):
        raise ValueError("Echo times must be finite and strictly positive.")
    if np.any(np.diff(tes) <= 0):
        raise ValueError("Echo times must be strictly increasing.")

    # Conservative unit inference:
    # MRI TEs represented in seconds are normally < 1; in ms they are > 1.
    if np.nanmax(tes) > 1.0:
        tes = tes / 1000.0

    if np.nanmax(tes) > 1.0:
        raise ValueError("Echo-time units remain implausible after ms-to-s conversion.")
    return tes


def resolve_te_from_metadata(meta: Dict[str, Any], n_echoes: int) -> np.ndarray:
    keys = [
        "echo_times_s",
        "echo_times_sec",
        "echo_times_seconds",
        "echo_times_ms",
        "echo_times",
        "te_s",
        "te_ms",
        "tes",
    ]
    values = recursively_find_values(meta, keys)
    candidates: List[np.ndarray] = []

    for value in values:
        if isinstance(value, dict):
            # Accept mappings like {"e1": 10.0, ...}
            try:
                seq = [float(v) for _, v in sorted(value.items())]
            except Exception:
                continue
        elif isinstance(value, (list, tuple)):
            try:
                seq = [float(v) for v in value]
            except Exception:
                continue
        else:
            continue

        try:
            candidates.append(normalize_echo_times(seq, n_echoes))
        except ValueError:
            continue

    unique: List[np.ndarray] = []
    for candidate in candidates:
        if not any(np.allclose(candidate, prior, rtol=0, atol=1e-9) for prior in unique):
            unique.append(candidate)

    if len(unique) != 1:
        raise RuntimeError(
            "Could not uniquely resolve echo times from rlx02 metadata. "
            "Store one explicit array such as acquisition.echo_times_ms."
        )
    return unique[0]


def resolve_mask_from_metadata(
    meta: Dict[str, Any],
    metadata_path: Path,
    project_root: Path,
    spatial_shape: Tuple[int, int, int],
) -> Optional[Path]:
    keys = [
        "authoritative_mask",
        "fit_mask",
        "brain_mask",
        "analysis_mask",
        "mask",
    ]
    raw_candidates: List[str] = []
    for value in recursively_find_values(meta, keys):
        raw_candidates.extend(flatten_strings(value))

    valid: List[Path] = []
    for raw in raw_candidates:
        low = raw.lower()
        if not (low.endswith(".nii") or low.endswith(".nii.gz")):
            continue
        try:
            path = resolve_path(raw, metadata_path, project_root)
            shape = nib.load(str(path)).shape
        except Exception:
            continue
        if tuple(shape[:3]) == tuple(spatial_shape):
            valid.append(path)

    unique = list(dict.fromkeys(valid))
    if len(unique) > 1:
        # Prefer names explicitly identifying rlx02 or fitting.
        preferred = [
            p for p in unique
            if any(token in p.name.lower() for token in ["fit_mask", "rlx02", "brain_mask"])
        ]
        if len(preferred) == 1:
            return preferred[0]
        formatted = "\n".join(f"  {p}" for p in unique)
        raise RuntimeError(
            "Multiple plausible fitting masks found in rlx02 metadata:\n"
            f"{formatted}\nMake the authoritative mask explicit."
        )
    return unique[0] if unique else None


def save_nifti(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(data.astype(dtype, copy=False), reference.affine, header)
    nib.save(img, str(path))


def candidate_grid(r2_min: float, r2_max: float, n: int) -> np.ndarray:
    if r2_min <= 0:
        return np.linspace(r2_min, r2_max, n, dtype=np.float64)
    return np.geomspace(r2_min, r2_max, n, dtype=np.float64)


def evaluate_grid(
    signal: np.ndarray,
    tes: np.ndarray,
    r2_grid: np.ndarray,
    s0_min: float,
    s0_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return best R2, S0 and SSE for signal shape (n_voxels, n_echoes).

    Memory is controlled by looping over R2 candidates while vectorizing across
    voxels. This avoids a candidate x voxel x echo tensor.
    """
    n_vox = signal.shape[0]
    best_sse = np.full(n_vox, np.inf, dtype=np.float64)
    best_r2 = np.full(n_vox, np.nan, dtype=np.float64)
    best_s0 = np.full(n_vox, np.nan, dtype=np.float64)

    for r2 in r2_grid:
        basis = np.exp(-r2 * tes)
        denom = float(np.dot(basis, basis))
        numer = signal @ basis
        s0 = numer / denom
        s0 = np.clip(s0, s0_min, s0_max)
        residual = signal - s0[:, None] * basis[None, :]
        sse = np.einsum("ij,ij->i", residual, residual)
        better = sse < best_sse
        if np.any(better):
            best_sse[better] = sse[better]
            best_r2[better] = r2
            best_s0[better] = s0[better]

    return best_r2, best_s0, best_sse


def fit_chunk(
    signal: np.ndarray,
    tes: np.ndarray,
    r2_min: float,
    r2_max: float,
    s0_min: float,
    s0_max_factor: float,
    coarse_points: int,
    refine_points: int,
    boundary_rtol: float,
    nrmse_eps: float,
) -> Dict[str, np.ndarray]:
    n_vox, n_echoes = signal.shape
    s0_max = np.maximum(np.nanmax(signal, axis=1) * s0_max_factor, s0_min + 1.0)

    coarse = candidate_grid(r2_min, r2_max, coarse_points)
    r2, s0, sse = evaluate_grid(signal, tes, coarse, s0_min, s0_max)

    # Local refinement around each voxel's coarse optimum. To stay vectorized,
    # group voxels by coarse-grid index and fit a denser local grid per group.
    coarse_idx = np.searchsorted(coarse, r2)
    coarse_idx = np.clip(coarse_idx, 0, coarse.size - 1)

    for idx in np.unique(coarse_idx):
        group = np.where(coarse_idx == idx)[0]
        lo = coarse[max(0, idx - 1)]
        hi = coarse[min(coarse.size - 1, idx + 1)]
        if hi <= lo:
            continue
        local = np.linspace(lo, hi, refine_points, dtype=np.float64)
        lr2, ls0, lsse = evaluate_grid(
            signal[group], tes, local, s0_min, s0_max[group]
        )
        r2[group] = lr2
        s0[group] = ls0
        sse[group] = lsse

    pred = s0[:, None] * np.exp(-r2[:, None] * tes[None, :])
    residual = signal - pred
    rmse = np.sqrt(np.mean(residual ** 2, axis=1))
    mean_signal = np.mean(signal, axis=1)
    nrmse = rmse / np.maximum(mean_signal, nrmse_eps)

    centered = signal - mean_signal[:, None]
    sst = np.einsum("ij,ij->i", centered, centered)
    r2coef = np.full(n_vox, np.nan, dtype=np.float64)
    nonconstant = sst > nrmse_eps
    r2coef[nonconstant] = 1.0 - sse[nonconstant] / sst[nonconstant]

    t2_ms = 1000.0 / r2

    r2_tol = boundary_rtol * max(abs(r2_max - r2_min), 1.0)
    s0_tol = boundary_rtol * np.maximum(s0_max - s0_min, 1.0)
    boundary = (
        (np.abs(r2 - r2_min) <= r2_tol)
        | (np.abs(r2 - r2_max) <= r2_tol)
        | (np.abs(s0 - s0_min) <= s0_tol)
        | (np.abs(s0 - s0_max) <= s0_tol)
    )

    finite = (
        np.isfinite(r2)
        & np.isfinite(t2_ms)
        & np.isfinite(s0)
        & np.isfinite(rmse)
        & np.isfinite(nrmse)
        & np.isfinite(r2coef)
    )
    failed = ~finite
    valid = finite & ~boundary & (r2 > 0) & (s0 > 0)

    return {
        "R2_s-1": r2,
        "T2_ms": t2_ms,
        "S0": s0,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2coef": r2coef,
        "fit_valid_mask": valid.astype(np.uint8),
        "boundary_hit_mask": boundary.astype(np.uint8),
        "failed_fit_mask": failed.astype(np.uint8),
    }


def fit_volume(
    data4d: np.ndarray,
    tes: np.ndarray,
    domain_mask: np.ndarray,
    min_signal: float,
    min_echoes: int,
    chunk_voxels: int,
    r2_min: float,
    r2_max: float,
    s0_min: float,
    s0_max_factor: float,
    coarse_points: int,
    refine_points: int,
    boundary_rtol: float,
    nrmse_eps: float,
) -> Dict[str, np.ndarray]:
    spatial_shape = data4d.shape[:3]
    n_echoes = data4d.shape[3]
    flat = data4d.reshape(-1, n_echoes).astype(np.float64, copy=False)
    mask_flat = domain_mask.reshape(-1).astype(bool)

    finite_count = np.sum(np.isfinite(flat), axis=1)
    positive_count = np.sum(flat > min_signal, axis=1)
    candidate = mask_flat & (finite_count >= min_echoes) & (positive_count >= min_echoes)

    # Current fit requires all selected echoes finite. Failed candidates are
    # retained in the failed-fit map rather than silently imputed.
    fully_finite = np.all(np.isfinite(flat), axis=1)
    fit_idx = np.where(candidate & fully_finite)[0]
    prefit_failed_idx = np.where(candidate & ~fully_finite)[0]

    outputs: Dict[str, np.ndarray] = {}
    float_names = ["R2_s-1", "T2_ms", "S0", "RMSE", "NRMSE", "R2coef"]
    mask_names = ["fit_valid_mask", "boundary_hit_mask", "failed_fit_mask"]

    for name in float_names:
        outputs[name] = np.full(flat.shape[0], np.nan, dtype=np.float32)
    for name in mask_names:
        outputs[name] = np.zeros(flat.shape[0], dtype=np.uint8)

    outputs["failed_fit_mask"][prefit_failed_idx] = 1

    for start in range(0, fit_idx.size, chunk_voxels):
        idx = fit_idx[start:start + chunk_voxels]
        result = fit_chunk(
            signal=flat[idx],
            tes=tes,
            r2_min=r2_min,
            r2_max=r2_max,
            s0_min=s0_min,
            s0_max_factor=s0_max_factor,
            coarse_points=coarse_points,
            refine_points=refine_points,
            boundary_rtol=boundary_rtol,
            nrmse_eps=nrmse_eps,
        )
        for name, values in result.items():
            outputs[name][idx] = values

    # Candidate voxels that reached the optimizer but returned no valid fit.
    invalid_after_fit = candidate & (outputs["fit_valid_mask"] == 0) & (
        outputs["boundary_hit_mask"] == 0
    )
    outputs["failed_fit_mask"][invalid_after_fit] = 1

    for name in outputs:
        outputs[name] = outputs[name].reshape(spatial_shape)

    outputs["analysis_domain_mask"] = candidate.reshape(spatial_shape).astype(np.uint8)
    return outputs


def robust_stats(values: np.ndarray) -> Dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0, "mean": None, "sd": None, "median": None,
            "p05": None, "p95": None,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_model(
    case: str,
    model: str,
    outputs: Dict[str, np.ndarray],
    n_echoes: int,
) -> Dict[str, Any]:
    domain = outputs["analysis_domain_mask"] > 0
    valid = outputs["fit_valid_mask"] > 0
    boundary = outputs["boundary_hit_mask"] > 0
    failed = outputs["failed_fit_mask"] > 0

    row: Dict[str, Any] = {
        "case": case,
        "model": model,
        "n_echoes": n_echoes,
        "domain_voxels": int(np.count_nonzero(domain)),
        "valid_voxels": int(np.count_nonzero(valid)),
        "boundary_hit_voxels": int(np.count_nonzero(boundary)),
        "failed_fit_voxels": int(np.count_nonzero(failed)),
        "valid_fraction": (
            float(np.count_nonzero(valid) / np.count_nonzero(domain))
            if np.count_nonzero(domain) else None
        ),
        "boundary_fraction": (
            float(np.count_nonzero(boundary) / np.count_nonzero(domain))
            if np.count_nonzero(domain) else None
        ),
        "failed_fraction": (
            float(np.count_nonzero(failed) / np.count_nonzero(domain))
            if np.count_nonzero(domain) else None
        ),
    }

    for metric in ["R2_s-1", "T2_ms", "S0", "RMSE", "NRMSE", "R2coef"]:
        stats = robust_stats(outputs[metric][valid])
        for key, value in stats.items():
            row[f"{metric}_{key}"] = value
    return row


def make_qc_png(
    data4d: np.ndarray,
    outputs_by_model: Dict[str, Dict[str, np.ndarray]],
    path: Path,
) -> None:
    ref = np.nanmean(data4d, axis=3)
    z_scores = np.sum(np.isfinite(ref) & (ref > 0), axis=(0, 1))
    z = int(np.argmax(z_scores))

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    panels = [
        ("Mean signal", ref[:, :, z]),
        ("all12 T2 [ms]", outputs_by_model[MODEL_ALL]["T2_ms"][:, :, z]),
        ("drop1 T2 [ms]", outputs_by_model[MODEL_DROP1]["T2_ms"][:, :, z]),
        ("T2 difference", (
            outputs_by_model[MODEL_DROP1]["T2_ms"]
            - outputs_by_model[MODEL_ALL]["T2_ms"]
        )[:, :, z]),
        ("all12 R2 [s-1]", outputs_by_model[MODEL_ALL]["R2_s-1"][:, :, z]),
        ("drop1 R2 [s-1]", outputs_by_model[MODEL_DROP1]["R2_s-1"][:, :, z]),
        ("all12 NRMSE", outputs_by_model[MODEL_ALL]["NRMSE"][:, :, z]),
        ("drop1 NRMSE", outputs_by_model[MODEL_DROP1]["NRMSE"][:, :, z]),
        ("all12 R²", outputs_by_model[MODEL_ALL]["R2coef"][:, :, z]),
        ("drop1 R²", outputs_by_model[MODEL_DROP1]["R2coef"][:, :, z]),
        ("Boundary hits", (
            outputs_by_model[MODEL_ALL]["boundary_hit_mask"]
            + 2 * outputs_by_model[MODEL_DROP1]["boundary_hit_mask"]
        )[:, :, z]),
        ("Failed fits", (
            outputs_by_model[MODEL_ALL]["failed_fit_mask"]
            + 2 * outputs_by_model[MODEL_DROP1]["failed_fit_mask"]
        )[:, :, z]),
    ]

    for ax, (title, panel) in zip(axes.flat, panels):
        ax.imshow(np.rot90(panel), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"rlx03 native-space mono-exponential QC, z={z}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def model_output_paths(out_dir: Path, case: str, model: str) -> Dict[str, Path]:
    stem = f"rlx03_{case}_{model}"
    return {
        "R2_s-1": out_dir / model / f"{stem}_R2_s-1.nii.gz",
        "T2_ms": out_dir / model / f"{stem}_T2_ms.nii.gz",
        "S0": out_dir / model / f"{stem}_S0.nii.gz",
        "RMSE": out_dir / model / f"{stem}_RMSE.nii.gz",
        "NRMSE": out_dir / model / f"{stem}_NRMSE.nii.gz",
        "R2coef": out_dir / model / f"{stem}_R2coef.nii.gz",
        "analysis_domain_mask": out_dir / model / f"{stem}_analysis_domain_mask.nii.gz",
        "fit_valid_mask": out_dir / model / f"{stem}_fit_valid_mask.nii.gz",
        "boundary_hit_mask": out_dir / model / f"{stem}_boundary_hit_mask.nii.gz",
        "failed_fit_mask": out_dir / model / f"{stem}_failed_fit_mask.nii.gz",
    }


def process_case(
    case: str,
    paths: APFPaths,
    overwrite: bool,
    dry_run: bool,
    r2_min: float,
    r2_max: float,
    s0_min: float,
    s0_max_factor: float,
    min_signal: float,
    min_echoes: int,
    chunk_voxels: int,
    coarse_points: int,
    refine_points: int,
    boundary_rtol: float,
) -> Dict[str, Any]:
    start = now_iso()
    print("=" * 72)
    print(f"{TAG}: {case}")
    print("=" * 72)

    metadata_path = find_rlx02_metadata(paths, case)
    rlx02_meta = load_metadata(metadata_path)

    rlx02_inputs = resolve_rlx02_inputs(
        meta=rlx02_meta,
        metadata_path=metadata_path,
        project_root=paths.root,
    )

    stack_path = rlx02_inputs["stack"]
    mask_path = rlx02_inputs["fit_mask"]

    stack_img = nib.load(str(stack_path))
    data4d = np.asarray(
        stack_img.dataobj,
        dtype=np.float32,
    )

    if data4d.ndim != 4:
        raise ValueError(
            f"Expected 4D rlx02 stack, got shape {data4d.shape}"
        )

    n_echoes = data4d.shape[3]

    tes_all = normalize_echo_times(
        rlx02_inputs["echo_times_s"],
        n_echoes,
    )

    mask_img = nib.load(str(mask_path))
    domain_mask = np.asarray(mask_img.dataobj) > 0
    mask_source = str(mask_path)

    if domain_mask.shape != data4d.shape[:3]:
        raise ValueError(
            "rlx02 fit-mask geometry does not match the echo stack:\n"
            f"  stack: {data4d.shape[:3]}\n"
            f"  mask:  {domain_mask.shape}"
        )

    out_dir = rlx03_dir(paths, case)
    expected_metadata = out_dir / f"rlx03_{case}_metadata.json"
    if expected_metadata.exists() and not overwrite:
        raise FileExistsError(
            f"rlx03 outputs already exist for {case}: {expected_metadata}\n"
            "Use --overwrite to replace them."
        )

    model_specs = {
        MODEL_ALL: (data4d, tes_all),
        MODEL_DROP1: (data4d[..., 1:], tes_all[1:]),
    }

    outputs_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    model_rows: List[Dict[str, Any]] = []
    output_metadata: Dict[str, Dict[str, str]] = {}

    for model, (model_data, model_tes) in model_specs.items():
        print(
            f"  Fitting {model}: {model_data.shape[3]} echoes, "
            f"TE {model_tes[0] * 1000:.3f}-{model_tes[-1] * 1000:.3f} ms"
        )
        if dry_run:
            continue

        outputs = fit_volume(
            data4d=model_data,
            tes=model_tes,
            domain_mask=domain_mask,
            min_signal=min_signal,
            min_echoes=min(min_echoes, model_data.shape[3]),
            chunk_voxels=chunk_voxels,
            r2_min=r2_min,
            r2_max=r2_max,
            s0_min=s0_min,
            s0_max_factor=s0_max_factor,
            coarse_points=coarse_points,
            refine_points=refine_points,
            boundary_rtol=boundary_rtol,
            nrmse_eps=DEFAULT_NRMSE_EPS,
        )
        outputs_by_model[model] = outputs

        paths_for_model = model_output_paths(out_dir, case, model)
        for name, output_path in paths_for_model.items():
            dtype = np.uint8 if name.endswith("_mask") else np.float32
            save_nifti(outputs[name], stack_img, output_path, dtype)

        row = summarize_model(
            case=case,
            model=model,
            outputs=outputs,
            n_echoes=model_data.shape[3],
        )
        model_rows.append(row)
        write_csv(
            out_dir / model / f"rlx03_{case}_{model}_summary.csv",
            [row],
        )
        output_metadata[model] = {
            key: str(value) for key, value in paths_for_model.items()
        }

    if dry_run:
        return {
            "case": case,
            "success": True,
            "dry_run": True,
            "rlx02_metadata": str(metadata_path),
            "stack": str(stack_path),
            "echo_times_ms": (tes_all * 1000.0).tolist(),
            "mask": mask_source,
        }

    qc_png = out_dir / f"rlx03_{case}_qc.png"
    make_qc_png(data4d, outputs_by_model, qc_png)
    write_csv(out_dir / f"rlx03_{case}_summary.csv", model_rows)

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
            "rlx02_metadata": str(metadata_path),
            "authoritative_4d_stack": str(stack_path),
            "authoritative_mask": mask_source,
            "native_space": True,
            "shape": list(data4d.shape),
            "echo_times_s": tes_all.tolist(),
            "echo_times_ms": (tes_all * 1000.0).tolist(),
        },
        "model": {
            "equation": "S(TE) = S0 * exp(-R2 * TE)",
            "fit_method": (
                "bounded variable-projection least squares with coarse-to-fine "
                "one-dimensional R2 search"
            ),
            "variants": {
                MODEL_ALL: "all echoes",
                MODEL_DROP1: "first echo excluded",
            },
            "bounds": {
                "R2_s-1": [r2_min, r2_max],
                "S0_min": s0_min,
                "S0_max_rule": f"{s0_max_factor} * voxel maximum signal",
            },
            "minimum_signal": min_signal,
            "minimum_positive_echoes": min_echoes,
            "coarse_grid_points": coarse_points,
            "refine_grid_points": refine_points,
            "boundary_relative_tolerance": boundary_rtol,
        },
        "qc_definitions": {
            "RMSE": "sqrt(mean((observed-predicted)^2)) in native signal units",
            "NRMSE": "RMSE / mean observed signal",
            "R2coef": "1 - SSE/SST; NaN for constant signal curves",
            "fit_valid_mask": (
                "finite positive fit within the search bounds and not classified "
                "as a boundary hit"
            ),
            "boundary_hit_mask": (
                "R2 or S0 lies within the configured relative tolerance of a bound"
            ),
            "failed_fit_mask": (
                "candidate fitting voxel with non-finite input or non-finite fit "
                "result, excluding boundary hits"
            ),
        },
        "outputs": output_metadata,
        "qc_png": str(qc_png),
        "summary_csv": str(out_dir / f"rlx03_{case}_summary.csv"),
        "model_summaries": model_rows,
    }
    write_json(expected_metadata, metadata)

    print(f"  DONE: {case}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APF rlx03: native-space mono-exponential T2 fitting."
    )
    add_case_selection_args(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--r2-min", type=float, default=DEFAULT_R2_MIN)
    parser.add_argument("--r2-max", type=float, default=DEFAULT_R2_MAX)
    parser.add_argument("--s0-min", type=float, default=DEFAULT_S0_MIN)
    parser.add_argument(
        "--s0-max-factor", type=float, default=DEFAULT_S0_MAX_FACTOR
    )
    parser.add_argument("--min-signal", type=float, default=DEFAULT_MIN_SIGNAL)
    parser.add_argument("--min-echoes", type=int, default=DEFAULT_MIN_ECHOES)
    parser.add_argument(
        "--chunk-voxels", type=int, default=DEFAULT_CHUNK_VOXELS
    )
    parser.add_argument(
        "--coarse-points", type=int, default=DEFAULT_COARSE_POINTS
    )
    parser.add_argument(
        "--refine-points", type=int, default=DEFAULT_REFINE_POINTS
    )
    parser.add_argument(
        "--boundary-rtol", type=float, default=DEFAULT_BOUNDARY_RTOL
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.r2_min < 0 or args.r2_max <= args.r2_min:
        raise SystemExit("Require 0 <= --r2-min < --r2-max.")
    if args.s0_min < 0:
        raise SystemExit("--s0-min must be non-negative.")
    if args.s0_max_factor <= 1:
        raise SystemExit("--s0-max-factor must be > 1.")
    if args.min_echoes < 3:
        raise SystemExit("--min-echoes must be >= 3.")
    if args.chunk_voxels < 1:
        raise SystemExit("--chunk-voxels must be positive.")
    if args.coarse_points < 16:
        raise SystemExit("--coarse-points must be >= 16.")
    if args.refine_points < 5:
        raise SystemExit("--refine-points must be >= 5.")
    if not (0 < args.boundary_rtol < 0.1):
        raise SystemExit("--boundary-rtol must be between 0 and 0.1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = get_paths()

    cases = resolve_cases(
        args,
        paths.ins_registry,
    )
    print_selected_cases(cases)

    cohort_rows: List[Dict[str, Any]] = []
    failures = 0

    for case in cases:
        try:
            metadata = process_case(
                case=case,
                paths=paths,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                r2_min=args.r2_min,
                r2_max=args.r2_max,
                s0_min=args.s0_min,
                s0_max_factor=args.s0_max_factor,
                min_signal=args.min_signal,
                min_echoes=args.min_echoes,
                chunk_voxels=args.chunk_voxels,
                coarse_points=args.coarse_points,
                refine_points=args.refine_points,
                boundary_rtol=args.boundary_rtol,
            )
            if args.dry_run:
                cohort_rows.append({
                    "case": case,
                    "success": True,
                    "dry_run": True,
                    "rlx02_metadata": metadata["rlx02_metadata"],
                    "stack": metadata["stack"],
                })
            else:
                for row in metadata["model_summaries"]:
                    cohort_rows.append({"success": True, **row})
        except Exception as exc:
            failures += 1
            print(f"ERROR in {case}: {exc}", file=sys.stderr)
            cohort_rows.append({
                "case": case,
                "success": False,
                "error": str(exc),
            })
            if len(cases) == 1:
                raise

    if not args.dry_run:
        out = summary_dir(paths)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "rlx03_cohort_summary.csv", cohort_rows)
        write_json(
            out / "rlx03_cohort_metadata.json",
            {
                "script": SCRIPT,
                "tag": TAG,
                "version": VERSION,
                "created": now_iso(),
                "command": " ".join(sys.argv),
                "n_cases_requested": len(cases),
                "n_failures": failures,
                "rows": cohort_rows,
            },
        )

    if failures:
        raise SystemExit(f"rlx03 completed with {failures} failed case(s).")


if __name__ == "__main__":
    main()
