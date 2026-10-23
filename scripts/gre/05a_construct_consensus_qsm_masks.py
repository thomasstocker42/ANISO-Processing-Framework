#!/usr/bin/env python3
"""
APF GRE05a v0.2.0: consensus QSM mask construction for background-field removal.

Purpose
-------
Construct the QSM masks recommended by the ISMRM QSM consensus before V-SHARP:

  Mask 1: non-eroded whole-brain GRE mask
  Mask 2: reliable multi-echo phase/field values based on relative residual
  Mask 1 x Mask 2: raw combined mask
  Mask 3: hole-filled mask used for background-field removal

The script deliberately preserves:
  - all voxels rejected by Mask 2
  - internal holes restored by hole filling

These are retained for GRE06 dipole inversion and QC.

Important upstream consistency check
------------------------------------
The accepted GRE03/GRE04 APF workflow historically used an eroded GRE mask. If
GRE04b total-field support does not cover the non-eroded Mask 1, this script
ABORTS by default rather than silently turning Mask 1 back into an eroded mask.

In that situation, GRE03/GRE04 must be rerun for QSM with support covering the
non-eroded GRE02b mask, or the user must explicitly override the check for
diagnostic purposes with --allow-incomplete-field-support.

GRE05a implements consensus mask construction and topology QC only. V-SHARP is implemented separately in GRE05b.

Relative residual
-----------------
Following the SEPIA/consensus example, the relative residual is

    sum_e |S_hat(TE_e) - S(TE_e)|^2
    --------------------------------
             sum_e |S(TE_e)|^2

where the measured complex signal is phase-referenced to the first echo and

    S_hat(TE_e) = S0 * exp(-R2star * TE_e) *
                  exp(i * [omega*(TE_e-TE1) + beta_OE*(parity_e-parity_1)])

omega = 2*pi*frequency_hz.

The parity term is inherited from the finalized parity-aware GRE04b model:

    phase = beta0 + omega*TE + beta_OE*parity

with parity = +1 for odd-numbered echoes and -1 for even-numbered echoes.
Including beta_OE here makes the GRE05a relative-residual model internally
consistent with the accepted GRE04b total-field estimator.

R2star and S0 are estimated from the magnitude echo train using the closed-form
log-linear mono-exponential fit:

    log(M) = log(S0) - R2star * TE

Only finite positive magnitude samples are used.

Default quality threshold
-------------------------
relative residual <= 0.30

This corresponds to the threshold used in the QSM consensus example pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


SCRIPT_VERSION = "0.2.0"
MODULE_NAME = "gre05a"
METHOD_NAME = "consensus_qsm_mask_construction_parity_aware"
OUTPUT_SUBDIR = Path("GRE") / "05a_bfr_masks"

DEFAULT_RELATIVE_RESIDUAL_THRESHOLD = 0.30
DEFAULT_MIN_FIELD_SUPPORT_FRACTION = 0.995


@dataclass(frozen=True)
class Inputs:
    case_id: str
    magnitude_4d: Path
    unwrapped_phase_4d: Path
    echo_times_file: Path
    mask1_brain: Path
    total_field_hz: Path
    gre04b_valid_fit_mask: Path
    gre04b_beta_oe_rad: Path


@dataclass
class CaseResult:
    case_id: str
    status: str
    message: str
    runtime_s: float | None = None
    n_echoes: int | None = None
    mask1_voxels: int | None = None
    field_supported_mask1_voxels: int | None = None
    field_support_fraction_mask1: float | None = None
    mask2_voxels: int | None = None
    mask12_voxels: int | None = None
    mask3_voxels: int | None = None
    rejected_voxels: int | None = None
    internal_hole_voxels: int | None = None
    relative_residual_median: float | None = None
    relative_residual_p95: float | None = None
    output_dir: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        root = cli_root.expanduser().resolve()
    elif os.environ.get("APF_ROOT"):
        root = Path(os.environ["APF_ROOT"]).expanduser().resolve()
    else:
        root = Path.cwd().resolve()

    if not root.exists():
        raise FileNotFoundError(f"APF root does not exist: {root}")
    return root


def list_cases(root: Path) -> list[str]:
    derivatives = root / "derivatives"
    if not derivatives.exists():
        return []
    return sorted(
        p.name
        for p in derivatives.iterdir()
        if p.is_dir() and p.name.startswith("ANISO_") and p.name.endswith("_InS")
    )


def unique_match(paths: Iterable[Path], description: str) -> Path:
    unique = sorted({p.resolve() for p in paths if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"No {description} found")
    if len(unique) > 1:
        joined = "\n  ".join(str(p) for p in unique)
        raise RuntimeError(
            f"Ambiguous {description}; found {len(unique)} files:\n  {joined}"
        )
    return unique[0]


def glob_many(roots: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            out.extend(root.rglob(pattern))
    return out


def case_search_roots(root: Path, case_id: str) -> list[Path]:
    candidates = [
        root / "derivatives" / case_id,
        root / "processing" / case_id,
        root / "nifti" / case_id,
    ]
    return [p for p in candidates if p.exists()]


def discover_magnitude_4d(roots: Sequence[Path]) -> Path:
    preferred = [
        p for p in glob_many(
            roots,
            [
                "GRE/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz",
                "GRE/phase_preparation_gre02/gre02_*_magnitude_4d.nii",
                "gre/phase_preparation_gre02/gre02_*_magnitude_4d.nii.gz",
                "gre/phase_preparation_gre02/gre02_*_magnitude_4d.nii",
            ],
        )
        if p.is_file()
    ]
    if preferred:
        return unique_match(preferred, "canonical GRE02 4D magnitude")

    fallback = [
        p for p in glob_many(roots, ["*magnitude_4d*.nii.gz", "*magnitude_4d*.nii"])
        if "relaxometry" not in str(p).lower()
        and "rlx" not in str(p).lower()
    ]
    return unique_match(fallback, "GRE 4D magnitude")


def discover_unwrapped_phase_4d(roots: Sequence[Path], case_id: str) -> Path:
    exact = [
        p for p in glob_many(
            roots,
            [
                f"GRE/03_romeo_unwrap/{case_id}_GRE_phase_unwrapped_romeo_rad_4d.nii.gz",
                f"GRE/03_romeo_unwrap/{case_id}_GRE_phase_unwrapped_romeo_rad_4d.nii",
                f"gre/03_romeo_unwrap/{case_id}_GRE_phase_unwrapped_romeo_rad_4d.nii.gz",
                f"gre/03_romeo_unwrap/{case_id}_GRE_phase_unwrapped_romeo_rad_4d.nii",
            ],
        )
        if p.is_file()
    ]
    if exact:
        return unique_match(exact, "GRE03 ROMEO unwrapped phase")

    fallback = [
        p for p in glob_many(
            roots,
            ["*phase_unwrapped_romeo_rad_4d*.nii.gz", "*phase_unwrapped_romeo_rad_4d*.nii"],
        )
        if p.is_file()
    ]
    return unique_match(fallback, "GRE03 ROMEO unwrapped phase")


def discover_echo_times(roots: Sequence[Path]) -> Path:
    candidates = [
        p for p in glob_many(
            roots,
            [
                "*echo_times_s.txt",
                "*echo_times_ms.txt",
                "*echo_times*.tsv",
                "*echo_times*.csv",
                "*echo_times*.json",
            ],
        )
        if p.is_file()
        and "relaxometry" not in str(p).lower()
        and "rlx" not in p.name.lower()
    ]

    gre02_s = [
        p for p in candidates
        if "phase_preparation_gre02" in str(p).lower()
        and "echo_times_s" in p.name.lower()
    ]
    if len(gre02_s) == 1:
        return gre02_s[0].resolve()

    gre02_ms = [
        p for p in candidates
        if "phase_preparation_gre02" in str(p).lower()
        and "echo_times_ms" in p.name.lower()
    ]
    if len(gre02_ms) == 1:
        return gre02_ms[0].resolve()

    return unique_match(candidates, "GRE02 echo-time file")


def discover_non_eroded_mask1(roots: Sequence[Path], case_id: str) -> Path:
    """
    Strictly select the non-eroded GRE02b HD-BET mask.

    Any filename containing ero/erod is rejected.
    """
    candidates = [
        p for p in glob_many(
            roots,
            [
                f"*{case_id}*GRE02b*brain_mask*.nii.gz",
                f"*{case_id}*GRE02b*brain_mask*.nii",
                "*GRE02b*brain_mask*.nii.gz",
                "*GRE02b*brain_mask*.nii",
                "*brain_mask*.nii.gz",
                "*brain_mask*.nii",
            ],
        )
        if p.is_file()
        and "gre" in str(p).lower()
        and "relaxometry" not in str(p).lower()
        and "rlx" not in str(p).lower()
        and "ero" not in p.name.lower()
        and "erod" not in p.name.lower()
    ]

    # Prefer GRE02b-labelled masks.
    gre02b = [p for p in candidates if "gre02b" in p.name.lower() or "02b_brain_mask" in str(p).lower()]
    if len(gre02b) == 1:
        return gre02b[0].resolve()
    if len(gre02b) > 1:
        return unique_match(gre02b, "non-eroded GRE02b brain mask")

    return unique_match(candidates, "non-eroded GRE brain mask (Mask 1)")


def discover_gre04b(root: Path, case_id: str) -> tuple[Path, Path, Path]:
    """
    Discover the finalized parity-aware GRE04b production outputs.

    GRE05a requires:
      - total field [Hz]
      - valid-fit mask
      - odd/even nuisance coefficient beta_OE [rad]

    The beta_OE map is required so that the complex-signal residual model
    matches the accepted GRE04b phase model.
    """
    wls = root / "derivatives" / case_id / "GRE" / "04b_field_wls"

    field = wls / f"{case_id}_GRE04b_total_field_hz.nii.gz"
    valid = wls / f"{case_id}_GRE04b_valid_fit_mask.nii.gz"

    beta_candidates = [
        wls / f"{case_id}_GRE04b_beta_odd_even_rad.nii.gz",
        wls / f"{case_id}_GRE04b_beta_oe_rad.nii.gz",
        wls / f"{case_id}_GRE04b_odd_even_beta_rad.nii.gz",
    ]

    if not field.is_file():
        roots = case_search_roots(root, case_id)
        field = unique_match(
            glob_many(
                roots,
                ["*GRE04b_total_field_hz.nii.gz", "*GRE04b_total_field_hz.nii"],
            ),
            "GRE04b total-field Hz map",
        )

    if not valid.is_file():
        roots = case_search_roots(root, case_id)
        valid = unique_match(
            glob_many(
                roots,
                ["*GRE04b_valid_fit_mask.nii.gz", "*GRE04b_valid_fit_mask.nii"],
            ),
            "GRE04b valid-fit mask",
        )

    beta_existing = [p for p in beta_candidates if p.is_file()]
    if len(beta_existing) == 1:
        beta_oe = beta_existing[0]
    elif len(beta_existing) > 1:
        beta_oe = unique_match(
            beta_existing,
            "canonical GRE04b beta_OE map",
        )
    else:
        roots = case_search_roots(root, case_id)
        beta_matches = glob_many(
            roots,
            [
                "*GRE04b*beta*odd*even*rad*.nii.gz",
                "*GRE04b*beta*odd*even*rad*.nii",
                "*GRE04b*beta_oe*rad*.nii.gz",
                "*GRE04b*beta_oe*rad*.nii",
                "*GRE04b*odd_even*beta*rad*.nii.gz",
                "*GRE04b*odd_even*beta*rad*.nii",
            ],
        )
        beta_oe = unique_match(
            beta_matches,
            "GRE04b odd/even nuisance coefficient beta_OE [rad]",
        )

    return field.resolve(), valid.resolve(), beta_oe.resolve()


def discover_inputs(root: Path, case_id: str) -> Inputs:
    roots = case_search_roots(root, case_id)
    if not roots:
        raise FileNotFoundError(f"No case search roots found for {case_id}")

    field, valid, beta_oe = discover_gre04b(root, case_id)

    return Inputs(
        case_id=case_id,
        magnitude_4d=discover_magnitude_4d(roots),
        unwrapped_phase_4d=discover_unwrapped_phase_4d(roots, case_id),
        echo_times_file=discover_echo_times(roots),
        mask1_brain=discover_non_eroded_mask1(roots, case_id),
        total_field_hz=field,
        gre04b_valid_fit_mask=valid,
        gre04b_beta_oe_rad=beta_oe,
    )


def read_echo_times_seconds(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("echo_times_s", "echo_times", "TE", "tes"):
                if key in obj:
                    values = obj[key]
                    break
            else:
                raise ValueError(f"No recognizable echo-time key in {path}")
        else:
            values = obj
        te = np.asarray(values, dtype=float).reshape(-1)
    else:
        tokens = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            text,
        )
        if not tokens:
            raise ValueError(f"No numeric echo times found in {path}")
        te = np.asarray([float(x) for x in tokens], dtype=float)

    name = path.name.lower()
    if "_ms" in name or "milliseconds" in name:
        te = te / 1000.0
    elif "_s" in name or "seconds" in name:
        pass
    else:
        # Guarded inference only for generic files.
        if np.nanmedian(te) > 1.0:
            te = te / 1000.0

    if te.ndim != 1 or te.size < 3:
        raise ValueError(f"Need >=3 echo times, got {te}")
    if not np.all(np.isfinite(te)) or np.any(te <= 0) or np.any(np.diff(te) <= 0):
        raise ValueError(f"Echo times must be finite, positive, strictly increasing: {te}")
    return te


def load_img(path: Path, expected_ndim: int | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if expected_ndim is not None and data.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim}D image, got {data.shape}: {path}")
    return img, data


def validate_geometry(reference: nib.Nifti1Image, other: nib.Nifti1Image, label: str) -> None:
    if tuple(reference.shape[:3]) != tuple(other.shape[:3]):
        raise ValueError(
            f"{label} spatial shape mismatch: {other.shape[:3]} vs {reference.shape[:3]}"
        )
    if not np.allclose(reference.affine, other.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{label} affine mismatch")


def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, path: Path, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    nib.save(out, str(path))


def closed_form_r2star_s0(
    magnitude_4d: np.ndarray,
    te_s: np.ndarray,
    spatial_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelwise log-linear magnitude fit:
        log(M) = log(S0) - R2* TE

    Returns R2* [s^-1], S0 [magnitude units], valid-fit mask.
    """
    shape3 = magnitude_4d.shape[:3]
    n_echoes = magnitude_4d.shape[3]

    r2star = np.full(shape3, np.nan, dtype=np.float32)
    s0 = np.full(shape3, np.nan, dtype=np.float32)
    valid = np.zeros(shape3, dtype=bool)

    idx = np.flatnonzero(spatial_mask.ravel())
    if idx.size == 0:
        return r2star, s0, valid

    mag = magnitude_4d.reshape(-1, n_echoes)[idx].astype(np.float64)
    good = np.isfinite(mag) & (mag > 0)

    # Require all echoes for a deterministic, comparable residual map.
    rows_ok = np.all(good, axis=1)
    if not np.any(rows_ok):
        return r2star, s0, valid

    mag_ok = mag[rows_ok]
    y = np.log(mag_ok)

    x = te_s.astype(np.float64)
    x_mean = float(np.mean(x))
    dx = x - x_mean
    denom = float(np.sum(dx * dx))
    if denom <= 0:
        raise ValueError("Degenerate echo-time vector")

    y_mean = np.mean(y, axis=1)
    slope = np.sum((y - y_mean[:, None]) * dx[None, :], axis=1) / denom
    intercept = y_mean - slope * x_mean

    r2 = -slope
    s0_vals = np.exp(intercept)

    phys_ok = (
        np.isfinite(r2)
        & np.isfinite(s0_vals)
        & (r2 >= 0.0)
        & (s0_vals > 0.0)
    )

    selected_flat = idx[rows_ok][phys_ok]
    r2star.ravel()[selected_flat] = r2[phys_ok].astype(np.float32)
    s0.ravel()[selected_flat] = s0_vals[phys_ok].astype(np.float32)
    valid.ravel()[selected_flat] = True

    return r2star, s0, valid


def compute_relative_residual(
    magnitude_4d: np.ndarray,
    phase_unwrapped_4d: np.ndarray,
    te_s: np.ndarray,
    frequency_hz: np.ndarray,
    beta_oe_rad: np.ndarray,
    spatial_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SEPIA-style relative residual.

    Measured signal is phase-referenced to echo 1:
      S(TE) = M(TE) * exp(i * [phi(TE)-phi(TE1)])

    Simulated signal:
      S_hat(TE_e) = S0 exp(-R2* TE_e) *
                    exp(i * [omega*(TE_e-TE1)
                             + beta_OE*(parity_e-parity_1)])

    This is the first-echo-referenced form of the finalized GRE04b model:
      phase = beta0 + omega*TE + beta_OE*parity

    parity = +1 for echoes 1,3,5,... and -1 for echoes 2,4,6,...
    """
    if magnitude_4d.shape != phase_unwrapped_4d.shape:
        raise ValueError(
            f"Magnitude/phase shape mismatch: "
            f"{magnitude_4d.shape} vs {phase_unwrapped_4d.shape}"
        )
    if magnitude_4d.shape[3] != te_s.size:
        raise ValueError(
            f"Echo count mismatch: image has {magnitude_4d.shape[3]}, "
            f"TE file has {te_s.size}"
        )
    if beta_oe_rad.shape != frequency_hz.shape:
        raise ValueError(
            f"beta_OE/frequency shape mismatch: "
            f"{beta_oe_rad.shape} vs {frequency_hz.shape}"
        )

    r2star, s0, magfit_valid = closed_form_r2star_s0(
        magnitude_4d, te_s, spatial_mask
    )

    residual = np.full(frequency_hz.shape, np.nan, dtype=np.float32)

    valid = (
        spatial_mask
        & magfit_valid
        & np.isfinite(frequency_hz)
        & np.isfinite(beta_oe_rad)
        & np.all(np.isfinite(magnitude_4d), axis=3)
        & np.all(np.isfinite(phase_unwrapped_4d), axis=3)
    )

    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return residual, r2star, s0, valid

    n_echoes = te_s.size
    mag = magnitude_4d.reshape(-1, n_echoes)[idx].astype(np.float64)
    phi = phase_unwrapped_4d.reshape(-1, n_echoes)[idx].astype(np.float64)
    freq = frequency_hz.ravel()[idx].astype(np.float64)
    beta_oe = beta_oe_rad.ravel()[idx].astype(np.float64)
    r2 = r2star.ravel()[idx].astype(np.float64)
    s0v = s0.ravel()[idx].astype(np.float64)

    phase_ref = phi - phi[:, [0]]
    measured = mag * np.exp(1j * phase_ref)

    omega = 2.0 * np.pi * freq
    dt = te_s - te_s[0]

    parity = np.where(
        np.arange(n_echoes) % 2 == 0,
        1.0,
        -1.0,
    ).astype(np.float64)
    parity_ref = parity - parity[0]

    predicted_phase = (
        omega[:, None] * dt[None, :]
        + beta_oe[:, None] * parity_ref[None, :]
    )

    predicted = (
        s0v[:, None]
        * np.exp(-r2[:, None] * te_s[None, :])
        * np.exp(1j * predicted_phase)
    )

    numerator = np.sum(np.abs(predicted - measured) ** 2, axis=1)
    denominator = np.sum(np.abs(measured) ** 2, axis=1)

    rr = np.full(idx.size, np.nan, dtype=np.float64)
    denom_ok = np.isfinite(denominator) & (denominator > 0)
    rr[denom_ok] = numerator[denom_ok] / denominator[denom_ok]

    residual.ravel()[idx] = rr.astype(np.float32)
    return residual, r2star, s0, valid


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "p05": None, "p95": None, "p99": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }



def analyze_rejected_components(
    mask1: np.ndarray,
    mask2_rejected: np.ndarray,
    voxel_sizes_mm: Sequence[float],
) -> tuple[dict[str, np.ndarray], list[dict[str, float | int | bool]]]:
    """
    Analyze 3-D connected components of rejected Mask-2 voxels.

    Connectivity deliberately matches scipy.ndimage.binary_fill_holes default
    topology: face-connected (6-connected) voxels in 3-D.

    For each rejected component, report:
      - voxel count and physical volume
      - whether it touches the Mask-1 boundary
      - number/fraction of voxels lying on the Mask-1 boundary
      - maximum and median Euclidean distance from the Mask-1 boundary

    Large maximum depth together with very small boundary contact is useful QC
    for components which are mostly internal but remain connected to the outer
    rejected layer through a narrow path. No automatic bridge threshold is used.
    """
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    eroded = ndimage.binary_erosion(mask1, structure=structure, border_value=0)
    mask1_boundary = mask1 & ~eroded

    voxel_sizes = tuple(float(v) for v in voxel_sizes_mm[:3])
    distance_to_outside_mm = ndimage.distance_transform_edt(
        mask1,
        sampling=voxel_sizes,
    )

    labels, n_components = ndimage.label(mask2_rejected, structure=structure)

    boundary_connected = np.zeros(mask1.shape, dtype=bool)
    enclosed = np.zeros(mask1.shape, dtype=bool)
    rows: list[dict[str, float | int | bool]] = []

    voxel_volume_mm3 = float(np.prod(voxel_sizes))

    for label_id in range(1, n_components + 1):
        comp = labels == label_id
        n_voxels = int(np.count_nonzero(comp))
        if n_voxels == 0:
            continue

        boundary_contact = comp & mask1_boundary
        n_boundary_contact = int(np.count_nonzero(boundary_contact))
        touches_boundary = n_boundary_contact > 0

        if touches_boundary:
            boundary_connected |= comp
        else:
            enclosed |= comp

        depths = distance_to_outside_mm[comp]
        rows.append(
            {
                "component_id": int(label_id),
                "n_voxels": n_voxels,
                "volume_mm3": n_voxels * voxel_volume_mm3,
                "touches_mask1_boundary": bool(touches_boundary),
                "boundary_contact_voxels": n_boundary_contact,
                "boundary_contact_fraction": (
                    n_boundary_contact / n_voxels if n_voxels else 0.0
                ),
                "median_depth_mm": float(np.median(depths)),
                "p95_depth_mm": float(np.percentile(depths, 95)),
                "max_depth_mm": float(np.max(depths)),
            }
        )

    rows.sort(key=lambda r: int(r["n_voxels"]), reverse=True)

    maps = {
        "mask1_boundary": mask1_boundary,
        "component_labels": labels.astype(np.int32),
        "boundary_connected": boundary_connected,
        "enclosed": enclosed,
    }
    return maps, rows


def write_component_csv(path: Path, rows: Sequence[dict[str, float | int | bool]]) -> None:
    fieldnames = [
        "component_id",
        "n_voxels",
        "volume_mm3",
        "touches_mask1_boundary",
        "boundary_contact_voxels",
        "boundary_contact_fraction",
        "median_depth_mm",
        "p95_depth_mm",
        "max_depth_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(root: Path, case_id: str) -> dict[str, Path]:
    outdir = root / "derivatives" / case_id / OUTPUT_SUBDIR
    p = f"{case_id}_GRE05"
    return {
        "dir": outdir,
        "relative_residual": outdir / f"{p}_relative_residual.nii.gz",
        "r2star_qc": outdir / f"{p}_relative_residual_r2star_s-1.nii.gz",
        "s0_qc": outdir / f"{p}_relative_residual_s0.nii.gz",
        "mask1": outdir / f"{p}_mask1_brain.nii.gz",
        "mask2": outdir / f"{p}_mask2_reliable_phase.nii.gz",
        "mask2_rejected": outdir / f"{p}_mask2_rejected.nii.gz",
        "mask12": outdir / f"{p}_mask12_raw.nii.gz",
        "mask2_internal_holes": outdir / f"{p}_mask2_internal_holes.nii.gz",
        "mask3": outdir / f"{p}_mask3_bfr_holefilled.nii.gz",
        "mask1_boundary": outdir / f"{p}_mask1_boundary.nii.gz",
        "rejected_component_labels": outdir / f"{p}_mask2_rejected_component_labels.nii.gz",
        "rejected_boundary_connected": outdir / f"{p}_mask2_rejected_boundary_connected.nii.gz",
        "rejected_enclosed": outdir / f"{p}_mask2_rejected_enclosed.nii.gz",
        "rejected_components_csv": outdir / f"{p}_mask2_rejected_components.csv",
        "qc": outdir / f"{p}_mask_qc.json",
    }


def process_case(root: Path, case_id: str, args: argparse.Namespace) -> CaseResult:
    started = time.perf_counter()
    paths = output_paths(root, case_id)

    try:
        inputs = discover_inputs(root, case_id)

        if args.dry_run:
            print(f"\n[{case_id}]")
            for key, value in asdict(inputs).items():
                if key != "case_id":
                    print(f"  {key}: {value}")
            return CaseResult(
                case_id=case_id,
                status="DRYRUN",
                message="Input discovery successful",
                output_dir=str(paths["dir"]),
            )

        if paths["qc"].exists() and not args.force:
            return CaseResult(
                case_id=case_id,
                status="SKIPPED",
                message="GRE05a mask outputs already exist; use --force",
                output_dir=str(paths["dir"]),
            )

        mag_img, magnitude = load_img(inputs.magnitude_4d, 4)
        phase_img, phase = load_img(inputs.unwrapped_phase_4d, 4)
        mask_img, mask1_data = load_img(inputs.mask1_brain, 3)
        field_img, field_hz = load_img(inputs.total_field_hz, 3)
        valid_img, gre04b_valid_data = load_img(inputs.gre04b_valid_fit_mask, 3)
        beta_oe_img, beta_oe_rad = load_img(inputs.gre04b_beta_oe_rad, 3)

        validate_geometry(mag_img, phase_img, "GRE03 phase")
        validate_geometry(mag_img, mask_img, "Mask 1")
        validate_geometry(mag_img, field_img, "GRE04b total field")
        validate_geometry(mag_img, valid_img, "GRE04b valid-fit mask")
        validate_geometry(mag_img, beta_oe_img, "GRE04b beta_OE map")

        if magnitude.shape != phase.shape:
            raise ValueError(
                f"Magnitude/phase 4D mismatch: {magnitude.shape} vs {phase.shape}"
            )

        te_s = read_echo_times_seconds(inputs.echo_times_file)
        if te_s.size != magnitude.shape[3]:
            raise ValueError(
                f"TE count {te_s.size} != image echoes {magnitude.shape[3]}"
            )

        mask1 = mask1_data > 0
        gre04b_valid = gre04b_valid_data > 0

        mask1_voxels = int(np.count_nonzero(mask1))
        if mask1_voxels == 0:
            raise ValueError("Mask 1 is empty")

        field_supported = (
            mask1
            & gre04b_valid
            & np.isfinite(field_hz)
            & np.isfinite(beta_oe_rad)
        )
        field_supported_voxels = int(np.count_nonzero(field_supported))
        field_support_fraction = field_supported_voxels / mask1_voxels

        if (
            field_support_fraction < args.min_field_support_fraction
            and not args.allow_incomplete_field_support
        ):
            raise RuntimeError(
                "GRE04b field support does not cover the non-eroded Mask 1: "
                f"{field_support_fraction:.6f} < "
                f"{args.min_field_support_fraction:.6f}. "
                "This is expected if GRE03/GRE04 were run using GRE02b mask_ero1. "
                "Do not silently use the eroded valid-fit mask as Mask 2. "
                "Rerun the QSM field-estimation path with the non-eroded GRE02b mask, "
                "or use --allow-incomplete-field-support only for diagnostic inspection."
            )

        relative_residual, r2star, s0, rr_model_valid = compute_relative_residual(
            magnitude_4d=magnitude,
            phase_unwrapped_4d=phase,
            te_s=te_s,
            frequency_hz=field_hz,
            beta_oe_rad=beta_oe_rad,
            spatial_mask=field_supported,
        )

        # Mask 2 is the quality threshold, restricted to locations where the
        # multi-echo model can genuinely be evaluated.
        mask2 = (
            mask1
            & field_supported
            & rr_model_valid
            & np.isfinite(relative_residual)
            & (relative_residual <= args.relative_residual_threshold)
        )

        mask12 = mask1 & mask2
        mask3 = ndimage.binary_fill_holes(mask12)

        mask2_rejected = mask1 & ~mask2
        mask2_internal_holes = mask3 & ~mask12

        topology_maps, component_rows = analyze_rejected_components(
            mask1=mask1,
            mask2_rejected=mask2_rejected,
            voxel_sizes_mm=mag_img.header.get_zooms()[:3],
        )

        if not np.array_equal(
            topology_maps["enclosed"],
            mask2_internal_holes,
        ):
            mismatch = int(
                np.count_nonzero(
                    topology_maps["enclosed"] ^ mask2_internal_holes
                )
            )
            raise RuntimeError(
                "Topology inconsistency: enclosed rejected components do not "
                f"match binary_fill_holes result ({mismatch} voxel mismatch)"
            )

        rr_inside_available = relative_residual[
            mask1 & field_supported & np.isfinite(relative_residual)
        ]
        rr_summary = finite_summary(rr_inside_available)

        paths["dir"].mkdir(parents=True, exist_ok=True)
        save_nifti(relative_residual, mag_img, paths["relative_residual"], np.float32)
        save_nifti(r2star, mag_img, paths["r2star_qc"], np.float32)
        save_nifti(s0, mag_img, paths["s0_qc"], np.float32)
        save_nifti(mask1, mag_img, paths["mask1"], np.uint8)
        save_nifti(mask2, mag_img, paths["mask2"], np.uint8)
        save_nifti(mask2_rejected, mag_img, paths["mask2_rejected"], np.uint8)
        save_nifti(mask12, mag_img, paths["mask12"], np.uint8)
        save_nifti(mask2_internal_holes, mag_img, paths["mask2_internal_holes"], np.uint8)
        save_nifti(mask3, mag_img, paths["mask3"], np.uint8)
        save_nifti(topology_maps["mask1_boundary"], mag_img, paths["mask1_boundary"], np.uint8)
        save_nifti(topology_maps["component_labels"], mag_img, paths["rejected_component_labels"], np.int32)
        save_nifti(topology_maps["boundary_connected"], mag_img, paths["rejected_boundary_connected"], np.uint8)
        save_nifti(topology_maps["enclosed"], mag_img, paths["rejected_enclosed"], np.uint8)
        write_component_csv(paths["rejected_components_csv"], component_rows)

        runtime_s = time.perf_counter() - started

        qc = {
            "module": MODULE_NAME,
            "script_version": SCRIPT_VERSION,
            "method": METHOD_NAME,
            "created_utc": now_iso(),
            "case_id": case_id,
            "status": "OK",
            "runtime_s": runtime_s,
            "inputs": {
                k: str(v)
                for k, v in asdict(inputs).items()
                if k != "case_id"
            },
            "configuration": {
                "mask1_policy": "non-eroded GRE02b HD-BET brain mask",
                "relative_residual_threshold": args.relative_residual_threshold,
                "min_field_support_fraction": args.min_field_support_fraction,
                "allow_incomplete_field_support": args.allow_incomplete_field_support,
                "mask3_hole_filling": "scipy.ndimage.binary_fill_holes",
                "vsharp_status": "V-SHARP is implemented separately in GRE05b",
            },
            "relative_residual_model": {
                "measured_signal": (
                    "M(TE)*exp(i*(phase(TE)-phase(TE1)))"
                ),
                "predicted_signal": (
                    "S0*exp(-R2star*TE + i*[2*pi*frequency_hz*(TE-TE1) "
                    "+ beta_OE*(parity-parity_echo1)])"
                ),
                "phase_model_source": (
                    "finalized parity-aware GRE04b: "
                    "phase = beta0 + omega*TE + beta_OE*parity"
                ),
                "parity_definition": (
                    "+1 for echoes 1,3,5,...; -1 for echoes 2,4,6,..."
                ),
                "magnitude_fit": (
                    "closed-form log-linear fit log(M)=log(S0)-R2star*TE"
                ),
                "summary_available_mask1_voxels": rr_summary,
            },
            "mask_counts": {
                "mask1_voxels": mask1_voxels,
                "field_supported_mask1_voxels": field_supported_voxels,
                "field_support_fraction_mask1": field_support_fraction,
                "mask2_reliable_voxels": int(np.count_nonzero(mask2)),
                "mask2_rejected_voxels": int(np.count_nonzero(mask2_rejected)),
                "mask12_raw_voxels": int(np.count_nonzero(mask12)),
                "mask3_holefilled_voxels": int(np.count_nonzero(mask3)),
                "internal_hole_voxels_restored": int(np.count_nonzero(mask2_internal_holes)),
            },
            "definitions": {
                "mask1": "non-eroded GRE02b brain mask",
                "mask2": (
                    "Mask1 AND GRE04b field support AND valid relative-residual "
                    f"model AND relative_residual <= {args.relative_residual_threshold}"
                ),
                "mask12_raw": "Mask1 AND Mask2",
                "mask3": "binary_fill_holes(mask12_raw)",
                "mask2_rejected": "Mask1 AND NOT Mask2",
                "mask2_internal_holes": "Mask3 AND NOT mask12_raw",
            },
            "rejected_component_topology": {
                "connectivity": "3D 6-connectivity (face connected)",
                "n_components": len(component_rows),
                "n_boundary_connected_components": sum(
                    bool(r["touches_mask1_boundary"]) for r in component_rows
                ),
                "n_enclosed_components": sum(
                    not bool(r["touches_mask1_boundary"]) for r in component_rows
                ),
                "boundary_connected_voxels": int(
                    np.count_nonzero(topology_maps["boundary_connected"])
                ),
                "enclosed_voxels": int(
                    np.count_nonzero(topology_maps["enclosed"])
                ),
                "largest_components": component_rows[:20],
                "interpretation_note": (
                    "Large max_depth_mm with a small boundary_contact_fraction can "
                    "indicate a component extending deep into Mask 1 while contacting "
                    "the boundary through a relatively narrow connection. Metrics are "
                    "descriptive; no automatic bridge threshold is applied."
                ),
            },
            "outputs": {
                k: str(v)
                for k, v in paths.items()
                if k != "dir"
            },
        }

        paths["qc"].write_text(json.dumps(qc, indent=2), encoding="utf-8")

        return CaseResult(
            case_id=case_id,
            status="OK",
            message="Consensus Mask 1-3 construction completed",
            runtime_s=runtime_s,
            n_echoes=int(te_s.size),
            mask1_voxels=mask1_voxels,
            field_supported_mask1_voxels=field_supported_voxels,
            field_support_fraction_mask1=field_support_fraction,
            mask2_voxels=int(np.count_nonzero(mask2)),
            mask12_voxels=int(np.count_nonzero(mask12)),
            mask3_voxels=int(np.count_nonzero(mask3)),
            rejected_voxels=int(np.count_nonzero(mask2_rejected)),
            internal_hole_voxels=int(np.count_nonzero(mask2_internal_holes)),
            relative_residual_median=rr_summary["median"],
            relative_residual_p95=rr_summary["p95"],
            output_dir=str(paths["dir"]),
        )

    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
            runtime_s=time.perf_counter() - started,
            output_dir=str(paths["dir"]),
        )


def write_summary(root: Path, results: Sequence[CaseResult]) -> None:
    summary_dir = root / "derivatives" / "summary" / "GRE"
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "gre05a_mask_construction_qc.csv"
    json_path = summary_dir / "gre05a_mask_construction_run.json"

    rows = [asdict(r) for r in results]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "module": MODULE_NAME,
                "script_version": SCRIPT_VERSION,
                "method": METHOD_NAME,
                "created_utc": now_iso(),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "APF GRE05a v0.2.0: construct consensus QSM Masks 1-3 "
            "and preserve phase-quality holes before V-SHARP."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="APF repository root; defaults to APF_ROOT or cwd",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID; repeat for multiple cases",
    )
    parser.add_argument(
        "--all-gre-ready",
        action="store_true",
        help="Attempt all discovered ANISO_*_InS derivative cases",
    )
    parser.add_argument(
        "--relative-residual-threshold",
        type=float,
        default=DEFAULT_RELATIVE_RESIDUAL_THRESHOLD,
        help="Maximum relative residual accepted for Mask 2",
    )
    parser.add_argument(
        "--min-field-support-fraction",
        type=float,
        default=DEFAULT_MIN_FIELD_SUPPORT_FRACTION,
        help=(
            "Required fraction of non-eroded Mask 1 covered by GRE04b valid field "
            "support before mask construction proceeds"
        ),
    )
    parser.add_argument(
        "--allow-incomplete-field-support",
        action="store_true",
        help=(
            "Diagnostic override only: allow Mask 2 construction even when GRE04b "
            "does not cover the non-eroded Mask 1"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print input discovery without loading image data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GRE05 mask outputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_root(args.root)

    if not (0.0 < args.relative_residual_threshold):
        raise SystemExit("--relative-residual-threshold must be > 0")
    if not (0.0 < args.min_field_support_fraction <= 1.0):
        raise SystemExit("--min-field-support-fraction must be in (0, 1]")

    cases = list(dict.fromkeys(args.case))
    if args.all_gre_ready:
        cases = list_cases(root)
    if not cases:
        raise SystemExit(
            "Select at least one --case CASE_ID or use --all-gre-ready"
        )

    print(f"APF GRE05a v{SCRIPT_VERSION}")
    print(f"Root: {root}")
    print(f"Method: {METHOD_NAME}")
    print(f"Cases: {len(cases)}")
    print(f"Relative residual threshold: {args.relative_residual_threshold}")
    print(
        "Minimum GRE04b support of non-eroded Mask 1: "
        f"{args.min_field_support_fraction:.3f}"
    )

    results: list[CaseResult] = []
    for case_id in cases:
        result = process_case(root, case_id, args)
        results.append(result)
        print(f"[{result.status}] {case_id}: {result.message}")
        if result.field_support_fraction_mask1 is not None:
            print(
                f"  GRE04b support of Mask 1: "
                f"{100.0 * result.field_support_fraction_mask1:.3f}%"
            )

    if not args.dry_run:
        write_summary(root, results)

    n_failed = sum(r.status == "FAILED" for r in results)
    n_ok = sum(r.status == "OK" for r in results)
    print(f"Completed: OK={n_ok}, FAILED={n_failed}, TOTAL={len(results)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
