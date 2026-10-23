#!/usr/bin/env python3
"""
APF rlx07: EPG sensitivity analysis for T2/R2 estimation.

Purpose
-------
Test whether imperfect refocusing and stimulated-echo pathways can plausibly
produce differences between:

1. the validated 12-echo drop1 mono-exponential R2 estimate; and
2. the high-resolution dual-echo R2app estimate.

This is a SIMULATION / SENSITIVITY analysis.

It is explicitly NOT intended to reproduce Siemens *tse2d2_7 exactly,
because the complete proprietary RF/k-space ordering is not available from
the exported DICOM metadata.

Acquisition approximation
-------------------------

12-echo reference:
    sequence family: *se2d12 / se_mc
    echo spacing:    9.8 ms
    ESP provenance:  measured TE increment
                     (9.8, 19.6, ..., 117.6 ms)
    echoes:          12
    nominal FA:      180 deg
    fit:             echoes 2--12 ("drop1")

Dual-echo TSE:
    sequence family:       *tse2d2_7
    nominal FA:            180 deg
    VariableFlipAngleFlag: N
    DICOM ETL:             14
    RF ETL:                7
    vendor echo spacing:   12.82 ms
    ESP provenance:        Siemens DICOM VendorReportedEchoSpacing
    effective TE:          13 and 103 ms
    EPG approximation:     echo-train positions 1 and 8

For the EPG sensitivity approximation, the dual-echo signal is sampled at
the echo locations nearest the reported effective TEs:

    13 / 12.82  ~= echo 1
    103 / 12.82 ~= echo 8

Thus:
    simulated signal points = EPG echo 1 and echo 8
    R2app calculation uses the actual reported TE labels 13 and 103 ms.

This approximation is intentionally separated from any claim about the
exact proprietary Siemens reconstruction.

EPG model
---------
Single-pool EPG with:
    - ideal 90-degree excitation
    - constant refocusing angle within each simulation
    - CPMG phase convention
    - T1 and T2 relaxation
    - gradient dephasing between RF pulses
    - no diffusion
    - no off-resonance
    - no explicit slice-profile integration
    - no magnetization transfer
    - no k-space point-spread-function modeling

The refocusing-angle sweep is interpreted as an EFFECTIVE flip angle
sensitivity analysis representing B1+/slice-profile imperfections around
the nominal 180-degree prescription.

Primary outputs
---------------
For each (T1, true T2, refocusing angle):

    R2_true
    R2_12echo_drop1
    R2_dual_app

and bias relative to true R2:

    bias_12 = (R2_12 - R2_true) / R2_true
    bias_dual = (R2_dual - R2_true) / R2_true

and method-to-method difference:

    dual_vs_12_relative =
        (R2_dual - R2_12) / R2_12

The empirical cohort target from rlx06b is:

    median relative delta R2 ~= -17.66 %

This is overlaid on the simulation figures as an empirical reference,
not used to fit the simulation.

Outputs
-------
derivatives/summary/relaxometry/rlx07/

    rlx07_simulation_grid.csv
    rlx07_angle_summary.csv
    rlx07_epg_echo_curves.png
    rlx07_dual_vs_reference_bias.png
    rlx07_bias_heatmap.png
    rlx07_metadata.json

Run
---
    ./apf scripts/relaxometry/07_epg_sensitivity.py

Optional:
    ./apf scripts/relaxometry/07_epg_sensitivity.py \
        --t1-ms 500 1000 1500 2000 \
        --t2-min-ms 40 \
        --t2-max-ms 250 \
        --t2-step-ms 2 \
        --fa-min-deg 120 \
        --fa-max-deg 180 \
        --fa-step-deg 5

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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SCRIPT = "07_epg_sensitivity.py"
TAG = "rlx07"
VERSION = "0.1.0"

# -------------------------------------------------------------------------
# Acquisition constants established from one representative acquisition's metadata.
# -------------------------------------------------------------------------

SE12_ESP_MS = 9.8
SE12_N_ECHOES = 12

DUAL_ESP_MS = 12.82
DUAL_TE1_MS = 13.0
DUAL_TE2_MS = 103.0
DUAL_N_SIM_ECHOES = 14

# Nearest echo centers to reported effective TE.
DUAL_ECHO1_INDEX = 1
DUAL_ECHO2_INDEX = 8

EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2 = -0.17657875262241746

DEFAULT_T1_MS = [
    500.0,
    1000.0,
    1500.0,
    2000.0,
]

DEFAULT_T2_MIN_MS = 40.0
DEFAULT_T2_MAX_MS = 250.0
DEFAULT_T2_STEP_MS = 2.0

DEFAULT_FA_MIN_DEG = 120.0
DEFAULT_FA_MAX_DEG = 180.0
DEFAULT_FA_STEP_DEG = 5.0


try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "rlx07 requires matplotlib."
    ) from exc

try:
    from apf_paths import APFPaths, get_paths
except ImportError as exc:
    raise ImportError(
        "Could not import APF infrastructure. Run through ./apf."
    ) from exc


# =========================================================================
# General utilities
# =========================================================================


def now_iso() -> str:
    return datetime.now().isoformat(
        timespec="seconds"
    )


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
                fieldnames.append(
                    key
                )
                seen.add(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def output_dir(
    paths: APFPaths,
) -> Path:
    return (
        paths.derivatives_dir
        / "summary"
        / "relaxometry"
        / "rlx07"
    )


# =========================================================================
# EPG implementation
# =========================================================================


def rf_rotation_matrix(
    alpha_rad: float,
    phi_rad: float,
) -> np.ndarray:
    """
    EPG RF transition matrix acting on [F+, F-, Z].

    Complex-valued EPG convention.

    Parameters
    ----------
    alpha_rad
        RF flip angle in radians.

    phi_rad
        RF phase in radians.
    """

    ca2 = math.cos(
        alpha_rad / 2.0
    ) ** 2

    sa2 = math.sin(
        alpha_rad / 2.0
    ) ** 2

    sa = math.sin(
        alpha_rad
    )

    ca = math.cos(
        alpha_rad
    )

    e2iphi = np.exp(
        2j * phi_rad
    )

    em2iphi = np.exp(
        -2j * phi_rad
    )

    eiphi = np.exp(
        1j * phi_rad
    )

    emiphi = np.exp(
        -1j * phi_rad
    )

    return np.asarray(
        [
            [
                ca2,
                e2iphi * sa2,
                -1j * eiphi * sa,
            ],
            [
                em2iphi * sa2,
                ca2,
                1j * emiphi * sa,
            ],
            [
                -0.5j * emiphi * sa,
                0.5j * eiphi * sa,
                ca,
            ],
        ],
        dtype=np.complex128,
    )


def apply_rf(
    Fp: np.ndarray,
    Fm: np.ndarray,
    Z: np.ndarray,
    alpha_rad: float,
    phi_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an RF pulse independently to all EPG configuration states.
    """

    R = rf_rotation_matrix(
        alpha_rad,
        phi_rad,
    )

    states = np.vstack(
        [
            Fp,
            Fm,
            Z,
        ]
    )

    rotated = (
        R @ states
    )

    return (
        rotated[0].copy(),
        rotated[1].copy(),
        rotated[2].copy(),
    )


def relax(
    Fp: np.ndarray,
    Fm: np.ndarray,
    Z: np.ndarray,
    dt_ms: float,
    t1_ms: float,
    t2_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply T1 and T2 relaxation for dt.
    """

    E1 = math.exp(
        -dt_ms / t1_ms
    )

    E2 = math.exp(
        -dt_ms / t2_ms
    )

    Fp = (
        Fp
        * E2
    )

    Fm = (
        Fm
        * E2
    )

    Z = (
        Z
        * E1
    )

    # Recovery only occurs in Z0.
    Z[0] += (
        1.0 - E1
    )

    return (
        Fp,
        Fm,
        Z,
    )


def shift_orders(
    Fp: np.ndarray,
    Fm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply one EPG gradient-dephasing shift.

    We store non-negative EPG orders:
        F+_0, F+_1, ...
        F-_0, F-_1, ...

    Under a positive unit gradient:
        F+_k -> F+_(k+1)
        F-_k -> F-_(k-1)

    The zero-order transverse states obey conjugate symmetry.
    """

    old_Fp = Fp.copy()
    old_Fm = Fm.copy()

    new_Fp = np.zeros_like(old_Fp)
    new_Fm = np.zeros_like(old_Fm)

    # Positive-order pathway moves upward.
    new_Fp[1:] = old_Fp[:-1]

    # Negative-order pathway moves toward zero.
    new_Fm[:-1] = old_Fm[1:]

    # A negative-order coherence reaching zero becomes the observable
    # positive zero-order transverse coherence.
    new_Fp[0] = np.conj(new_Fm[0])

    # Enforce F-_0 = conj(F+_0).
    new_Fm[0] = np.conj(new_Fp[0])

    return new_Fp, new_Fm


def epg_cpmg_echo_train(
    t1_ms: float,
    t2_ms: float,
    echo_spacing_ms: float,
    n_echoes: int,
    refocusing_angle_deg: float,
    n_states: int = 128,
) -> np.ndarray:
    """
    Simulate a CPMG-like multi-spin-echo train using EPG.

    Sequence:
        equilibrium
        90 deg excitation about x
        [half-ESP evolution
         refocusing pulse about y
         half-ESP evolution
         observe F0] x N

    Gradient dephasing is split across each half echo.

    Returns
    -------
    np.ndarray
        Magnitude of F0 at each echo center.
    """

    if t1_ms <= 0:
        raise ValueError(
            "T1 must be > 0."
        )

    if t2_ms <= 0:
        raise ValueError(
            "T2 must be > 0."
        )

    if echo_spacing_ms <= 0:
        raise ValueError(
            "Echo spacing must be > 0."
        )

    if n_echoes < 1:
        raise ValueError(
            "n_echoes must be >= 1."
        )

    if n_states < (
        2 * n_echoes + 8
    ):
        raise ValueError(
            "n_states too small for requested echo train."
        )

    Fp = np.zeros(
        n_states,
        dtype=np.complex128,
    )

    Fm = np.zeros(
        n_states,
        dtype=np.complex128,
    )

    Z = np.zeros(
        n_states,
        dtype=np.complex128,
    )

    # Thermal equilibrium.
    Z[0] = 1.0

    # Ideal 90-degree excitation.
    #
    # Excitation phase x = 0.
    Fp, Fm, Z = apply_rf(
        Fp,
        Fm,
        Z,
        alpha_rad=np.deg2rad(
            90.0
        ),
        phi_rad=0.0,
    )

    refocus_rad = np.deg2rad(
        refocusing_angle_deg
    )

    # CPMG refocusing phase along y.
    refocus_phase = (
        np.pi / 2.0
    )

    half_esp = (
        echo_spacing_ms / 2.0
    )

    echoes: List[float] = []

    for _ in range(
        n_echoes
    ):
        # First half echo.
        Fp, Fm, Z = relax(
            Fp,
            Fm,
            Z,
            dt_ms=half_esp,
            t1_ms=t1_ms,
            t2_ms=t2_ms,
        )

        Fp, Fm = shift_orders(
            Fp,
            Fm,
        )

        # Refocusing RF.
        Fp, Fm, Z = apply_rf(
            Fp,
            Fm,
            Z,
            alpha_rad=refocus_rad,
            phi_rad=refocus_phase,
        )

        # Second half echo.
        Fp, Fm, Z = relax(
            Fp,
            Fm,
            Z,
            dt_ms=half_esp,
            t1_ms=t1_ms,
            t2_ms=t2_ms,
        )

        Fp, Fm = shift_orders(
            Fp,
            Fm,
        )

        echoes.append(
            float(
                np.abs(
                    Fp[0]
                )
            )
        )

    return np.asarray(
        echoes,
        dtype=np.float64,
    )


# =========================================================================
# Validation of EPG implementation
# =========================================================================


def validate_ideal_180_case() -> Dict[str, float]:
    """
    Sanity check.

    With ideal 180-degree refocusing, the simulated echo envelope should
    reduce to the expected mono-exponential spin-echo decay to numerical
    precision.

    This check runs before the sensitivity analysis. Failure stops rlx07.
    """

    t1_ms = 1000.0
    t2_ms = 100.0
    esp_ms = 10.0
    n_echoes = 12

    simulated = epg_cpmg_echo_train(
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        echo_spacing_ms=esp_ms,
        n_echoes=n_echoes,
        refocusing_angle_deg=180.0,
    )

    te = (
        np.arange(
            1,
            n_echoes + 1,
            dtype=np.float64,
        )
        * esp_ms
    )

    if (
        not np.all(np.isfinite(simulated))
        or simulated[0] <= 0
    ):
        raise RuntimeError(
            "EPG implementation produced invalid or zero echo amplitudes "
            "during the ideal-180 sanity check.\\n"
            f"Simulated echoes: {simulated}"
        )

    simulated_norm = simulated / simulated[0]

    expected_norm = np.exp(
        -(te - te[0]) / t2_ms
    )

    if not np.all(np.isfinite(simulated_norm)):
        raise RuntimeError(
            "EPG ideal-180 sanity check produced non-finite normalized echoes."
        )

    max_abs_error = float(
        np.max(
            np.abs(
                simulated_norm - expected_norm
            )
        )
    )

    max_rel_error = float(
        np.max(
            np.abs(
                (simulated_norm - expected_norm)
                / expected_norm
            )
        )
    )

    tolerance = 1e-8

    if (
        not np.isfinite(max_rel_error)
        or max_rel_error > tolerance
    ):
        raise RuntimeError(
            "EPG implementation failed ideal-180 sanity check.\\n"
            f"  max absolute error: {max_abs_error:.3e}\\n"
            f"  max relative error: {max_rel_error:.3e}\\n"
            f"  tolerance:          {tolerance:.3e}\\n"
            f"  simulated:          {simulated_norm}\\n"
            f"  expected:           {expected_norm}"
        )

    return {
        "max_absolute_error":
            max_abs_error,

        "max_relative_error":
            max_rel_error,

        "tolerance":
            tolerance,
    }


# =========================================================================
# R2 estimators
# =========================================================================


def fit_monoexponential_r2(
    te_ms: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float]:
    """
    Log-linear mono-exponential fit:

        ln(S) = ln(S0) - R2 * TE

    Returns
    -------
    r2_s_1, s0
    """

    te_ms = np.asarray(
        te_ms,
        dtype=np.float64,
    )

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            te_ms
        )
        & np.isfinite(
            signal
        )
        & (
            signal > 0
        )
    )

    if np.count_nonzero(
        valid
    ) < 2:
        return (
            float("nan"),
            float("nan"),
        )

    x_s = (
        te_ms[
            valid
        ]
        / 1000.0
    )

    y = np.log(
        signal[
            valid
        ]
    )

    slope, intercept = np.polyfit(
        x_s,
        y,
        deg=1,
    )

    r2_s_1 = (
        -float(
            slope
        )
    )

    s0 = float(
        np.exp(
            intercept
        )
    )

    return (
        r2_s_1,
        s0,
    )


def dual_echo_r2app(
    signal1: float,
    signal2: float,
    te1_ms: float = DUAL_TE1_MS,
    te2_ms: float = DUAL_TE2_MS,
) -> float:
    """
    Analytical two-point apparent R2.
    """

    if (
        not np.isfinite(
            signal1
        )
        or not np.isfinite(
            signal2
        )
        or signal1 <= 0
        or signal2 <= 0
        or signal2 >= signal1
    ):
        return float(
            "nan"
        )

    dt_s = (
        (
            te2_ms
            - te1_ms
        )
        / 1000.0
    )

    return float(
        np.log(
            signal1
            / signal2
        )
        / dt_s
    )


# =========================================================================
# Simulation
# =========================================================================


def simulate_condition(
    t1_ms: float,
    t2_ms: float,
    fa_deg: float,
) -> Dict[str, Any]:
    """
    Simulate both acquisition approximations for one parameter combination.
    """

    # ------------------------------------------------------------------
    # 12-echo sequence
    # ------------------------------------------------------------------

    se12_signal = epg_cpmg_echo_train(
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        echo_spacing_ms=SE12_ESP_MS,
        n_echoes=SE12_N_ECHOES,
        refocusing_angle_deg=fa_deg,
    )

    se12_te_ms = (
        np.arange(
            1,
            SE12_N_ECHOES + 1,
            dtype=np.float64,
        )
        * SE12_ESP_MS
    )

    # Adopted experimental model: drop echo 1.
    r2_12_s_1, _ = fit_monoexponential_r2(
        te_ms=se12_te_ms[1:],
        signal=se12_signal[1:],
    )

    # Also retain all12 for sensitivity/reference.
    r2_12_all_s_1, _ = fit_monoexponential_r2(
        te_ms=se12_te_ms,
        signal=se12_signal,
    )

    # ------------------------------------------------------------------
    # Dual-TSE approximation
    # ------------------------------------------------------------------

    dual_signal = epg_cpmg_echo_train(
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        echo_spacing_ms=DUAL_ESP_MS,
        n_echoes=DUAL_N_SIM_ECHOES,
        refocusing_angle_deg=fa_deg,
    )

    s1 = float(
        dual_signal[
            DUAL_ECHO1_INDEX - 1
        ]
    )

    s2 = float(
        dual_signal[
            DUAL_ECHO2_INDEX - 1
        ]
    )

    r2_dual_s_1 = dual_echo_r2app(
        signal1=s1,
        signal2=s2,
    )

    # ------------------------------------------------------------------
    # Bias calculations
    # ------------------------------------------------------------------

    true_r2_s_1 = (
        1000.0
        / t2_ms
    )

    true_t2_ms = (
        t2_ms
    )

    def relative_bias(
        estimate: float,
        reference: float,
    ) -> float:
        if (
            not np.isfinite(
                estimate
            )
            or reference == 0
        ):
            return float(
                "nan"
            )

        return float(
            (
                estimate
                - reference
            )
            / reference
        )

    dual_vs_12_relative = relative_bias(
        r2_dual_s_1,
        r2_12_s_1,
    )

    return {
        "T1_ms":
            float(
                t1_ms
            ),

        "T2_true_ms":
            float(
                true_t2_ms
            ),

        "R2_true_s-1":
            float(
                true_r2_s_1
            ),

        "refocusing_FA_deg":
            float(
                fa_deg
            ),

        "R2_12echo_all_s-1":
            float(
                r2_12_all_s_1
            ),

        "R2_12echo_drop1_s-1":
            float(
                r2_12_s_1
            ),

        "R2_dual_app_s-1":
            float(
                r2_dual_s_1
            ),

        "bias_12echo_drop1_relative":
            relative_bias(
                r2_12_s_1,
                true_r2_s_1,
            ),

        "bias_dual_relative":
            relative_bias(
                r2_dual_s_1,
                true_r2_s_1,
            ),

        "dual_vs_12_deltaR2_s-1":
            float(
                r2_dual_s_1
                - r2_12_s_1
            )
            if (
                np.isfinite(
                    r2_dual_s_1
                )
                and np.isfinite(
                    r2_12_s_1
                )
            )
            else float(
                "nan"
            ),

        "dual_vs_12_relative":
            dual_vs_12_relative,

        "dual_vs_12_relative_percent":
            float(
                100.0
                * dual_vs_12_relative
            )
            if np.isfinite(
                dual_vs_12_relative
            )
            else float(
                "nan"
            ),

        "dual_signal_echo1":
            s1,

        "dual_signal_echo8":
            s2,
    }


def run_grid(
    t1_values_ms: Sequence[float],
    t2_values_ms: Sequence[float],
    fa_values_deg: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    for t1_ms in t1_values_ms:
        for fa_deg in fa_values_deg:
            for t2_ms in t2_values_ms:
                rows.append(
                    simulate_condition(
                        t1_ms=float(
                            t1_ms
                        ),
                        t2_ms=float(
                            t2_ms
                        ),
                        fa_deg=float(
                            fa_deg
                        ),
                    )
                )

    return rows


# =========================================================================
# Summaries
# =========================================================================


def nearest_rows_to_empirical_bias(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each T1 and refocusing angle, find the simulated T2 at which
    dual-vs-12 relative bias is closest to the empirical cohort target.
    """

    groups: Dict[
        Tuple[float, float],
        List[Dict[str, Any]],
    ] = {}

    for row in rows:
        key = (
            float(
                row[
                    "T1_ms"
                ]
            ),
            float(
                row[
                    "refocusing_FA_deg"
                ]
            ),
        )

        groups.setdefault(
            key,
            [],
        ).append(
            row
        )

    result: List[
        Dict[str, Any]
    ] = []

    for (
        t1_ms,
        fa_deg,
    ), group in sorted(
        groups.items()
    ):
        finite = [
            row
            for row in group
            if np.isfinite(
                row[
                    "dual_vs_12_relative"
                ]
            )
        ]

        if not finite:
            continue

        best = min(
            finite,
            key=lambda row: abs(
                float(
                    row[
                        "dual_vs_12_relative"
                    ]
                )
                - EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
            ),
        )

        result.append(
            {
                "T1_ms":
                    t1_ms,

                "refocusing_FA_deg":
                    fa_deg,

                "closest_T2_true_ms":
                    best[
                        "T2_true_ms"
                    ],

                "closest_R2_true_s-1":
                    best[
                        "R2_true_s-1"
                    ],

                "simulated_dual_vs_12_relative":
                    best[
                        "dual_vs_12_relative"
                    ],

                "simulated_dual_vs_12_percent":
                    best[
                        "dual_vs_12_relative_percent"
                    ],

                "empirical_target_relative":
                    EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2,

                "empirical_target_percent":
                    (
                        100.0
                        * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
                    ),

                "absolute_target_error_percentage_points":
                    abs(
                        float(
                            best[
                                "dual_vs_12_relative_percent"
                            ]
                        )
                        - (
                            100.0
                            * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
                        )
                    ),
            }
        )

    return result


# =========================================================================
# Figures
# =========================================================================


def select_representative_t1(
    t1_values: Sequence[float],
) -> float:
    """
    Select T1 closest to 1000 ms for compact figures.
    """

    arr = np.asarray(
        t1_values,
        dtype=np.float64,
    )

    return float(
        arr[
            np.argmin(
                np.abs(
                    arr
                    - 1000.0
                )
            )
        ]
    )


def select_plot_angles(
    fa_values: Sequence[float],
) -> List[float]:
    """
    Keep the main figures readable.
    """

    targets = [
        180.0,
        170.0,
        160.0,
        150.0,
        140.0,
        130.0,
        120.0,
    ]

    available = np.asarray(
        fa_values,
        dtype=np.float64,
    )

    selected: List[
        float
    ] = []

    for target in targets:
        idx = int(
            np.argmin(
                np.abs(
                    available
                    - target
                )
            )
        )

        value = float(
            available[
                idx
            ]
        )

        if value not in selected:
            selected.append(
                value
            )

    return sorted(
        selected,
        reverse=True,
    )


def make_echo_curves_figure(
    out_path: Path,
    t1_ms: float,
    t2_ms: float,
    fa_values: Sequence[float],
) -> None:
    """
    Illustrate how stimulated echoes distort the two echo trains.
    """

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for fa in select_plot_angles(
        fa_values
    ):
        sig = epg_cpmg_echo_train(
            t1_ms=t1_ms,
            t2_ms=t2_ms,
            echo_spacing_ms=SE12_ESP_MS,
            n_echoes=SE12_N_ECHOES,
            refocusing_angle_deg=fa,
        )

        sig = (
            sig
            / sig[0]
        )

        te = (
            np.arange(
                1,
                SE12_N_ECHOES + 1
            )
            * SE12_ESP_MS
        )

        ax.plot(
            te,
            sig,
            marker="o",
            markersize=3,
            label=f"{fa:.0f}°",
        )

    ideal_te = (
        np.arange(
            1,
            SE12_N_ECHOES + 1
        )
        * SE12_ESP_MS
    )

    ideal = np.exp(
        -(
            ideal_te
            - ideal_te[0]
        )
        / t2_ms
    )

    ax.plot(
        ideal_te,
        ideal,
        linestyle="--",
        linewidth=1.5,
        label="Ideal mono-exponential",
    )

    ax.set_xlabel(
        "Echo time [ms]"
    )

    ax.set_ylabel(
        "Normalized echo magnitude"
    )

    ax.set_title(
        "EPG echo-train sensitivity to effective refocusing angle\n"
        f"T1={t1_ms:.0f} ms, T2={t2_ms:.0f} ms"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Refocusing FA"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_bias_curve_figure(
    rows: List[Dict[str, Any]],
    out_path: Path,
    representative_t1_ms: float,
    fa_values: Sequence[float],
) -> None:
    """
    Plot simulated method-to-method relative R2 difference vs true T2.
    """

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for fa in select_plot_angles(
        fa_values
    ):
        subset = [
            row
            for row in rows
            if math.isclose(
                float(
                    row[
                        "T1_ms"
                    ]
                ),
                representative_t1_ms,
            )
            and math.isclose(
                float(
                    row[
                        "refocusing_FA_deg"
                    ]
                ),
                fa,
            )
        ]

        subset = sorted(
            subset,
            key=lambda row: float(
                row[
                    "T2_true_ms"
                ]
            ),
        )

        x = np.asarray(
            [
                row[
                    "T2_true_ms"
                ]
                for row in subset
            ],
            dtype=np.float64,
        )

        y = np.asarray(
            [
                row[
                    "dual_vs_12_relative_percent"
                ]
                for row in subset
            ],
            dtype=np.float64,
        )

        ax.plot(
            x,
            y,
            label=f"{fa:.0f}°",
        )

    empirical_percent = (
        100.0
        * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
    )

    ax.axhline(
        empirical_percent,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Empirical cohort median "
            f"({empirical_percent:.1f}%)"
        ),
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "True T2 [ms]"
    )

    ax.set_ylabel(
        "Dual vs 12-echo ΔR2 [%]"
    )

    ax.set_title(
        "EPG sensitivity: apparent dual-echo bias relative to\n"
        "12-echo drop1 estimate "
        f"(T1={representative_t1_ms:.0f} ms)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Effective refocusing FA",
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_heatmap_figure(
    rows: List[Dict[str, Any]],
    out_path: Path,
    representative_t1_ms: float,
    t2_values: Sequence[float],
    fa_values: Sequence[float],
) -> None:
    """
    Heatmap of dual-vs-12 relative bias for one representative T1.
    """

    t2_sorted = np.asarray(
        sorted(
            set(
                float(x)
                for x in t2_values
            )
        ),
        dtype=np.float64,
    )

    fa_sorted = np.asarray(
        sorted(
            set(
                float(x)
                for x in fa_values
            )
        ),
        dtype=np.float64,
    )

    matrix = np.full(
        (
            fa_sorted.size,
            t2_sorted.size,
        ),
        np.nan,
        dtype=np.float64,
    )

    t2_index = {
        value: i
        for i, value in enumerate(
            t2_sorted
        )
    }

    fa_index = {
        value: i
        for i, value in enumerate(
            fa_sorted
        )
    }

    for row in rows:
        if not math.isclose(
            float(
                row[
                    "T1_ms"
                ]
            ),
            representative_t1_ms,
        ):
            continue

        t2 = float(
            row[
                "T2_true_ms"
            ]
        )

        fa = float(
            row[
                "refocusing_FA_deg"
            ]
        )

        matrix[
            fa_index[
                fa
            ],
            t2_index[
                t2
            ],
        ] = float(
            row[
                "dual_vs_12_relative_percent"
            ]
        )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[
            t2_sorted[0],
            t2_sorted[-1],
            fa_sorted[0],
            fa_sorted[-1],
        ],
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Dual vs 12-echo ΔR2 [%]"
    )

    empirical_percent = (
        100.0
        * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
    )

    try:
        contour = ax.contour(
            t2_sorted,
            fa_sorted,
            matrix,
            levels=[
                empirical_percent
            ],
            linewidths=1.5,
        )

        ax.clabel(
            contour,
            fmt={
                empirical_percent:
                    f"Observed {empirical_percent:.1f}%"
            },
        )

    except ValueError:
        # Target may lie outside simulated range.
        pass

    ax.set_xlabel(
        "True T2 [ms]"
    )

    ax.set_ylabel(
        "Effective refocusing FA [deg]"
    )

    ax.set_title(
        "EPG sensitivity surface\n"
        f"T1={representative_t1_ms:.0f} ms"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )



def make_t1_sensitivity_figure(
    rows: List[Dict[str, Any]],
    out_path: Path,
    t1_values: Sequence[float],
    t2_values: Sequence[float],
) -> None:
    """
    Plot the effective refocusing angle required to reproduce the empirical
    cohort dual-vs-12-echo R2 bias as a function of true T2, separately for
    each simulated T1.

    For each (T1, T2), the refocusing angle is estimated by interpolation
    across the simulated FA grid.

    This figure directly tests how strongly the inferred effective
    refocusing-angle requirement depends on the assumed T1.
    """

    target_percent = (
        100.0
        * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for t1_ms in sorted(
        float(x)
        for x in t1_values
    ):
        result_t2 = []
        result_fa = []

        for t2_ms in sorted(
            float(x)
            for x in t2_values
        ):
            subset = [
                row
                for row in rows
                if math.isclose(
                    float(row["T1_ms"]),
                    t1_ms,
                )
                and math.isclose(
                    float(row["T2_true_ms"]),
                    t2_ms,
                )
                and np.isfinite(
                    float(row["dual_vs_12_relative_percent"])
                )
            ]

            if len(subset) < 2:
                continue

            subset = sorted(
                subset,
                key=lambda row: float(
                    row["refocusing_FA_deg"]
                ),
            )

            fa = np.asarray(
                [
                    float(row["refocusing_FA_deg"])
                    for row in subset
                ],
                dtype=np.float64,
            )

            bias = np.asarray(
                [
                    float(row["dual_vs_12_relative_percent"])
                    for row in subset
                ],
                dtype=np.float64,
            )

            # Interpolation requires monotonic x. Sort by simulated bias.
            order = np.argsort(
                bias
            )

            bias_sorted = bias[order]
            fa_sorted = fa[order]

            # Only interpolate when the empirical target lies within the
            # simulated bias range for this T1/T2 combination.
            if not (
                bias_sorted[0]
                <= target_percent
                <= bias_sorted[-1]
            ):
                continue

            effective_fa = float(
                np.interp(
                    target_percent,
                    bias_sorted,
                    fa_sorted,
                )
            )

            result_t2.append(
                t2_ms
            )

            result_fa.append(
                effective_fa
            )

        if result_t2:
            ax.plot(
                result_t2,
                result_fa,
                linewidth=2,
                label=f"T1 = {t1_ms:.0f} ms",
            )

    ax.set_xlabel(
        "True T2 [ms]"
    )

    ax.set_ylabel(
        "Effective refocusing FA reproducing observed bias [deg]"
    )

    ax.set_title(
        "EPG sensitivity to assumed T1\n"
        f"Target dual-vs-12-echo R2 bias = {target_percent:.1f}%"
    )

    ax.set_ylim(
        115,
        180,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "APF rlx07: EPG sensitivity analysis for "
            "multi-echo and dual-echo T2/R2 estimation."
        )
    )

    parser.add_argument(
        "--t1-ms",
        type=float,
        nargs="+",
        default=DEFAULT_T1_MS,
        help=(
            "T1 sensitivity values in ms. "
            "Default: 500 1000 1500 2000."
        ),
    )

    parser.add_argument(
        "--t2-min-ms",
        type=float,
        default=DEFAULT_T2_MIN_MS,
    )

    parser.add_argument(
        "--t2-max-ms",
        type=float,
        default=DEFAULT_T2_MAX_MS,
    )

    parser.add_argument(
        "--t2-step-ms",
        type=float,
        default=DEFAULT_T2_STEP_MS,
    )

    parser.add_argument(
        "--fa-min-deg",
        type=float,
        default=DEFAULT_FA_MIN_DEG,
    )

    parser.add_argument(
        "--fa-max-deg",
        type=float,
        default=DEFAULT_FA_MAX_DEG,
    )

    parser.add_argument(
        "--fa-step-deg",
        type=float,
        default=DEFAULT_FA_STEP_DEG,
    )

    parser.add_argument(
        "--representative-t2-ms",
        type=float,
        default=100.0,
        help=(
            "True T2 used for the illustrative echo-train figure."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rlx07 outputs.",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if any(
        value <= 0
        for value in args.t1_ms
    ):
        raise SystemExit(
            "All T1 values must be > 0."
        )

    if (
        args.t2_min_ms <= 0
        or args.t2_max_ms <= 0
        or args.t2_step_ms <= 0
        or args.t2_max_ms
        < args.t2_min_ms
    ):
        raise SystemExit(
            "Invalid T2 range."
        )

    if (
        args.fa_min_deg <= 0
        or args.fa_max_deg > 180
        or args.fa_step_deg <= 0
        or args.fa_max_deg
        < args.fa_min_deg
    ):
        raise SystemExit(
            "Refocusing FA range must satisfy "
            "0 < min <= max <= 180."
        )

    if args.representative_t2_ms <= 0:
        raise SystemExit(
            "--representative-t2-ms must be > 0."
        )


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    paths = get_paths()

    out_dir = output_dir(
        paths
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_csv = (
        out_dir
        / "rlx07_simulation_grid.csv"
    )

    angle_summary_csv = (
        out_dir
        / "rlx07_angle_summary.csv"
    )

    echo_figure = (
        out_dir
        / "rlx07_epg_echo_curves.png"
    )

    bias_figure = (
        out_dir
        / "rlx07_dual_vs_reference_bias.png"
    )

    heatmap_figure = (
        out_dir
        / "rlx07_bias_heatmap.png"
    )

    t1_sensitivity_figure = (
        out_dir
        / "rlx07_t1_sensitivity.png"
    )

    metadata_path = (
        out_dir
        / "rlx07_metadata.json"
    )

    if (
        metadata_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"rlx07 outputs already exist:\n"
            f"  {metadata_path}\n"
            "Use --overwrite to replace them."
        )

    print(
        "=" * 72
    )

    print(
        "rlx07: EPG sensitivity analysis"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Mandatory internal validation.
    # ------------------------------------------------------------------

    sanity = validate_ideal_180_case()

    print(
        "  EPG ideal-180 sanity check: PASS"
    )

    print(
        f"    max relative error: "
        f"{sanity['max_relative_error']:.3e}"
    )

    # ------------------------------------------------------------------
    # Build parameter grids.
    # ------------------------------------------------------------------

    t1_values = np.asarray(
        args.t1_ms,
        dtype=np.float64,
    )

    t2_values = np.arange(
        args.t2_min_ms,
        args.t2_max_ms
        + 0.5
        * args.t2_step_ms,
        args.t2_step_ms,
        dtype=np.float64,
    )

    fa_values = np.arange(
        args.fa_min_deg,
        args.fa_max_deg
        + 0.5
        * args.fa_step_deg,
        args.fa_step_deg,
        dtype=np.float64,
    )

    print(
        f"  T1 values: "
        f"{', '.join(f'{x:.0f}' for x in t1_values)} ms"
    )

    print(
        f"  T2 range: "
        f"{t2_values[0]:.1f}--{t2_values[-1]:.1f} ms "
        f"(step {args.t2_step_ms:.1f})"
    )

    print(
        f"  Refocusing FA: "
        f"{fa_values[0]:.1f}--{fa_values[-1]:.1f} deg "
        f"(step {args.fa_step_deg:.1f})"
    )

    print(
        "  Acquisition provenance:"
    )

    print(
        f"    12-echo SE: ESP={SE12_ESP_MS:.2f} ms "
        "(source: measured TE increment 9.8, 19.6, ..., 117.6 ms), "
        f"N={SE12_N_ECHOES}, fit=drop1"
    )

    print(
        f"    Dual-echo TSE: ESP={DUAL_ESP_MS:.2f} ms "
        "(source: Siemens DICOM VendorReportedEchoSpacing)"
    )

    print(
        f"      effective TE={DUAL_TE1_MS:.0f}/{DUAL_TE2_MS:.0f} ms, "
        f"EPG approximation=echo-train positions "
        f"{DUAL_ECHO1_INDEX}/{DUAL_ECHO2_INDEX}"
    )

    print(
        "      nominal refocusing FA=180 deg, "
        "VariableFlipAngleFlag=N"
    )

    # ------------------------------------------------------------------
    # Run.
    # ------------------------------------------------------------------

    rows = run_grid(
        t1_values_ms=t1_values,
        t2_values_ms=t2_values,
        fa_values_deg=fa_values,
    )

    write_csv(
        grid_csv,
        rows,
    )

    target_rows = nearest_rows_to_empirical_bias(
        rows
    )

    write_csv(
        angle_summary_csv,
        target_rows,
    )

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------

    representative_t1 = select_representative_t1(
        t1_values
    )

    make_echo_curves_figure(
        out_path=echo_figure,
        t1_ms=representative_t1,
        t2_ms=float(
            args.representative_t2_ms
        ),
        fa_values=fa_values,
    )

    make_bias_curve_figure(
        rows=rows,
        out_path=bias_figure,
        representative_t1_ms=representative_t1,
        fa_values=fa_values,
    )

    make_heatmap_figure(
        rows=rows,
        out_path=heatmap_figure,
        representative_t1_ms=representative_t1,
        t2_values=t2_values,
        fa_values=fa_values,
    )

    make_t1_sensitivity_figure(
        rows=rows,
        out_path=t1_sensitivity_figure,
        t1_values=t1_values,
        t2_values=t2_values,
    )

    # ------------------------------------------------------------------
    # Compact diagnostic near experimental median T2.
    # ------------------------------------------------------------------

    target_t2 = 100.0

    print()
    print(
        "Sensitivity near T2 = 100 ms"
    )

    print(
        "-" * 72
    )

    for fa in sorted(
        fa_values,
        reverse=True,
    ):
        subset = [
            row
            for row in rows
            if math.isclose(
                float(
                    row[
                        "T1_ms"
                    ]
                ),
                representative_t1,
            )
            and math.isclose(
                float(
                    row[
                        "refocusing_FA_deg"
                    ]
                ),
                float(
                    fa
                ),
            )
        ]

        if not subset:
            continue

        nearest = min(
            subset,
            key=lambda row: abs(
                float(
                    row[
                        "T2_true_ms"
                    ]
                )
                - target_t2
            ),
        )

        print(
            f"  FA={fa:6.1f} deg: "
            f"R2true={nearest['R2_true_s-1']:6.3f}, "
            f"R2_12={nearest['R2_12echo_drop1_s-1']:6.3f}, "
            f"R2dual={nearest['R2_dual_app_s-1']:6.3f}, "
            f"dual-vs-12="
            f"{nearest['dual_vs_12_relative_percent']:7.2f}%"
        )

    # ------------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------------

    metadata = {
        "script":
            SCRIPT,

        "tag":
            TAG,

        "version":
            VERSION,

        "created":
            now_iso(),

        "execution": {
            "host":
                platform.node(),

            "platform":
                platform.platform(),

            "python":
                sys.version.split()[0],

            "command":
                " ".join(
                    sys.argv
                ),
        },

        "scientific_role": {
            "analysis_type":
                "EPG sensitivity simulation",

            "primary_question":
                (
                    "Can imperfect constant refocusing angles and "
                    "stimulated-echo pathways plausibly produce the "
                    "systematic difference observed between the "
                    "dual-echo R2app and 12-echo drop1 R2 estimates?"
                ),

            "not_an_exact_sequence_reconstruction":
                True,

            "reason":
                (
                    "Complete proprietary Siemens RF timing and k-space "
                    "ordering for *tse2d2_7 are not available from the "
                    "exported DICOM metadata."
                ),
        },

        "acquisition_model": {
            "12echo": {
                "sequence":
                    "*se2d12",

                "echo_spacing_ms":
                    SE12_ESP_MS,

                "n_echoes":
                    SE12_N_ECHOES,

                "fit":
                    "echoes 2-12",

                "nominal_refocusing_angle_deg":
                    180.0,
            },

            "dual_echo": {
                "sequence":
                    "*tse2d2_7",

                "vendor_echo_spacing_ms":
                    DUAL_ESP_MS,

                "reported_effective_TE_ms":
                    [
                        DUAL_TE1_MS,
                        DUAL_TE2_MS,
                    ],

                "simulation_sampling_echo_indices":
                    [
                        DUAL_ECHO1_INDEX,
                        DUAL_ECHO2_INDEX,
                    ],

                "DICOM_EchoTrainLength":
                    14,

                "DICOM_RFEchoTrainLength":
                    7,

                "nominal_refocusing_angle_deg":
                    180.0,

                "variable_flip_angle_flag":
                    "N",

                "approximation_warning":
                    (
                        "Echo 1 and echo 8 sampling is a sensitivity "
                        "approximation based on reported effective TE "
                        "and vendor echo spacing. It is not asserted "
                        "to reproduce exact Siemens image contrast or "
                        "k-space ordering."
                    ),
            },
        },

        "epg_model": {
            "single_pool":
                True,

            "excitation_angle_deg":
                90.0,

            "refocusing_phase":
                "CPMG y-phase",

            "constant_refocusing_angle_within_simulation":
                True,

            "includes_T1":
                True,

            "includes_T2":
                True,

            "includes_stimulated_echo_pathways":
                True,

            "slice_profile_model":
                False,

            "diffusion_model":
                False,

            "off_resonance_model":
                False,

            "magnetization_transfer_model":
                False,

            "kspace_ordering_model":
                False,
        },

        "parameter_grid": {
            "T1_ms":
                [
                    float(x)
                    for x in t1_values
                ],

            "T2_min_ms":
                float(
                    t2_values[0]
                ),

            "T2_max_ms":
                float(
                    t2_values[-1]
                ),

            "T2_step_ms":
                float(
                    args.t2_step_ms
                ),

            "refocusing_FA_min_deg":
                float(
                    fa_values[0]
                ),

            "refocusing_FA_max_deg":
                float(
                    fa_values[-1]
                ),

            "refocusing_FA_step_deg":
                float(
                    args.fa_step_deg
                ),
        },

        "empirical_reference": {
            "source_stage":
                "rlx06b",

            "median_relative_delta_R2":
                EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2,

            "median_relative_delta_R2_percent":
                (
                    100.0
                    * EMPIRICAL_MEDIAN_RELATIVE_DELTA_R2
                ),

            "definition":
                (
                    "(dual R2app - 12echo drop1 R2) / "
                    "12echo drop1 R2"
                ),
        },

        "internal_validation": {
            "ideal_180_monoexponential_check":
                sanity,
        },

        "outputs": {
            "simulation_grid_csv":
                str(
                    grid_csv
                ),

            "angle_summary_csv":
                str(
                    angle_summary_csv
                ),

            "echo_curves_figure":
                str(
                    echo_figure
                ),

            "bias_figure":
                str(
                    bias_figure
                ),

            "heatmap_figure":
                str(
                    heatmap_figure
                ),
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    print()
    print(
        f"  Grid rows: {len(rows)}"
    )

    print(
        f"  Simulation CSV: {grid_csv}"
    )

    print(
        f"  Angle summary: {angle_summary_csv}"
    )

    print(
        f"  Echo curves: {echo_figure}"
    )

    print(
        f"  Bias curves: {bias_figure}"
    )

    print(
        f"  Bias heatmap: {heatmap_figure}"
    )

    print(
        f"  T1 sensitivity: {t1_sensitivity_figure}"
    )

    print(
        f"  DONE: {TAG}"
    )


if __name__ == "__main__":
    main()