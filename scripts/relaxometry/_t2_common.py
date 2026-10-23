#!/usr/bin/env python3
"""Shared utilities for APF relaxometry scripts.

The case-selection implementation is deliberately isolated here. It supports
single cases, explicit case lists, and registry-backed --all-t2-ready. If the
project's canonical shared case-selection helper exposes a different API,
only resolve_cases() needs to be replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np


ECHO_RE = re.compile(r"(?P<prefix>.*?)(?:_e(?P<echo>\d+))$", re.IGNORECASE)


@dataclass(frozen=True)
class EchoInput:
    image: str
    json: str
    echo_number: int
    echo_time_s: float
    series_number: int | None
    series_description: str
    protocol_name: str
    pulse_sequence_name: str
    pulse_sequence_details: str
    scanning_sequence: str
    echo_train_length: int | None
    flip_angle_deg: float | None


@dataclass(frozen=True)
class Acquisition:
    acquisition_id: str
    source_dir: str
    classification: str
    echoes: list[EchoInput]
    shape: tuple[int, ...]
    voxel_size_mm: tuple[float, float, float]
    affine_match: bool
    geometry_match: bool
    ready: bool
    warnings: list[str]


def project_root() -> Path:
    return Path(os.environ.get("APF_ROOT", Path.cwd())).expanduser().resolve()


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case", nargs="?", help="Single case ID.")
    parser.add_argument("--cases", nargs="+", default=None, help="Explicit case IDs.")
    parser.add_argument(
        "--all-t2-ready",
        action="store_true",
        help="Select registry cases for which a valid T2 acquisition is discovered.",
    )


def _registry_cases(root: Path) -> list[str]:
    registry = root / "config" / "InS_cases.csv"
    if not registry.exists():
        raise FileNotFoundError(
            f"Registry not found: {registry}. Use CASE or --cases, or set APF_ROOT."
        )

    cases: list[str] = []
    with registry.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "case" not in reader.fieldnames:
            raise ValueError(f"Registry lacks required 'case' column: {registry}")
        for row in reader:
            case = (row.get("case") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if case and status not in {"excluded", "inactive", "archived"}:
                cases.append(case)
    return cases


def resolve_cases(args: argparse.Namespace, root: Path) -> list[str]:
    selectors = int(bool(args.case)) + int(bool(args.cases)) + int(args.all_t2_ready)
    if selectors != 1:
        raise ValueError(
            "Select exactly one of CASE, --cases, or --all-t2-ready."
        )

    if args.case:
        return [args.case]
    if args.cases:
        return list(dict.fromkeys(args.cases))

    ready: list[str] = []
    for case in _registry_cases(root):
        try:
            acquisitions = discover_acquisitions(root, case)
        except Exception:
            continue
        if any(a.ready for a in acquisitions):
            ready.append(case)
    return ready


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _series_key(json_path: Path, meta: dict[str, Any]) -> str:
    stem = json_path.name.removesuffix(".json")
    match = ECHO_RE.match(stem)
    prefix = match.group("prefix") if match else stem
    series_number = meta.get("SeriesNumber")
    return f"{json_path.parent}:{series_number}:{prefix}"


def _to_echo(json_path: Path, meta: dict[str, Any]) -> EchoInput:
    image = json_path.with_suffix(".nii.gz")
    if not image.exists():
        image = json_path.with_suffix(".nii")
    if not image.exists():
        raise FileNotFoundError(f"No NIfTI matching {json_path}")

    echo_time = meta.get("EchoTime")
    if echo_time is None:
        raise ValueError(f"Missing EchoTime: {json_path}")

    echo_number = meta.get("EchoNumber")
    if echo_number is None:
        m = ECHO_RE.match(json_path.name.removesuffix(".json"))
        echo_number = int(m.group("echo")) if m and m.group("echo") else 0

    return EchoInput(
        image=str(image),
        json=str(json_path),
        echo_number=int(echo_number),
        echo_time_s=float(echo_time),
        series_number=int(meta["SeriesNumber"]) if meta.get("SeriesNumber") is not None else None,
        series_description=str(meta.get("SeriesDescription", "")),
        protocol_name=str(meta.get("ProtocolName", "")),
        pulse_sequence_name=str(meta.get("PulseSequenceName", "")),
        pulse_sequence_details=str(meta.get("PulseSequenceDetails", "")),
        scanning_sequence=str(meta.get("ScanningSequence", "")),
        echo_train_length=int(meta["EchoTrainLength"]) if meta.get("EchoTrainLength") is not None else None,
        flip_angle_deg=float(meta["FlipAngle"]) if meta.get("FlipAngle") is not None else None,
    )


def _classify(echoes: Sequence[EchoInput]) -> str:
    text = " ".join(
        [
            echoes[0].series_description,
            echoes[0].protocol_name,
            echoes[0].pulse_sequence_name,
            echoes[0].pulse_sequence_details,
        ]
    ).lower()
    is_tse = "tse" in text or "turbo" in text
    if len(echoes) == 2 and is_tse:
        return "dual_tse"
    if len(echoes) >= 4:
        return "multi_echo"
    return "unsupported"


def _geometry(echoes: Sequence[EchoInput]) -> tuple[tuple[int, ...], tuple[float, float, float], bool, bool]:
    images = [nib.load(e.image) for e in echoes]
    shape0 = tuple(images[0].shape)
    zoom0 = tuple(float(v) for v in images[0].header.get_zooms()[:3])
    affine0 = images[0].affine
    geometry_match = all(tuple(img.shape) == shape0 and np.allclose(img.header.get_zooms()[:3], zoom0, atol=1e-5) for img in images[1:])
    affine_match = all(np.allclose(img.affine, affine0, atol=1e-4) for img in images[1:])
    return shape0, zoom0, affine_match, geometry_match


def discover_acquisitions(root: Path, case: str) -> list[Acquisition]:
    candidate_dirs = [
        root / "nifti" / case / "T2",
        root / "nifti" / case / "SE",
    ]
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for json_path in sorted(directory.glob("*.json")):
            try:
                meta = _load_json(json_path)
            except Exception:
                continue
            if meta.get("EchoTime") is None:
                continue
            key = _series_key(json_path, meta)
            groups.setdefault(key, []).append((json_path, meta))

    acquisitions: list[Acquisition] = []
    for key, entries in groups.items():
        warnings: list[str] = []
        echoes: list[EchoInput] = []
        try:
            echoes = [_to_echo(path, meta) for path, meta in entries]
            echoes.sort(key=lambda e: (e.echo_time_s, e.echo_number))
            shape, voxel_size, affine_match, geometry_match = _geometry(echoes)
        except Exception as exc:
            warnings.append(str(exc))
            continue

        classification = _classify(echoes)
        if len({round(e.echo_time_s, 9) for e in echoes}) != len(echoes):
            warnings.append("Duplicate echo times.")
        if any(np.diff([e.echo_time_s for e in echoes]) <= 0):
            warnings.append("Echo times are not strictly increasing.")
        if not geometry_match:
            warnings.append("Echo geometry mismatch.")
        if not affine_match:
            warnings.append("Echo affine mismatch.")
        if classification == "unsupported":
            warnings.append("Series does not match dual-TSE or multi-echo criteria.")

        ready = (
            classification in {"dual_tse", "multi_echo"}
            and geometry_match
            and affine_match
            and not any("Duplicate echo" in w for w in warnings)
        )

        acquisitions.append(
            Acquisition(
                acquisition_id=key,
                source_dir=str(Path(echoes[0].image).parent),
                classification=classification,
                echoes=echoes,
                shape=shape,
                voxel_size_mm=voxel_size,
                affine_match=affine_match,
                geometry_match=geometry_match,
                ready=ready,
                warnings=warnings,
            )
        )

    acquisitions.sort(
        key=lambda a: (
            {"dual_tse": 0, "multi_echo": 1, "unsupported": 2}.get(a.classification, 9),
            a.acquisition_id,
        )
    )
    return acquisitions


def acquisition_to_dict(acq: Acquisition) -> dict[str, Any]:
    data = asdict(acq)
    data["echo_times_ms"] = [1000.0 * e.echo_time_s for e in acq.echoes]
    return data


def output_root(root: Path, case: str, stage: str) -> Path:
    path = root / "derivatives" / case / "relaxometry" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")
