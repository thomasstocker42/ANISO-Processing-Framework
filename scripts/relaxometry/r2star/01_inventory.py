#!/usr/bin/env python3
"""
R2STAR01 — GRE R2* input inventory

Purpose
-------
Audit all inputs required for magnitude-derived R2* reconstruction.

Expected inputs per case:
  - 12 GRE magnitude echoes
  - corresponding JSON sidecars
  - EchoNumber and EchoTime metadata
  - finalized GRE02b non-eroded brain mask

No quantitative R2* fitting is performed here.

Outputs
-------
derivatives/summary/relaxometry/r2star/r2star01_inventory/
    r2star01_inventory.csv
    r2star01_inventory.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


SCRIPT_ID = "R2STAR01"
SCRIPT_NAME = Path(__file__).name
VERSION = "0.1.0"

DEFAULT_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_ECHOES = tuple(range(1, 13))


class InventoryError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_registry_cases(root: Path) -> list[str]:
    """Use APF registry, with CSV fallback."""
    try:
        from apf_registry import get_cases  # type: ignore

        records = get_cases()
        cases: list[str] = []

        for record in records:
            if isinstance(record, str):
                case = record
            elif isinstance(record, dict):
                case = record.get("case") or record.get("case_id") or ""
            else:
                case = getattr(record, "case", "") or getattr(record, "case_id", "")

            if case:
                cases.append(str(case))

        if cases:
            return sorted(set(cases))

    except (ImportError, TypeError):
        pass

    csv_path = root / "config" / "InS_cases.csv"

    if not csv_path.is_file():
        raise InventoryError(
            f"APF registry unavailable and case CSV not found: {csv_path}"
        )

    cases: list[str] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lowered = {
                str(k).lower(): str(v).strip()
                for k, v in row.items()
            }

            case = lowered.get("case") or lowered.get("case_id") or ""
            active = lowered.get("active", "true").lower()

            if case and active not in {
                "0", "false", "no", "inactive", "excluded"
            }:
                cases.append(case)

    return sorted(set(cases))


def echo_number_from_name(path: Path) -> int | None:
    match = re.search(r"_e(\d+)\.(?:nii(?:\.gz)?|json)$", path.name)

    if match:
        return int(match.group(1))

    return None


def find_echo_table(root: Path, case: str) -> list[dict[str, Any]]:
    mag_dir = root / "nifti" / case / "GRE" / "mag"

    rows: list[dict[str, Any]] = []

    for echo in EXPECTED_ECHOES:
        nii_candidates = sorted(
            list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii.gz"))
            + list(mag_dir.glob(f"{case}_GRE_12echo_*_e{echo}.nii"))
        )

        if len(nii_candidates) != 1:
            rows.append(
                {
                    "echo": echo,
                    "nii": None,
                    "json": None,
                    "echo_number_json": None,
                    "te_s": None,
                    "status": (
                        "missing_nifti"
                        if len(nii_candidates) == 0
                        else f"duplicate_nifti_{len(nii_candidates)}"
                    ),
                }
            )
            continue

        nii = nii_candidates[0]

        if nii.name.endswith(".nii.gz"):
            js = nii.with_name(nii.name[:-7] + ".json")
        else:
            js = nii.with_suffix(".json")

        if not js.is_file():
            rows.append(
                {
                    "echo": echo,
                    "nii": nii,
                    "json": js,
                    "echo_number_json": None,
                    "te_s": None,
                    "status": "missing_json",
                }
            )
            continue

        try:
            metadata = json.loads(js.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "echo": echo,
                    "nii": nii,
                    "json": js,
                    "echo_number_json": None,
                    "te_s": None,
                    "status": f"json_error:{exc}",
                }
            )
            continue

        echo_json = metadata.get("EchoNumber")
        te_s = metadata.get("EchoTime")

        if echo_json is None or te_s is None:
            status = "missing_echo_metadata"
        elif int(echo_json) != echo:
            status = "echo_number_mismatch"
        else:
            status = "OK"

        rows.append(
            {
                "echo": echo,
                "nii": nii,
                "json": js,
                "echo_number_json": (
                    int(echo_json) if echo_json is not None else None
                ),
                "te_s": float(te_s) if te_s is not None else None,
                "status": status,
            }
        )

    return rows


def nifti_geometry(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "shape": None,
            "zooms_mm": None,
            "affine": None,
            "finite": False,
        }

    image = nib.load(str(path))

    return {
        "shape": list(image.shape),
        "zooms_mm": [float(x) for x in image.header.get_zooms()[:3]],
        "affine": np.asarray(image.affine).tolist(),
        "finite": bool(np.isfinite(image.affine).all()),
    }


def same_geometry(a: Path, b: Path) -> bool:
    ia = nib.load(str(a))
    ib = nib.load(str(b))

    return (
        ia.shape == ib.shape
        and np.allclose(ia.affine, ib.affine, atol=1e-5)
    )


def inspect_case(root: Path, case: str) -> dict[str, Any]:
    table = find_echo_table(root, case)

    mask = (
        root
        / "derivatives"
        / case
        / "GRE"
        / "02b_brain_mask"
        / f"{case}_GRE02b_brain_mask.nii.gz"
    )

    echo_ok = [row for row in table if row["status"] == "OK"]

    first_echo = (
        echo_ok[0]["nii"]
        if echo_ok
        else None
    )

    geometry_ok = False
    mask_geometry_ok = False

    if len(echo_ok) == 12:
        ref = echo_ok[0]["nii"]

        geometry_ok = all(
            same_geometry(ref, row["nii"])
            for row in echo_ok[1:]
        )

        if mask.is_file():
            mask_geometry_ok = same_geometry(ref, mask)

    te_s = [
        row["te_s"]
        for row in table
        if row["status"] == "OK"
    ]

    te_monotonic = (
        len(te_s) == 12
        and all(te_s[i] < te_s[i + 1] for i in range(11))
    )

    ready = (
        len(echo_ok) == 12
        and mask.is_file()
        and geometry_ok
        and mask_geometry_ok
        and te_monotonic
    )

    return {
        "case": case,
        "ready": ready,
        "n_echoes_ok": len(echo_ok),
        "mask_exists": mask.is_file(),
        "echo_geometry_ok": geometry_ok,
        "mask_geometry_ok": mask_geometry_ok,
        "te_monotonic": te_monotonic,
        "te_ms": [
            round(float(x) * 1000.0, 6)
            for x in te_s
        ],
        "mask": str(mask),
        "first_echo_geometry": nifti_geometry(first_echo),
        "mask_geometry": nifti_geometry(mask if mask.is_file() else None),
        "echoes": [
            {
                **row,
                "nii": str(row["nii"]) if row["nii"] else None,
                "json": str(row["json"]) if row["json"] else None,
            }
            for row in table
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory GRE magnitude inputs for APF R2* fitting."
    )

    parser.add_argument("cases", nargs="*", help="Case IDs.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all active registry cases.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    args = parser.parse_args()

    if args.all and args.cases:
        parser.error("Use explicit cases or --all, not both.")

    if not args.all and not args.cases:
        parser.error("Provide case ID(s) or use --all.")

    return args


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    cases = (
        get_registry_cases(root)
        if args.all
        else sorted(set(args.cases))
    )

    out_dir = (
        root
        / "derivatives"
        / "summary"
        / "relaxometry"
        / "r2star"
        / "r2star01_inventory"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "r2star01_inventory.json"
    csv_path = out_dir / "r2star01_inventory.csv"

    print("=" * 72)
    print(f"{SCRIPT_ID} — GRE R2* input inventory")
    print(f"Version: {VERSION}")
    print(f"Root:    {root}")
    print("=" * 72)

    results = []

    for case in cases:
        result = inspect_case(root, case)
        results.append(result)

        print(
            f"{case:20s} "
            f"echoes={result['n_echoes_ok']:2d}/12 "
            f"mask={'OK' if result['mask_exists'] else '-':2s} "
            f"geom={'OK' if result['echo_geometry_ok'] else '-':2s} "
            f"maskgeom={'OK' if result['mask_geometry_ok'] else '-':2s} "
            f"TE={'OK' if result['te_monotonic'] else '-':2s} "
            f"READY={'YES' if result['ready'] else 'NO'}"
        )

    payload = {
        "script": SCRIPT_NAME,
        "script_id": SCRIPT_ID,
        "version": VERSION,
        "timestamp": now_iso(),
        "root": str(root),
        "n_cases": len(results),
        "n_ready": sum(bool(x["ready"]) for x in results),
        "cases": results,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    csv_fields = [
        "case",
        "ready",
        "n_echoes_ok",
        "mask_exists",
        "echo_geometry_ok",
        "mask_geometry_ok",
        "te_monotonic",
        "te_ms",
        "mask",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "case": result["case"],
                    "ready": result["ready"],
                    "n_echoes_ok": result["n_echoes_ok"],
                    "mask_exists": result["mask_exists"],
                    "echo_geometry_ok": result["echo_geometry_ok"],
                    "mask_geometry_ok": result["mask_geometry_ok"],
                    "te_monotonic": result["te_monotonic"],
                    "te_ms": ", ".join(
                        f"{x:.3f}" for x in result["te_ms"]
                    ),
                    "mask": result["mask"],
                }
            )

    print()
    print(f"Ready: {payload['n_ready']}/{payload['n_cases']}")
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {json_path}")

    return 0 if payload["n_ready"] == payload["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())