#!/usr/bin/env python3
"""
APF infrastructure tool: create_manifest.py

Create or refresh one APF per-case manifest.

Example
-------
python scripts/infrastructure/create_manifest.py ANISO_00_000_InS
python scripts/infrastructure/create_manifest.py ANISO_00_000_InS --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from apf_manifest import create_manifest, manifest_path, validate_manifest_schema
from apf_paths import get_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an APF manifest.json for one case."
    )
    parser.add_argument("case", help="Case ID, e.g. ANISO_00_000_InS")
    parser.add_argument(
        "--case-type",
        default="InS",
        choices=["InS", "ExS"],
        help="Case type. Default: InS",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_paths()

    manifest = create_manifest(
        args.case,
        case_type=args.case_type,
        paths=paths,
        overwrite=args.overwrite,
    )

    valid, errors = validate_manifest_schema(manifest)
    path = manifest_path(args.case, paths=paths)

    print("=" * 70)
    print("APF create_manifest.py")
    print("=" * 70)
    print(f"Case:      {args.case}")
    print(f"Case type: {args.case_type}")
    print(f"Manifest:  {path}")

    if valid:
        print("Schema:    valid")
    else:
        print("Schema:    INVALID")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())