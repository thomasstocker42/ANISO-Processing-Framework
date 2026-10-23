#!/usr/bin/env python3
"""
APF infrastructure tool: migrate_manifests.py

Upgrade all existing APF manifests to the current manifest schema.

This tool is intentionally idempotent:
- existing manifest information is preserved
- missing schema fields are added
- missing modules/steps/step fields are added
- already-current manifests remain unchanged
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from apf_manifest import (
    load_manifest,
    manifest_path,
    save_manifest,
    upgrade_manifest_schema,
    validate_manifest_schema,
)
from apf_paths import get_paths


def main() -> int:
    paths = get_paths()
    manifest_files = sorted(paths.derivatives_dir.glob("*/processing/manifest.json"))

    updated = 0
    unchanged = 0
    missing = 0
    failed = 0

    print("=" * 70)
    print("APF migrate_manifests.py")
    print("=" * 70)

    for path in manifest_files:
        case_id = path.parents[1].name

        try:
            manifest = load_manifest(case_id, paths=paths)
            upgraded, changes = upgrade_manifest_schema(manifest)

            valid, errors = validate_manifest_schema(upgraded)
            if not valid:
                failed += 1
                print(f"[FAILED] {case_id}")
                for error in errors:
                    print(f"   - {error}")
                continue

            if changes:
                save_manifest(upgraded, paths=paths)
                updated += 1
                print(f"[UPDATED] {case_id}")
                for change in changes:
                    print(f"   + {change}")
            else:
                unchanged += 1
                print(f"[OK] {case_id}")

        except Exception as exc:
            failed += 1
            print(f"[ERROR] {case_id}: {exc}")

    print()
    print("=" * 70)
    print("Migration summary")
    print("=" * 70)
    print(f"Updated  : {updated}")
    print(f"Unchanged: {unchanged}")
    print(f"Missing  : {missing}")
    print(f"Failed   : {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())