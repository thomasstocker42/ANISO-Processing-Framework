#!/usr/bin/env python3
"""
Small smoke test for APF path and registry modules.
Run from ANISO_processing root after installing apf_paths.py and apf_registry.py.
"""

from apf_paths import get_paths
from apf_registry import get_cases

paths = get_paths()
print(f"ROOT: {paths.root}")
print(f"InS registry: {paths.ins_registry}")

cases = get_cases(case_type="InS")
print(f"N active/new InS cases: {len(cases)}")
for c in cases:
    print(f"  {c.case:18s} {c.status:8s} {c.type:4s} {c.raw_root}")
