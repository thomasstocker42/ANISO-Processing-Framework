#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
apf_manifest.py

Purpose
-------
Manage per-case processing manifests.

Hybrid provenance model
-----------------------
- manifest.json stores the current processing state.
- timestamped log files under processing/logs/ store execution history.

This keeps the manifest lightweight and queryable while preserving detailed run history
outside the manifest.

Version
-------
APF core version: 0.2
Created: 2026-06-17
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

from apf_paths import APFPaths, get_paths
from apf_registry import APFCase, get_case


# APF versioning

APF_CORE_VERSION = "0.2"
MANIFEST_SCHEMA_VERSION = "0.2"

# Standard output policy
OUTPUT_POLICY = {
    "images": ".nii.gz",
    "masks": ".nii.gz",
    "maps": ".nii.gz",
    "tables": ".csv",
    "metadata": ".json",
    "qc_figures": ".png",
    "final_figures": ".pdf",
    "logs": ".log",
}


def now_iso() -> str:
    """Return local timestamp in ISO-like format without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat()


def empty_step() -> dict[str, Any]:
    """Return an empty processing-step state object."""
    return {
        "completed": False,
        "valid": False,
        "script": "",
        "script_version": "",
        "algorithm_version": "",
        "date_started": "",
        "date_finished": "",
        "duration_seconds": None,
        "command": "",
        "inputs": [],
        "outputs": [],
        "parameters": {},
        "qc": {
            "available": False,
            "files": [],
            "summary": {},
        },
        "warnings": [],
        "errors": [],
        "notes": "",
    }


def default_modules() -> dict[str, dict[str, dict[str, Any]]]:
    """Return the default APF module/step tree."""
    step = empty_step
    return {
        "GRE": {
            "conversion": step(),
            "brain_extraction": step(),
            "vesselness": step(),
            "R2star": step(),
            "local_frequency": step(),
            "QSM": step(),
            "chi_separation": step(),
        },
        "MP2RAGE": {
            "inventory": step(),
            "tissue_segmentation": step(),
            "registration_to_GRE": step(),
        },
        "DTI": {
            "inventory": step(),
            "topup": step(),
            "eddy": step(),
            "dtifit_allb": step(),
            "dtifit_b1000": step(),
            "dtifit_b2000": step(),
            "orientation_maps": step(),
            "registration_to_GRE": step(),
        },
        "Analysis": {
            "vessel_R2star": step(),
            "vessel_local_frequency": step(),
            "orientation_R2star": step(),
            "combined_models": step(),
        },
    }


def build_manifest(case_entry: APFCase) -> dict[str, Any]:
    """Create a new manifest dictionary from a registry case entry."""
    t = now_iso()
    return {
        "case": case_entry.case,
        "case_type": case_entry.type,
        "created": t,
        "last_updated": t,

        "apf_core_version": APF_CORE_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,

        "output_policy": deepcopy(OUTPUT_POLICY),

        "registry_status": case_entry.status,
        "raw_root": case_entry.raw_root,
        "notes": case_entry.notes,

        "modules": default_modules(),
    }


def required_top_level_fields() -> list[str]:
    """Return required top-level manifest fields."""
    return [
        "case",
        "case_type",
        "created",
        "last_updated",
        "apf_core_version",
        "manifest_schema_version",
        "output_policy",
        "registry_status",
        "raw_root",
        "notes",
        "modules",
    ]


def validate_manifest_schema(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate manifest structure.

    Returns
    -------
    valid:
        True if no schema errors were found.
    errors:
        List of human-readable schema errors.
    """
    errors: list[str] = []

    for key in required_top_level_fields():
        if key not in manifest:
            errors.append(f"Missing top-level field: {key}")

    if "modules" not in manifest or not isinstance(manifest.get("modules"), dict):
        errors.append("Missing or invalid top-level field: modules")
        return False, errors

    default = default_modules()

    for module_name, default_steps in default.items():
        if module_name not in manifest["modules"]:
            errors.append(f"Missing module: {module_name}")
            continue

        module = manifest["modules"][module_name]
        if not isinstance(module, dict):
            errors.append(f"Invalid module object: {module_name}")
            continue

        for step_name, default_step_state in default_steps.items():
            if step_name not in module:
                errors.append(f"Missing step: {module_name}/{step_name}")
                continue

            step_state = module[step_name]
            if not isinstance(step_state, dict):
                errors.append(f"Invalid step object: {module_name}/{step_name}")
                continue

            for field in default_step_state.keys():
                if field not in step_state:
                    errors.append(
                        f"Missing field in step {module_name}/{step_name}: {field}"
                    )

    return len(errors) == 0, errors


def infer_case_type(case: str) -> str:
    """Infer case type from case ID."""
    if case.endswith("_ExS"):
        return "ExS"
    return "InS"


def upgrade_manifest_schema(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Upgrade a manifest dictionary/schema to the current schema.

    This function is non-destructive:
    - existing values are preserved
    - missing top-level fields are added
    - missing modules are added
    - missing steps are added
    - missing step fields are added

    Returns
    -------
    upgraded_manifest:
        The upgraded manifest dictionary.
    changes:
        Human-readable list of changes applied.
    """
    upgraded = deepcopy(manifest)
    changes: list[str] = []

    t = now_iso()

    top_level_defaults = {
        "case_type": upgraded.get("case_type", infer_case_type(upgraded.get("case", ""))),
        "created": upgraded.get("created", t),
        "last_updated": upgraded.get("last_updated", t),
        "apf_core_version": APF_CORE_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "output_policy": deepcopy(OUTPUT_POLICY),
        "registry_status": upgraded.get("registry_status", ""),
        "raw_root": upgraded.get("raw_root", ""),
        "notes": upgraded.get("notes", ""),
        "modules": {},
    }

    # Convert legacy framework_version if present, but do not rely on it.
    if "framework_version" in upgraded:
        changes.append("Detected legacy field: framework_version")

    for key, value in top_level_defaults.items():
        if key not in upgraded:
            upgraded[key] = value
            changes.append(f"Added top-level field: {key}")

    # Always update schema/core version to current APF version.
    if upgraded.get("apf_core_version") != APF_CORE_VERSION:
        upgraded["apf_core_version"] = APF_CORE_VERSION
        changes.append("Updated apf_core_version")

    if upgraded.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        upgraded["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
        changes.append("Updated manifest_schema_version")

    # Ensure output policy contains all required entries without removing extras.
    if not isinstance(upgraded.get("output_policy"), dict):
        upgraded["output_policy"] = deepcopy(OUTPUT_POLICY)
        changes.append("Replaced invalid output_policy")
    else:
        for key, value in OUTPUT_POLICY.items():
            if key not in upgraded["output_policy"]:
                upgraded["output_policy"][key] = value
                changes.append(f"Added output_policy entry: {key}")

    if not isinstance(upgraded.get("modules"), dict):
        upgraded["modules"] = {}
        changes.append("Replaced invalid modules object")

    default = default_modules()

    for module_name, default_steps in default.items():
        if module_name not in upgraded["modules"]:
            upgraded["modules"][module_name] = deepcopy(default_steps)
            changes.append(f"Added module: {module_name}")
            continue

        if not isinstance(upgraded["modules"][module_name], dict):
            upgraded["modules"][module_name] = deepcopy(default_steps)
            changes.append(f"Replaced invalid module: {module_name}")
            continue

        for step_name, default_step_state in default_steps.items():
            if step_name not in upgraded["modules"][module_name]:
                upgraded["modules"][module_name][step_name] = deepcopy(default_step_state)
                changes.append(f"Added step: {module_name}/{step_name}")
                continue

            step_state = upgraded["modules"][module_name][step_name]

            if not isinstance(step_state, dict):
                upgraded["modules"][module_name][step_name] = deepcopy(default_step_state)
                changes.append(f"Replaced invalid step: {module_name}/{step_name}")
                continue

            for field, default_value in default_step_state.items():
                if field not in step_state:
                    step_state[field] = deepcopy(default_value)
                    changes.append(
                        f"Added field to step {module_name}/{step_name}: {field}"
                    )

    upgraded["last_updated"] = now_iso()

    return upgraded, changes


def manifest_path(
    case: str,
    *,
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> Path:
    """Return manifest path for one case."""
    paths = paths or get_paths(root)
    return paths.case_manifest(case)


def create_manifest(
    case: str,
    *,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create manifest.json for one case.

    If overwrite=False and a manifest already exists, it is loaded and returned.
    """
    paths = paths or get_paths(root)
    case_entry = get_case(case, case_type=case_type, paths=paths)
    path = paths.case_manifest(case)

    if path.exists() and not overwrite:
        return load_manifest(case, paths=paths)

    manifest = build_manifest(case_entry)
    save_manifest(manifest, paths=paths)
    return manifest


def load_manifest(
    case: str,
    *,
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """Load one case manifest."""
    paths = paths or get_paths(root)
    path = paths.case_manifest(case)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest, changes = upgrade_manifest_schema(manifest)

    if changes:
        save_manifest(manifest, paths=paths)

    return manifest

    #with path.open("r", encoding="utf-8") as f:
    #    return json.load(f)


def save_manifest(
    manifest: dict[str, Any],
    *,
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> Path:
    """Save manifest dictionary to its standard path."""
    paths = paths or get_paths(root)
    case = manifest.get("case")
    if not case:
        raise ValueError("Manifest is missing top-level 'case' field.")

    valid, errors = validate_manifest_schema(manifest)
    if not valid:
        joined = "\n".join(errors)
        raise ValueError(f"Manifest schema validation failed:\n{joined}")

    manifest["last_updated"] = now_iso()

    path = paths.case_manifest(case)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return path


def ensure_manifest(
    case: str,
    *,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """Load manifest if present, otherwise create it."""
    paths = paths or get_paths(root)
    path = paths.case_manifest(case)
    if path.exists():
        return load_manifest(case, paths=paths)
    return create_manifest(case, case_type=case_type, paths=paths)


def get_step(
    manifest: dict[str, Any],
    module: str,
    step: str,
) -> dict[str, Any]:
    """Return a processing step object from a manifest."""
    try:
        return manifest["modules"][module][step]
    except KeyError as exc:
        raise KeyError(f"Unknown manifest module/step: {module}/{step}") from exc


def is_step_valid(
    case: str,
    module: str,
    step: str,
    *,
    algorithm_version: Optional[str] = None,
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> bool:
    """
    Return True if a step is completed and valid.

    If algorithm_version is supplied, the stored algorithm version must also match.
    """
    manifest = load_manifest(case, root=root, paths=paths)
    state = get_step(manifest, module, step)

    if not (state.get("completed") and state.get("valid")):
        return False

    if algorithm_version is not None:
        return state.get("algorithm_version") == algorithm_version

    return True


def update_step(
    case: str,
    module: str,
    step: str,
    *,
    completed: Optional[bool] = None,
    valid: Optional[bool] = None,
    script: Optional[str] = None,
    script_version: Optional[str] = None,
    algorithm_version: Optional[str] = None,
    date_started: Optional[str] = None,
    date_finished: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    command: Optional[str] = None,
    inputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
    qc: Optional[dict[str, Any]] = None,
    warnings: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
    notes: Optional[str] = None,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """
    Update one processing step in one case manifest.

    Only supplied fields are changed.
    """
    paths = paths or get_paths(root)
    manifest = ensure_manifest(case, case_type=case_type, paths=paths)
    state = get_step(manifest, module, step)

    updates = {
        "completed": completed,
        "valid": valid,
        "script": script,
        "script_version": script_version,
        "algorithm_version": algorithm_version,
        "date_started": date_started,
        "date_finished": date_finished,
        "duration_seconds": duration_seconds,
        "command": command,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": parameters,
        "qc": qc,
        "warnings": warnings,
        "errors": errors,
        "notes": notes,
    }

    for key, value in updates.items():
        if value is not None:
            state[key] = value

    save_manifest(manifest, paths=paths)
    return manifest


def reset_step(
    case: str,
    module: str,
    step: str,
    *,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """Reset a step to the empty state."""
    paths = paths or get_paths(root)
    manifest = ensure_manifest(case, case_type=case_type, paths=paths)
    manifest["modules"][module][step] = empty_step()
    save_manifest(manifest, paths=paths)
    return manifest


def mark_step_started(
    case: str,
    module: str,
    step: str,
    *,
    script: str,
    script_version: str,
    algorithm_version: str,
    command: str = "",
    inputs: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """Convenience function for marking a step as started."""
    return update_step(
        case,
        module,
        step,
        completed=False,
        valid=False,
        script=script,
        script_version=script_version,
        algorithm_version=algorithm_version,
        date_started=now_iso(),
        command=command,
        inputs=inputs or [],
        parameters=parameters or {},
        warnings=[],
        errors=[],
        notes="started",
        case_type=case_type,
        root=root,
        paths=paths,
    )


def mark_step_finished(
    case: str,
    module: str,
    step: str,
    *,
    valid: bool,
    outputs: Optional[list[str]] = None,
    qc: Optional[dict[str, Any]] = None,
    warnings: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
    duration_seconds: Optional[float] = None,
    notes: str = "",
    case_type: str = "InS",
    root: Optional[str | Path] = None,
    paths: Optional[APFPaths] = None,
) -> dict[str, Any]:
    """Convenience function for marking a step as finished."""
    return update_step(
        case,
        module,
        step,
        completed=True,
        valid=valid,
        date_finished=now_iso(),
        duration_seconds=duration_seconds,
        outputs=outputs or [],
        qc=qc or {
            "available": False,
            "files": [],
            "summary": {},
        },
        warnings=warnings or [],
        errors=errors or [],
        notes=notes,
        case_type=case_type,
        root=root,
        paths=paths,
    )
