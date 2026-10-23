#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
apf_paths.py

Purpose
-------
Centralize project path handling for APF scripts.

Scientific scripts should not hardcode project paths internally. They should
obtain project paths through this module.

Version
-------
APF core version: 0.2
Created: 2026-06-17
Updated: 2026-07-16
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# APF project root
#
# Override on another machine or project location with:
#
#     export APF_ROOT=/path/to/ANISO_processing
#
# ---------------------------------------------------------------------
DEFAULT_ROOT = Path(
    os.environ.get("APF_ROOT")
    or Path(__file__).resolve().parents[2]
).expanduser().resolve()

@dataclass(frozen=True)
class APFPaths:
    """Container for standard ANISO Processing Framework paths."""

    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def infrastructure_dir(self) -> Path:
        return self.scripts_dir / "infrastructure"

    @property
    def raw_dicom_dir(self) -> Path:
        return self.root / "raw_dicom"

    @property
    def raw_study_dir(self) -> Path:
        return self.raw_dicom_dir / "03_Studiendaten"

    @property
    def nifti_dir(self) -> Path:
        return self.root / "nifti"

    @property
    def derivatives_dir(self) -> Path:
        return self.root / "derivatives"

    @property
    def summary_dir(self) -> Path:
        return self.derivatives_dir / "summary"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def ins_registry(self) -> Path:
        return self.config_dir / "InS_cases.csv"

    @property
    def exs_registry(self) -> Path:
        return self.config_dir / "ExS_cases.csv"

    @property
    def pipeline_config(self) -> Path:
        return self.config_dir / "pipeline_config.json"

    def case_raw_dir(self, case: str) -> Path:
        """Historical cohort location."""
        return self.raw_study_dir / case

    def case_addcase_dir(self, case: str) -> Path:
        """New cases awaiting integration."""
        return self.raw_study_dir / "addCase" / case

    def find_case_raw_dir(self, case: str) -> Path | None:
        """Return the raw DICOM directory for a case.

        Checks the historical cohort location first and then the addCase
        staging area.
        """
        if self.case_raw_dir(case).is_dir():
            return self.case_raw_dir(case)

        if self.case_addcase_dir(case).is_dir():
            return self.case_addcase_dir(case)

        return None

    def case_nifti_dir(self, case: str) -> Path:
        return self.nifti_dir / case

    def case_derivatives_dir(self, case: str) -> Path:
        return self.derivatives_dir / case

    def case_processing_dir(self, case: str) -> Path:
        return self.case_derivatives_dir(case) / "processing"

    def case_manifest(self, case: str) -> Path:
        return self.case_processing_dir(case) / "manifest.json"

    def case_logs_dir(self, case: str) -> Path:
        return self.case_derivatives_dir(case) / "logs"

    def case_qc_dir(self, case: str) -> Path:
        return self.case_derivatives_dir(case) / "qc"

    # -----------------------------------------------------------------
    # Generic module/stage helpers
    #
    # These are additive and backward-compatible. Existing scripts that
    # use summary_dir or case_derivatives_dir continue to work unchanged.
    # -----------------------------------------------------------------
    def summary_module_dir(self, module: str) -> Path:
        """Return derivatives/summary/<module>."""
        return self.summary_dir / module

    def case_module_dir(self, case: str, module: str) -> Path:
        """Return derivatives/<case>/<module>."""
        return self.case_derivatives_dir(case) / module

    def case_stage_dir(
        self,
        case: str,
        module: str,
        stage: str,
    ) -> Path:
        """Return derivatives/<case>/<module>/<stage>."""
        return self.case_module_dir(case, module) / stage


def get_paths(root: Optional[str | Path] = None) -> APFPaths:
    """Return APFPaths for the given root or the default project root."""
    root_path = (
        Path(root).expanduser().resolve()
        if root is not None
        else DEFAULT_ROOT
    )
    return APFPaths(root=root_path)


def ensure_base_dirs(paths: APFPaths) -> None:
    """Create APF base directories if they do not already exist."""
    for directory in [
        paths.config_dir,
        paths.scripts_dir,
        paths.infrastructure_dir,
        paths.raw_dicom_dir,
        paths.nifti_dir,
        paths.derivatives_dir,
        paths.summary_dir,
        paths.results_dir,
        paths.figures_dir,
        paths.docs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
