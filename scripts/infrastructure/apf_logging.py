#!/usr/bin/env python3
"""
ANISO Processing Framework (APF)
apf_logging.py

Purpose
-------
Provide standardized APF logging for infrastructure and processing scripts.

Responsibilities
----------------
- Write timestamped per-case log files
- Mirror messages to terminal and log file
- Track start/finish time and duration
- Collect warnings and errors
- Provide section headers and timed blocks
- Export metadata compatible with APF manifests
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Iterator, Optional

from apf_paths import APFPaths, get_paths


LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FILENAME_TIME_FORMAT = "%Y%m%d_%H%M%S"


def now_iso() -> str:
    """Return local timestamp in ISO-like format without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat()


class APFLogger:
    """
    APF execution logger.

    One APFLogger instance represents one script execution for one case and one
    processing step.
    """

    def __init__(
        self,
        *,
        case: str,
        module: str,
        step: str,
        script: str,
        script_version: str = "",
        algorithm_version: str = "",
        command: str = "",
        paths: Optional[APFPaths] = None,
        root: Optional[str | Path] = None,
        echo: bool = True,
    ) -> None:
        self.case = case
        self.module = module
        self.step = step
        self.script = script
        self.script_version = script_version
        self.algorithm_version = algorithm_version
        self.command = command

        self.paths = paths or get_paths(root)
        self.echo = echo

        self.start_time = datetime.now().replace(microsecond=0)
        self.finish_time: Optional[datetime] = None
        self.duration_seconds: Optional[float] = None

        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.outputs: list[str] = []

        self.hostname = socket.gethostname()
        self.python_version = sys.version.replace("\n", " ")
        self.platform = platform.platform()

        self.log_dir = (
            self.paths.derivatives_dir
            / self.case
            / "processing"
            / "logs"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        stamp = self.start_time.strftime(FILENAME_TIME_FORMAT)
        safe_module = self._safe_name(self.module)
        safe_step = self._safe_name(self.step)
        self.log_path = self.log_dir / f"{stamp}_{safe_module}_{safe_step}.log"

        self._file = self.log_path.open("w", encoding="utf-8")

        self._write_header()

    @staticmethod
    def _safe_name(text: str) -> str:
        """Return filesystem-safe string."""
        return (
            text.replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )

    def _timestamp(self) -> str:
        return datetime.now().strftime(LOG_TIME_FORMAT)

    def _write(self, level: str, message: str) -> None:
        line = f"{self._timestamp()} [{level}] {message}"

        if self.echo:
            print(line)

        self._file.write(line + "\n")
        self._file.flush()

    def _write_header(self) -> None:
        self._file.write("=" * 70 + "\n")
        self._file.write("APF execution log\n")
        self._file.write("=" * 70 + "\n")
        self._file.write(f"Case:              {self.case}\n")
        self._file.write(f"Module:            {self.module}\n")
        self._file.write(f"Step:              {self.step}\n")
        self._file.write(f"Script:            {self.script}\n")
        self._file.write(f"Script version:    {self.script_version}\n")
        self._file.write(f"Algorithm version: {self.algorithm_version}\n")
        self._file.write(f"Command:           {self.command}\n")
        self._file.write(f"Started:           {self.start_time.isoformat()}\n")
        self._file.write(f"Hostname:          {self.hostname}\n")
        self._file.write(f"Platform:          {self.platform}\n")
        self._file.write(f"Python:            {self.python_version}\n")
        self._file.write("=" * 70 + "\n\n")
        self._file.flush()

        if self.echo:
            print("=" * 70)
            print("APF execution log")
            print("=" * 70)
            print(f"Case:   {self.case}")
            print(f"Module: {self.module}")
            print(f"Step:   {self.step}")
            print(f"Script: {self.script}")
            print(f"Log:    {self.log_path}")
            print("=" * 70)

    def info(self, message: str) -> None:
        """Log an informational message."""
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        """Log a warning and store it."""
        self.warnings.append(message)
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        """Log an error and store it."""
        self.errors.append(message)
        self._write("ERROR", message)

    def add_output(self, path: str | Path) -> None:
        """Register an output file produced by the script."""
        self.outputs.append(str(path))

    def add_outputs(self, *paths: str | Path) -> None:
        """Register multiple output files."""
        for path in paths:
            self.add_output(path)

    def exception(self, message: str, exc: BaseException) -> None:
        """Log an exception with traceback and store it as an error."""
        error_message = f"{message}: {exc}"
        self.errors.append(error_message)
        self._write("ERROR", error_message)

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for line in tb.rstrip().splitlines():
            self._write("TRACEBACK", line)

    def section(self, title: str) -> None:
        """Write a visual section header."""
        border = "=" * 70
        if self.echo:
            print(border)
            print(title)
            print(border)

        self._file.write(border + "\n")
        self._file.write(title + "\n")
        self._file.write(border + "\n")
        self._file.flush()

    @contextmanager
    def timer(self, label: str) -> Iterator[None]:
        """Context manager for timed blocks."""
        self.info(f"{label} started")
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            raise
        finally:
            dt = time.perf_counter() - t0
            self.info(f"{label} finished in {dt:.2f} s")

    def metadata(self) -> dict[str, Any]:
        """Return execution metadata suitable for JSON or manifest usage."""
        return {
            "case": self.case,
            "module": self.module,
            "step": self.step,
            "script": self.script,
            "script_version": self.script_version,
            "algorithm_version": self.algorithm_version,
            "command": self.command,
            "log_file": str(self.log_path),
            "date_started": self.start_time.isoformat(),
            "date_finished": (
                self.finish_time.isoformat() if self.finish_time else ""
            ),
            "duration_seconds": self.duration_seconds,
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "outputs": list(self.outputs),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def manifest_fields(self) -> dict[str, Any]:
        """
        Return fields directly compatible with apf_manifest.update_step().

        This dictionary can be unpacked directly into update_step().
        """
        return {
            "script": self.script,
            "script_version": self.script_version,
            "algorithm_version": self.algorithm_version,
            "date_started": self.start_time.isoformat(),
            "date_finished": (
                self.finish_time.isoformat() if self.finish_time else ""
            ),
            "duration_seconds": self.duration_seconds,
            "command": self.command,
            "outputs": list(self.outputs),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "log_file": str(self.log_path),
        }

    def finish(self, *, success: bool = True) -> None:
        """Finish the log and close the file."""
        self.finish_time = datetime.now().replace(microsecond=0)
        self.duration_seconds = (
            self.finish_time - self.start_time
        ).total_seconds()

        self._file.write("\n")
        self._file.write("=" * 70 + "\n")
        self._file.write("APF execution finished\n")
        self._file.write("=" * 70 + "\n")
        self._file.write(f"Finished:          {self.finish_time.isoformat()}\n")
        self._file.write(f"Duration seconds:  {self.duration_seconds:.2f}\n")
        self._file.write(f"Success:           {success}\n")
        self._file.write(f"Warnings:          {len(self.warnings)}\n")
        self._file.write(f"Errors:            {len(self.errors)}\n")
        self._file.write("=" * 70 + "\n")
        self._file.flush()

        if self.echo:
            print("=" * 70)
            print("APF execution finished")
            print("=" * 70)
            print(f"Success:  {success}")
            print(f"Duration: {self.duration_seconds:.2f} s")
            print(f"Warnings: {len(self.warnings)}")
            print(f"Errors:   {len(self.errors)}")
            print(f"Log:      {self.log_path}")

        self.close()

    def close(self) -> None:
        """Close the log file if open."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "APFLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.exception("Unhandled exception", exc)
            self.finish(success=False)
            return False

        self.finish(success=True)
        return False