"""Restricted subprocess execution and path safety helpers."""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

ALLOWED_COMMANDS = frozenset({"bcftools", "samtools", "bwa", "wgsim", "nextflow"})
PROJECT_ROOT = Path(os.environ.get("GTA_ROOT") or Path(__file__).resolve().parents[2]).resolve()
RUNS_DIR = PROJECT_ROOT / "runs"
WORK_DIR = RUNS_DIR / "work"
_BLOCKED_PARTS = {".env", ".git"}


class ToolError(Exception):
    """A recoverable error that is reported back to the LLM as structured data."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class SandboxError(ToolError):
    """Raised when a command or path violates sandbox policy."""


@dataclass
class RunResult:
    """Outcome of a sandboxed command."""

    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def new_run_dir(prefix: str) -> Path:
    """Create and return a fresh folder under runs/work/."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = WORK_DIR / f"{prefix}-{stamp}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path(path: str | os.PathLike[str], *, must_exist: bool = True) -> Path:
    """Resolve a user-supplied path and ensure it stays inside the project."""
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.is_relative_to(PROJECT_ROOT):
        raise SandboxError(f"Path is outside the project: {path}", "Only files inside the project directory are accessible.")
    if _BLOCKED_PARTS & set(p.relative_to(PROJECT_ROOT).parts):
        raise SandboxError(f"Access to this path is not allowed: {path}", "Secrets and git internals are off limits.")
    if must_exist and not p.exists():
        raise ToolError(f"File not found: {path}", "Check the path; data files live under data/.")
    return p


def run(cmd: Sequence[str], *, cwd: Path | str | None = None, timeout: int = 300, check: bool = True) -> RunResult:
    """Run an allowlisted command without a shell, inside a run folder."""
    if not cmd or not all(isinstance(c, str) for c in cmd):
        raise SandboxError("Command must be a non-empty list of strings.")
    exe = cmd[0]
    if exe not in ALLOWED_COMMANDS:
        raise SandboxError(f"Command not allowed: {exe}", f"Allowed commands: {sorted(ALLOWED_COMMANDS)}")
    resolved = shutil.which(exe)
    if resolved is None:
        raise SandboxError(f"Executable not installed: {exe}", "Install it via conda (bioconda channel).")
    workdir = Path(cwd).resolve() if cwd else new_run_dir(exe)
    if not workdir.is_relative_to(RUNS_DIR):
        raise SandboxError(f"Working directory must be inside {RUNS_DIR}")
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([resolved, *cmd[1:]], cwd=workdir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"{exe} timed out after {timeout}s", "Try a smaller input.") from exc
    result = RunResult(list(cmd), proc.returncode, proc.stdout, proc.stderr)
    if check and proc.returncode != 0:
        raise ToolError(f"{exe} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}", "Check input files and parameters.")
    return result
