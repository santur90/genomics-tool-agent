"""Environment checks behind `gta doctor`."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata

PYTHON_DEPS = [
    "openai",
    "pydantic",
    "typer",
    "rich",
    "pandas",
    "numpy",
    "scipy",
    "pydeseq2",
    "pyyaml",
    "rank-bm25",
]
# name -> (version command, install hint)
BINARIES: dict[str, tuple[list[str], str]] = {
    "bcftools": (["bcftools", "--version"], "conda install -c conda-forge -c bioconda bcftools"),
    "samtools": (["samtools", "--version"], "conda install -c conda-forge -c bioconda samtools"),
    "bwa": (["bwa"], "conda install -c conda-forge -c bioconda bwa"),
    "nextflow": (["nextflow", "-version"], "conda install -c conda-forge -c bioconda nextflow"),
    "java": (["java", "-version"], "conda install -c conda-forge openjdk"),
}
ENV_VARS = {
    "LLM_BASE_URL": ("OpenAI-compatible endpoint URL", False),
    "LLM_API_KEY": ("API key (value is never shown)", True),
    "LLM_MODEL": ("model name supporting tool calling", False),
}
_VERSION = re.compile(r"\d+(?:\.\d+)+(?:[-+.\w]*)?")


@dataclass
class Check:
    """One row of the doctor table."""

    kind: str
    name: str
    ok: bool
    detail: str
    hint: str = ""
    required: bool = True


def python_version_check() -> Check:
    """The interpreter itself must be >= 3.11."""
    v = sys.version_info
    ok = v >= (3, 11)
    return Check("python", "python", ok, f"{v.major}.{v.minor}.{v.micro}", "" if ok else "Use Python >= 3.11")


def check_python_deps() -> list[Check]:
    """Installed version of each runtime dependency."""
    rows = []
    for name in PYTHON_DEPS:
        try:
            rows.append(Check("package", name, True, metadata.version(name)))
        except metadata.PackageNotFoundError:
            rows.append(Check("package", name, False, "MISSING", f"pip install {name}"))
    return rows


def _first_version(text: str) -> str:
    """Pick the first version-like token out of a tool's banner."""
    for line in text.splitlines():
        found = _VERSION.search(line)
        if found:
            return found.group(0)
    return "unknown"


def binary_version(cmd: list[str]) -> str:
    """Run a version command (read-only probe) and parse its output; java/bwa print to stderr."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return _first_version((proc.stdout or "") + "\n" + (proc.stderr or ""))


def check_binaries() -> list[Check]:
    """Presence and version of external programs."""
    rows = []
    for name, (cmd, hint) in BINARIES.items():
        if shutil.which(name) is None:
            rows.append(Check("binary", name, False, "MISSING", hint))
            continue
        version = binary_version(cmd)
        if name == "java" and version == "unknown":  # e.g. the macOS /usr/bin/java stub without a JDK
            rows.append(Check("binary", name, False, "MISSING", hint))
        else:
            rows.append(Check("binary", name, True, version))
    return rows


def check_env() -> list[Check]:
    """LLM configuration; needed for `gta run` but not for tools or the mock eval."""
    rows = []
    for name, (desc, secret) in ENV_VARS.items():
        value = os.environ.get(name)
        if value:
            rows.append(Check("env", name, True, "set" if secret else value, required=False))
        else:
            rows.append(Check("env", name, False, "MISSING", f"export {name}=...  # {desc}", required=False))
    return rows


def check_knowledge() -> list[Check]:
    """Presence of the GO/Reactome store; optional, needed by search_knowledge/get_term/annotate_genes."""
    from gta.rag import store

    kdir = store.knowledge_dir()
    ok = (kdir / store.DOCS_FILE).exists() or (kdir / store.REACTOME_TXT).exists()
    hint = "" if ok else "gta fetch-knowledge  # downloads GO + Reactome (or set GTA_KNOWLEDGE_DIR)"
    return [Check("data", "knowledge store", ok, str(kdir) if ok else "MISSING", hint, required=False)]


def run_checks() -> list[Check]:
    """All checks in display order."""
    return [python_version_check(), *check_python_deps(), *check_binaries(), *check_knowledge(), *check_env()]


def exit_code(checks: list[Check]) -> int:
    """1 if any required check (python, packages, binaries) failed; missing LLM env vars only warn."""
    return int(any(not c.ok and c.required for c in checks))
