"""Run allowlisted Nextflow pipelines."""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.registry import tool

CONFIG_PATH = sandbox.PROJECT_ROOT / "config" / "pipelines.yaml"
_PARAM_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class NextflowInput(BaseModel):
    """Arguments for run_nextflow."""

    model_config = ConfigDict(extra="forbid")
    pipeline: str = Field(description="Pipeline name from config/pipelines.yaml (not a path)")
    params: dict[str, str] = Field(
        default_factory=dict, description="Pipeline parameters, e.g. {'counts': 'data/rnaseq/counts.tsv'}"
    )
    stub: bool = Field(True, description="Use -stub-run (fast dry run without real computation)")


def load_pipelines() -> dict[str, dict[str, Any]]:
    """Read the pipeline allowlist."""
    return yaml.safe_load(CONFIG_PATH.read_text()).get("pipelines", {})


def _build_params(spec: dict[str, Any], params: dict[str, str]) -> list[str]:
    allowed = set(spec.get("allowed_params", []))
    path_params = set(spec.get("path_params", []))
    args: list[str] = []
    for key, value in params.items():
        if not _PARAM_KEY.match(key) or key not in allowed:
            raise ToolError(f"Parameter not allowed: {key}", f"Allowed parameters: {sorted(allowed)}")
        if key == "outdir":
            raise ToolError("outdir is managed by the tool.")
        value = str(sandbox.safe_path(value)) if key in path_params else str(value)
        if value.startswith("-"):
            raise ToolError(f"Invalid value for {key}")
        args += [f"--{key}", value]
    return args


@tool(
    "run_nextflow",
    "Run a registered Nextflow pipeline by name (stub mode by default) and list its output files.",
    NextflowInput,
)
def run_nextflow(pipeline: str, params: dict[str, str] | None = None, stub: bool = True) -> dict[str, Any]:
    """Run an allowlisted pipeline in a run folder."""
    pipelines = load_pipelines()
    if pipeline not in pipelines:
        raise ToolError(f"Unknown pipeline: {pipeline}", f"Registered pipelines: {sorted(pipelines)}")
    spec = pipelines[pipeline]
    script = sandbox.safe_path(spec["path"])
    extra = _build_params(spec, params or {})
    out = sandbox.new_run_dir(f"nf-{pipeline}")
    cmd = [
        "nextflow",
        "run",
        str(script),
        "-ansi-log",
        "false",
        "-work-dir",
        str(out / "work"),
        "--outdir",
        str(out / "results"),
    ]
    if stub:
        cmd.append("-stub-run")
    proc = sandbox.run(cmd + extra, cwd=out, timeout=900)
    files = sorted(str(p) for p in (out / "results").rglob("*") if p.is_file())
    return {
        "summary": {
            "pipeline": pipeline,
            "stub": stub,
            "returncode": proc.returncode,
            "n_output_files": len(files),
        },
        "outputs": {"outdir": str(out / "results"), "files": files},
    }
