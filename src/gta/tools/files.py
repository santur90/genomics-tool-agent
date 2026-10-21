"""File inspection tool."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.registry import tool


class InspectFileInput(BaseModel):
    """Arguments for inspect_file."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Path of a tsv/csv/vcf/vcf.gz/gmt file inside the project")


def read_table(path: Path) -> pd.DataFrame:
    """Read a tsv/csv with the first column as the index."""
    sep = "," if path.suffix == ".csv" else "\t"
    try:
        return pd.read_csv(path, sep=sep, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise ToolError(f"File is empty: {path.name}") from exc


def _inspect_vcf(path: Path) -> dict[str, Any]:
    header = sandbox.run(["bcftools", "view", "-h", str(path)], cwd=sandbox.new_run_dir("inspect")).stdout.splitlines()
    chrom_line = next((h for h in header if h.startswith("#CHROM")), "")
    ids = sandbox.run(["bcftools", "query", "-f", "%CHROM\n", str(path)], cwd=sandbox.new_run_dir("inspect")).stdout.splitlines()
    return {
        "summary": {"type": "vcf", "n_records": len(ids), "n_header_lines": len(header), "columns": chrom_line.lstrip("#").split("\t")},
        "outputs": {},
    }


def _inspect_table(path: Path) -> dict[str, Any]:
    sep = "," if path.suffix == ".csv" else "\t"
    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError:
        return {"summary": {"type": "table", "empty": True}, "outputs": {}}
    head = df.head(5).astype(str).to_dict(orient="records")
    return {
        "summary": {"type": "table", "n_rows": len(df), "n_columns": df.shape[1], "columns": list(df.columns), "head": head},
        "outputs": {},
    }


def _inspect_gmt(path: Path) -> dict[str, Any]:
    names = [ln.split("\t")[0] for ln in path.read_text().splitlines() if ln.strip()]
    return {"summary": {"type": "gmt", "n_sets": len(names), "set_names": names[:20]}, "outputs": {}}


@tool("inspect_file", "Inspect a tsv/csv table (shape, columns, head), a VCF (header and record count), or a GMT file.", InspectFileInput)
def inspect_file(path: str) -> dict[str, Any]:
    """Return basic structure information about a data file."""
    p = sandbox.safe_path(path)
    name = p.name.lower()
    if name.endswith((".vcf", ".vcf.gz")):
        return _inspect_vcf(p)
    if name.endswith((".tsv", ".csv", ".txt")):
        return _inspect_table(p)
    if name.endswith(".gmt"):
        return _inspect_gmt(p)
    raise ToolError(f"Unsupported file type: {p.name}", "Supported: .tsv, .csv, .txt, .vcf, .vcf.gz, .gmt")
