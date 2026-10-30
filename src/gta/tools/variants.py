"""Variant calling, VCF filtering and summarising via bcftools."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.registry import ToolInput, tool

_QUERY_FMT = "%CHROM\t%POS\t%REF\t%ALT\t%QUAL\t%INFO/DP\n"


class CallVariantsInput(ToolInput):
    """Arguments for call_variants."""

    model_config = ConfigDict(extra="forbid")
    bam: str = Field(description="Sorted BAM file")
    ref: str = Field(description="Reference FASTA")
    min_qual: float = Field(20.0, ge=0, description="Minimum variant QUAL to keep")


class FilterVCFInput(ToolInput):
    """Arguments for filter_vcf."""

    model_config = ConfigDict(extra="forbid")
    vcf: str = Field(description="Input VCF or VCF.gz")
    min_qual: float = Field(0.0, ge=0, description="Minimum QUAL")
    min_dp: int = Field(0, ge=0, description="Minimum INFO/DP")
    gene: str | None = Field(None, description="Keep only records whose INFO/GENE equals this value")

    @field_validator("gene", mode="before")
    @classmethod
    def _normalize_gene(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return None if v.lower() in ("", "null", "none") else v
        return v


class SummarizeVCFInput(ToolInput):
    """Arguments for summarize_vcf."""

    model_config = ConfigDict(extra="forbid")
    vcf: str = Field(description="VCF or VCF.gz to summarise")


def _vcf_has_info(vcf: Path, tag: str) -> bool:
    header = sandbox.run(["bcftools", "view", "-h", str(vcf)], cwd=sandbox.new_run_dir("hdr")).stdout
    return f"##INFO=<ID={tag}," in header


def _count_records(vcf: Path) -> int:
    out = sandbox.run(
        ["bcftools", "query", "-f", "%POS\n", str(vcf)], cwd=sandbox.new_run_dir("count")
    ).stdout
    return len(out.splitlines())


@tool(
    "call_variants",
    "Call variants from a sorted BAM with bcftools mpileup|call; returns a VCF.gz path.",
    CallVariantsInput,
)
def call_variants(bam: str, ref: str, min_qual: float = 20.0) -> dict[str, Any]:
    """Call SNPs/indels and filter by QUAL."""
    bam_p, ref_p = sandbox.safe_path(bam), sandbox.safe_path(ref)
    out = sandbox.new_run_dir("callvar")
    sandbox.run(
        ["bcftools", "mpileup", "-d", "1000", "-f", str(ref_p), "-Ob", "-o", "raw.bcf", str(bam_p)], cwd=out
    )
    sandbox.run(["bcftools", "call", "-mv", "-Ob", "-o", "calls.bcf", "raw.bcf"], cwd=out)
    sandbox.run(
        ["bcftools", "view", "-i", f"QUAL>={min_qual}", "-Oz", "-o", "calls.vcf.gz", "calls.bcf"], cwd=out
    )
    vcf = out / "calls.vcf.gz"
    return {
        "summary": {"n_variants": _count_records(vcf), "min_qual": min_qual},
        "outputs": {"vcf": str(vcf)},
    }


@tool(
    "filter_vcf",
    "Filter a VCF by QUAL, INFO/DP and optionally INFO/GENE; returns the filtered VCF and record counts.",
    FilterVCFInput,
)
def filter_vcf(vcf: str, min_qual: float = 0.0, min_dp: int = 0, gene: str | None = None) -> dict[str, Any]:
    """Filter records with a bcftools include expression."""
    src = sandbox.safe_path(vcf)
    if gene is not None and not _vcf_has_info(src, "GENE"):
        raise ToolError(
            "This VCF has no INFO/GENE annotation.", "Omit the gene argument or use an annotated VCF."
        )
    if gene is not None and not gene.replace("_", "").replace("-", "").isalnum():
        raise ToolError(f"Invalid gene name: {gene}")
    clauses = [f"QUAL>={min_qual}"]
    if min_dp:
        clauses.append(f"INFO/DP>={min_dp}")
    if gene is not None:
        clauses.append(f'INFO/GENE="{gene}"')
    out = sandbox.new_run_dir("filtervcf")
    sandbox.run(
        ["bcftools", "view", "-i", " && ".join(clauses), "-Oz", "-o", "filtered.vcf.gz", str(src)], cwd=out
    )
    n_in, n_out = _count_records(src), _count_records(out / "filtered.vcf.gz")
    return {
        "summary": {"n_input": n_in, "n_output": n_out, "min_qual": min_qual, "min_dp": min_dp, "gene": gene},
        "outputs": {"vcf": str(out / "filtered.vcf.gz")},
    }


@tool(
    "summarize_vcf",
    "Summarise a VCF: total records, counts by variant type (SNP/INDEL) and per gene when INFO/GENE exists.",
    SummarizeVCFInput,
)
def summarize_vcf(vcf: str) -> dict[str, Any]:
    """Count variants by type and gene using bcftools query."""
    src = sandbox.safe_path(vcf)
    has_gene = _vcf_has_info(src, "GENE")
    fmt = "%REF\t%ALT" + ("\t%INFO/GENE" if has_gene else "") + "\n"
    rows = sandbox.run(
        ["bcftools", "query", "-f", fmt, str(src)], cwd=sandbox.new_run_dir("summ")
    ).stdout.splitlines()
    types: Counter[str] = Counter()
    genes: Counter[str] = Counter()
    for row in rows:
        ref, alt, *rest = row.split("\t")
        first_alt = alt.split(",")[0]
        types["SNP" if len(ref) == 1 and len(first_alt) == 1 else "INDEL"] += 1
        if rest:
            genes[rest[0]] += 1
    summary: dict[str, Any] = {"n_variants": len(rows), "by_type": dict(types)}
    if has_gene:
        summary["by_gene"] = dict(genes)
    return {"summary": summary, "outputs": {}}
