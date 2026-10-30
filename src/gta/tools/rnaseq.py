"""Differential expression with PyDESeq2."""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any

import pandas as pd
from pydantic import ConfigDict, Field

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.files import read_table
from gta.tools.registry import ToolInput, tool


class DESeq2Input(ToolInput):
    """Arguments for run_deseq2."""

    model_config = ConfigDict(extra="forbid")
    counts: str = Field(description="Counts TSV: genes as rows, samples as columns, first column = gene id")
    metadata: str = Field(description="Metadata TSV: first column = sample id, plus the design factor column")
    design_factor: str = Field(description="Metadata column to test, e.g. 'condition'")
    ref_level: str = Field(description="Reference (baseline) level of the factor, e.g. 'control'")
    test_level: str | None = Field(
        None, description="Level compared to the reference; inferred if the factor has two levels"
    )
    padj_cutoff: float = Field(0.05, gt=0, le=1, description="Adjusted p-value cutoff")
    lfc_cutoff: float = Field(1.0, ge=0, description="Absolute log2 fold-change cutoff")


def _build_dds(counts: pd.DataFrame, meta: pd.DataFrame, factor: str):
    from pydeseq2.dds import DeseqDataSet

    try:
        return DeseqDataSet(counts=counts, metadata=meta, design=f"~{factor}", quiet=True)
    except TypeError:  # older pydeseq2 releases
        return DeseqDataSet(counts=counts, metadata=meta, design_factors=factor, quiet=True)


def _fit(counts_t: pd.DataFrame, meta: pd.DataFrame, factor: str, test: str, ref: str) -> pd.DataFrame:
    from pydeseq2.ds import DeseqStats

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The dispersion trend curve fitting did not converge")
        dds = _build_dds(counts_t, meta, factor)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[factor, test, ref], quiet=True)
        stats.summary()
    return stats.results_df


def _prepare(counts: str, metadata: str, factor: str, ref: str, test: str | None):
    counts_df = read_table(sandbox.safe_path(counts))
    meta = read_table(sandbox.safe_path(metadata))
    if factor not in meta.columns:
        raise ToolError(
            f"Column '{factor}' not found in metadata.", f"Available columns: {list(meta.columns)}"
        )
    samples = [s for s in counts_df.columns if s in meta.index]
    if len(samples) < 4:
        raise ToolError(
            "Fewer than 4 samples shared between counts and metadata.",
            "Sample ids in metadata must match count columns.",
        )
    meta = meta.loc[samples, [factor]].astype(str)
    levels = sorted(meta[factor].unique())
    if ref not in levels:
        raise ToolError(f"ref_level '{ref}' not in column '{factor}'.", f"Levels: {levels}")
    others = [lv for lv in levels if lv != ref]
    if test is None:
        if len(others) != 1:
            raise ToolError(
                "test_level is required when the factor has more than two levels.", f"Levels: {levels}"
            )
        test = others[0]
    if test not in levels:
        raise ToolError(f"test_level '{test}' not in column '{factor}'.", f"Levels: {levels}")
    counts_t = counts_df[samples].T.round().astype(int)
    counts_t = counts_t.loc[:, counts_t.sum(axis=0) >= 10]
    return counts_t, meta, test


def _sig(x: float) -> float:
    return float(f"{x:.4g}")


@tool(
    "run_deseq2",
    "Run RNA-seq differential expression (PyDESeq2) and report up/down-regulated gene counts and top genes.",
    DESeq2Input,
)
def run_deseq2(
    counts: str,
    metadata: str,
    design_factor: str,
    ref_level: str,
    test_level: str | None = None,
    padj_cutoff: float = 0.05,
    lfc_cutoff: float = 1.0,
) -> dict[str, Any]:
    """Run DESeq2 and write results plus up/down gene lists."""
    counts_t, meta, test = _prepare(counts, metadata, design_factor, ref_level, test_level)
    res = _fit(counts_t, meta, design_factor, test, ref_level)
    sig = res.dropna(subset=["padj"])
    sig = sig[sig["padj"] < padj_cutoff]
    up = sig[sig["log2FoldChange"] >= lfc_cutoff].sort_values("log2FoldChange", ascending=False)
    down = sig[sig["log2FoldChange"] <= -lfc_cutoff].sort_values("log2FoldChange")
    out = sandbox.new_run_dir("deseq2")
    res.to_csv(out / "deseq2_results.tsv", sep="\t", index_label="gene")
    (out / "up_genes.txt").write_text("\n".join(up.index) + ("\n" if len(up) else ""))
    (out / "down_genes.txt").write_text("\n".join(down.index) + ("\n" if len(down) else ""))
    top = res.dropna(subset=["padj"]).sort_values("padj").head(10)
    top_genes = [
        {"gene": g, "log2FoldChange": _sig(r.log2FoldChange), "padj": _sig(r.padj)} for g, r in top.iterrows()
    ]
    return {
        "summary": {
            "contrast": f"{design_factor}: {test} vs {ref_level}",
            "n_genes_tested": int(len(res.dropna(subset=["padj"]))),
            "padj_cutoff": padj_cutoff,
            "lfc_cutoff": lfc_cutoff,
            "n_up": int(len(up)),
            "n_down": int(len(down)),
            "top_genes": top_genes,
        },
        "outputs": {
            "results_tsv": str(out / "deseq2_results.tsv"),
            "up_genes": str(out / "up_genes.txt"),
            "down_genes": str(out / "down_genes.txt"),
        },
    }
