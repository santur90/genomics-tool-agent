"""Over-representation analysis (hypergeometric test + BH)."""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import hypergeom

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.registry import tool


class ORAInput(BaseModel):
    """Arguments for run_ora."""

    model_config = ConfigDict(extra="forbid")
    gmt: str = Field(description="Path to a GMT gene-set file")
    genes: list[str] | None = Field(None, description="Query gene symbols (alternative to gene_list_file)")
    gene_list_file: str | None = Field(None, description="Text file with one gene per line, e.g. an up_genes.txt output")
    background_size: int | None = Field(None, ge=1, description="Gene universe size; defaults to the union of GMT genes and the query")


def read_gmt(path: str) -> dict[str, set[str]]:
    """Parse a GMT file into {set name: genes}."""
    sets: dict[str, set[str]] = {}
    for line in sandbox.safe_path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = {g for g in parts[2:] if g}
    if not sets:
        raise ToolError(f"No gene sets found in {path}", "GMT format: name<TAB>description<TAB>gene1<TAB>gene2...")
    return sets


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def _load_query(genes: list[str] | None, gene_list_file: str | None) -> set[str]:
    if (genes is None) == (gene_list_file is None):
        raise ToolError("Provide exactly one of 'genes' or 'gene_list_file'.")
    if genes is not None:
        query = set(genes)
    else:
        lines = sandbox.safe_path(gene_list_file).read_text().splitlines()
        query = {ln.split("\t")[0].strip() for ln in lines if ln.strip() and not ln.startswith("#")}
    if not query:
        raise ToolError("The query gene list is empty.", "Relax the DE thresholds or provide genes.")
    return query


@tool("run_ora", "Over-representation (enrichment) analysis of a gene list against a GMT file; hypergeometric test with BH correction.", ORAInput)
def run_ora(gmt: str, genes: list[str] | None = None, gene_list_file: str | None = None, background_size: int | None = None) -> dict[str, Any]:
    """Test each gene set for enrichment and return ranked results."""
    sets = read_gmt(gmt)
    query = _load_query(genes, gene_list_file)
    universe = set().union(*sets.values()) | query
    n_bg = background_size or len(universe)
    if len(query) > n_bg:
        raise ToolError("background_size is smaller than the query gene list.")
    names = list(sets)
    k = np.array([len(query & sets[s]) for s in names])
    size = np.array([len(sets[s]) for s in names])
    pvals = hypergeom.sf(k - 1, n_bg, np.minimum(size, n_bg), len(query))
    padj = bh_adjust(pvals)
    rows = sorted(zip(names, k, size, pvals, padj), key=lambda r: (r[4], r[3]))
    out = sandbox.new_run_dir("ora")
    with open(out / "ora_results.tsv", "w") as fh:
        fh.write("set\toverlap\tset_size\tp_value\tpadj\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6g}\t{r[4]:.6g}\n")
    top = [{"set": r[0], "overlap": int(r[1]), "set_size": int(r[2]), "p_value": float(f"{r[3]:.4g}"), "padj": float(f"{r[4]:.4g}")} for r in rows[:5]]
    return {
        "summary": {"n_query": len(query), "background_size": int(n_bg), "n_sets_tested": len(names),
                    "n_significant": int((padj < 0.05).sum()), "top": top},
        "outputs": {"results_tsv": str(out / "ora_results.tsv")},
    }
