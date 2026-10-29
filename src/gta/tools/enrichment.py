"""Over-representation analysis (hypergeometric test + BH)."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.stats import hypergeom

from gta import sandbox
from gta.sandbox import ToolError
from gta.tools.files import read_table
from gta.tools.registry import tool


class ORAInput(BaseModel):
    """Arguments for run_ora."""

    model_config = ConfigDict(extra="forbid")
    gmt: str = Field(description="Path to a GMT gene-set file")
    genes: list[str] | None = Field(None, description="Query gene symbols (alternative to gene_list_file)")
    gene_list_file: str | None = Field(
        None, description="Text file with one gene per line, e.g. an up_genes.txt output"
    )
    de_results: str | None = Field(
        None,
        description="results_tsv path returned by run_deseq2; the background becomes the genes tested there that are also in the GMT",
    )
    universe: list[str] | None = Field(
        None,
        description="Explicit background gene list (overrides de_results); intersected with the GMT genes",
    )
    background_size: int | None = Field(
        None,
        ge=1,
        description="Explicit background size only (overrides everything; no gene-level restriction)",
    )

    @field_validator("genes", "universe", mode="before")
    @classmethod
    def _coerce_genes(cls, value: Any) -> list[str] | None:
        """Accept a list, a JSON list string, or a comma/newline/space separated string."""
        return parse_genes(value)


def parse_genes(value: Any) -> list[str] | None:
    """Normalise gene input to a list of names (None if empty)."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError:
                value = re.split(r"[\s,;\[\]\"']+", text)
        else:
            value = re.split(r"[\s,;]+", text)
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("genes must be a list or a delimited string")
    genes = [str(g).strip() for g in value if str(g).strip()]
    return genes or None


def read_gmt(path: str) -> dict[str, set[str]]:
    """Parse a GMT file into {set name: genes}."""
    sets: dict[str, set[str]] = {}
    for line in sandbox.safe_path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = {g for g in parts[2:] if g}
    if not sets:
        raise ToolError(
            f"No gene sets found in {path}", "GMT format: name<TAB>description<TAB>gene1<TAB>gene2..."
        )
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


def ora_rows(
    sets: dict[str, set[str]], query: set[str], bg: set[str] | None, n_bg: int
) -> list[tuple[str, int, int, float, float]]:
    """One-sided Fisher exact (hypergeometric) test per set with BH correction.

    Returns (set, overlap, set_size, p_value, padj) tuples sorted by padj then p-value.
    """
    members = {s: (g & bg if bg is not None else g) for s, g in sets.items()}
    names = list(sets)
    k = np.array([len(query & members[s]) for s in names])
    size = np.array([len(members[s]) for s in names])
    pvals = hypergeom.sf(k - 1, n_bg, np.minimum(size, n_bg), len(query))
    padj = bh_adjust(pvals)
    rows = zip(names, k.tolist(), size.tolist(), pvals.tolist(), padj.tolist(), strict=True)
    return sorted(rows, key=lambda r: (r[4], r[3]))


def _load_query(genes: list[str] | None, gene_list_file: str | None) -> set[str]:
    if not genes and not gene_list_file:
        raise ToolError(
            "Provide 'genes' or 'gene_list_file'.",
            "Use the up_genes.txt/down_genes.txt path returned by run_deseq2 as gene_list_file.",
        )
    query = set(genes or [])
    if gene_list_file:
        lines = sandbox.safe_path(gene_list_file).read_text().splitlines()
        query |= {ln.split("\t")[0].strip() for ln in lines if ln.strip() and not ln.startswith("#")}
    if not query:
        raise ToolError("The query gene list is empty.", "Relax the DE thresholds or provide genes.")
    return query


def _background(
    sets: dict[str, set[str]], de_results: str | None, universe: list[str] | None, background_size: int | None
) -> tuple[set[str] | None, int, str]:
    """Return (gene-level background or None, background size, source label)."""
    gmt_genes = set().union(*sets.values())
    if background_size:
        return None, background_size, "user_background_size"
    if universe:
        bg = set(universe) & gmt_genes
        return bg, len(bg), "user_universe"
    if de_results:
        df = read_table(sandbox.safe_path(de_results))
        if "pvalue" not in df.columns:
            raise ToolError(
                "de_results has no 'pvalue' column.", "Pass the results_tsv path returned by run_deseq2."
            )
        bg = set(df.index[df["pvalue"].notna()].astype(str)) & gmt_genes
        return bg, len(bg), "de_results_tested_intersect_gmt"
    return gmt_genes, len(gmt_genes), "gmt_union"


@tool(
    "run_ora",
    "Over-representation test. Typically use the up_genes.txt/down_genes.txt path returned by run_deseq2 as gene_list_file, "
    "and pass the results_tsv path from the same run_deseq2 call as de_results so the background is the set of tested genes; "
    "do not guess gene names.",
    ORAInput,
)
def run_ora(
    gmt: str,
    genes: list[str] | None = None,
    gene_list_file: str | None = None,
    de_results: str | None = None,
    universe: list[str] | None = None,
    background_size: int | None = None,
) -> dict[str, Any]:
    """Test each gene set for enrichment and return ranked results."""
    sets = read_gmt(gmt)
    query_input = _load_query(genes, gene_list_file)
    bg, n_bg, source = _background(sets, de_results, universe, background_size)
    query = query_input & bg if bg is not None else query_input
    if not query:
        raise ToolError(
            "None of the query genes are in the background.",
            "Check gene identifiers against the GMT and the DE results.",
        )
    if len(query) > n_bg:
        raise ToolError("background_size is smaller than the query gene list.")
    rows = ora_rows(sets, query, bg, n_bg)
    names = [r[0] for r in rows]
    padj = np.array([r[4] for r in rows])
    out = sandbox.new_run_dir("ora")
    with open(out / "ora_results.tsv", "w") as fh:
        fh.write("set\toverlap\tset_size\tp_value\tpadj\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6g}\t{r[4]:.6g}\n")
    top = [
        {
            "set": r[0],
            "overlap": int(r[1]),
            "set_size": int(r[2]),
            "p_value": float(f"{r[3]:.4g}"),
            "padj": float(f"{r[4]:.4g}"),
        }
        for r in rows[:5]
    ]
    return {
        "summary": {
            "n_query_input": len(query_input),
            "n_query": len(query),
            "background_size": int(n_bg),
            "background_source": source,
            "n_sets_tested": len(names),
            "n_significant": int((padj < 0.05).sum()),
            "top": top,
        },
        "outputs": {"results_tsv": str(out / "ora_results.tsv")},
    }
