"""Knowledge tools: search GO/Reactome, fetch a term, and annotate gene lists."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gta import sandbox
from gta.rag.index import get_index
from gta.sandbox import ToolError
from gta.tools.enrichment import _background, _load_query, ora_rows, parse_genes
from gta.tools.registry import tool

Source = Literal["go", "reactome", "all"]
_GENES_SHOWN = 40
_ID_FORMAT = re.compile(r"^(GO:\d{7}|R-HSA-\d+)$")


class SearchInput(BaseModel):
    """Arguments for search_knowledge."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        min_length=1, description="Free text, optionally with gene symbols such as TP53 or MDM2"
    )
    source: Source = Field(
        "all", description="Restrict to Gene Ontology ('go'), Reactome ('reactome') or both"
    )
    top_k: int = Field(5, ge=1, le=25, description="Number of hits to return")


class GetTermInput(BaseModel):
    """Arguments for get_term."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(
        description="GO or Reactome id returned by search_knowledge/annotate_genes, e.g. GO:0006915"
    )


class AnnotateInput(BaseModel):
    """Arguments for annotate_genes."""

    model_config = ConfigDict(extra="forbid")
    genes: list[str] | None = Field(None, description="Gene symbols (alternative to gene_list_file)")
    gene_list_file: str | None = Field(
        None, description="Text file with one gene per line, e.g. up_genes.txt"
    )
    source: Source = Field("all", description="Gene sets to test: 'go', 'reactome' or 'all'")
    top_k: int = Field(10, ge=1, le=50, description="Number of top terms to report")
    de_results: str | None = Field(
        None,
        description="results_tsv path from run_deseq2; background = genes tested there that are in the store",
    )
    min_set_size: int = Field(3, ge=1, description="Skip terms with fewer annotated genes")
    max_set_size: int = Field(2000, ge=1, description="Skip terms with more annotated genes")

    @field_validator("genes", mode="before")
    @classmethod
    def _coerce_genes(cls, value: Any) -> list[str] | None:
        return parse_genes(value)


@tool(
    "search_knowledge",
    "Search Gene Ontology terms and Reactome pathways by free text and/or gene symbols. Returns ids, names and "
    "snippets. Use it before making biological claims, and cite only ids returned by tools, e.g. [GO:0006915].",
    SearchInput,
)
def search_knowledge(query: str, source: str = "all", top_k: int = 5) -> dict[str, Any]:
    """Rank knowledge documents for a query."""
    index = get_index()
    hits = index.search(query, source=source, top_k=top_k)
    return {
        "summary": {
            "query": query,
            "source": source,
            "backend": index.backend,
            "query_genes_recognised": index.query_genes(query),
            "n_hits": len(hits),
            "hits": hits,
        },
        "outputs": {},
    }


def _normalise_id(raw: str) -> str:
    term = raw.strip()
    return term.upper() if term.upper().startswith(("GO:", "R-HSA-")) else term


@tool(
    "get_term",
    "Get the full record (name, definition, parents, annotated human genes) for a GO or Reactome id. "
    "Errors for ids that do not exist; never invent ids.",
    GetTermInput,
)
def get_term(id: str) -> dict[str, Any]:
    """Return one knowledge record; the full gene list is written to a file."""
    term_id = _normalise_id(id)
    doc = get_index().by_id.get(term_id)
    if doc is None:
        reason = "unknown id" if _ID_FORMAT.match(term_id) else "not a GO/Reactome id"
        raise ToolError(
            f"Unknown term id ({reason}): {id}",
            "Use search_knowledge to find valid ids; they look like GO:0006915 or R-HSA-109581.",
        )
    out = sandbox.new_run_dir("term")
    (out / f"{term_id.replace(':', '_')}.json").write_text(json.dumps(doc, indent=2))
    parents = [
        {"id": p, "name": get_index().by_id[p]["name"]} for p in doc["parents"] if p in get_index().by_id
    ]
    return {
        "summary": {
            "id": doc["id"],
            "name": doc["name"],
            "source": doc["source"],
            "namespace": doc["namespace"],
            "definition": doc["definition"],
            "parents": parents,
            "n_genes": len(doc["genes"]),
            "genes": doc["genes"][:_GENES_SHOWN],
            "genes_truncated": len(doc["genes"]) > _GENES_SHOWN,
        },
        "outputs": {"record_json": str(out / f"{term_id.replace(':', '_')}.json")},
    }


@tool(
    "annotate_genes",
    "Rank GO terms / Reactome pathways by over-representation of a gene list (one-sided Fisher test, BH-adjusted). "
    "Typically pass the up_genes.txt/down_genes.txt path from run_deseq2 as gene_list_file and its results_tsv as "
    "de_results; do not guess gene names. Cite the returned ids.",
    AnnotateInput,
)
def annotate_genes(
    genes: list[str] | None = None,
    gene_list_file: str | None = None,
    source: str = "all",
    top_k: int = 10,
    de_results: str | None = None,
    min_set_size: int = 3,
    max_set_size: int = 2000,
) -> dict[str, Any]:
    """Enrichment of a gene list against the knowledge store's gene sets."""
    index = get_index()
    docs = [
        d
        for d in index.docs
        if (source == "all" or d["source"] == source) and min_set_size <= len(d["genes"]) <= max_set_size
    ]
    if not docs:
        raise ToolError(
            "No knowledge terms match the size/source filters.", "Relax min_set_size/max_set_size."
        )
    sets = {d["id"]: set(d["genes"]) for d in docs}
    query_input = _load_query(genes, gene_list_file)
    bg, n_bg, bg_source = _background(sets, de_results, None, None)
    query = query_input & bg if bg is not None else query_input
    if not query:
        raise ToolError(
            "None of the query genes are in the knowledge store background.",
            "Use official human gene symbols (e.g. TP53).",
        )
    rows = ora_rows(sets, query, bg, n_bg)
    padj = np.array([r[4] for r in rows])
    out = sandbox.new_run_dir("annotate")
    with open(out / "annotation_results.tsv", "w") as fh:
        fh.write("id\tname\tsource\toverlap\tset_size\tp_value\tpadj\n")
        for tid, k, size, p, q in rows:
            d = index.by_id[tid]
            fh.write(f"{tid}\t{d['name']}\t{d['source']}\t{k}\t{size}\t{p:.6g}\t{q:.6g}\n")
    top = []
    for tid, k, size, p, q in rows[:top_k]:
        d = index.by_id[tid]
        top.append(
            {
                "id": tid,
                "name": d["name"],
                "source": d["source"],
                "overlap": k,
                "set_size": size,
                "p_value": float(f"{p:.4g}"),
                "padj": float(f"{q:.4g}"),
                "overlap_genes": sorted(query & set(d["genes"]))[:8],
            }
        )
    return {
        "summary": {
            "n_query_input": len(query_input),
            "n_query": len(query),
            "background_size": int(n_bg),
            "background_source": "knowledge_store_union" if bg_source == "gmt_union" else bg_source,
            "n_terms_tested": len(rows),
            "n_significant": int((padj < 0.05).sum()),
            "top": top,
        },
        "outputs": {"results_tsv": str(out / "annotation_results.tsv")},
    }
