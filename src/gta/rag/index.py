"""Search index over knowledge documents: BM25 (default) or embeddings, plus gene-overlap boosting."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gta.rag import store
from gta.sandbox import ToolError

GENE_BOOST = 0.5  # added to the normalised text score per fraction of query genes found in a document
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_WORD = re.compile(r"[a-z0-9]+")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
Encoder = Callable[[Sequence[str]], np.ndarray]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def doc_tokens(doc: dict[str, Any]) -> list[str]:
    """BM25 tokens: name (double weight), definition, and genes as g_<symbol> tokens."""
    name = _words(doc["name"])
    return name + name + _words(doc.get("definition", "")) + ["g_" + g.lower() for g in doc["genes"]]


def doc_text(doc: dict[str, Any], max_genes: int = 20) -> str:
    """Compact text used for embeddings."""
    return f"{doc['name']}. {doc.get('definition', '')} Genes: {' '.join(doc['genes'][:max_genes])}"


def snippet(doc: dict[str, Any], limit: int = 160) -> str:
    """Short human-readable description of a document."""
    text = doc.get("definition") or doc.get("description") or ""
    if not text:
        shown = ", ".join(doc["genes"][:6])
        text = f"{doc['namespace']} with {len(doc['genes'])} human genes" + (
            f" (e.g. {shown})" if shown else ""
        )
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def sentence_transformer_encoder(model_name: str | None = None) -> Encoder:
    """Embedding encoder backed by sentence-transformers (the optional [rag] extra)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ToolError(
            "GTA_RAG_BACKEND=embed needs sentence-transformers.",
            'Install it with: pip install "genomics-tool-agent[rag]" (or use GTA_RAG_BACKEND=bm25).',
        ) from exc
    model = SentenceTransformer(model_name or os.environ.get("GTA_RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL)
    return lambda texts: np.asarray(
        model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    )


class KnowledgeIndex:
    """Ranks documents for a free-text query; gene symbols in the query boost documents containing them."""

    def __init__(
        self,
        docs: list[dict[str, Any]],
        backend: str = "bm25",
        encoder: Encoder | None = None,
        cache_file: Path | None = None,
    ) -> None:
        if backend not in ("bm25", "embed"):
            raise ToolError(f"Unknown GTA_RAG_BACKEND: {backend}", "Use 'bm25' or 'embed'.")
        self.docs = docs
        self.backend = backend
        self.by_id = {d["id"]: d for d in docs}
        self.gene_sets = [set(d["genes"]) for d in docs]
        self.vocab = set().union(*self.gene_sets) if docs else set()
        self._bm25 = None
        self._vectors: np.ndarray | None = None
        self._encode: Encoder | None = None
        if backend == "bm25":
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([doc_tokens(d) or ["_"] for d in docs])
        else:
            self._encode = encoder or sentence_transformer_encoder()
            self._vectors = self._load_vectors(cache_file)

    def _load_vectors(self, cache_file: Path | None) -> np.ndarray:
        if cache_file is not None and cache_file.exists():
            cached = np.load(cache_file)
            if len(cached) == len(self.docs):
                return cached
        vectors = np.asarray(self._encode([doc_text(d) for d in self.docs]), dtype=float)
        if cache_file is not None:
            np.save(cache_file, vectors)
        return vectors

    def query_genes(self, query: str) -> list[str]:
        """Tokens of the query that are exact (case-sensitive) gene symbols in the knowledge store."""
        seen: list[str] = []
        for tok in _SYMBOL.findall(query):
            if tok in self.vocab and tok not in seen:
                seen.append(tok)
        return seen

    def _text_scores(self, query: str, genes: list[str]) -> np.ndarray:
        if self._bm25 is not None:
            tokens = _words(query) + ["g_" + g.lower() for g in genes]
            return np.asarray(self._bm25.get_scores(tokens), dtype=float)
        vec = np.asarray(self._encode([query]), dtype=float)[0]
        return np.clip(self._vectors @ vec, 0.0, None)

    def search(self, query: str, source: str = "all", top_k: int = 5) -> list[dict[str, Any]]:
        """Top documents as {id, name, source, snippet, score, matched_genes}."""
        genes = self.query_genes(query)
        text = self._text_scores(query, genes)
        top = text.max() if len(text) else 0.0
        score = text / top if top > 0 else np.zeros(len(self.docs))
        if genes:
            matched = np.array([len(set(genes) & gs) / len(genes) for gs in self.gene_sets])
            score = score + GENE_BOOST * matched
        order = np.argsort(-score, kind="stable")
        hits = []
        for i in order:
            doc = self.docs[i]
            if score[i] <= 0:
                break
            if source != "all" and doc["source"] != source:
                continue
            hits.append(
                {
                    "id": doc["id"],
                    "name": doc["name"],
                    "source": doc["source"],
                    "snippet": snippet(doc),
                    "score": round(float(score[i]), 4),
                    "matched_genes": sorted(set(genes) & self.gene_sets[i]),
                }
            )
            if len(hits) == top_k:
                break
        return hits


_CACHE: dict[tuple, KnowledgeIndex] = {}


def get_index() -> KnowledgeIndex:
    """Index for the current knowledge dir and GTA_RAG_BACKEND (cached per process)."""
    backend = (os.environ.get("GTA_RAG_BACKEND") or "bm25").lower()
    path = store.docs_path()
    key = (str(path), path.stat().st_mtime_ns, backend)
    if key not in _CACHE:
        model = os.environ.get("GTA_RAG_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
        cache_file = path.with_name(f"embeddings_{slug}.npy") if backend == "embed" else None
        _CACHE.clear()
        _CACHE[key] = KnowledgeIndex(store.load_docs(), backend, cache_file=cache_file)
    return _CACHE[key]
