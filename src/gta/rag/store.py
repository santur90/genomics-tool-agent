"""Knowledge document store built from Reactome and Gene Ontology files (one JSONL document per term)."""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gta.sandbox import ToolError

DEFAULT_DIR = Path.home() / ".cache" / "gta" / "knowledge"
DOCS_FILE = "docs.jsonl"
MANIFEST_FILE = "manifest.json"
REACTOME_TXT = "ReactomePathways.txt"
REACTOME_GMT = "ReactomePathways.gmt"
GO_OBO = "go-basic.obo"
GOA_GAF = "goa_human.gaf.gz"
# Experimental evidence plus curator-reviewed codes (TAS traceable author, IC inferred by curator).
EVIDENCE_CODES = frozenset(
    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP", "TAS", "IC"}
)
_ID_REL = re.compile(r"(GO:\d{7})")
_DEF = re.compile(r'^"(.*?)(?<!\\)"')


def knowledge_dir() -> Path:
    """Where downloaded files and the document store live (GTA_KNOWLEDGE_DIR overrides)."""
    return Path(os.environ.get("GTA_KNOWLEDGE_DIR") or DEFAULT_DIR).expanduser()


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def parse_obo(lines: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Parse [Term] stanzas of an OBO file; obsolete terms are dropped.

    Parents follow is_a and part_of so annotations can be propagated to ancestors.
    """
    terms: dict[str, dict[str, Any]] = {}
    cur: dict[str, Any] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("["):
            cur = {"parents": [], "obsolete": False} if line == "[Term]" else None
            continue
        if cur is None or ": " not in line:
            continue
        key, _, value = line.partition(": ")
        if key == "id":
            cur["id"] = value
            terms[value] = cur
        elif key == "name":
            cur["name"] = value
        elif key == "namespace":
            cur["namespace"] = value
        elif key == "def":
            m = _DEF.match(value)
            cur["definition"] = (m.group(1) if m else value).replace('\\"', '"')
        elif key == "is_a" or (key == "relationship" and value.startswith("part_of ")):
            m = _ID_REL.search(value)
            if m:
                cur["parents"].append(m.group(1))
        elif key == "is_obsolete" and value == "true":
            cur["obsolete"] = True
    return {gid: t for gid, t in terms.items() if not t["obsolete"] and "name" in t}


def parse_gaf(lines: Iterable[str], evidence: frozenset[str] = EVIDENCE_CODES) -> dict[str, set[str]]:
    """Direct human annotations {GO id: gene symbols}, keeping only the given evidence codes and no NOT."""
    direct: dict[str, set[str]] = {}
    for line in lines:
        if line.startswith("!"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 13 or "taxon:9606" not in f[12] or "NOT" in f[3] or f[6] not in evidence:
            continue
        direct.setdefault(f[4], set()).add(f[2])
    return direct


def propagate(terms: dict[str, dict[str, Any]], direct: dict[str, set[str]]) -> dict[str, set[str]]:
    """Add each term's genes to all of its ancestors (true-path rule)."""
    children: dict[str, list[str]] = {}
    for gid, t in terms.items():
        for parent in t["parents"]:
            children.setdefault(parent, []).append(gid)
    memo: dict[str, set[str]] = {}

    def genes_of(gid: str) -> set[str]:
        if gid not in memo:
            memo[gid] = set()  # guards against cycles in malformed input
            acc = set(direct.get(gid, ()))
            for child in children.get(gid, ()):
                acc |= genes_of(child)
            memo[gid] = acc
        return memo[gid]

    return {gid: genes_of(gid) for gid in terms}


def go_documents(obo: Path, gaf: Path) -> list[dict[str, Any]]:
    """One document per non-obsolete GO term."""
    with _open_text(obo) as fh:
        terms = parse_obo(fh)
    with _open_text(gaf) as fh:
        genes = propagate(terms, parse_gaf(fh))
    return [
        {
            "id": gid,
            "name": t["name"],
            "source": "go",
            "namespace": t.get("namespace", ""),
            "definition": t.get("definition", ""),
            "description": "",
            "parents": sorted(set(t["parents"]) & set(terms)),
            "genes": sorted(genes[gid]),
        }
        for gid, t in sorted(terms.items())
    ]


def reactome_documents(txt: Path, gmt: Path) -> list[dict[str, Any]]:
    """One document per human Reactome pathway (name from the pathway list, genes from the GMT)."""
    members: dict[str, list[str]] = {}
    for line in gmt.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            members[parts[1]] = sorted({g for g in parts[2:] if g})
    docs = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] == "Homo sapiens":
            docs.append(
                {
                    "id": parts[0],
                    "name": parts[1],
                    "source": "reactome",
                    "namespace": "pathway",
                    "definition": "",
                    "description": "",
                    "parents": [],
                    "genes": members.get(parts[0], []),
                }
            )
    return sorted(docs, key=lambda d: d["id"])


def build_store(raw_dir: Path, out_dir: Path | None = None) -> dict[str, int]:
    """Build docs.jsonl from raw downloads in raw_dir; returns document counts."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir or raw_dir)
    gaf = next((raw_dir / n for n in (GOA_GAF, "goa_human.gaf") if (raw_dir / n).exists()), None)
    missing = [n for n in (REACTOME_TXT, REACTOME_GMT, GO_OBO) if not (raw_dir / n).exists()]
    if gaf is None:
        missing.append(GOA_GAF)
    if missing:
        raise ToolError(
            f"Missing knowledge source files in {raw_dir}: {missing}", "Run `gta fetch-knowledge`."
        )
    docs = reactome_documents(raw_dir / REACTOME_TXT, raw_dir / REACTOME_GMT) + go_documents(
        raw_dir / GO_OBO, gaf
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (DOCS_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    tmp.replace(out_dir / DOCS_FILE)
    n_reactome = sum(d["source"] == "reactome" for d in docs)
    return {"n_reactome": n_reactome, "n_go": len(docs) - n_reactome}


def docs_path() -> Path:
    """Path to docs.jsonl, building it from raw files in the knowledge dir if only those exist."""
    kdir = knowledge_dir()
    path = kdir / DOCS_FILE
    if path.exists():
        return path
    if (kdir / REACTOME_TXT).exists() or (kdir / GO_OBO).exists():
        build_store(kdir)
        return path
    raise ToolError(
        f"Knowledge base not found in {kdir}",
        "Run `gta fetch-knowledge` (or python scripts/fetch_knowledge.py) once, or set GTA_KNOWLEDGE_DIR.",
    )


def load_docs() -> list[dict[str, Any]]:
    """All documents of the current knowledge store."""
    with open(docs_path(), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
