import gzip
import json
import shutil
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gta import cli, doctor
from gta.agent import SYSTEM_PROMPT
from gta.rag import fetch, store
from gta.rag.index import KnowledgeIndex, get_index
from gta.tools import registry
from tests.conftest import needs_data

FIXTURE = Path(__file__).parent / "fixtures" / "knowledge_mini"
ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture(scope="session")
def built_store(tmp_path_factory):
    """The offline fixture copied to a temp dir and turned into a document store."""
    dest = tmp_path_factory.mktemp("knowledge")
    for f in FIXTURE.iterdir():
        shutil.copy(f, dest / f.name)
    store.build_store(dest)
    return dest


@pytest.fixture
def kb(built_store, monkeypatch):
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", str(built_store))
    monkeypatch.delenv("GTA_RAG_BACKEND", raising=False)
    return built_store


def docs_by_id(kb):
    return {d["id"]: d for d in store.load_docs()}


# ---------- document store ----------


def test_fixture_store_has_expected_documents(kb):
    docs = docs_by_id(kb)
    assert sum(d["source"] == "reactome" for d in docs.values()) == 20  # other-species rows are ignored
    assert sum(d["source"] == "go" for d in docs.values()) == 31
    assert "GO:0006977" not in docs  # obsolete term dropped
    assert (
        docs["GO:0006915"]["name"] == "apoptotic process"
        and "programmed cell death" in docs["GO:0006915"]["definition"]
    )
    assert docs["R-HSA-913531"]["name"] == "Interferon Signaling" and "STAT1" in docs["R-HSA-913531"]["genes"]


def test_obo_parser_handles_obsolete_relationships_and_quoted_definitions():
    obo = """[Term]
id: GO:0000001
name: parent term
namespace: biological_process
def: "A \\"quoted\\" definition." [GOC:x]

[Term]
id: GO:0000002
name: child term
is_a: GO:0000001 ! parent term
relationship: part_of GO:0000003 ! other
relationship: regulates GO:0000009 ! ignored

[Term]
id: GO:0000004
name: gone
is_obsolete: true

[Typedef]
id: part_of
""".splitlines(keepends=True)
    terms = store.parse_obo(obo)
    assert set(terms) == {"GO:0000001", "GO:0000002"}
    assert terms["GO:0000001"]["definition"] == 'A "quoted" definition.'
    assert terms["GO:0000002"]["parents"] == ["GO:0000001", "GO:0000003"]


def test_gaf_filter_keeps_experimental_and_curated_but_not_iea_or_not_or_other_species():
    def row(gene, go, ev, qual="involved_in", taxon="taxon:9606"):
        return (
            "\t".join(
                [
                    "UniProtKB",
                    "P1",
                    gene,
                    qual,
                    go,
                    "PMID:1",
                    ev,
                    "",
                    "P",
                    "",
                    "",
                    "protein",
                    taxon,
                    "20200101",
                    "X",
                ]
            )
            + "\n"
        )

    lines = [
        "!gaf-version: 2.2\n",
        row("KEEP1", "GO:1", "IDA"),
        row("KEEP2", "GO:1", "TAS"),
        row("IEAGENE", "GO:1", "IEA"),
        row("NOTGENE", "GO:1", "IDA", qual="NOT|involved_in"),
        row("MOUSE", "GO:1", "IDA", taxon="taxon:10090"),
    ]
    assert store.parse_gaf(lines) == {"GO:1": {"KEEP1", "KEEP2"}}


def test_annotations_propagate_to_ancestors_only():
    terms = {
        "GO:1": {"parents": []},
        "GO:2": {"parents": ["GO:1"]},
        "GO:3": {"parents": ["GO:2"]},
        "GO:4": {"parents": ["GO:1"]},
    }
    genes = store.propagate(terms, {"GO:3": {"A"}, "GO:4": {"B"}, "GO:1": {"C"}})
    assert genes["GO:3"] == {"A"} and genes["GO:2"] == {"A"} and genes["GO:4"] == {"B"}
    assert genes["GO:1"] == {"A", "B", "C"}


def test_fixture_go_gene_lists_exclude_iea_only_genes(kb):
    direct = store.parse_gaf(
        line for line in (FIXTURE / "goa_human.gaf").read_text().splitlines(keepends=True)
    )
    kept = set().union(*direct.values())
    assert set(docs_by_id(kb)["GO:0006955"]["genes"]) <= kept
    all_lines = [
        ln.split("\t")
        for ln in (FIXTURE / "goa_human.gaf").read_text().splitlines()
        if not ln.startswith("!")
    ]
    iea_only = {f[2] for f in all_lines if f[6] == "IEA"} - kept
    assert iea_only and not iea_only & set().union(
        *(set(d["genes"]) for d in docs_by_id(kb).values() if d["source"] == "go")
    )


def test_cell_cycle_inherits_genes_from_child_terms(kb):
    docs = docs_by_id(kb)
    assert docs["GO:0007049"]["genes"] and set(docs["GO:0000278"]["genes"]) <= set(
        docs["GO:0007049"]["genes"]
    )


def test_missing_store_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", str(tmp_path / "nothing"))
    out = registry.call("search_knowledge", {"query": "apoptosis"})
    assert out["kind"] == "tool_error" and "fetch-knowledge" in out["hint"]


def test_store_builds_lazily_from_raw_files(tmp_path, monkeypatch):
    for f in FIXTURE.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", str(tmp_path))
    assert not (tmp_path / "docs.jsonl").exists()
    assert registry.call("get_term", {"id": "R-HSA-109581"})["summary"]["name"] == "Apoptosis"
    assert (tmp_path / "docs.jsonl").exists()


def test_build_reports_missing_raw_files(tmp_path):
    with pytest.raises(store.ToolError, match="Missing knowledge source files"):
        store.build_store(tmp_path)


# ---------- retrieval ----------

QUERIES = [
    ("apoptosis", {"R-HSA-109581"}),
    ("programmed cell death", {"GO:0006915"}),
    ("interferon signaling", {"R-HSA-913531"}),
    ("response to type I interferon", {"GO:0034340"}),
    ("TP53 MDM2", {"R-HSA-5633007"}),
    ("double-strand break repair by homologous recombination", {"GO:0000724"}),
    ("glycolysis", {"R-HSA-70171", "GO:0006096"}),
    ("Wnt signaling", {"R-HSA-195721", "GO:0016055"}),
    ("ribosomal RNA processing", {"R-HSA-6791226", "GO:0006364"}),
    ("mitotic cell cycle checkpoints", {"R-HSA-69620"}),
]


def test_recall_at_5_on_fixed_queries(kb):
    index = get_index()
    found = total = 0
    misses = []
    for query, expected in QUERIES:
        ids = {h["id"] for h in index.search(query, top_k=5)}
        found += len(expected & ids)
        total += len(expected)
        misses += [(query, e) for e in expected - ids]
    assert found / total >= 0.9, misses


def test_search_tool_output_shape_and_filters(kb):
    out = registry.call("search_knowledge", {"query": "programmed cell death", "source": "go", "top_k": 3})
    hits = out["summary"]["hits"]
    assert out["summary"]["backend"] == "bm25" and len(hits) == 3
    assert all(h["source"] == "go" and h["id"].startswith("GO:") for h in hits)
    assert set(hits[0]) >= {"id", "name", "source", "snippet", "score"}
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    reactome = registry.call("search_knowledge", {"query": "apoptosis", "source": "reactome"})["summary"][
        "hits"
    ]
    assert reactome and all(h["id"].startswith("R-HSA-") for h in reactome)


def test_query_without_matches_returns_no_hits(kb):
    assert registry.call("search_knowledge", {"query": "qqqzzz"})["summary"]["n_hits"] == 0


def test_gene_symbols_in_query_boost_documents_containing_them(kb):
    index = get_index()
    assert index.query_genes("role of BRCA1 and brca2 and TP53") == [
        "BRCA1",
        "TP53",
    ]  # symbols are case-sensitive
    hits = index.search("BRCA1 BARD1", top_k=5)
    assert hits and all("BRCA1" in h["matched_genes"] for h in hits[:3])
    plain = index.search("repair", top_k=5)
    assert plain  # text-only queries still work and recognise no genes
    assert not any(h["matched_genes"] for h in plain)


def test_gene_boost_reorders_results_with_the_same_text_score():
    docs = [
        {
            "id": "A",
            "name": "kinase signaling",
            "source": "go",
            "namespace": "x",
            "definition": "",
            "genes": ["EGFR"],
        },
        {
            "id": "B",
            "name": "kinase signaling",
            "source": "go",
            "namespace": "x",
            "definition": "",
            "genes": ["KRAS"],
        },
        {
            "id": "C",
            "name": "unrelated thing",
            "source": "go",
            "namespace": "x",
            "definition": "",
            "genes": ["TP53"],
        },
    ]
    idx = KnowledgeIndex(docs)
    assert idx.search("kinase signaling KRAS")[0]["id"] == "B"
    assert idx.search("kinase signaling EGFR")[0]["id"] == "A"


# ---------- embedding backend (fake encoder; sentence-transformers is optional) ----------


def fake_encoder(texts):
    vecs = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            vecs[i, zlib.crc32(w.encode()) % 64] += 1.0
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def test_embed_backend_ranks_and_caches_vectors(kb, tmp_path):
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_encoder(texts)

    cache = tmp_path / "emb.npy"
    docs = store.load_docs()
    idx = KnowledgeIndex(docs, "embed", encoder=counting, cache_file=cache)
    assert idx.backend == "embed" and cache.exists()
    assert "R-HSA-913531" in {h["id"] for h in idx.search("Interferon Signaling", top_k=3)}
    n_calls = len(calls)
    KnowledgeIndex(docs, "embed", encoder=counting, cache_file=cache).search("apoptosis")
    assert len(calls) == n_calls + 1  # only the query was encoded; document vectors came from the cache


def test_embed_backend_without_extra_installed_explains_how_to_fix(kb, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setenv("GTA_RAG_BACKEND", "embed")
    out = registry.call("search_knowledge", {"query": "apoptosis"})
    assert "sentence-transformers" in out["error"] and "[rag]" in out["hint"]


def test_unknown_backend_is_rejected(kb, monkeypatch):
    monkeypatch.setenv("GTA_RAG_BACKEND", "magic")
    assert "Unknown GTA_RAG_BACKEND" in registry.call("search_knowledge", {"query": "x"})["error"]


# ---------- get_term ----------


def test_get_term_returns_record_and_writes_full_gene_list(kb):
    out = registry.call("get_term", {"id": "R-HSA-6798695"})
    s = out["summary"]
    assert s["name"] == "Neutrophil degranulation" and s["n_genes"] > 40
    assert len(s["genes"]) == 40 and s["genes_truncated"] is True
    record = json.loads(Path(out["outputs"]["record_json"]).read_text())
    assert len(record["genes"]) == s["n_genes"]


def test_get_term_normalises_case_and_lists_parents(kb):
    s = registry.call("get_term", {"id": " go:0000278 "})["summary"]
    assert s["id"] == "GO:0000278" and s["namespace"] == "biological_process"
    assert {"id": "GO:0007049", "name": "cell cycle"} in s["parents"]


@pytest.mark.parametrize("bad", ["GO:9999999", "R-HSA-1", "not an id", "GO:6977"])
def test_get_term_unknown_id_is_a_structured_error(kb, bad):
    out = registry.call("get_term", {"id": bad})
    assert (
        out["kind"] == "tool_error"
        and "Unknown term id" in out["error"]
        and "search_knowledge" in out["hint"]
    )


# ---------- annotate_genes ----------


def test_annotate_genes_inline_list_uses_store_background(kb):
    genes = docs_by_id(kb)["R-HSA-70171"]["genes"]
    out = registry.call("annotate_genes", {"genes": ",".join(genes), "top_k": 3})["summary"]
    assert out["background_source"] == "knowledge_store_union" and out["top"][0]["id"] == "R-HSA-70171"
    assert out["top"][0]["overlap"] == len(genes) and out["top"][0]["padj"] < 1e-10
    assert out["n_query"] == len(genes) and len(out["top"]) == 3


def test_annotate_genes_source_filter_and_input_errors(kb):
    genes = docs_by_id(kb)["R-HSA-70171"]["genes"]
    top = registry.call("annotate_genes", {"genes": genes, "source": "go", "top_k": 5})["summary"]["top"]
    assert top and all(t["id"].startswith("GO:") for t in top)
    assert "Provide 'genes'" in registry.call("annotate_genes", {})["error"]
    assert "None of the query genes" in registry.call("annotate_genes", {"genes": ["NOTAGENE1"]})["error"]
    assert (
        "No knowledge terms"
        in registry.call("annotate_genes", {"genes": genes, "min_set_size": 5000})["error"]
    )


@needs_data("data/rnaseq_real/counts.tsv", "data/rnaseq_real/ground_truth.json")
def test_annotate_genes_recovers_planted_pathways_after_deseq2(kb):
    truth = json.loads((ROOT / "data/rnaseq_real/ground_truth.json").read_text())
    de = registry.call(
        "run_deseq2",
        {
            "counts": "data/rnaseq_real/counts.tsv",
            "metadata": "data/rnaseq_real/metadata.tsv",
            "design_factor": "condition",
            "ref_level": "control",
        },
    )
    called_up = set(Path(de["outputs"]["up_genes"]).read_text().split())
    assert len(called_up & set(truth["up_genes"])) / len(truth["up_genes"]) >= 0.8
    ann = registry.call(
        "annotate_genes",
        {
            "gene_list_file": de["outputs"]["up_genes"],
            "de_results": de["outputs"]["results_tsv"],
            "source": "reactome",
        },
    )["summary"]
    assert ann["background_size"] == truth["n_genes"]
    top_ids = [t["id"] for t in ann["top"]]
    assert set(truth["planted_pathway_ids"]) <= set(top_ids[:3])
    assert all(t["padj"] < 1e-10 for t in ann["top"][:2])


def test_planted_pathways_exist_in_fixture(kb):
    path = ROOT / "data/rnaseq_real/ground_truth.json"
    if not path.exists():
        pytest.skip("run `make data`")
    truth = json.loads(path.read_text())
    assert set(truth["planted_pathway_ids"]) <= set(docs_by_id(kb))


# ---------- tool specs and prompt ----------


def test_new_tools_registered_with_schemas():
    specs = {s["function"]["name"]: s["function"] for s in registry.openai_tool_specs()}
    assert {"search_knowledge", "get_term", "annotate_genes"} <= set(specs)
    assert specs["search_knowledge"]["parameters"]["properties"]["source"]["enum"] == [
        "go",
        "reactome",
        "all",
    ]
    assert specs["get_term"]["parameters"]["required"] == ["id"]


def test_system_prompt_requires_tool_backed_citations():
    assert "[GO:0006915]" in SYSTEM_PROMPT and "[R-HSA-109581]" in SYSTEM_PROMPT
    assert "Never cite an id that was not returned by a tool" in SYSTEM_PROMPT


# ---------- downloader ----------


def fake_downloader(calls):
    def download(url, dest):
        calls.append(url)
        name = dest.name
        if name.endswith(".gmt.zip"):
            with zipfile.ZipFile(dest, "w") as z:
                z.write(FIXTURE / "ReactomePathways.gmt", "ReactomePathways.gmt")
        elif name.endswith(".gz"):
            dest.write_bytes(gzip.compress((FIXTURE / "goa_human.gaf").read_bytes()))
        else:
            shutil.copy(FIXTURE / name, dest)

    return download


def test_fetch_records_manifest_builds_store_and_skips_existing(tmp_path):
    calls: list[str] = []
    manifest = fetch.fetch_all(tmp_path, downloader=fake_downloader(calls))
    assert set(calls) == set(fetch.SOURCES.values())
    for name, url in fetch.SOURCES.items():
        entry = manifest["files"][name]
        assert entry["url"] == url and entry["sha256"] == fetch.sha256(tmp_path / name)
        assert len(entry["downloaded"]) == 10 and entry["bytes"] == (tmp_path / name).stat().st_size
    assert json.loads((tmp_path / "manifest.json").read_text())["store"]["n_go"] == 31
    assert (
        manifest["store"]["n_reactome"] == 20
        and manifest["store"]["go_data_version"] == "subset-of-go-basic (test fixture)"
    )
    assert "IEA" not in manifest["store"]["go_evidence_codes"]

    fetch.fetch_all(tmp_path, downloader=fake_downloader(calls))
    assert len(calls) == 4  # nothing re-downloaded
    fetch.fetch_all(tmp_path, force=True, downloader=fake_downloader(calls))
    assert len(calls) == 8


def test_download_writes_via_part_file(tmp_path, monkeypatch):
    class Resp:
        def __init__(self):
            self.data = [b"abc", b""]

        def read(self, n=-1):
            return self.data.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda req, timeout: Resp())
    dest = tmp_path / "f.txt"
    fetch.download("http://example.org/f.txt", dest)
    assert dest.read_bytes() == b"abc" and not (tmp_path / "f.txt.part").exists()


def test_fetch_knowledge_command_forwards_options(monkeypatch):
    seen = {}
    monkeypatch.setattr("gta.rag.fetch.main", lambda argv: seen.setdefault("argv", argv) and 0)
    res = runner.invoke(cli.app, ["fetch-knowledge", "--dest", "/tmp/k", "--force"])
    assert res.exit_code == 0 and seen["argv"] == ["--dest", "/tmp/k", "--force"]


def test_doctor_reports_knowledge_store(kb, monkeypatch):
    row = doctor.check_knowledge()[0]
    assert row.ok and row.kind == "data" and not row.required
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", str(kb / "missing"))
    row = doctor.check_knowledge()[0]
    assert not row.ok and "fetch-knowledge" in row.hint and doctor.exit_code([row]) == 0
