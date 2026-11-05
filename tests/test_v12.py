"""Benchmark v1.2: prompt-echo citations, offline rescoring, id placeholders, pinned store identity."""

import json
import re
import shutil

import pytest

from eval import run_eval as ev
from eval.graders import KNOWLEDGE_TOOLS, citation_correctness, citation_validity, grade_task
from eval.run_eval import (
    KNOWLEDGE_FIXTURE,
    KNOWLEDGE_RAW_FILES,
    knowledge_store_identity,
    load_tasks,
    rescore,
    write_summary,
)
from gta.agent import SYSTEM_PROMPT, run_agent
from gta.llm import MockLLM
from gta.rag import store
from gta.tools import registry

REAL_ID = re.compile(r"GO:\d{7}|R-HSA-\d+")


def _task(task_id):
    return next(t for t in load_tasks() if t["id"] == task_id)


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """Private copy of the pinned fixture; generated files never land in tests/fixtures."""
    dest = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_FIXTURE, dest)
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", str(dest))
    monkeypatch.delenv("GTA_RAG_BACKEND", raising=False)
    return dest


def _grade_trap(answer, tmp_path):
    task = _task("rag_trap_missing_id")
    calls = [{"name": "get_term", "arguments": {"id": "GO:9999999"}}]
    result = run_agent(task["prompt"], MockLLM.from_calls(calls, answer), trace_dir=tmp_path)
    return grade_task(task, result, {})


# ---------- 1. prompt-echoed ids are not citations ----------


def test_trap_task_passes_when_answer_only_echoes_prompt_id(kb, tmp_path):
    m = _grade_trap("GO:9999999 does not exist in the knowledge base.", tmp_path)
    assert m["citation_precision"] == 1.0 and m["citation_validity"] is None
    assert m["task_success"] == 1.0


def test_trap_task_still_fails_when_answer_cites_a_different_unretrieved_id(kb, tmp_path):
    m = _grade_trap("GO:9999999 does not exist, but you probably mean GO:1234567.", tmp_path)
    assert m["citation_precision"] == 0.0 and m["citation_validity"] == 0.0
    assert m["task_success"] == 0.0


def test_prompt_ids_excluded_from_precision_and_validity():
    assert citation_correctness("GO:1 and GO:0000001", "about GO:0000001", None, []) == (1.0, None)
    assert citation_validity("GO:0000001 is missing", "about GO:0000001", []) is None
    assert citation_validity("GO:0000001 and GO:0000002", "about GO:0000001", []) == 0.0


# ---------- 2. offline rescoring ----------


def _saved_row(answer, old_success, old_precision):
    return {
        "id": "rag_trap_missing_id",
        "category": "rag",
        "repeat": 0,
        "model": "fake-model",
        "tasks_version": "1.1",
        "answer": answer,
        "tools_called": ["get_term"],
        "calls": [
            {
                "name": "get_term",
                "args": {"id": "GO:9999999"},
                "ok": False,
                "error": "Unknown term id",
                "result_summary": None,
            }
        ],
        "metrics": {
            "task_success": old_success,
            "citation_precision": old_precision,
            "citation_validity": 0.0,
            "invalid_call_rate": 0.0,
            "n_steps": 2.0,
            "n_duplicate_calls": 0.0,
            "latency_s": 1.5,
            "tokens": 99.0,
        },
    }


def test_rescore_writes_new_file_reports_diff_and_never_touches_input(tmp_path, monkeypatch):
    def no_llm(*a, **k):
        raise AssertionError("rescoring must not call an LLM")

    monkeypatch.setattr(ev, "OpenAICompatLLM", no_llm)
    src = tmp_path / "fake_20260101-0000.done.jsonl"
    src.write_text(json.dumps(_saved_row("GO:9999999 does not exist.", 0.0, 0.0)) + "\n")
    before = src.read_bytes()

    out, diffs = rescore(src, truth={})

    assert src.read_bytes() == before
    assert out == tmp_path / "fake_20260101-0000.rescored_v1.2.jsonl"
    (d,) = diffs
    assert (d["id"], d["old"], d["new"]) == ("rag_trap_missing_id", 0.0, 1.0)
    assert "task_success" in d["changed"] and "citation_precision" in d["changed"]
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["metrics"]["task_success"] == 1.0
    assert row["metrics"]["latency_s"] == 1.5 and row["metrics"]["n_steps"] == 2.0  # run metrics carried over


def test_rescore_refuses_to_overwrite_input(tmp_path):
    src = tmp_path / "x.rescored_v1.2.jsonl"
    src.write_text(json.dumps(_saved_row("GO:9999999 does not exist.", 0.0, 0.0)) + "\n")
    with pytest.raises(SystemExit):
        rescore(src, truth={})


def test_summary_keeps_v11_table_and_adds_labelled_rescored_table(tmp_path):
    agg = {"overall": {"task_success": {"mean": 0.5, "sd": 0.0}}, "by_category": {}}
    rep = {"model": "m", "repeats": 1, "tasks_version": "1.1", "aggregate": agg}
    (tmp_path / "m_1.json").write_text(json.dumps(rep))
    write_summary(tmp_path)
    v11_only = (tmp_path / "summary.md").read_text()

    row = _saved_row("GO:9999999 does not exist.", 0.0, 0.0)
    row["metrics"] = {"task_success": 1.0}
    (tmp_path / "m_1.rescored_v1.2.jsonl").write_text(json.dumps(row) + "\n")
    write_summary(tmp_path)
    text = (tmp_path / "summary.md").read_text()

    assert text.startswith(v11_only.rstrip("\n"))
    assert "## v1.1 rescored with v1.2 graders" in text
    assert "50±0%" in text and "100±0%" in text


# ---------- 3. no real-looking ids in prompt/hints ----------


def test_no_real_looking_ids_in_system_prompt_hints_or_descriptions(kb):
    texts = [SYSTEM_PROMPT, json.dumps(registry.openai_tool_specs())]
    for name in registry.tool_names():
        res = registry.call(name, {"nonsense_argument": 1})
        texts.append(f"{res.get('error', '')} {res.get('hint', '')}")
    texts.append(json.dumps(registry.call("get_term", {"id": "not-an-id"})))
    texts.append(json.dumps(registry.call("annotate_genes", {"genes": ["NOT_A_GENE"]})))
    for text in texts:
        assert not REAL_ID.search(text), REAL_ID.search(text).group()
    assert "GO:XXXXXXX" in SYSTEM_PROMPT and "R-HSA-XXXXXX" in SYSTEM_PROMPT
    assert "GO:XXXXXXX" in registry.call("get_term", {"id": "not-an-id"})["hint"]


# ---------- 4. pinned store identity ----------


def test_store_identity_covers_raw_files_only(tmp_path):
    base = knowledge_store_identity()
    assert base["fixture"] == "tests/fixtures/knowledge_mini" and len(base["sha256"]) == 64
    copy = tmp_path / "kb"
    shutil.copytree(KNOWLEDGE_FIXTURE, copy)
    (copy / "docs.jsonl").write_text("generated")
    assert knowledge_store_identity(copy)["sha256"] == base["sha256"]
    gmt = copy / "ReactomePathways.gmt"
    gmt.write_text(gmt.read_text() + "\n")
    assert knowledge_store_identity(copy)["sha256"] != base["sha256"]
    assert set(KNOWLEDGE_RAW_FILES) == {
        "ReactomePathways.gmt",
        "ReactomePathways.txt",
        "go-basic.obo",
        "goa_human.gaf",
    }


def test_rows_and_summary_carry_store_identity(tmp_path):
    pytest.importorskip("yaml")
    from tests.conftest import ROOT

    if not (ROOT / "data/rnaseq/ground_truth.json").exists():
        pytest.skip("missing data; run `make data`")
    rep = ev.run_eval(mock=True, tasks="rag_trap_missing_id", results_dir=tmp_path)
    ident = knowledge_store_identity()
    assert rep["knowledge_store"] == ident
    assert all(r["knowledge_store"] == ident for r in rep["tasks"])
    fake = {**rep, "model": "real-model"}
    (tmp_path / "real_1.json").write_text(json.dumps(fake))
    write_summary(tmp_path)
    assert ident["sha256"] in (tmp_path / "summary.md").read_text()


# ---------- 5. expected ids are retrievable from the pinned store ----------


def _literal(args):
    return "{" not in json.dumps(args)


def test_expected_ids_retrievable_from_pinned_store(kb, tmp_path):
    before = sorted(p.name for p in KNOWLEDGE_FIXTURE.iterdir())
    store.build_store(kb)
    failures = []
    for task in load_tasks():
        expected = task.get("expected_ids")
        if task["category"] != "rag" or not expected:
            continue
        calls = [
            c
            for c in task["mock"].get("calls", [])
            if c["name"] in KNOWLEDGE_TOOLS and _literal(c["arguments"])
        ]
        if not any(c["name"] == "search_knowledge" or c["name"] == "get_term" for c in calls):
            calls = [{"name": "get_term", "arguments": {"id": i}} for i in expected]  # DE-dependent task
        retrieved = " ".join(json.dumps(registry.call(c["name"], c["arguments"])) for c in calls)
        missing = [i for i in expected if i not in set(REAL_ID.findall(retrieved))]
        if missing:
            failures.append(f"{task['id']}: not retrievable: {missing}")
    assert not failures, "\n".join(failures)
    assert sorted(p.name for p in KNOWLEDGE_FIXTURE.iterdir()) == before, "fixture dir was modified"


# ---------- result_text in saved rows ----------


def test_call_log_has_truncated_redacted_result_text(monkeypatch):
    from gta.agent import MAX_TOOL_CHARS, ToolCallRecord

    monkeypatch.setenv("LLM_API_KEY", "supersecretvalue123")
    result = {
        "summary": {"note": "key supersecretvalue123 api_key=abc123xyz Bearer abcdefgh12345 " + "x" * 6000}
    }
    log = ToolCallRecord("t", "{}", result, 0.0).to_log()["result_text"]
    assert "supersecretvalue123" not in log and "abc123xyz" not in log and "abcdefgh12345" not in log
    assert "[REDACTED]" in log and len(log) <= MAX_TOOL_CHARS + len("...[truncated]")


def test_rescore_uses_result_text_for_citation_validity(tmp_path):
    def row(with_text):
        r = _saved_row("GO:0006915 is apoptotic process.", 0.0, 1.0)
        r["id"] = "rag_find_go_term"
        r["calls"] = [
            {"name": "search_knowledge", "args": {"query": "apoptosis"}, "ok": True, "error": None,
             "result_summary": {"n_hits": 1}}
        ]  # fmt: skip
        if with_text:
            r["calls"][0]["result_text"] = '{"summary": {"hits": [{"id": "GO:0006915"}]}}'
        return r

    src = tmp_path / "a.done.jsonl"
    src.write_text(json.dumps(row(True)) + "\n")
    _, diffs = rescore(src, truth={})
    assert diffs[0]["has_result_text"]
    out = [json.loads(x) for x in (tmp_path / "a.rescored_v1.2.jsonl").read_text().splitlines()]
    assert out[0]["metrics"]["citation_validity"] == 1.0
    src.write_text(json.dumps(row(False)) + "\n")
    _, diffs = rescore(src, truth={})
    assert not diffs[0]["has_result_text"]
    out = [json.loads(x) for x in (tmp_path / "a.rescored_v1.2.jsonl").read_text().splitlines()]
    assert out[0]["metrics"]["citation_validity"] == 0.0


def test_cli_rescore_warns_for_rows_without_result_text(tmp_path):
    from typer.testing import CliRunner

    from gta.cli import app

    src = tmp_path / "b.done.jsonl"
    src.write_text(json.dumps(_saved_row("GO:9999999 does not exist.", 0.0, 0.0)) + "\n")
    res = CliRunner().invoke(app, ["eval", "--rescore", str(src)])
    assert res.exit_code == 0, res.output
    assert "citation validity is unreliable" in " ".join(res.output.split())
