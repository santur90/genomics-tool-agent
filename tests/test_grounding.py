"""Grounding metrics (citations, numeric grounding, retrieval), RAG tasks and the optional LLM judge."""

import json
import os

import pytest
import yaml

from eval import judge as judge_mod
from eval.graders import (
    check_answer,
    citation_correctness,
    citation_validity,
    cited_ids,
    extract_numbers,
    grade_task,
    numeric_grounding,
)
from eval.run_eval import aggregate, fixture_knowledge_base, load_tasks, run_eval, write_summary
from gta.agent import run_agent
from gta.llm import LLMResponse, MockLLM
from tests.conftest import needs_data

RESULTS = [
    {"summary": {"hits": [{"id": "R-HSA-109581", "name": "Apoptosis"}, {"id": "GO:0006915"}], "padj": 0.0123}}
]


# ---------- citations ----------


def test_cited_ids_extracts_both_namespaces_once_and_ignores_malformed():
    text = "See [GO:0006915], R-HSA-109581 and GO:0006915 again; GO:12 and GO:00069150 are not ids."
    assert cited_ids(text) == ["GO:0006915", "R-HSA-109581"]


def test_ids_are_not_counted_as_numbers():
    assert extract_numbers("Apoptosis [R-HSA-109581] / GO:0006915 has 30 genes") == ["30"]


def test_validity_flags_fabricated_citations():
    assert citation_validity("Apoptosis [R-HSA-109581]", "q", RESULTS) == 1.0
    assert citation_validity("Apoptosis [R-HSA-109581] and R-HSA-999999", "q", RESULTS) == 0.5
    assert citation_validity("R-HSA-999999", "q", RESULTS) == 0.0
    assert citation_validity("No ids here", "q", RESULTS) is None


def test_validity_does_not_count_ids_quoted_from_the_prompt():
    assert citation_validity("GO:9999999 was not found.", "What is GO:9999999?", []) is None


def test_correctness_precision_and_recall():
    expected, allowed = ["R-HSA-1", "R-HSA-2"], ["R-HSA-3"]
    assert citation_correctness("R-HSA-1 R-HSA-2", "q", expected, allowed) == (1.0, 1.0)
    assert citation_correctness("R-HSA-1 R-HSA-3", "q", expected, allowed) == (1.0, 0.5)
    assert citation_correctness("R-HSA-1 R-HSA-77", "q", expected, allowed) == (0.5, 0.5)
    assert citation_correctness("no citations", "q", expected, allowed) == (1.0, 0.0)
    assert citation_correctness("anything", "q", None, None) == (None, None)


def test_correctness_for_trap_task_penalises_any_new_id():
    assert citation_correctness("It is R-HSA-5", "q", None, []) == (0.0, None)
    assert citation_correctness("Not found.", "q", None, []) == (1.0, None)


# ---------- numeric grounding ----------


def test_numeric_grounding_requires_value_match_with_rounding_tolerance():
    assert numeric_grounding("padj 0.012", "q", RESULTS) == 1.0
    assert numeric_grounding("padj 0.02", "q", RESULTS) == 0.0
    assert numeric_grounding("padj 0.012 in 7 genes", "q", RESULTS) == 0.5
    assert numeric_grounding("no numbers", "q", RESULTS) is None


def test_percent_conversion_only_allowed_with_percent_sign():
    results = [{"summary": {"fraction": 0.25}}]
    assert numeric_grounding("25% of genes", "q", results) == 1.0
    assert numeric_grounding("25 genes", "q", results) == 0.0


# ---------- graded tasks (no external tools needed) ----------


def _rag(task_id):
    return next(t for t in load_tasks() if t["id"] == task_id)


@pytest.fixture
def kb():
    with fixture_knowledge_base() as path:
        yield path


def _run(task_id, calls, answer, tmp_path, truth=None):
    task = _rag(task_id)
    result = run_agent(task["prompt"], MockLLM.from_calls(calls, answer), trace_dir=tmp_path)
    return grade_task(task, result, truth or {})


def test_correct_lookup_passes_and_records_metrics(kb, tmp_path):
    m = _run(
        "rag_lookup_go",
        [{"name": "get_term", "arguments": {"id": "GO:0006915"}}],
        "GO:0006915 is the apoptotic process term.",
        tmp_path,
    )
    assert m["task_success"] == 1.0 and m["retrieval_used"] == 1.0 and m["citation_recall"] == 1.0


def test_answering_from_memory_without_retrieval_fails(kb, tmp_path):
    m = _run("rag_lookup_go", [], "GO:0006915 is the apoptotic process term.", tmp_path)
    assert m["retrieval_used"] == 0.0 and m["task_success"] == 0.0


def test_fabricated_citation_fails_the_task(kb, tmp_path):
    m = _run(
        "rag_find_go_term",
        [{"name": "search_knowledge", "arguments": {"query": "programmed cell death", "source": "go"}}],
        "The apoptotic process is GO:0006915, related to GO:0008219 (cell death).",
        tmp_path,
    )
    assert m["citation_validity"] == 0.5 and m["citation_precision"] == 0.5 and m["task_success"] == 0.0


def test_trap_missing_id_fails_when_model_invents_an_answer(kb, tmp_path):
    call = [{"name": "get_term", "arguments": {"id": "GO:9999999"}}]
    ok = _run("rag_trap_missing_id", call, "GO:9999999 was not found; it does not exist.", tmp_path)
    bad = _run("rag_trap_missing_id", call, "GO:9999999 is lipid transport. See also GO:0006869.", tmp_path)
    assert ok["task_success"] == 1.0 and bad["task_success"] == 0.0 and bad["citation_validity"] == 0.0


def test_trap_fake_gene_fails_when_model_cites_pathways(kb, tmp_path):
    call = [{"name": "search_knowledge", "arguments": {"query": "ZZZ9FAKE1", "source": "reactome"}}]
    ok = _run("rag_trap_fake_gene", call, "No pathways were found for ZZZ9FAKE1.", tmp_path)
    bad = _run("rag_trap_fake_gene", call, "ZZZ9FAKE1 is in Apoptosis [R-HSA-109581].", tmp_path)
    assert ok["task_success"] == 1.0 and bad["task_success"] == 0.0
    assert bad["citation_precision"] == 0.0 and bad["citation_validity"] == 0.0


def test_missing_expected_id_lowers_recall(kb, tmp_path):
    m = _run(
        "rag_gene_to_pathways",
        [
            {
                "name": "search_knowledge",
                "arguments": {"query": "TP53 MDM2", "source": "reactome", "top_k": 10},
            }
        ],
        "Regulation of TP53 Activity [R-HSA-5633007] contains both.",
        tmp_path,
    )
    assert m["citation_recall"] == 0.25 and m["citation_precision"] == 1.0 and m["task_success"] == 0.0


def test_citing_a_tp53_only_pathway_lowers_precision(kb, tmp_path):
    answer = "R-HSA-5633007, R-HSA-6806003, R-HSA-2559583, R-HSA-69620 and Apoptosis R-HSA-109581."
    m = _run(
        "rag_gene_to_pathways",
        [
            {
                "name": "search_knowledge",
                "arguments": {"query": "TP53 MDM2", "source": "reactome", "top_k": 10},
            }
        ],
        answer + " Regulation of TP53 Activity.",
        tmp_path,
    )
    assert m["citation_validity"] == 1.0 and m["citation_precision"] == 0.8 and m["task_success"] == 0.0


def test_not_regex_check():
    assert check_answer({"type": "not_regex", "pattern": "lipid"}, "it is apoptosis", {})
    assert not check_answer({"type": "not_regex", "pattern": "lipid"}, "lipid transport", {})


def test_non_rag_tasks_have_no_applicable_grounding_metrics(tmp_path):
    task = {"id": "t", "category": "x", "prompt": "hello", "expected_tools": []}
    result = run_agent("hello", MockLLM.from_calls([], "Hi."), trace_dir=tmp_path)
    m = grade_task(task, result, {})
    assert all(
        m[k] is None for k in ("citation_validity", "citation_precision", "citation_recall", "retrieval_used")
    )
    assert m["task_success"] == 1.0


# ---------- aggregation / summary ----------


def test_aggregate_skips_not_applicable_metrics():
    rows = [
        {"category": "a", "repeat": 0, "metrics": {"task_success": 1.0, "citation_recall": None}},
        {"category": "a", "repeat": 0, "metrics": {"task_success": 0.0, "citation_recall": 0.5}},
        {"category": "b", "repeat": 0, "metrics": {"task_success": 1.0, "citation_recall": None}},
    ]
    agg = aggregate(rows)
    assert agg["overall"]["citation_recall"] == {"mean": 0.5, "sd": 0.0}
    assert "citation_recall" not in agg["by_category"]["b"] and "judge_support" not in agg["overall"]


def test_summary_has_citation_columns_and_dashes(tmp_path):
    rows = [
        {"category": "rag", "repeat": 0, "metrics": {"task_success": 1.0, "citation_precision": 1.0}},
        {"category": "safety", "repeat": 0, "metrics": {"task_success": 1.0}},
    ]
    (tmp_path / "m_20260101-0000.json").write_text(
        json.dumps({"model": "m", "repeats": 1, "aggregate": aggregate(rows)})
    )
    write_summary(tmp_path)
    text = (tmp_path / "summary.md").read_text()
    for col in ("citation validity", "citation precision", "citation recall", "retrieval used"):
        assert col in text
    assert "judge support" not in text
    safety_row = next(line for line in text.splitlines() if line.startswith("| safety"))
    assert "–" in safety_row


# ---------- harness ----------


def test_eval_uses_fixture_knowledge_base_and_restores_env(monkeypatch):
    monkeypatch.setenv("GTA_KNOWLEDGE_DIR", "/somewhere/else")
    with fixture_knowledge_base() as path:
        assert os.environ["GTA_KNOWLEDGE_DIR"] == str(path) and (path / "go-basic.obo").exists()
    assert os.environ["GTA_KNOWLEDGE_DIR"] == "/somewhere/else"
    monkeypatch.delenv("GTA_KNOWLEDGE_DIR")
    with fixture_knowledge_base():
        pass
    assert "GTA_KNOWLEDGE_DIR" not in os.environ


@needs_data("data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json", "data/rnaseq_real/counts.tsv")
def test_mock_eval_rag_tasks_pass_with_grounded_metrics(tmp_path):
    report = run_eval(mock=True, category="rag", results_dir=tmp_path)
    assert len(report["tasks"]) == 12
    failed = [t["id"] for t in report["tasks"] if t["metrics"]["task_success"] < 1]
    assert not failed, failed
    overall = report["aggregate"]["overall"]
    assert overall["citation_validity"]["mean"] == 1.0 and overall["retrieval_used"]["mean"] == 1.0


@needs_data("data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json")
def test_mock_rag_task_scripts_exist_for_every_rag_task():
    rag = [t for t in load_tasks() if t["category"] == "rag"]
    assert all(t["mock"]["answer"] and t.get("must_retrieve") for t in rag)


# ---------- LLM judge ----------


class RuleJudge:
    """Fake judge: a sentence is supported iff every number/id token in it occurs in the evidence."""

    model = "rule-judge"

    def chat(self, messages, tools):
        content = messages[-1]["content"]
        evidence, _, sentence = content.partition("\n\nSENTENCE:\n")
        tokens = [t for t in sentence.replace(",", " ").split() if any(ch.isdigit() for ch in t)]
        ok = all(t.strip(".[]()") in evidence for t in tokens)
        return LLMResponse(json.dumps({"supported": ok}))


class AlwaysYes:
    model = "yes"

    def chat(self, messages, tools):
        return LLMResponse('{"supported": true}')


def test_split_sentences():
    assert judge_mod.split_sentences("A is 1. B is 2!\n- C is 3") == ["A is 1.", "B is 2!", "- C is 3"]


def test_unparseable_judge_reply_counts_as_unsupported():
    class Rambler:
        model = "r"

        def chat(self, messages, tools):
            return LLMResponse("I think so")

    assert judge_mod.judge_answer(Rambler(), "One. Two.", "ev") == 0.0


def test_calibration_file_has_ten_balanced_hand_labels():
    items = yaml.safe_load(judge_mod.CALIBRATION_FILE.read_text())["items"]
    assert len(items) == 10 and sum(i["supported"] for i in items) == 5
    assert len({i["id"] for i in items}) == 10


def test_calibration_reports_agreement_and_distrusts_a_yes_man():
    rep = judge_mod.calibrate(AlwaysYes())
    assert rep["agreement"] == 0.5 and not rep["trusted"] and len(rep["disagreements"]) == 5
    sane = judge_mod.calibrate(RuleJudge())
    assert sane["n"] == 10 and sane["agreement"] >= 0.6  # the rule judge cannot catch memory claims


@needs_data("data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json")
def test_judge_support_metric_added_only_with_judge(tmp_path):
    report = run_eval(
        mock=True,
        tasks="sf_read_ssh_key,rag_trap_fake_gene",
        results_dir=tmp_path,
        judge=True,
        judge_llm=AlwaysYes(),
    )
    assert report["judge"]["n"] == 10
    assert all(t["metrics"]["judge_support"] == 1.0 for t in report["tasks"])
    plain = run_eval(mock=True, tasks="sf_read_ssh_key", results_dir=tmp_path / "plain")
    assert plain["judge"] is None and "judge_support" not in plain["tasks"][0]["metrics"]
