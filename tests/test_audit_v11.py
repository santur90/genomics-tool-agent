"""Tool-call audit logging, argument grading, gene normalization, repeated validation errors, versioning."""

import json

import pytest
from pydantic import ValidationError

from eval.graders import arg_accuracy, grade_task, values_equal
from eval.run_eval import TASKS_VERSION, load_tasks, run_eval, write_summary
from gta.agent import ToolCallRecord, run_agent
from gta.llm import MockLLM
from gta.tools.variants import FilterVCFInput, filter_vcf
from tests.conftest import ROOT, needs_bin, needs_data

VCF = "data/variants/toy.vcf.gz"


def _rec(name, args, result=None, validated=None):
    return ToolCallRecord(name, json.dumps(args), result or {"summary": {}}, 0.0, False, validated)


def _check(tool, arg, **kw):
    return {"tool": tool, "arg": arg, **kw}


# ---- argument grading ----


def test_numeric_equivalence():
    assert values_equal(30, 30.0) and values_equal("30", 30) and values_equal(30.0, "30")
    assert not values_equal(30, 31) and not values_equal("abc", 30)


def test_string_case_and_whitespace_insensitive():
    assert values_equal("  Reactome ", "reactome")
    assert not values_equal("go", "reactome")


@pytest.mark.parametrize("val", [30, 30.0, "30"])
def test_arg_check_numeric_forms(val):
    calls = [_rec("filter_vcf", {"vcf": "x", "min_qual": val})]
    assert arg_accuracy([_check("filter_vcf", "min_qual", equals=30)], calls) == 1.0


def test_arg_check_string_forms():
    calls = [_rec("search_knowledge", {"query": "q", "source": " Reactome "})]
    assert arg_accuracy([_check("search_knowledge", "source", equals="reactome")], calls) == 1.0


def test_omitted_args_resolve_from_model_defaults():
    calls = [_rec("filter_vcf", {"vcf": "x"})]
    assert arg_accuracy([_check("filter_vcf", "min_dp", equals=0)], calls) == 1.0
    assert arg_accuracy([_check("filter_vcf", "gene", equals=None)], calls) == 1.0
    assert arg_accuracy([_check("filter_vcf", "min_dp", equals=5)], calls) == 0.0
    nf = [_rec("run_nextflow", {"pipeline": "rnaseq_mini"})]
    assert arg_accuracy([_check("run_nextflow", "stub", equals=True)], nf) == 1.0


def test_last_successful_call_is_graded():
    bad = _rec("filter_vcf", {"vcf": "x", "min_qual": 30}, {"error": "boom", "kind": "tool_error"})
    good1 = _rec("filter_vcf", {"vcf": "x", "min_qual": 10})
    good2 = _rec("filter_vcf", {"vcf": "x", "min_qual": 30})
    check = [_check("filter_vcf", "min_qual", equals=30)]
    assert arg_accuracy(check, [good1, good2]) == 1.0
    assert arg_accuracy(check, [good2, good1]) == 0.0
    assert arg_accuracy(check, [good1, bad]) == 0.0  # failed call never counts
    assert arg_accuracy(check, [bad]) == 0.0


def test_failed_calls_still_count_as_invalid():
    task = {"prompt": "p", "expected_tools": ["filter_vcf"], "arg_checks": []}
    bad = _rec("filter_vcf", {"min_qual": 1}, {"error": "x", "kind": "invalid_arguments"})
    good = _rec("filter_vcf", {"vcf": "x"})
    from gta.agent import AgentResult

    m = grade_task(task, AgentResult(task="p", answer="a", calls=[bad, good]), {})
    assert m["invalid_call_rate"] == 0.5


# ---- gene normalization ----


@pytest.mark.parametrize("raw", [None, "", "   ", "null", "None", " NULL ", "none"])
def test_gene_null_like_normalized(raw):
    assert FilterVCFInput(vcf="x", gene=raw).gene is None


def test_gene_real_value_preserved():
    assert FilterVCFInput(vcf="x", gene=" EGFR ").gene == "EGFR"


def test_gene_non_string_rejected():
    with pytest.raises(ValidationError):
        FilterVCFInput(vcf="x", gene=5)


@needs_bin("bcftools")
@needs_data(VCF, "data/variants/vcf_truth.json")
class TestFilterTruth:
    t = property(lambda self: json.loads((ROOT / "data/variants/vcf_truth.json").read_text()))

    def test_qual30_dp10(self):
        assert filter_vcf(VCF, min_qual=30, min_dp=10)["summary"]["n_output"] == self.t["n_qual30_dp10"] == 17

    def test_qual50_dp12(self):
        assert filter_vcf(VCF, min_qual=50, min_dp=12)["summary"]["n_output"] == self.t["n_qual50_dp12"] == 10

    def test_egfr_qual20(self):
        assert filter_vcf(VCF, min_qual=20, gene="EGFR")["summary"]["n_output"] == self.t["n_egfr_qual20"] == 5

    def test_empty_gene_equals_none(self):
        from gta.tools import registry

        a = registry.call("filter_vcf", {"vcf": VCF, "min_qual": 20, "gene": ""})
        b = registry.call("filter_vcf", {"vcf": VCF, "min_qual": 20, "gene": None})
        assert a["summary"] == b["summary"] and "error" not in a

    def test_invalid_gene_still_rejected(self):
        from gta.tools import registry

        out = registry.call("filter_vcf", {"vcf": VCF, "gene": "EG;FR"})
        assert out["kind"] == "tool_error" and "Invalid gene name" in out["error"]


# ---- call logging ----


def test_calls_logged_with_validated_args(tmp_path):
    llm = MockLLM.from_calls(
        [
            {"name": "filter_vcf", "arguments": {"vcf": VCF, "min_qual": "30", "gene": ""}},
            {"name": "filter_vcf", "arguments": {"min_qual": 1}},
        ],
        "done",
    )
    res = run_agent("p SECRET-PROMPT", llm, trace_dir=tmp_path)
    logs = [c.to_log() for c in res.calls]
    assert set(logs[0]) == {"name", "args", "ok", "error", "result_summary", "result_text"}
    assert logs[0]["args"] == {"vcf": VCF, "min_qual": 30.0, "min_dp": 0, "gene": None}
    assert logs[1]["ok"] is False and logs[1]["error"] and logs[1]["result_summary"] is None
    assert "SECRET-PROMPT" not in json.dumps(logs)


# ---- repeated validation failures ----


def test_repeated_validation_error_stops_and_reports(tmp_path):
    bad = {"tool_calls": [{"name": "filter_vcf", "arguments": {"min_qual": 1}}]}
    llm = MockLLM([bad] * 3 + [bad, {"content": "The tool failed: missing vcf argument."}])
    res = run_agent("p", llm, max_steps=10, trace_dir=tmp_path)
    assert res.termination_reason == "repeated_validation_error"
    assert len(res.calls) == 3
    assert "vcf" in res.answer


def test_nonconsecutive_validation_errors_do_not_stop(tmp_path):
    bad = {"tool_calls": [{"name": "filter_vcf", "arguments": {"min_qual": 1}}]}
    ok = {"tool_calls": [{"name": "inspect_file", "arguments": {"path": "data/nope.tsv"}}]}
    llm = MockLLM([bad, bad, ok, bad, {"content": "end"}])
    res = run_agent("p", llm, trace_dir=tmp_path)
    assert res.termination_reason == "final_answer" and len(res.calls) == 4


# ---- task corrections & versioning ----


def _task(tid):
    return next(t for t in load_tasks() if t["id"] == tid)


def test_task_corrections():
    from eval.graders import check_answer

    assert TASKS_VERSION == "1.2"
    go = _task("rag_find_go_term")
    assert go["prompt"] == "What is the GO term for apoptosis?"
    chk = go["answer_checks"][0]
    assert check_answer(chk, "GO:0006915", {}) and check_answer(chk, "apoptotic process", {})
    path = _task("rag_gene_to_pathways")["answer_checks"][0]
    assert check_answer(path, "see R-HSA-5633007", {}) and check_answer(path, "Regulation of TP53 Activity", {})
    for tid in ("eh_missing_vcf", "eh_missing_gmt"):
        assert check_answer(_task(tid)["answer_checks"][0], "the file is not present", {})
    assert not check_answer(_task("eh_missing_counts")["answer_checks"][0], "the file is not present", {})


def test_eval_rows_and_report_carry_version_and_calls(tmp_path):
    rep = run_eval(mock=True, tasks="st_vcf_filter_count", results_dir=tmp_path)
    assert rep["tasks_version"] == "1.2"
    row = rep["tasks"][0]
    assert row["tasks_version"] == "1.2" and row["termination_reason"]
    assert set(row["calls"][0]) == {"name", "args", "ok", "error", "result_summary", "result_text"}


def test_summary_never_merges_versions(tmp_path):
    def rep(version, model):
        r = {"model": model, "repeats": 1, "aggregate": {"overall": {}, "by_category": {}}}
        if version:
            r["tasks_version"] = version
        return r

    (tmp_path / "a_1.json").write_text(json.dumps(rep(None, "m1")))
    (tmp_path / "b_1.json").write_text(json.dumps(rep("1.1", "m1")))
    write_summary(tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert "## Tasks v1.1" in text and "## Tasks v1.0" in text


# ---- provider compatibility / robustness ----


@pytest.mark.parametrize("tools", [[], None])
def test_empty_tools_omits_tools_and_tool_choice(monkeypatch, tools):
    from types import SimpleNamespace

    from gta.llm import OpenAICompatLLM

    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(k, "x")
    sent = {}

    def create(**kw):
        sent.update(kw)
        msg = SimpleNamespace(content="hi", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    llm = OpenAICompatLLM()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.chat([{"role": "user", "content": "q"}], tools)
    assert "tools" not in sent and "tool_choice" not in sent
    llm.chat([{"role": "user", "content": "q"}], [{"type": "function"}])
    assert sent["tools"] and sent["tool_choice"] == "auto"


def test_forced_final_arbitrary_exception_uses_fallback(tmp_path):
    bad = {"tool_calls": [{"name": "filter_vcf", "arguments": {"min_qual": 1}}]}

    class Flaky(MockLLM):
        def chat(self, messages, tools):
            if not tools:
                raise RuntimeError("provider exploded")
            return super().chat(messages, tools)

    res = run_agent("p", Flaky([bad] * 5), max_steps=10, trace_dir=tmp_path)
    assert res.termination_reason == "repeated_validation_error"
    assert res.answer.startswith("I could not complete the request: filter_vcf repeatedly failed validation.")
    events = [json.loads(line)["type"] for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "forced_final_error" in events


def test_forced_final_quota_still_propagates(tmp_path):
    from gta.llm import QuotaExhausted

    bad = {"tool_calls": [{"name": "filter_vcf", "arguments": {"min_qual": 1}}]}

    class Quota(MockLLM):
        def chat(self, messages, tools):
            if not tools:
                raise QuotaExhausted("q", [])
            return super().chat(messages, tools)

    with pytest.raises(QuotaExhausted):
        run_agent("p", Quota([bad] * 5), max_steps=10, trace_dir=tmp_path)


# ---- null-like strings for optional fields ----


def test_run_ora_null_strings_behave_like_omitted():
    from gta.tools.enrichment import ORAInput

    nulls = ORAInput.model_validate(
        {"gmt": "g.gmt", "background_size": "null", "de_results": " None ", "genes": "null", "universe": ""}
    )
    assert nulls == ORAInput.model_validate({"gmt": "g.gmt"})
    assert nulls.genes is None and nulls.de_results is None and nulls.background_size is None


def test_null_strings_not_applied_to_required_or_plain_string_fields():
    from gta.tools.enrichment import ORAInput

    with pytest.raises(ValidationError):
        ORAInput.model_validate({})
    assert ORAInput.model_validate({"gmt": "null"}).gmt == "null"  # required str untouched
    from gta.tools.rnaseq import DESeq2Input

    assert DESeq2Input.model_validate(
        {"counts": "c", "metadata": "m", "design_factor": "none", "ref_level": ""}
    ).design_factor == "none"


def test_run_ora_null_strings_via_registry():
    from gta.tools import registry

    out = registry.call("run_ora", {"gmt": "nope.gmt", "background_size": "null", "genes": "null"})
    assert out["kind"] == "tool_error"  # validated; failed later on the missing file, not on arguments
