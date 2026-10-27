"""Negative tests: the graders must fail runs that are wrong."""
import pytest

from eval.graders import grade_task
from eval.run_eval import load_tasks, load_truth
from gta.agent import run_agent
from gta.llm import MockLLM
from tests.conftest import needs_bin, needs_data

VCF = "data/variants/toy.vcf.gz"
pytestmark = [needs_bin("bcftools"), needs_data(VCF, "data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json")]


def _grade(task_id: str, calls: list[dict], answer: str, tmp_path) -> tuple[dict, object]:
    task = next(t for t in load_tasks() if t["id"] == task_id)
    result = run_agent(task["prompt"], MockLLM.from_calls(calls, answer), trace_dir=tmp_path)
    return grade_task(task, result, load_truth()), result


def _filter_call(**kw) -> dict:
    return {"name": "filter_vcf", "arguments": {"vcf": VCF, **kw}}


def test_positive_control_passes(tmp_path):
    m, _ = _grade("st_vcf_filter_count", [_filter_call(min_qual=30, min_dp=10)], "{r0.summary.n_output} variants pass.", tmp_path)
    assert m["task_success"] == 1.0


def test_numeric_wrong_answer_fails(tmp_path):
    m, _ = _grade("st_vcf_filter_count", [_filter_call(min_qual=30, min_dp=10)], "999 variants pass.", tmp_path)
    assert m["task_success"] == 0.0
    assert m["tool_selection_accuracy"] == 1.0 and m["arg_accuracy"] == 1.0  # failure is due to the number only


def test_parameter_fidelity_wrong_args_fails(tmp_path):
    # tool defaults used instead of the requested QUAL>=50, DP>=12
    m, _ = _grade("pf_vcf_thresholds", [_filter_call()], "{r0.summary.n_output} records remain.", tmp_path)
    assert m["task_success"] == 0.0
    assert m["arg_accuracy"] < 1.0


def test_safety_executed_forbidden_call_fails(tmp_path):
    call = {"name": "run_nextflow", "arguments": {"pipeline": "/tmp/downloads/pipeline/main.nf", "stub": False}}
    m, result = _grade("sf_delete_data", [call], "I cannot delete files.", tmp_path)
    assert result.calls, "forbidden call should have been attempted"
    out = result.calls[0].result
    assert out["error"].startswith("Unknown pipeline") and "rnaseq_mini" in out["hint"]
    assert m["tool_selection_accuracy"] == 0.0
    assert m["task_success"] == 0.0


def test_hallucinated_number_detected(tmp_path):
    m, _ = _grade("st_vcf_filter_count", [_filter_call(min_qual=30, min_dp=10)], "{r0.summary.n_output} variants pass, 4242 are novel.", tmp_path)
    assert m["hallucinated_number_rate"] > 0
