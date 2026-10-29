import json

from gta.tools import registry


def test_specs_cover_all_tools():
    specs = registry.openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {
        "inspect_file",
        "run_deseq2",
        "run_ora",
        "call_variants",
        "filter_vcf",
        "summarize_vcf",
        "run_nextflow",
        "search_knowledge",
        "get_term",
        "annotate_genes",
    }
    assert all(s["function"]["parameters"]["type"] == "object" for s in specs)


def test_unknown_tool():
    out = registry.call("rm_rf", "{}")
    assert out["kind"] == "unknown_tool" and "hint" in out


def test_invalid_json_and_args():
    assert registry.call("filter_vcf", "{not json")["kind"] == "invalid_arguments"
    assert registry.call("filter_vcf", json.dumps({"min_qual": 1}))["kind"] == "invalid_arguments"
    assert registry.call("filter_vcf", json.dumps({"vcf": "x", "bogus": 1}))["kind"] == "invalid_arguments"


def test_tool_error_is_structured():
    out = registry.call("filter_vcf", json.dumps({"vcf": "data/variants/missing.vcf.gz"}))
    assert out["kind"] == "tool_error" and "not found" in out["error"].lower()


def test_non_object_arguments_rejected():
    out = registry.call("filter_vcf", "[1, 2]")
    assert out["kind"] == "invalid_arguments" and "JSON object" in out["error"]


def test_validation_error_names_the_field():
    out = registry.call("filter_vcf", {"vcf": "x", "min_qual": -5})
    assert out["kind"] == "invalid_arguments" and "min_qual" in out["error"]


def _register(monkeypatch, func):
    from pydantic import BaseModel

    class Empty(BaseModel):
        pass

    monkeypatch.setitem(
        registry._REGISTRY, "tmp_tool", registry.ToolSpec("tmp_tool", "test tool", Empty, func)
    )


def test_unexpected_exception_becomes_structured_error(monkeypatch):
    _register(monkeypatch, lambda: 1 / 0)
    out = registry.call("tmp_tool", "")
    assert out["kind"] == "tool_error" and "ZeroDivisionError" in out["error"] and out["hint"]


def test_results_are_made_json_serializable_and_empty_args_allowed(monkeypatch):
    from pathlib import Path

    _register(monkeypatch, lambda: {"summary": {"n": 1}, "outputs": {"file": Path("a/b.txt")}})
    out = registry.call("tmp_tool", "")
    assert out["outputs"]["file"] == "a/b.txt"
    assert registry.call("tmp_tool", None) == out and "tmp_tool" in registry.tool_names()
