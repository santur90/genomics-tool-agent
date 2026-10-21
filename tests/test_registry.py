import json

from gta.tools import registry


def test_specs_cover_all_tools():
    specs = registry.openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"inspect_file", "run_deseq2", "run_ora", "call_variants", "filter_vcf", "summarize_vcf", "run_nextflow"}
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
