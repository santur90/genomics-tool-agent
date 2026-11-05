"""Tests for the rule-based answer verifier (src/gta/verify.py)."""

from pathlib import Path

import gta.verify as verify_mod
from gta.verify import verify


def _ok(name, args, summary=None, text=None):
    return {"name": name, "args": args, "ok": True, "error": None, "result_summary": summary, "result_text": text}


def _err(name, args, error):
    return {"name": name, "args": args, "ok": False, "error": error, "result_summary": None}


def _types(rep):
    return sorted(rep.types)


# ---- numbers --------------------------------------------------------------------------------------


def test_correct_answer_is_not_flagged():
    prompt = "How many variants in toy.vcf.gz have QUAL>=30 and DP>=10?"
    calls = [_ok("filter_vcf", {"min_qual": 30, "min_dp": 10}, {"n_input": 30, "n_output": 17})]
    rep = verify(prompt, calls, "17 variants pass QUAL>=30 and DP>=10.")
    assert not rep.flagged


def test_number_not_in_any_source_is_flagged():
    prompt = "How many variants have QUAL>=30?"
    calls = [_ok("filter_vcf", {"min_qual": 30}, {"n_output": 17})]
    rep = verify(prompt, calls, "There are 999 variants.")
    assert _types(rep) == ["unsupported_number"]


def test_number_echoed_from_prompt_is_not_flagged():
    prompt = "Use padj < 0.01 and lfc 1.5. How many genes are up?"
    calls = [_ok("run_deseq2", {"padj_cutoff": 0.01}, {"n_up": 49})]
    assert not verify(prompt, calls, "With padj < 0.01 and lfc 1.5 there are 49 up-regulated genes.").flagged


def test_rounded_and_scientific_numbers_are_supported():
    calls = [_ok("run_ora", {}, None, '{"p_value": 0.25131, "padj": 2.762e-49}')]
    assert not verify("q", calls, "p = 0.2513 and padj = 2.762e-49").flagged
    assert verify("q", calls, "p = 0.31").flagged


def test_ids_and_gene_tokens_are_not_numbers():
    calls = [_ok("search_knowledge", {}, None, '{"id": "R-HSA-109581", "gene": "GENE_0076"}')]
    assert not verify("q", calls, "GENE_0076 is in R-HSA-109581 (SET_02).").flagged


# ---- citations ------------------------------------------------------------------------------------


def test_citation_never_retrieved_is_flagged():
    calls = [_ok("search_knowledge", {}, None, '{"id": "GO:0006915"}')]
    rep = verify("q", calls, "See GO:0006915 and GO:0012501.")
    assert _types(rep) == ["unsupported_citation"]
    assert rep.issues[0].evidence == "GO:0012501"


def test_citation_quoted_from_prompt_is_not_flagged():
    prompt = "What does GO:9999999 refer to?"
    calls = [_err("get_term", {"id": "GO:9999999"}, "Unknown term id (unknown id): GO:9999999")]
    assert not verify(prompt, calls, "GO:9999999 does not exist in the knowledge base.").flagged


# ---- input substitution ---------------------------------------------------------------------------

_PROMPT_COUNTS = "Run DESeq2 on data/rnaseq/counts_v2.tsv with data/rnaseq/metadata.tsv and give the number of up genes."


def _substituted_calls():
    return [
        _err("run_deseq2", {"counts": "data/rnaseq/counts_v2.tsv"}, "File not found: data/rnaseq/counts_v2.tsv"),
        _ok("run_deseq2", {"counts": "data/rnaseq/counts.tsv"}, {"n_up": 52}),
    ]


def test_silent_input_substitution_is_flagged():
    rep = verify(_PROMPT_COUNTS, _substituted_calls(), "There are 52 up-regulated genes.")
    assert _types(rep) == ["input_substitution"]


def test_disclosed_substitution_is_not_flagged():
    ans = "counts_v2.tsv was not found, so I used counts.tsv instead: 52 up-regulated genes."
    assert not verify(_PROMPT_COUNTS, _substituted_calls(), ans).flagged


def test_wrong_column_substitution_is_flagged():
    prompt = "Run DESeq2 using the column 'treatment' as the design factor."
    calls = [
        _err("run_deseq2", {"design_factor": "treatment"}, "Column 'treatment' not found in metadata."),
        _ok("run_deseq2", {"design_factor": "condition"}, {"n_up": 52}),
    ]
    assert _types(verify(prompt, calls, "Used the 'condition' column: 52 genes up.")) == ["input_substitution"]


def test_validation_error_retry_is_not_substitution():
    prompt = "Run DESeq2 with condition treated vs control."
    calls = [
        _err("run_deseq2", {"padj_cutoff": "0"}, "Invalid arguments for run_deseq2: ['padj_cutoff: Input should be greater than 0']"),
        _ok("run_deseq2", {"padj_cutoff": 0.05}, {"n_up": 52}),
    ]
    assert not verify(prompt, calls, "52 genes are up-regulated.").flagged


def test_value_chosen_by_model_is_not_substitution():
    calls = [
        _err("run_ora", {"gmt": "data/genesets/other.gmt"}, "File not found: data/genesets/other.gmt"),
        _ok("run_ora", {"gmt": "data/genesets/toy.gmt"}, {"n_query": 52}),
    ]
    assert not verify("Run enrichment.", calls, "52 genes were tested.").flagged


# ---- final failure --------------------------------------------------------------------------------


def test_empty_answer_after_failure_is_flagged():
    calls = [_err("run_ora", {"gmt": "data/genesets/hallmark.gmt"}, "File not found: data/genesets/hallmark.gmt")]
    assert _types(verify("Run ORA", calls, "")) == ["unresolved_error"]


def test_answer_that_does_not_name_the_missing_file_is_flagged():
    calls = [_err("run_ora", {"gmt": "data/genesets/hallmark.gmt"}, "File not found: data/genesets/hallmark.gmt")]
    ans = "I cannot perform this task as it requires additional information about the files and their locations."
    assert _types(verify("Run ORA against data/genesets/hallmark.gmt", calls, ans)) == ["unreported_error"]


def test_answer_that_reports_missing_file_is_not_flagged():
    calls = [_err("filter_vcf", {"vcf": "data/variants/patient_001.vcf.gz"}, "File not found: data/variants/patient_001.vcf.gz")]
    prompt = "Filter data/variants/patient_001.vcf.gz with QUAL>=30 and tell me how many remain."
    assert not verify(prompt, calls, "The file patient_001.vcf.gz was not found, so I cannot count variants.").flagged
    assert not verify(prompt, calls, "I cannot do this because the file is not present in the system.").flagged


def test_results_stated_after_failed_last_call_is_flagged():
    calls = [
        _ok("run_deseq2", {}, {"n_up": 52}),
        _err("run_ora", {"gmt": "data/genesets/x.gmt"}, "Invalid arguments for run_ora: ['gmt: bad']"),
    ]
    assert "unresolved_error" in _types(verify("q", calls, "52 genes are up."))


# ---- skipped / purity -----------------------------------------------------------------------------


def test_numbers_are_skipped_when_no_output_is_stored():
    calls = [{"name": "filter_vcf", "args": {}, "ok": True, "error": None, "result_summary": None}]
    rep = verify("q", calls, "There are 17 variants.")
    assert not rep.flagged and rep.skipped


def test_verifier_has_no_access_to_eval_or_task_files():
    src = Path(verify_mod.__file__).read_text()
    for forbidden in ("import eval", "from eval", "import yaml", "tasks.yaml", "graders", "expected"):
        assert forbidden not in src.replace('"""', "").split("from __future__")[1], forbidden


def test_run_directory_paths_are_not_numbers():
    calls = [_ok("filter_vcf", {}, {"n_output": 10})]
    ans = "Output: /Users/q/Documents/Github/genomics-tool-agent/runs/work/filtervcf-20261029-161342-7579/filtered.vcf.gz (10 records)."
    assert not verify("q", calls, ans).flagged


def test_ratios_like_overlap_are_still_checked():
    calls = [_ok("run_ora", {}, None, '{"overlap": 40, "set_size": 50}')]
    assert not verify("q", calls, "Overlap 40/50.").flagged
    assert verify("q", calls, "Overlap 41/50.").flagged


def test_real_unsupported_number_next_to_a_path_is_still_flagged():
    calls = [_ok("filter_vcf", {}, {"n_output": 10})]
    assert verify("q", calls, "Wrote runs/work/filtervcf-20261029-1/out.vcf.gz with 999 records.").flagged
