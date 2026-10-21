from eval.graders import arg_accuracy, check_answer, extract_numbers, hallucinated_number_rate, tool_selection
from eval.run_eval import load_tasks
from gta.agent import ToolCallRecord
from tests.conftest import needs_bin, needs_data


def test_extract_numbers_ignores_identifiers_and_list_markers():
    assert extract_numbers("1. GENE_0001 has padj 1.2e-5 and 2,000 reads (30%)") == ["1.2e-5", "2000", "30"]


def test_hallucination_rate():
    results = [{"summary": {"n_up": 50, "padj": 0.0123}}]
    assert hallucinated_number_rate("50 genes, padj 0.01", "run it", results) == 0.0
    assert hallucinated_number_rate("51 genes", "run it", results) == 1.0
    assert hallucinated_number_rate("cutoff 0.05", "use 0.05", results) == 0.0


def test_tool_selection_order_and_refusal():
    assert tool_selection(["a", "b"], ["a", "x", "b"], False) == 1.0
    assert tool_selection(["a", "b"], ["b", "a"], False) == 0.0
    assert tool_selection([], [], True) == 1.0
    assert tool_selection([], ["a"], True) == 0.0


def test_arg_accuracy():
    calls = [ToolCallRecord("run_deseq2", '{"padj_cutoff": 0.01}', {}, 0.0)]
    assert arg_accuracy([{"tool": "run_deseq2", "arg": "padj_cutoff", "equals": 0.01}], calls) == 1.0
    assert arg_accuracy([{"tool": "run_deseq2", "arg": "lfc_cutoff", "equals": 2}], calls) == 0.0


def test_check_answer_tolerance():
    truth = {"rnaseq": {"n_up": 50}}
    assert check_answer({"type": "number", "truth": "rnaseq.n_up", "tol": 5}, "I found 53 genes", truth)
    assert not check_answer({"type": "number", "truth": "rnaseq.n_up", "tol": 5}, "I found 70 genes", truth)


def test_task_file_shape():
    tasks = load_tasks()
    assert len(tasks) == 25
    cats = {t["category"] for t in tasks}
    assert cats == {"single_tool", "multi_step", "parameter_fidelity", "error_handling", "safety"}
    assert len({t["id"] for t in tasks}) == 25
    assert all(t["mock"]["answer"] for t in tasks)


@needs_bin("bcftools", "samtools", "nextflow")
@needs_data("data/rnaseq/ground_truth.json", "data/variants/vcf_truth.json", "data/variants/sample.bam")
def test_mock_eval_all_pass(tmp_path):
    from eval.run_eval import run_eval

    report = run_eval(mock=True, repeats=1, results_dir=tmp_path)
    failed = [t["id"] for t in report["tasks"] if t["metrics"]["task_success"] < 1]
    assert not failed, failed
    assert (tmp_path / "summary.md").exists()
