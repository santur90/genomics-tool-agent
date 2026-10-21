from gta.tools.nextflow import run_nextflow
from tests.conftest import needs_bin, needs_data

import pytest

from gta.sandbox import ToolError


@needs_bin("nextflow")
@needs_data("data/rnaseq/counts.tsv", "data/rnaseq/metadata.tsv")
def test_rnaseq_mini_stub_succeeds():
    res = run_nextflow("rnaseq_mini", {"counts": "data/rnaseq/counts.tsv", "metadata": "data/rnaseq/metadata.tsv"}, stub=True)
    assert res["summary"]["returncode"] == 0
    assert any(f.endswith("deseq2_results.tsv") for f in res["outputs"]["files"])


def test_unknown_pipeline_rejected():
    with pytest.raises(ToolError) as exc:
        run_nextflow("/tmp/evil/main.nf")
    assert "rnaseq_mini" in exc.value.hint


def test_disallowed_param_rejected():
    with pytest.raises(ToolError):
        run_nextflow("rnaseq_mini", {"profile": "docker"})
