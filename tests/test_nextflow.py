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


@pytest.mark.parametrize("bad", [
    "/tmp/evil/main.nf", "pipelines/rnaseq_mini/main.nf", "../pipelines/rnaseq_mini/main.nf",
    "pipelines/rnaseq_mini", "rnaseq_mini/", "RNASEQ_MINI", "", "rnaseq_mini; rm -rf /",
])
def test_non_allowlisted_pipeline_never_reaches_subprocess(bad, monkeypatch):
    import subprocess

    from gta import sandbox
    from gta.tools import registry

    def boom(*a, **k):
        raise AssertionError("subprocess executed for a non-allowlisted pipeline")

    monkeypatch.setattr(sandbox, "run", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    out = registry.call("run_nextflow", {"pipeline": bad, "stub": False})
    assert out["error"] and out["hint"]
    assert "rnaseq_mini" in out["hint"]


def test_disallowed_param_rejected():
    with pytest.raises(ToolError):
        run_nextflow("rnaseq_mini", {"profile": "docker"})
