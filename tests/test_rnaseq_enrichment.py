import json

from gta.tools.enrichment import bh_adjust, run_ora
from gta.tools.files import inspect_file
from gta.tools.rnaseq import run_deseq2
from tests.conftest import ROOT, needs_data

import numpy as np

COUNTS, META = "data/rnaseq/counts.tsv", "data/rnaseq/metadata.tsv"


def test_bh_adjust_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = bh_adjust(p)
    assert adj.max() <= 1 and adj[0] <= adj[2] <= adj[1] <= adj[3]


@needs_data(COUNTS, META)
def test_inspect_counts():
    s = inspect_file("data/rnaseq/counts.tsv")["summary"]
    assert s["n_rows"] == 2000 and s["n_columns"] == 13


@needs_data(COUNTS, META, "data/rnaseq/ground_truth.json")
def test_deseq2_recovers_planted_genes():
    truth = json.loads((ROOT / "data/rnaseq/ground_truth.json").read_text())
    res = run_deseq2(COUNTS, META, "condition", "control")
    up = set(open(res["outputs"]["up_genes"]).read().split())
    down = set(open(res["outputs"]["down_genes"]).read().split())
    assert len(up & set(truth["up_genes"])) / truth["n_up"] >= 0.9
    assert len(down & set(truth["down_genes"])) / truth["n_down"] >= 0.9
    assert len(up - set(truth["up_genes"])) <= 5  # few false positives
    assert res["summary"]["n_up"] == len(up)


@needs_data(COUNTS, META)
def test_deseq2_bad_column_raises_hint():
    import pytest

    from gta.sandbox import ToolError

    with pytest.raises(ToolError) as exc:
        run_deseq2(COUNTS, META, "treatment", "control")
    assert "condition" in exc.value.hint


@needs_data("data/genesets/toy.gmt", "data/rnaseq/planted_up.txt")
def test_ora_top_is_pathway_up():
    res = run_ora("data/genesets/toy.gmt", gene_list_file="data/rnaseq/planted_up.txt")
    top = res["summary"]["top"][0]
    assert top["set"] == "PATHWAY_UP" and top["padj"] < 1e-10
