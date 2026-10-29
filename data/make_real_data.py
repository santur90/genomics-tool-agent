"""Simulate an RNA-seq count matrix with real HGNC symbols and planted Reactome pathway activation.

Genes come from the offline Reactome fixture (tests/fixtures/knowledge_mini), so no download is needed.
Up-regulation is planted in two pathways: Interferon Signaling (R-HSA-913531) and
Regulation of TP53 Activity (R-HSA-5633007). Output: data/rnaseq_real/{counts,metadata}.tsv, ground_truth.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SEED = 7
N_GENES = 1500
N_PER_GROUP = 6
FOLD = 4.0
DISPERSION = 0.1
FRACTION_UP = 0.8  # share of each planted pathway's genes that is actually up-regulated
PLANTED = {"R-HSA-913531": "Interferon Signaling", "R-HSA-5633007": "Regulation of TP53 Activity"}
ROOT = Path(__file__).resolve().parent
FIXTURE_GMT = ROOT.parent / "tests" / "fixtures" / "knowledge_mini" / "ReactomePathways.gmt"
SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]*$")  # official human symbols; drops viral/lowercase entries


def read_pathways(path: Path) -> dict[str, tuple[str, list[str]]]:
    """{pathway id: (name, HGNC-style gene symbols)} from a Reactome GMT."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        out[parts[1]] = (parts[0], sorted(g for g in set(parts[2:]) if SYMBOL.match(g)))
    return out


def choose_genes(pathways: dict, rng: np.random.Generator) -> tuple[list[str], dict[str, list[str]]]:
    """Pick N_GENES symbols (all planted-pathway genes plus filler outside them) and the up genes per pathway."""
    planted_genes = sorted(set().union(*(set(pathways[p][1]) for p in PLANTED)))
    filler_pool = sorted(
        {g for pid, (_, genes) in pathways.items() if pid not in PLANTED for g in genes} - set(planted_genes)
    )
    n_filler = N_GENES - len(planted_genes)
    filler = list(rng.choice(filler_pool, n_filler, replace=False))
    genes = sorted(planted_genes + filler)
    up = {}
    for pid in PLANTED:
        members = pathways[pid][1]
        up[pid] = sorted(rng.choice(members, int(round(FRACTION_UP * len(members))), replace=False))
    return genes, up


def simulate(genes: list[str], up_genes: set[str], rng: np.random.Generator):
    """Genes x samples NB counts; planted genes are FOLD-times higher in treated samples."""
    base = np.exp(rng.normal(5.0, 1.2, len(genes)))
    is_up = np.array([g in up_genes for g in genes])
    base[is_up] = np.maximum(base[is_up], 80.0)
    size_factors = np.exp(rng.normal(0, 0.1, 2 * N_PER_GROUP))
    conditions = ["control"] * N_PER_GROUP + ["treated"] * N_PER_GROUP
    counts = np.zeros((len(genes), len(conditions)), dtype=int)
    for j, cond in enumerate(conditions):
        mu = base * size_factors[j] * (np.where(is_up, FOLD, 1.0) if cond == "treated" else 1.0)
        counts[:, j] = rng.poisson(rng.gamma(shape=1.0 / DISPERSION, scale=mu * DISPERSION))
    return counts, conditions


def main() -> None:
    """Write the demo dataset and its ground truth."""
    rng = np.random.default_rng(SEED)
    pathways = read_pathways(FIXTURE_GMT)
    genes, up = choose_genes(pathways, rng)
    up_all = set().union(*map(set, up.values()))
    counts, conditions = simulate(genes, up_all, rng)
    samples = [f"ctrl_{i}" for i in range(1, N_PER_GROUP + 1)] + [
        f"treat_{i}" for i in range(1, N_PER_GROUP + 1)
    ]
    out = ROOT / "rnaseq_real"
    out.mkdir(exist_ok=True)
    with open(out / "counts.tsv", "w") as fh:
        fh.write("gene\t" + "\t".join(samples) + "\n")
        for g, row in zip(genes, counts, strict=True):
            fh.write(g + "\t" + "\t".join(map(str, row)) + "\n")
    with open(out / "metadata.tsv", "w") as fh:
        fh.write("sample\tcondition\n")
        for s, c in zip(samples, conditions, strict=True):
            fh.write(f"{s}\t{c}\n")
    truth = {
        "seed": SEED,
        "n_genes": len(genes),
        "fold_change": FOLD,
        "fraction_of_pathway_up": FRACTION_UP,
        "planted_pathways": [
            {"id": pid, "name": name, "n_up_genes": len(up[pid])} for pid, name in PLANTED.items()
        ],
        "planted_pathway_ids": list(PLANTED),
        "up_genes": sorted(up_all),
        "note": "Pathway genes come from tests/fixtures/knowledge_mini (Reactome subset); child pathways overlap.",
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote data/rnaseq_real: {len(genes)} genes, {len(up_all)} planted up genes")


if __name__ == "__main__":
    main()
