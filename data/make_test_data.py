"""Simulate an RNA-seq count matrix with planted DE genes and a toy GMT file."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 42
N_GENES = 2000
N_PER_GROUP = 6
FOLD = 4.0
DISPERSION = 0.1
N_UP, N_DOWN = 50, 30

ROOT = Path(__file__).resolve().parent


def gene_name(i: int) -> str:
    """1-based gene identifier, e.g. GENE_0001."""
    return f"GENE_{i:04d}"


def simulate_counts(rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Return a genes x samples NB count matrix and the condition per sample."""
    base = np.exp(rng.normal(5.0, 1.2, N_GENES))
    base[: N_UP + N_DOWN] = np.maximum(base[: N_UP + N_DOWN], 80.0)  # planted genes stay detectable
    size_factors = np.exp(rng.normal(0, 0.1, 2 * N_PER_GROUP))
    conditions = ["control"] * N_PER_GROUP + ["treated"] * N_PER_GROUP
    fc = np.ones(N_GENES)
    fc[:N_UP] = FOLD
    fc[N_UP : N_UP + N_DOWN] = 1.0 / FOLD
    counts = np.zeros((N_GENES, len(conditions)), dtype=int)
    for j, cond in enumerate(conditions):
        mu = base * size_factors[j] * (fc if cond == "treated" else 1.0)
        lam = rng.gamma(shape=1.0 / DISPERSION, scale=mu * DISPERSION)
        counts[:, j] = rng.poisson(lam)
    return counts, conditions


def write_rnaseq(counts: np.ndarray, conditions: list[str]) -> dict:
    """Write counts, metadata, planted gene list and ground_truth.json."""
    out = ROOT / "rnaseq"
    out.mkdir(parents=True, exist_ok=True)
    samples = [f"ctrl_{i}" for i in range(1, N_PER_GROUP + 1)] + [f"treat_{i}" for i in range(1, N_PER_GROUP + 1)]
    genes = [gene_name(i) for i in range(1, N_GENES + 1)]
    with open(out / "counts.tsv", "w") as fh:
        fh.write("gene\t" + "\t".join(samples) + "\n")
        for g, row in zip(genes, counts):
            fh.write(g + "\t" + "\t".join(map(str, row)) + "\n")
    with open(out / "metadata.tsv", "w") as fh:
        fh.write("sample\tcondition\n")
        for s, c in zip(samples, conditions):
            fh.write(f"{s}\t{c}\n")
    up, down = genes[:N_UP], genes[N_UP : N_UP + N_DOWN]
    (out / "planted_up.txt").write_text("\n".join(up) + "\n")
    truth = {
        "seed": SEED, "n_genes": N_GENES, "n_samples": len(samples), "fold_change": FOLD,
        "n_up": N_UP, "n_down": N_DOWN, "up_genes": up, "down_genes": down,
        "enriched_pathway": "PATHWAY_UP",
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def write_genesets(rng: np.random.Generator, truth: dict) -> None:
    """Write toy.gmt: PATHWAY_UP plus 19 random sets."""
    out = ROOT / "genesets"
    out.mkdir(parents=True, exist_ok=True)
    all_genes = [gene_name(i) for i in range(1, N_GENES + 1)]
    non_planted = all_genes[N_UP + N_DOWN :]
    up_part = list(rng.choice(truth["up_genes"], 40, replace=False))
    noise = list(rng.choice(non_planted, 10, replace=False))
    sets = {"PATHWAY_UP": up_part + noise}
    for i in range(1, 20):
        size = int(rng.integers(30, 61))
        sets[f"SET_{i:02d}"] = list(rng.choice(all_genes, size, replace=False))
    with open(out / "toy.gmt", "w") as fh:
        for name, members in sets.items():
            fh.write(f"{name}\ttoy set\t" + "\t".join(members) + "\n")


def main() -> None:
    """Generate all RNA-seq test data."""
    rng = np.random.default_rng(SEED)
    counts, conditions = simulate_counts(rng)
    truth = write_rnaseq(counts, conditions)
    write_genesets(rng, truth)
    print("wrote data/rnaseq and data/genesets")


if __name__ == "__main__":
    main()
