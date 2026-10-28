"""Nextflow DESEQ2 step: reuse gta.tools.rnaseq and copy outputs to the work dir."""

import argparse
import shutil

from gta.tools.rnaseq import run_deseq2


def main() -> None:
    """Run DESeq2 and expose results where Nextflow expects them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--factor", default="condition")
    ap.add_argument("--ref-level", default="control")
    a = ap.parse_args()
    res = run_deseq2(counts=a.counts, metadata=a.metadata, design_factor=a.factor, ref_level=a.ref_level)
    shutil.copy(res["outputs"]["results_tsv"], "deseq2_results.tsv")
    shutil.copy(res["outputs"]["up_genes"], "up_genes.txt")


if __name__ == "__main__":
    main()
