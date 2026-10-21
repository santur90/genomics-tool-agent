import json

from gta.tools.files import inspect_file
from gta.tools.variants import call_variants, filter_vcf, summarize_vcf
from tests.conftest import ROOT, needs_bin, needs_data

VCF = "data/variants/toy.vcf.gz"
pytestmark = needs_bin("bcftools")


def _truth() -> dict:
    return json.loads((ROOT / "data/variants/vcf_truth.json").read_text())


@needs_data(VCF, "data/variants/vcf_truth.json")
def test_inspect_vcf():
    assert inspect_file(VCF)["summary"]["n_records"] == 30


@needs_data(VCF, "data/variants/vcf_truth.json")
def test_filter_matches_truth():
    t = _truth()
    assert filter_vcf(VCF, min_qual=30, min_dp=10)["summary"]["n_output"] == t["n_qual30_dp10"]
    assert filter_vcf(VCF, min_qual=30, min_dp=10, gene="TP53")["summary"]["n_output"] == t["n_tp53_qual30_dp10"]
    assert filter_vcf(VCF, min_qual=20, gene="EGFR")["summary"]["n_output"] == t["n_egfr_qual20"]


@needs_data(VCF, "data/variants/vcf_truth.json")
def test_summarize_matches_truth():
    t = _truth()
    s = summarize_vcf(VCF)["summary"]
    assert s["n_variants"] == t["n_total"]
    assert s["by_type"] == {"SNP": t["n_snp"], "INDEL": t["n_indel"]}
    assert s["by_gene"] == t["per_gene_total"]


def _truth_snp_positions() -> set[int]:
    """Parse wgsim's mutation list (chrom, 1-based pos, ref, alt); keep SNPs."""
    positions = set()
    for line in (ROOT / "data/variants/truth.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and len(parts[2]) == 1 and len(parts[3]) == 1 and parts[2] != "-" and parts[3] != "-":
            positions.add(int(parts[1]))
    return positions


@needs_bin("bwa", "samtools")
@needs_data("data/variants/sample.bam", "data/variants/ref.fa", "data/variants/truth.txt")
def test_call_variants_recall():
    import subprocess

    res = call_variants("data/variants/sample.bam", "data/variants/ref.fa", min_qual=20)
    out = subprocess.run(["bcftools", "query", "-f", "%POS\n", res["outputs"]["vcf"]], capture_output=True, text=True, check=True).stdout
    called = {int(x) for x in out.split()}
    truth = _truth_snp_positions()
    assert truth, "no SNPs parsed from truth.txt"
    assert len(truth & called) / len(truth) >= 0.9
