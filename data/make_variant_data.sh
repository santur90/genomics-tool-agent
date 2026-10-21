#!/usr/bin/env bash
# Simulate a 50 kb reference, reads (wgsim), alignment (bwa), and a hand-crafted toy VCF.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/variants"
mkdir -p "$DIR"
cd "$DIR"

python - <<'PY'
import random
rng = random.Random(7)
seq = "".join(rng.choice("ACGT") for _ in range(50000))
with open("ref.fa", "w") as fh:
    fh.write(">chr1\n")
    for i in range(0, len(seq), 60):
        fh.write(seq[i:i + 60] + "\n")
PY

samtools faidx ref.fa
bwa index ref.fa 2> /dev/null

# wgsim prints its true mutation list on stdout
wgsim -S 11 -r 0.001 -N 100000 ref.fa r1.fq r2.fq > truth.txt
bwa mem -t 2 ref.fa r1.fq r2.fq 2> /dev/null | samtools sort -l 9 -o sample.bam -
samtools index sample.bam
rm -f r1.fq r2.fq

# hand-crafted VCF with known QUAL / DP / GENE values
python - <<'PY'
import json
from collections import Counter

genes = ["TP53", "KRAS", "EGFR", "BRCA1", "NRAS"]
bases = "ACGT"
records = []
for i in range(30):
    pos = 1000 + 1500 * i
    ref = bases[i % 4]
    indel = i % 6 == 5
    alt = ref + "T" if indel else bases[(i + 1) % 4]
    records.append(dict(pos=pos, ref=ref, alt=alt, qual=10 + 3 * i, dp=5 + (i * 7) % 20,
                        gene=genes[i % 5], indel=indel))

with open("toy.vcf", "w") as fh:
    fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=50000>\n")
    fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
    fh.write('##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">\n')
    fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for r in records:
        fh.write(f"chr1\t{r['pos']}\t.\t{r['ref']}\t{r['alt']}\t{r['qual']}\tPASS\tDP={r['dp']};GENE={r['gene']}\n")

def keep(q=0, d=0, gene=None):
    return [r for r in records if r["qual"] >= q and r["dp"] >= d and (gene is None or r["gene"] == gene)]

truth = {
    "n_total": len(records),
    "n_snp": sum(not r["indel"] for r in records),
    "n_indel": sum(r["indel"] for r in records),
    "n_qual30_dp10": len(keep(30, 10)),
    "n_qual50_dp12": len(keep(50, 12)),
    "n_tp53_qual30_dp10": len(keep(30, 10, "TP53")),
    "n_egfr_qual20": len(keep(20, 0, "EGFR")),
    "per_gene_total": dict(Counter(r["gene"] for r in records)),
}
json.dump(truth, open("vcf_truth.json", "w"), indent=2)
PY

bcftools view -Oz -o toy.vcf.gz toy.vcf
bcftools index -f toy.vcf.gz
rm -f toy.vcf
du -sh "$DIR"
