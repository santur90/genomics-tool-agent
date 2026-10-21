nextflow.enable.dsl = 2

params.counts    = "${projectDir}/../../data/rnaseq/counts.tsv"
params.metadata  = "${projectDir}/../../data/rnaseq/metadata.tsv"
params.factor    = 'condition'
params.ref_level = 'control'
params.outdir    = 'results'

process FASTQC_STUB {
    tag "$sample"

    input:
    val sample

    output:
    path "${sample}_fastqc.txt"

    script:
    """
    echo "QC placeholder for ${sample}" > ${sample}_fastqc.txt
    """

    stub:
    """
    touch ${sample}_fastqc.txt
    """
}

process COUNT_STUB {
    input:
    path qc_files

    output:
    path "counts_checked.txt"

    script:
    """
    cat ${qc_files} > counts_checked.txt
    """

    stub:
    """
    touch counts_checked.txt
    """
}

process DESEQ2 {
    publishDir params.outdir, mode: 'copy'

    input:
    path counts
    path metadata
    path checked

    output:
    path "deseq2_results.tsv"
    path "up_genes.txt"

    script:
    """
    python ${projectDir}/bin/run_deseq2.py --counts ${counts} --metadata ${metadata} --factor ${params.factor} --ref-level ${params.ref_level}
    """

    stub:
    """
    printf 'gene\\tbaseMean\\tlog2FoldChange\\tpadj\\n' > deseq2_results.tsv
    touch up_genes.txt
    """
}

workflow {
    samples = Channel.of('ctrl_1', 'ctrl_2', 'treat_1', 'treat_2')
    FASTQC_STUB(samples)
    COUNT_STUB(FASTQC_STUB.out.collect())
    DESEQ2(file(params.counts), file(params.metadata), COUNT_STUB.out)
}
