process DOWNLOAD_SRA_PAIRED {
  label 'container', 'bigmem'

  publishDir "${params.outdir}/sra_downloads/rnaseq_sra_paired/", mode: 'copy'

  input:
    val acc

  output:
    tuple val(acc), path("${acc}_1.fastq.gz"), path("${acc}_2.fastq.gz")

  script:
  """
  fasterq-dump --split-files --threads ${params.threads} ${acc}
  gzip ${acc}_1.fastq ${acc}_2.fastq
  """
}

process DOWNLOAD_SRA_SINGLE {
  label 'container', 'bigmem'

  publishDir "${params.outdir}/sra_downloads/rnaseq_sra_single/", mode: 'copy'

  input:
    val acc

  output:
    tuple val(acc), path("${acc}.fastq.gz")

  script:
  """
  fasterq-dump --threads ${params.threads} ${acc}
  gzip ${acc}.fastq
  """
}

process DOWNLOAD_SRA_ISOSEQ {
  label 'container', 'bigmem'

  publishDir "${params.outdir}/sra_downloads/isoseq_sra/", mode: 'copy'

  input:
    val acc

  output:
    tuple val(acc), path("${acc}.fastq.gz")

  script:
  """
  fasterq-dump --threads ${params.threads} ${acc}
  gzip ${acc}.fastq
  """
}
