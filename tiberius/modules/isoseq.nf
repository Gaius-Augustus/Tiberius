nextflow.enable.dsl=2

process MINIMAP2_MAP {
  label 'container', 'bigmem'
  input: path genome; path reads
  output: path "isoseq/${reads.baseName}.bam", emit: bam
  script: """
  mkdir -p isoseq
  ${params.tools.minimap2} -ax splice:hq -uf ${genome} ${reads} -t ${params.threads} \
    | ${params.tools.samtools} sort -@ ${params.threads} -o isoseq/${reads.baseName}.bam
  """
}
