nextflow.enable.dsl=2
process HISAT2_BUILD {
  label 'container', 'bigmem'
  input:
    path genome

  output:
    path "hisat2_idx", emit: idxdir

  script:
  """
  mkdir -p hisat2_idx
  ${params.tools.hisat2_build} -p ${task.cpus} ${genome} hisat2_idx/genome
  """
}

process HISAT2_MAP_SINGLE {
  label 'container', 'bigmem'
  input:
    path idxdir
    path reads
  output:
    path "${reads.simpleName}.bam", emit: bam

  script:
  """
  ${params.tools.hisat2} -x ${idxdir}/genome -U ${reads} --dta -p ${task.cpus} \
    | ${params.tools.samtools} sort -@ ${task.cpus} -o ${reads.simpleName}.bam
  """
}

process HISAT2_MAP_PAIRED {
  label 'container', 'bigmem'
  input:
    path idxdir
    tuple val(sample), path(reads)
  output:
    path "${sample}.bam", emit: bam

  script:
  """
  mkdir -p tmp
  ${params.tools.hisat2} -x ${idxdir}/genome -1 ${reads[0]} -2 ${reads[1]} --dta -p ${task.cpus} \
    | ${params.tools.samtools} sort -@ ${task.cpus} -o ${sample}.bam
  """
}

process SAMTOOLS_MERGE {
  label 'container', 'bigmem'
  input:
    path bams

  output:
    path "merged.bam", emit: bam

  script:
  """
  ${params.tools.samtools} merge -f -@ ${task.cpus} merged.bam ${bams}
  """
}


process BAM2HINTS {
  label 'container', 'bigmem'
  input: path bam; path genome
  output: path "${bam.simpleName}.hints.gff", emit: hints
  script: """
  samtools sort -@ ${task.cpus} -o sorted.bam ${bam}
  ${params.tools.bam2hints} --intronsonly --in=sorted.bam --out=${bam}.temp
  filterIntronsFindStrand.pl ${genome} ${bam}.temp --score > ${bam.simpleName}.hints.gff
  """
}
