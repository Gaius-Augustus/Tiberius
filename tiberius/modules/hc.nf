nextflow.enable.dsl=2

process HC_SUPPORTED {
  // publishDir "${params.outdir}", pattern: "hc/*", mode: 'copy'
  
  input:
    path diamond_revised
    path revised_pep
    path proteins
    path stringtie_gtf
    path stringtie_gff3
    path transcripts_fasta
    path miniprot_gff

  output:
    path "hc/training.gff", emit: training_gff
    path "hc/hc_genes.pep", emit: hc_pep
    path "hc/lc_genes.pep", emit: lc_pep

  script:
  """
  mkdir -p hc
  python3 ${projectDir}/scripts/generate_hc_genes.py \
    --diamond_revised ${diamond_revised} \
    --revised_pep     ${revised_pep} \
    --proteins        ${proteins} \
    --stringtie_gtf     ${stringtie_gtf} \
    --stringtie_gff3    ${stringtie_gff3} \
    --transcripts_fasta ${transcripts_fasta} \
    --miniprot_gff      ${miniprot_gff} \
    --transdecoder_util ${params.tools.transdecoder_util_orf2genome} \
    --bedtools_path     ${params.tools.bedtools} \
    --outdir            hc 
  """
}

process HC_FORMAT_FILTER {
  publishDir "${params.outdir}", pattern: "training.gff", mode: 'copy'

  input: 
    path traingff,  stageAs: 'training_original.gff'
    path genome

  output:
    path "training.gff"

  script:
  """
  python3 ${projectDir}/scripts/extend_cds_with_stop_codon.py ${traingff} > training_extended.gff
  python3 ${projectDir}/scripts/check_stop_codons.py \
    training_extended.gff ${genome} \
    --write-passing-gff training.gff > filter.tsv
  """
}