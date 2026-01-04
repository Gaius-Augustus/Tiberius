nextflow.enable.dsl=2

//  TransDecoder: gtf->fasta, LongOrfs, Predict
process TD_ALL {
  label 'container'
  input:
    path gtf
    path genome

  output:
    path "transdecoder/transcripts.fasta",                    emit: cdna
    path "transdecoder/transcripts.fasta.transdecoder.pep",   emit: pep
    path "transdecoder/*.transdecoder_dir",                   emit: longdir

  script:
  """
  mkdir -p transdecoder

  # 1) GTF -> transcript FASTA
  ${params.tools.transdecoder_util_gtf2fa} ${gtf} ${genome} > transdecoder/transcripts.fasta

  # 2) Long ORFs
  ${params.tools.transdecoder_longorfs} -O transdecoder -t transdecoder/transcripts.fasta

  # 3) Predict CDS/peptides
  ${params.tools.transdecoder_predict}  -O transdecoder -t transdecoder/transcripts.fasta
  """
}

process SHORTEN_INCOMPLETE_ORFS {
  input:
    path pep
  output:
    path "shortened_candidates.pep", emit: pep_short
  script:
  """
  python ${projectDir}/scripts/shorten_incomplete.py ${pep} -o shortened_candidates.pep
  """
}

process CDS_CLASSIFY_AND_REVISE {
  input:
    path diamond_normal,  stageAs: 'diamond_normal.tsv'
    path diamond_short,   stageAs: 'diamond_short.tsv'
    path transdecoder_pep
    path shortened_pep

  output:
    path "revised_candidates.pep", emit: revised_pep
    path "classifications.json",   emit: classes

  script:
  """
  python ${projectDir}/scripts/revise_pep.py \
    --diamond_normal diamond_normal.tsv \
    --diamond_short  diamond_short.tsv  \
    --transdecoder_pep ${transdecoder_pep} \
    --shortened_pep    ${shortened_pep} \
    --revised_pep      revised_candidates.pep \
    --classifications_json classifications.json
  """
}