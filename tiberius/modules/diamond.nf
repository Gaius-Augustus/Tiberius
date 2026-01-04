nextflow.enable.dsl=2

process DIAMOND_MAKEDB {
  label 'container'
  input: path proteins
  output: path "diamond/protein_db.dmnd", emit: db
  script: """
  mkdir -p diamond
  ${params.tools.diamond} makedb --in ${proteins} -d diamond/protein_db
  """
}
process DIAMOND_BLASTP {
  label 'container'
  input: path pep; path db
  output: path "diamond/diamond.tsv", emit: tsv
  script: """
  mkdir -p diamond
  ${params.tools.diamond} blastp -q ${pep} -d ${db} -o diamond/diamond.tsv -k 5 --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
  """
}
