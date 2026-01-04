nextflow.enable.dsl=2

process CONCAT_PROTEINS {
  tag "concat_proteins"

  input:
  path proteins, stageAs:  "?/*" 

  output:
  path "proteins_concat.faa"

  shell:
  def input_str = proteins instanceof List ? proteins.join(" ") : proteins
  """
    cat ${input_str} > proteins_concat.faa
  """
}

process DOWNLOAD_ODB12_PARTITIONS {
  tag "odb12_partitions"

  input:
  val partitions

  output:
  path "odb12_partitions.faa"

  script:
  """
  set -euo pipefail
  : > odb12_partitions.faa
  for part in ${partitions}; do
      url="https://bioinf.uni-greifswald.de/bioinf/partitioned_odb12/\${part}.fa.gz"
      curl -fsSL "\${url}" | gunzip -c >> odb12_partitions.faa
  done
  """
}


process CONCAT_HINTS {
  publishDir "${params.outdir}", mode: 'copy'   

  input:
    path(prot)
    path(rnaseq), stageAs: 'rnaseq_hints.gff'
    path(isoseq), stageAs: 'isoseq_hints.gff'

  output:
    path "hintsfile.gff", emit: hints

  script:
  """
  cat ${prot} ${ rnaseq ?: "/dev/null" } ${ isoseq ?: "/dev/null" } > hintsfile.gff
  """
}

process EMPTY_FILE {
  output:
    path 'empty.txt'

  label 'local_only'
  
  script:
  """
    touch empty.txt
  """
}

process CALC_ALIGNMENT_RATE {
    tag "${aln_file.simpleName}"

    input:
    path aln_file

    output:
    tuple path(aln_file), path("alignment_rate.txt")

    """
    pct=\$(samtools flagstat ${aln_file} \\
        | awk '/ mapped \\(/ { 
              # take 5th field, e.g. (99.79%
              val=\$5
              # remove everything that is not a digit or a dot
              gsub(/[^0-9.]/, "", val)
              print val
              exit
          }')
    echo "\$pct" > alignment_rate.txt
    """
}


workflow FILTER_ALIGNMENT {
    take:
    aln_ch

    main:
    CALC_ALIGNMENT_RATE(aln_ch)

    CALC_ALIGNMENT_RATE.out
        .map { file, ratefile ->
            def pct = ratefile.text.trim().toFloat()
            tuple(file, pct)
        }
        .set { file_pct_ch }

    file_pct_ch
        .filter { file, pct -> pct < 80 }
        .view { file, pct ->
            "Removing ${file.simpleName} (alignment ${String.format('%.2f', pct)}%)"
        }

    file_pct_ch
        .filter { file, pct -> pct >= 80 }
        .map { file, pct -> file }
        .set { filtered_ch }

    emit:
    filtered_ch
}