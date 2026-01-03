nextflow.enable.dsl=2

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