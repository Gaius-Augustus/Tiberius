process RUN_TIBERIUS {
    label 'gpu', 'container', 'bigmem'

    memory '180 GB'
    input:
        path genome
        path model_cfg

    output:
        path "tiberius.${genome.name}.gtf"

    script:
    """
     python3 /home/gabriell/evidence_pipeline/Tiberius/tiberius.py \
        --genome ${genome} \
        --model_cfg ${model_cfg}\
        --out tiberius.${genome.name}.gtf
    """
}

process SPLIT_GENOME {
    input:
      path genome

    output:
      path "chunks/*.fa", emit: chunks

    script:
    """
    mkdir -p chunks
    python3 ${projectDir}/scripts/split_genome_fasta.py \\
        --genome ${genome} \\
        --outdir chunks \\
        --prefix genome_chunk \\
        --min-size ${params.tiberius.min_split_size ?: 20000000} \\
        --max-files ${params.tiberius.max_files ?: 20}
    """
}

process MERGE_TIBERIUS {
    publishDir {"${params.outdir}/"}, mode:'copy'

    input:
      path gff_files, stageAs: "?/*"

    output:
      path "tiberius.gff3"

    script:
    """
    python3 ${projectDir}/scripts/merge_annotations.py --mode full \\
        ${gff_files} > tiberius.gff3
    """
}

process MERGE_TIBERIUS_TRAIN {
    publishDir {"${params.outdir}/"}, mode:'copy'

    input:
      path tiberius 
      path traingenes

    output:
      path "tiberius_train.gff3", emit: merged

    script:
    """
    python3 ${projectDir}/scripts/merge_annotations.py --mode full \\
        ${tiberius} ${traingenes} > tiberius_train.gff3
    """
}

process MERGE_TIBERIUS_TRAIN_PRIO {
    publishDir {"${params.outdir}/"}, mode:'copy'

    input:
      path tiberius 
      path traingenes

    output:
      path "tiberius_train_prio.gff3", emit: merged

    script:
    """
    python3 ${projectDir}/scripts/merge_annotations.py --mode priority \\
        --priority-file ${traingenes} ${tiberius} ${traingenes} > tiberius_train_prio.gff3
    """
}
process PROTEIN_FROM_GFF {
  publishDir {"${params.outdir}/"}, mode:'copy'
  
  label 'container'

  input:
      path tiberius 
      path genome
    
  output:
      path "tiberius_proteins.fa"

  script:
    """
    gffread ${tiberius} \
        -g ${genome} \
        -y tiberius_proteins.fa
    """
}