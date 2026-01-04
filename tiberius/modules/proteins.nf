nextflow.enable.dsl=2

process MINIPROT_ALIGN {
  label 'container', 'bigmem'

  input: path genome; path proteins

  output: 
    path "miniprot/miniprot.aln", emit: aln

  script: """
  mkdir -p miniprot
  ${params.tools.miniprot} -t ${params.threads} --aln ${genome} ${proteins} > miniprot/miniprot.aln
  """
}

process MINIPROT_BOUNDARY_SCORE {
  label 'container'

  input:
    path aln
    path score_matrix

  output:
    path "miniprot/miniprot_parsed.gff", emit: gff

  script:
  """
  mkdir -p miniprot
  ${params.tools.miniprot_boundary_scorer} \
    -s ${score_matrix} \
    -o miniprot/miniprot_parsed.gff \
    < ${aln}
  """
}

process MINIPROTHINT_CONVERT {
  label 'container'

  input: path gff

  output: 
    path "miniprot/miniprot.gtf", emit: gtf
    path "miniprot/miniprot_trainingGenes.gff", emit: traingff

  script: """
  mkdir -p miniprot
  ${params.tools.miniprothint} ${gff} --workdir miniprot --ignoreCoverage --topNperSeed 10 --minScoreFraction 0.5
  """
}

process ALN2HINTS {
  input: path gtf

  output: path "hints_protein.gff", emit: hints
  
  script: """
  ${projectDir}/scripts/aln2hints.pl --in=${gtf} --out=prot_hintsfile.aln2hints.temp.gff --prg=miniprot --priority=4
  cp prot_hintsfile.aln2hints.temp.gff hints_protein.gff
  """
}

process PREPROCESS_PROTEINDB {
  label 'container', 'bigmem'

  input: 
    path proteinDB 
    path tiberius_prot

  output: path "protein_preprocessed.fa"

  script: """
    # Count protein sequences (FASTA headers start with '>')
    N_PROT=\$(grep -c '^>' "${proteinDB}" || echo 0)

    echo "[PREPROCESS_PROTEINDB] Number of proteins in input: \$N_PROT" >&2

    if [[ "\$N_PROT" -le 1000000 ]]; then
        echo "[PREPROCESS_PROTEINDB] <= 1,000,000 proteins – using full DB." >&2
        ln -s "${proteinDB}" protein_preprocessed.fa
    else
        echo "[PREPROCESS_PROTEINDB] > 1,000,000 proteins – running DIAMOND soft filter." >&2

        diamond makedb \
          --in "${proteinDB}" \
          --db prot_db

        diamond blastp \
          --query "${tiberius_prot}" \
          --db prot_db \
          --out diamond_hits.tsv \
          --outfmt 6 qseqid sseqid pident length evalue bitscore qlen slen \
          --evalue 1e-5 \
          --max-target-seqs 200 \
          --very-sensitive \
          --threads ${params.threads} 

        python3 ${projectDir}/scripts/rank_species_from_diamond.py diamond_hits.tsv 13 > species_rank.tsv

        awk '
        BEGIN {
          while ((getline < "top_species.txt") > 0) {
            wanted[\$1] = 1
          }
        }
        /^>/ {
          hdr = substr(\$0, 2)
          split(hdr, a, /[ \t]/)
          id = a[1]                 # e.g. 101020_0:000003
          species = id
          sub(/_.*/, "", species)   # species = 101020
          keep = (species in wanted)
        }
        keep { print }
      ' ${proteinDB} > protein_preprocessed.fa
    fi
  """
}