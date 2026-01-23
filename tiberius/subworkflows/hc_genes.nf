nextflow.enable.dsl=2

include { STRINGTIE_MERGE } from '../modules/assembly.nf'
include { TD_ALL; SHORTEN_INCOMPLETE_ORFS; CDS_CLASSIFY_AND_REVISE } from '../modules/transdecoder.nf'
include { DIAMOND_MAKEDB; DIAMOND_BLASTP as DIAMOND_BLASTP_NORM; DIAMOND_BLASTP as DIAMOND_BLASTP_SHORT; DIAMOND_BLASTP as DIAMOND_BLASTP_REV } from '../modules/diamond.nf'
include { HC_SUPPORTED; HC_FORMAT_FILTER } from '../modules/hc.nf'

process EMPTY_TSV {
  output:
  path "empty.tsv", emit: tsv
  script:
  """
  : > empty.tsv
  """
}


workflow HC_GENES {

    take:
    asm_gtf_ch
    CH_GENOME
    proteindb
    asm_gff3_ch
    td_scored_gff   
    params_map

    main:
    asm      = STRINGTIE_MERGE(asm_gtf_ch.collect())
    td_all   = TD_ALL(asm.gtf, CH_GENOME)

    pep_short = SHORTEN_INCOMPLETE_ORFS(td_all.pep)
    db        = DIAMOND_MAKEDB(proteindb)

    dia_norm  = DIAMOND_BLASTP_NORM(td_all.pep, db.db)
    dia_short = DIAMOND_BLASTP_SHORT(pep_short.pep_short, db.db)

    rev       = CDS_CLASSIFY_AND_REVISE(dia_norm.tsv, dia_short.tsv, td_all.pep, pep_short.pep_short)
    dia_rev   = DIAMOND_BLASTP_REV(rev.revised_pep, db.db)

    hc = HC_SUPPORTED(
        dia_rev.tsv,
        rev.revised_pep,
        proteindb,
        asm.gtf,
        asm.gff3,    
        td_all.cdna,
        td_scored_gff
    )

    train_final = HC_FORMAT_FILTER(hc.training_gff, params_map.genome)

    emit:
    train_gff = train_final
}
