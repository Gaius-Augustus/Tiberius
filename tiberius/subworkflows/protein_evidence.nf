nextflow.enable.dsl=2

include { MINIPROT_ALIGN; MINIPROT_BOUNDARY_SCORE; MINIPROTHINT_CONVERT; ALN2HINTS; PREPROCESS_PROTEINDB } from '../modules/proteins.nf'
include { RUN_TIBERIUS; SPLIT_GENOME; PROTEIN_FROM_GFF; MERGE_TIBERIUS; MERGE_TIBERIUS as REFORMAT_TIBERIUS } from '../modules/tiberius.nf'

workflow PROTEIN_EVIDENCE {

    // Declare in workflow scope so emit: can access them
    def proteindb_ch
    def tiberius_gff_ch = Channel.empty()
    def scored_ch
    def prot_gtf_ch
    def prot_hints_ch

    take:
    CH_GENOME
    CH_PROTEINS
    CH_SCORE
    params_map

    main:
    if( !params_map.proteins )        error "params.proteins is required for protein evidence modes"
    if( !params_map.scoring_matrix )  error "params.scoring_matrix is required for protein evidence modes"

    if( params_map.tiberius?.run ) {
        if( params_map.tiberius?.result && file(params_map.tiberius.result).exists() ) {
            tiberius_gff_ch = Channel.fromPath(params_map.tiberius.result, checkIfExists: true)
            tiberius_gff_ch = REFORMAT_TIBERIUS(tiberius_gff_ch.toList())
        } else {
            genome_split   = SPLIT_GENOME(CH_GENOME)
            chunks_ch      = genome_split.chunks.flatten()
            tiberius_split = RUN_TIBERIUS(chunks_ch, params_map.tiberius.model_cfg)
            tiberius_gff_ch = MERGE_TIBERIUS(tiberius_split.toList())
        }

        tiberius_prot = PROTEIN_FROM_GFF(tiberius_gff_ch, CH_GENOME)
        proteindb_ch = PREPROCESS_PROTEINDB(CH_PROTEINS, tiberius_prot)
    }
    else {
        proteindb_ch = CH_PROTEINS
    }

    prot_aln    = MINIPROT_ALIGN(CH_GENOME, proteindb_ch)
    scored_ch   = MINIPROT_BOUNDARY_SCORE(prot_aln.aln, CH_SCORE)
    prot_gtf_ch = MINIPROTHINT_CONVERT(scored_ch.gff)
    prot_hints_ch = ALN2HINTS(prot_gtf_ch.gtf)

    emit:
    proteindb     = proteindb_ch
    tiberius_gff  = tiberius_gff_ch
    scored_gff    = scored_ch.gff
    prot_gtf      = prot_gtf_ch.gtf
    prot_traingff = prot_gtf_ch.traingff
    prot_hints    = prot_hints_ch.hints
}
