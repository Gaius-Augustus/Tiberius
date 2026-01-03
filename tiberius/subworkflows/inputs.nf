nextflow.enable.dsl=2

workflow INPUTS {

    take:
    params_map

    main:
    // ---- required genome ----
    if( !params_map.genome ) error "params.genome is required"
    def genomeFile = file(params_map.genome)
    if( !genomeFile.exists() ) error "Genome file not found: ${genomeFile}"
    CH_GENOME = Channel.value(genomeFile)

    CH_PROTEINS = Channel.empty()
    if( params_map.proteins ) {
        def proteinsFile = file(params_map.proteins)
        if( !proteinsFile.exists() ) error "Proteins file not found: ${proteinsFile}"
        CH_PROTEINS = Channel.value(proteinsFile)
    }

    // ---- scoring matrix (only needed when protein evidence runs) ----
    CH_SCORE = Channel.empty()
    if( params_map.scoring_matrix ) {
        def SCORE = file(params_map.scoring_matrix)
        if( !SCORE.exists() ) error "Score matrix not found: ${SCORE}"
        CH_SCORE = Channel.value(SCORE)
    }

    // ---- flags for local inputs ----
    def DO_SE_LOCAL  = params_map.rnaseq_single  && params_map.rnaseq_single.size()  > 0
    def DO_PE_LOCAL  = params_map.rnaseq_paired  && params_map.rnaseq_paired.size()  > 0
    def DO_ISO_LOCAL = params_map.isoseq         && params_map.isoseq.size()         > 0

    // ---- flags for any inputs ----
    def DO_SE  = DO_SE_LOCAL  || (params_map.rnaseq_sra_single && params_map.rnaseq_sra_single.size() > 0)
    def DO_PE  = DO_PE_LOCAL  || (params_map.rnaseq_sra_paired && params_map.rnaseq_sra_paired.size() > 0)
    def DO_ISO = DO_ISO_LOCAL || (params_map.isoseq_sra        && params_map.isoseq_sra.size()        > 0)

    // ---- infer mode ----
    def hasPaired   = DO_PE
    def hasSingle   = DO_SE
    def hasIso      = DO_ISO
    def hasProteins = params_map.proteins != null

    def MODE
    if( params_map.mode ) {
        MODE = params_map.mode
    } else {
        if( hasIso && (hasPaired || hasSingle) && hasProteins ) MODE = 'mixed'
        else if( hasIso && hasProteins )                        MODE = 'isoseq'
        else if( hasPaired || hasSingle && hasProteins )        MODE = 'rnaseq'
        else if( hasProteins )                   MODE = 'proteins'
        else                                     MODE = 'tiberius' 
    }

    emit:
    genome      = CH_GENOME
    proteins    = CH_PROTEINS
    score       = CH_SCORE
    mode        = MODE
    do_se       = Channel.value(DO_SE)
    do_pe       = Channel.value(DO_PE)
    do_iso      = Channel.value(DO_ISO)
    do_se_local = Channel.value(DO_SE_LOCAL)
    do_pe_local = Channel.value(DO_PE_LOCAL)
    do_iso_local= Channel.value(DO_ISO_LOCAL)
}
