nextflow.enable.dsl=2

include { 
    CONCAT_PROTEINS as CONCAT_PROTEINS_1 
    CONCAT_PROTEINS as CONCAT_PROTEINS_2
    DOWNLOAD_ODB12_PARTITIONS} from '../modules/util.nf'


workflow INPUTS {

    take:
    params_map

    main:
    // ---- genome ----
    if( !params_map.genome ) error "params.genome is required"
    def genomeFile = file(params_map.genome)
    if( !genomeFile.exists() ) error "Genome file not found: ${genomeFile}"
    CH_GENOME = nextflow.Channel.value(genomeFile)

    // ---- proteins ----
    CH_PROTEINS = nextflow.Channel.empty()
    def proteinsList = []
    def proteinsFiles = []
    def localProteinsCh = nextflow.Channel.empty()
    if( params_map.proteins ) {
        def rawList = (params_map.proteins instanceof List) ? params_map.proteins : [params_map.proteins]
        proteinsList = rawList.findAll { it }
        proteinsFiles = proteinsList.collect { file(it) }
        proteinsFiles.each { if( !it.exists() ) error "Proteins file not found: ${it}" }
        if( proteinsFiles.size() == 1 ) {
            localProteinsCh = nextflow.Channel.value(proteinsFiles[0])
        } else if( proteinsFiles.size() > 1 ) {
            localProteinsCh = CONCAT_PROTEINS_1(nextflow.Channel.value(proteinsFiles))
        }
    }

    def odb12List = []
    if( params_map.odb12Partitions ) {
        def rawOdb = (params_map.odb12Partitions instanceof List) ? params_map.odb12Partitions : [params_map.odb12Partitions]
        odb12List = rawOdb.findAll { it }
        def allowed = [
            'Metazoa', 'Vertebrata', 'Viridiplantae', 'Arthropoda', 'Fungi',
            'Alveolata', 'Stramenopiles', 'Amoebozoa', 'Euglenozoa', 'Eukaryota'
        ]
        def invalid = odb12List.findAll { !(it in allowed) }
        if( invalid ) error "Unsupported odb12Partitions: ${invalid.join(', ')}"
    }

    def odb12Ch = nextflow.Channel.empty()
    if( odb12List.size() > 0 ) {
        def odb12Arg = odb12List.join(' ')
        odb12Ch = DOWNLOAD_ODB12_PARTITIONS(odb12Arg)
    }

    def hasLocalProteins = proteinsFiles.size() > 0
    def hasOdb12Proteins = odb12List.size() > 0
    if( hasLocalProteins && !hasOdb12Proteins ) {
        CH_PROTEINS = localProteinsCh
    } else if( !hasLocalProteins && hasOdb12Proteins ) {
        CH_PROTEINS = odb12Ch
    } else if( hasLocalProteins && hasOdb12Proteins ) {
        def combinedList = localProteinsCh.mix(odb12Ch).toList()
        CH_PROTEINS = CONCAT_PROTEINS_2(combinedList)
    }

    // ---- scoring matrix (only needed when protein evidence runs) ----
    CH_SCORE = nextflow.Channel.empty()
    if( params_map.scoring_matrix ) {
        def SCORE = file(params_map.scoring_matrix)
        if( !SCORE.exists() ) error "Score matrix not found: ${SCORE}"
        CH_SCORE = nextflow.Channel.value(SCORE)
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
    def hasProteins = proteinsList.size() > 0 || odb12List.size() > 0

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
    do_se       = nextflow.Channel.value(DO_SE)
    do_pe       = nextflow.Channel.value(DO_PE)
    do_iso      = nextflow.Channel.value(DO_ISO)
    do_se_local = nextflow.Channel.value(DO_SE_LOCAL)
    do_pe_local = nextflow.Channel.value(DO_PE_LOCAL)
    do_iso_local= nextflow.Channel.value(DO_ISO_LOCAL)
}
