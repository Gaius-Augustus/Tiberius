nextflow.enable.dsl=2

include { RUN_TIBERIUS; SPLIT_GENOME; MERGE_TIBERIUS; MERGE_TIBERIUS as REFORMAT_TIBERIUS } from '../modules/tiberius.nf'

workflow TIBERIUS_ONLY {

    take:
    CH_GENOME
    params_map

    main:
    if( !params_map.tiberius?.run ) {
        error "Mode 'tiberius' requires params.tiberius.run=true."
    }

    def tiberius
    if( params_map.tiberius?.result && file(params_map.tiberius.result).exists() ) {
        tiberius = Channel.fromPath(params_map.tiberius.result, checkIfExists: true)
        tiberius = REFORMAT_TIBERIUS(tiberius.toList())
    } else {
        genome_split   = SPLIT_GENOME(CH_GENOME)
        chunks_ch      = genome_split.chunks.flatten()
        tiberius_split = RUN_TIBERIUS(chunks_ch, params_map.tiberius.model_cfg)
        tiberius       = MERGE_TIBERIUS(tiberius_split.toList())
    }

    emit:
    gff = tiberius
}
