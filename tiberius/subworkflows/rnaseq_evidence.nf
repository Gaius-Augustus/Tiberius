nextflow.enable.dsl=2

include { HISAT2_BUILD; HISAT2_MAP_SINGLE; HISAT2_MAP_PAIRED;
          SAMTOOLS_VIEW_SORT as SAMTOOLS_VIEW_SORT_SINGLE;
          SAMTOOLS_VIEW_SORT as SAMTOOLS_VIEW_SORT_PAIRED;
          SAMTOOLS_MERGE as SAMTOOLS_MERGE_RNA;
          BAM2HINTS as BAM2HINTS_RNA } from '../modules/rnaseq.nf'

include { FILTER_ALIGNMENT as FILTER_PE; FILTER_ALIGNMENT as FILTER_SE } from '../modules/util.nf'
include { DOWNLOAD_SRA_PAIRED; DOWNLOAD_SRA_SINGLE } from '../modules/download.nf'
include { STRINGTIE_ASSEMBLE_RNA } from '../modules/assembly.nf'
include { EMPTY_FILE } from '../modules/util.nf'

workflow RNASEQ_EVIDENCE {

    take:
    CH_GENOME
    params_map

    main:
    empty_file = EMPTY_FILE()

    def DO_SE_LOCAL = params_map.rnaseq_single && params_map.rnaseq_single.size() > 0
    def DO_PE_LOCAL = params_map.rnaseq_paired && params_map.rnaseq_paired.size() > 0
    def DO_SE = DO_SE_LOCAL || (params_map.rnaseq_sra_single && params_map.rnaseq_sra_single.size() > 0)
    def DO_PE = DO_PE_LOCAL || (params_map.rnaseq_sra_paired && params_map.rnaseq_sra_paired.size() > 0)

    if( !(DO_SE || DO_PE) ) {
        emit:
        hints = empty_file
        asm_gtf = Channel.empty()
        asm_gff3 = Channel.empty()
        return
    }

    // Build channels (local + SRA)
    CH_PAIRED = Channel.empty()
    if( DO_PE ) {
        CH_PAIRED_LOCAL = DO_PE_LOCAL ? Channel.fromFilePairs(params_map.rnaseq_paired, flat:true, checkIfExists:true) : Channel.empty()
        if( params_map.rnaseq_sra_paired ) {
            CH_RNASEQ_SRA_IDS_PAIRED = Channel.from(params_map.rnaseq_sra_paired)
            CH_RNASEQ_PAIRED_SRA = DOWNLOAD_SRA_PAIRED(CH_RNASEQ_SRA_IDS_PAIRED)
            CH_PAIRED_SRA = CH_RNASEQ_PAIRED_SRA.map { acc, r1, r2 -> tuple(acc, [r1, r2]) }
            CH_PAIRED = CH_PAIRED_LOCAL.mix(CH_PAIRED_SRA)
        } else CH_PAIRED = CH_PAIRED_LOCAL
    }

    CH_SINGLE = Channel.empty()
    if( DO_SE ) {
        CH_SINGLE_LOCAL = DO_SE_LOCAL ? Channel.fromPath(params_map.rnaseq_single, checkIfExists:true) : Channel.empty()
        if( params_map.rnaseq_sra_single ) {
            CH_RNASEQ_SRA_IDS_SINGLE = Channel.from(params_map.rnaseq_sra_single)
            CH_RNASEQ_SINGLE_SRA = DOWNLOAD_SRA_SINGLE(CH_RNASEQ_SRA_IDS_SINGLE)
            CH_SINGLE = CH_SINGLE_LOCAL.mix(CH_RNASEQ_SINGLE_SRA.map { acc, f -> f })
        } else CH_SINGLE = CH_SINGLE_LOCAL
    }

    index = HISAT2_BUILD(CH_GENOME)
    rnaseq_bams = Channel.empty()

    if( DO_SE ) {
        map_se = HISAT2_MAP_SINGLE(index.idxdir, CH_SINGLE)
        FILTER_SE(map_se.sam)
        sort_se = SAMTOOLS_VIEW_SORT_SINGLE(FILTER_SE.out)
        rnaseq_bams = rnaseq_bams.mix(sort_se.bam)
    }

    if( DO_PE ) {
        map_pe = HISAT2_MAP_PAIRED(index.idxdir, CH_PAIRED)
        FILTER_PE(map_pe.sam)
        sort_pe = SAMTOOLS_VIEW_SORT_PAIRED(FILTER_PE.out)
        rnaseq_bams = rnaseq_bams.mix(sort_pe.bam)
    }

    asm = STRINGTIE_ASSEMBLE_RNA(rnaseq_bams)

    rnaseq_merged = SAMTOOLS_MERGE_RNA(rnaseq_bams.collect())
    rnaseq_hints  = BAM2HINTS_RNA(rnaseq_merged.bam, CH_GENOME)

    emit:
    hints   = rnaseq_hints.hints
    asm_gtf = asm.gtf
    asm_gff3= asm.gff3
}
