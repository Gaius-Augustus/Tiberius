nextflow.enable.dsl=2

include { MINIMAP2_MAP } from '../modules/isoseq.nf'
include { FILTER_ALIGNMENT as FILTER_ISOSEQ; EMPTY_FILE } from '../modules/util.nf'
include { SAMTOOLS_VIEW_SORT as SAMTOOLS_VIEW_SORT_ISO; SAMTOOLS_MERGE as SAMTOOLS_MERGE_ISO; BAM2HINTS as BAM2HINTS_ISO } from '../modules/rnaseq.nf'
include { DOWNLOAD_SRA_ISOSEQ } from '../modules/download.nf'
include { STRINGTIE_ASSEMBLE_ISO } from '../modules/assembly.nf'

workflow ISOSEQ_EVIDENCE {

    take:
    CH_GENOME
    params_map

    main:
    empty_file = EMPTY_FILE()

    def DO_ISO_LOCAL = params_map.isoseq && params_map.isoseq.size() > 0
    def DO_ISO = DO_ISO_LOCAL || (params_map.isoseq_sra && params_map.isoseq_sra.size() > 0)

    if( !DO_ISO ) {
        emit:
        hints = empty_file
        asm_gtf = Channel.empty()
        asm_gff3 = Channel.empty()
        return
    }

    CH_ISO_LOCAL = DO_ISO_LOCAL ? Channel.fromPath(params_map.isoseq, checkIfExists:true) : Channel.empty()

    CH_ISO = CH_ISO_LOCAL
    if( params_map.isoseq_sra ) {
        CH_ISO_SRA_IDS = Channel.from(params_map.isoseq_sra)
        CH_RNASEQ_ISO_SRA = DOWNLOAD_SRA_ISOSEQ(CH_ISO_SRA_IDS)
        CH_ISO = CH_ISO_LOCAL.mix(CH_RNASEQ_ISO_SRA.map { acc, f -> f })
    }

    iso_sam  = MINIMAP2_MAP(CH_GENOME, CH_ISO)
    FILTER_ISOSEQ(iso_sam.sam)
    iso_sort = SAMTOOLS_VIEW_SORT_ISO(FILTER_ISOSEQ.out)

    iso_bams = Channel.empty().mix(iso_sort.bam)
    stringtie_isoseq = STRINGTIE_ASSEMBLE_ISO(iso_bams)

    iso_merged = SAMTOOLS_MERGE_ISO(iso_bams.collect())
    iso_hints  = BAM2HINTS_ISO(iso_merged.bam, CH_GENOME)

    emit:
    hints   = iso_hints.hints
    asm_gtf = stringtie_isoseq.gtf
    asm_gff3= stringtie_isoseq.gff3
}
