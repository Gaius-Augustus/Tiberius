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
    CH_PAIRED_LOCAL = Channel.empty()
    if( DO_PE ) {
        def pe = params_map.rnaseq_paired

        /*
        * Case 1: glob pattern or string → use fromFilePairs (NON-flat)
        *   "/path/*_{1,2}.fastq.gz"
        */
        if( pe instanceof CharSequence ) {

            CH_PAIRED_LOCAL =
                Channel.fromFilePairs(pe, size: 2, checkIfExists: true)
                // emits: tuple(id, [r1, r2])
        }

        /*
        * Case 2: list of explicit PAIRS
        *   - [r1, r2]
        *   - [r1, r2]
        */
        else if(
            pe instanceof List &&
            pe.every { it instanceof List && it.size() == 2 }
        ) {

            CH_PAIRED_LOCAL =
                Channel.from(pe)
                .map { pair ->
                    def r1 = file(pair[0])
                    def r2 = file(pair[1])

                    def id = r1.baseName
                        .replaceFirst(/([._-]R?1|[._-]1)$/, '')

                    tuple(id, [r1, r2])
                }
        }
        else if (DO_PE_LOCAL) {
            error "Invalid rnaseq_paired format. Expected glob string, list of pairs, or exactly two files."
        }

        /*
        * Case 3: user accidentally gives a flat list of two files
        *   - r1
        *   - r2
        * → treat as ONE library (fail-safe)
        */
        else if(
            pe instanceof List &&
            pe.size() == 2 &&
            pe.every { it instanceof CharSequence }
        ) {

            def r1 = file(pe[0])
            def r2 = file(pe[1])

            def id = r1.baseName
                .replaceFirst(/([._-]R?1|[._-]1)$/, '')

            CH_PAIRED_LOCAL =
                Channel.of( tuple(id, [r1, r2]) )
        }
        
        // CH_PAIRED_LOCAL = DO_PE_LOCAL ? Channel.fromFilePairs(params_map.rnaseq_paired, flat:true, checkIfExists:true) : Channel.empty()
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
