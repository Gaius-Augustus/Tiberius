nextflow.enable.dsl=2

include { CONCAT_HINTS; EMPTY_FILE } from './modules/util.nf'
include { MERGE_TIBERIUS_TRAIN; MERGE_TIBERIUS_TRAIN_PRIO } from './modules/tiberius.nf'
include { HC_FORMAT_FILTER } from './modules/hc.nf'

include { INPUTS } from './subworkflows/inputs.nf'
include { PROTEIN_EVIDENCE } from './subworkflows/protein_evidence.nf'
include { RNASEQ_EVIDENCE } from './subworkflows/rnaseq_evidence.nf'
include { ISOSEQ_EVIDENCE } from './subworkflows/isoseq_evidence.nf'
include { HC_GENES } from './subworkflows/hc_genes.nf'
include { TIBERIUS_ONLY } from './subworkflows/tiberius_only.nf'

def inferMode(boolean hasPaired, boolean hasSingle, boolean hasIso, boolean hasProteins) {
  if ((hasPaired || hasSingle) && hasIso && hasProteins) return 'mixed'
  if (hasIso && hasProteins) return 'isoseq'
  if (hasPaired || hasSingle && hasProteins) return 'rnaseq'
  if (hasProteins && hasProteins) return 'proteins'
  return 'tiberius'
}

workflow {
  OUT_CH = null

  main:
    def outdir = params.outdir ?: "results"
    file(outdir).mkdirs()

    // infer from params
    def hasPaired   = params.rnaseq_paired?.size()  > 0 || params.rnaseq_sra_paired?.size() > 0
    def hasSingle   = params.rnaseq_single?.size()  > 0 || params.rnaseq_sra_single?.size() > 0
    def hasIso      = params.isoseq?.size()         > 0 || params.isoseq_sra?.size() > 0
    def proteinsList = []
    if( params.proteins ) {
      def rawList = (params.proteins instanceof List) ? params.proteins : [params.proteins]
      proteinsList = rawList.findAll { it }
    }
    def odb12List = []
    if( params.odb12Partitions ) {
      def rawOdb = (params.odb12Partitions instanceof List) ? params.odb12Partitions : [params.odb12Partitions]
      odb12List = rawOdb.findAll { it }
    }
    def hasProteins = proteinsList.size() > 0 || odb12List.size() > 0

    def tiberiusRunVal = params.tiberius?.run
    def tiberiusRun = (tiberiusRunVal instanceof Boolean) \
      ? tiberiusRunVal \
      : (tiberiusRunVal?.toString()?.trim()?.toLowerCase() in ['true','1','yes','y'])

    def MODE = params.mode ?: inferMode(hasPaired, hasSingle, hasIso, hasProteins)
    log.info "Running mode: ${MODE}"

    def inp  = INPUTS(params)

    if( MODE == 'tiberius' ) {
      def tOnly = TIBERIUS_ONLY(inp.genome, params)
      OUT_CH = tOnly.gff

    } else {

      def pe = PROTEIN_EVIDENCE(inp.genome, inp.proteins, inp.score, params)

      def empty_file = EMPTY_FILE()

      // Use fully-qualified Channel to avoid any name shadowing issues
      def re = (MODE in ['mixed','rnaseq']) \
        ? RNASEQ_EVIDENCE(inp.genome, params) \
        : [hints: empty_file, asm_gtf: nextflow.Channel.empty(), asm_gff3: nextflow.Channel.empty()]

      def ie = (MODE in ['mixed','isoseq']) \
        ? ISOSEQ_EVIDENCE(inp.genome, params) \
        : [hints: empty_file, asm_gtf: nextflow.Channel.empty(), asm_gff3: nextflow.Channel.empty()]

      def asm_gtf  = nextflow.Channel.empty()
      def asm_gff3 = nextflow.Channel.empty()

      if( MODE == 'mixed' ) {
        asm_gtf  = re.asm_gtf.mix(ie.asm_gtf)
        asm_gff3 = re.asm_gff3.mix(ie.asm_gff3)
      } else if( MODE == 'rnaseq' ) {
        asm_gtf  = re.asm_gtf
        asm_gff3 = re.asm_gff3
      } else if( MODE == 'isoseq' ) {
        asm_gtf  = ie.asm_gtf
        asm_gff3 = ie.asm_gff3
      }

      def all_hints = CONCAT_HINTS(pe.prot_hints, re.hints, ie.hints)

      def train_final
      if( MODE in ['mixed','rnaseq','isoseq'] ) {
        def tr = HC_GENES(asm_gtf, inp.genome, pe.proteindb, asm_gff3, pe.scored_gff, params)
        train_final = tr.train_gff
      } else {
        train_final = HC_FORMAT_FILTER(pe.prot_traingff, params.genome)
      }

      if( tiberiusRun ) {
        MERGE_TIBERIUS_TRAIN(pe.tiberius_gff, train_final)
        // MERGE_TIBERIUS_TRAIN_PRIO(pe.tiberius_gff, train_final)
      }

      OUT_CH = all_hints.hints
    }

  emit:
    OUT_CH
}
