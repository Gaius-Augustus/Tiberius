nextflow.enable.dsl=2

process REMOVE_CONFLICTING_WITH_PROTHINT {
  tag "prothint_conflicts"
  input: path pred_gtf; path prothint_spaln_gff
  output: path "hc/predictions.noconflict.gtf", emit: filtered
  when: params.prothint_conflict_filter
  script: """
  mkdir -p hc
  python3 ${projectDir}/scripts/removeConflictingPredictions.py     ${pred_gtf} ${prothint_spaln_gff} hc/predictions.noconflict.gtf --intronMargin 100
  """
}
