import argparse

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Tiberius predicts gene structures from nucleotide sequences.\n"
            "Use direct Tiberius inference or launch the Nextflow pipeline."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Direct Tiberius:\n"
            "    tiberius.py --genome genome.fa --model_cfg eudicotyledons --out tiberius.gtf\n"
            "  Nextflow pipeline (params file):\n"
            "    tiberius.py --params_yaml params.yaml --nf_config conf/base.config\n"
            "  Nextflow pipeline:\n"
            "    tiberius.py --nf_config conf/base.config --genome genome.fa --model_cfg eudicotyledons\n"
        ),
    )
    # parser.add_argument('--model_lstm', type=str, default='',
    #     help='LSTM model file that can be used with --model_hmm to add a custom HMM layer, otherwise a default HMM layer is added.')
    general = parser.add_argument_group("General")
    general.add_argument('-p', '--params_yaml',
        help='Path to parameters in a YAML file to specify input options and input files. See example at docs/params.yaml', default='')
    general.add_argument('--genome', type=str,
        help='Genome sequence file in FASTA format.')
    general.add_argument('--model_cfg', type=str, default='',
        help='Model config path or name in model_cfg/ (e.g. diatoms or diatoms.yaml).')
    general.add_argument('--show_cfg', action='store_true',
        help='Print the model config file in a readable format.')
    general.add_argument('--list_cfg', action='store_true',
        help='List every file in model_cfg/ with its target species.')
    
    tiberius_grp = parser.add_argument_group("Direct Tiberius only")    
    model_grp = tiberius_grp.add_mutually_exclusive_group(required=False)
    model_grp.add_argument('--model', type=str,
        help='Tiberius model with weight file (.h5) without the HMM layer.', default='')
    tiberius_grp.add_argument('--model_hmm', type=str, default='',
        help='HMM layer file that can be used instead of the default HMM.')
    tiberius_grp.add_argument('--model_lstm_old', type=str, default='',
        help=argparse.SUPPRESS)
    tiberius_grp.add_argument('--model_old', type=str,
        help=argparse.SUPPRESS, default='')
    tiberius_grp.add_argument('--out', type=str,
        help='Output GTF file with Tiberius gene prediction.', default='tiberius.gtf')
    tiberius_grp.add_argument('--parallel_factor', type=int, default=0,
        help='Parallel factor used in Viterbi (default uses sqrt(seq_len)).')
    tiberius_grp.add_argument('--no_softmasking', action='store_true',
        help='Disable softmasking.')
    tiberius_grp.add_argument('--clamsa', type=str, default='',
        help='Clamsa prefix for additional input features.')
    tiberius_grp.add_argument('--learnMSA', type=str, default='',
        help=argparse.SUPPRESS)
    tiberius_grp.add_argument('--codingseq', type=str, default='',
        help='Output coding sequences as FASTA.')
    tiberius_grp.add_argument('--protseq', type=str, default='',
        help='Output protein sequences as FASTA.')
    tiberius_grp.add_argument('--strand', type=str, default='+,-',
        help='Either "+" or "-" or "+,-".')
    tiberius_grp.add_argument('--seq_len', type=int, default=500004,
        help='Length of sub-sequences used for parallelizing the prediction.')
    tiberius_grp.add_argument('--batch_size', type=int, default=16,
        help='Number of sub-sequences per batch.')
    tiberius_grp.add_argument('--id_prefix', type=str, default='',
        help='Prefix for gene and transcript IDs in output GTF file.')
    tiberius_grp.add_argument('--min_genome_seqlen', type=int, default=0,
        help='Minimum length of input sequences used for predictions.')
    tiberius_grp.add_argument('--singularity', action='store_true',
        help='Run Tiberius inside the Singularity image (auto-download if missing).')
    
    nf_grp = parser.add_argument_group("Nextflow Pipeline")
    nf_grp.add_argument('-c', '--nf_config',
        help='Path to the Nextflow config file. See examples in conf/*.config', default='')
    nf_grp.add_argument('--profile',
        help='Nextflow profile(s) to activate (comma-separated).')
    nf_grp.add_argument('--nextflow_bin', default='nextflow',
        help='Path to the Nextflow executable to use.')
    nf_grp.add_argument('--resume', action='store_true',
        help='Pass -resume to Nextflow to continue a previous execution.')
    nf_grp.add_argument('--work_dir',
        help='Optional custom Nextflow work directory (passed to -work-dir).')
    nf_grp.add_argument('--check_tools', action='store_true',
        help='Validate native tool executables (useful when not relying on Singularity).')
    nf_grp.add_argument('--skip_singularity_check', action='store_true',
        help='Skip Singularity executable validation.')
    nf_grp.add_argument('--dry_run', action='store_true',
        help='Only run validation; do not start Nextflow.')
    nf_grp.add_argument('nextflow_args', nargs=argparse.REMAINDER,
        help='Additional arguments forwarded verbatim to Nextflow (prefix them with "--").')

    nf_params_grp = parser.add_argument_group("Nextflow Params")
    nf_params_grp.add_argument('--outdir', help='Output directory for Nextflow/Tiberius results.')
    nf_params_grp.add_argument('--threads', type=int, help='Thread count for pipeline processes.')
    nf_params_grp.add_argument('--proteins', nargs='*', default=[],
        help='Protein FASTA input(s).')
    nf_params_grp.add_argument('--odb12Partitions', nargs='*', default=[],
        help='ODB12 partition name(s) to download and append to proteins.')
    nf_params_grp.add_argument('--rnaseq_single', nargs='*', default=[],
        help='RNA-Seq single-end FASTQ input(s).')
    nf_params_grp.add_argument('--rnaseq_paired', nargs='*', default=[],
        help='RNA-Seq paired-end FASTQ input(s).')
    nf_params_grp.add_argument('--rnaseq_sra_single', nargs='*', default=[],
        help='RNA-Seq single-end SRA accession(s).')
    nf_params_grp.add_argument('--rnaseq_sra_paired', nargs='*', default=[],
        help='RNA-Seq paired-end SRA accession(s).')
    nf_params_grp.add_argument('--isoseq', nargs='*', default=[],
        help='Iso-Seq FASTQ input(s).')
    nf_params_grp.add_argument('--isoseq_sra', nargs='*', default=[],
        help='Iso-Seq SRA accession(s).')
    nf_params_grp.add_argument('--mode', help='Pipeline mode (see Nextflow docs).')
    nf_params_grp.add_argument('--scoring_matrix', help='Path to scoring matrix CSV.')
    nf_params_grp.add_argument('--prothint_conflict_filter', action='store_true',
        help='Enable prothint_conflict_filter in pipeline params.')
    nf_params_grp.add_argument('--tiberius_result', help='Path for pipeline tiberius.result param.')
    return parser.parse_args()
