import argparse

def parseCmd():
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Tiberius predicts gene structures from nucleotide sequences.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "    tiberius.py --genome genome.fa --model_cfg eudicotyledons --out tiberius.gtf\n"
            "Evidence integration and multi-GPU runs with Nextflow are done by Paludamentum,\n"
            "which runs Tiberius as a gene finder: https://github.com/Gaius-Augustus/Paludamentum\n"
        ),
    )
    # parser.add_argument('--model_lstm', type=str, default='',
    #     help='LSTM model file that can be used with --model_hmm to add a custom HMM layer, otherwise a default HMM layer is added.')
    general = parser.add_argument_group("General")
    general.add_argument('-p', '--params_yaml',
        help=('YAML file with "genome" and "tiberius: {model_cfg: ...}", e.g. the params file '
              'of a Paludamentum run. Fills --genome and --model_cfg if they are not given.'), default='')
    general.add_argument('--genome', type=str,
        help='Genome sequence file in FASTA format.')
    general.add_argument('--model_cfg', type=str, default='',
        help='Model config path or name in model_cfg/ (e.g. diatoms or diatoms.yaml).')
    general.add_argument('--show_cfg', action='store_true',
        help='Print the model config file in a readable format.')
    general.add_argument('--list_cfg', action='store_true',
        help='List every file in model_cfg/ with its target species.')

    tiberius_grp = parser.add_argument_group("Prediction")
    model_grp = tiberius_grp.add_mutually_exclusive_group(required=False)
    model_grp.add_argument('--model', type=str,
        help='Tiberius model with weight file (.h5) without the HMM layer.', default='')
    tiberius_grp.add_argument('--model_hmm', type=str, default='',
        help='HMM layer file that can be used instead of the default HMM.')
    tiberius_grp.add_argument('--model_lstm_old', type=str, default='',
        help=argparse.SUPPRESS)
    tiberius_grp.add_argument('--model_old', type=str,
        help=argparse.SUPPRESS, default='')
    tiberius_grp.add_argument('--out', type=str, nargs='+',
        help=('Output annotation file(s). Each path must end in .gtf, .gff, or .gff3. '
              'Specify multiple paths to write several formats simultaneously '
              '(e.g. --out tiberius.gtf tiberius.gff3).'),
        default=['tiberius.gtf'])
    tiberius_grp.add_argument('--parallel_factor', type=int, default=0,
        help='Parallel factor used in Viterbi (default uses sqrt(seq_len)).')

    tiberius_grp.add_argument('--hmm_eps', type=float, default=0.01,
        help='Deviation from the identity matrix of the HMM emitter.')
    tiberius_grp.add_argument('--hmm_initial_exon_len', type=int, default=200,
        help='Sets the transitions of the exon classes in the HMM.')
    tiberius_grp.add_argument('--hmm_initial_intron_len', type=int, default=4500,
        help='Sets the transitions of the intron classes in the HMM.')
    tiberius_grp.add_argument('--hmm_initial_ir_len', type=int, default=10000,
        help='Sets the transitions of the intergenic region in the HMM.')
    tiberius_grp.add_argument('--group_size_limit', type=int, default=100000000,
        help='This sets the maximum number of groups that can be processed by Brick2Marble at a time. Reducing this value will reduce CPU memory usage.')

    tiberius_grp.add_argument('--no_softmasking', action='store_true',
        help='Disable softmasking.')
    tiberius_grp.add_argument('--clamsa', type=str, default=None,
        help='Clamsa prefix for additional input features.')
    tiberius_grp.add_argument('--codingseq', type=str, default='',
        help='Output coding sequences as FASTA.')
    tiberius_grp.add_argument('--protseq', type=str, default='',
        help='Output protein sequences as FASTA.')
    tiberius_grp.add_argument('--seq_len', type=int, default=None,
        help='Length of sub-sequences used for parallelizing the prediction.')
    tiberius_grp.add_argument('--batch_size', type=int, default=None,
        help='Number of sub-sequences per batch.')
    tiberius_grp.add_argument('--id_prefix', type=str, default='',
        help='Prefix for gene and transcript IDs in output file(s). Works for both GTF and GFF3.')
    tiberius_grp.add_argument('--min_genome_seqlen', type=int, default=1000,
        help='Minimum length of input sequences used for predictions.')
    tiberius_grp.add_argument('--singularity', action='store_true',
        help='Run Tiberius inside the Singularity image (auto-download if missing).')
    tiberius_grp.add_argument('--cleanup_old_singularity_images', action='store_true',
        help='Delete locally cached Singularity images that do not match the pinned version.')

    return parser.parse_args()
