import argparse

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(
        
        description="""Tiberius predicts gene structures from a nucleotide sequences that can have repeat softmasking.

    There are flexible configuration to load the model:
    - Automatic detection of correct model for mammalian species (if no model is specified)
    - Download model using a model config file that includes URL of the model (see model_cfg/README.md for requirements)
    - Use a local model file (use --model /path/to/model/directory)

    Example usage:
        Automatic model detection:
        tiberius.py --genome genome.fa --out tiberius.gtf

        Download specific model model:
        tiberius.py --genome genome.fa --model_cfg model_cfg/mammalia_nosofttmasking_v2.yaml --out tiberius.gtf        
    """)
    # parser.add_argument('--model_lstm', type=str, default='',
    #     help='LSTM model file that can be used with --model_hmm to add a custom HMM layer, otherwise a default HMM layer is added.')
    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument('--model', type=str,    
        help='Tiberius model with weight file (.h5) without the HMM layer.', default='')
    grp.add_argument('--model_cfg', type=str, default='',
        help='Yaml file with model infomation, including URL for download. Can be used instead of --mode to specify a model.')

    parser.add_argument('--model_hmm', type=str, default='',
        help='HMM layer file that can be used instead of the default HMM.')
    parser.add_argument('--model_lstm_old', type=str, default='',
        help=argparse.SUPPRESS)
    parser.add_argument('--model_old', type=str,
        help=argparse.SUPPRESS, default='')
    parser.add_argument('--out', type=str,
        help='Output GTF file with Tiberius gene prediction.', default='tiberius.gtf')
    parser.add_argument('--genome',  type=str, #required=True,
        help='Genome sequence file in FASTA format.')
    parser.add_argument('--parallel_factor',  type=int, default=0,
        help='Parallel factor used in Viterbi. Use the factor of w_size that is closest to sqrt(w_size) (817 works well for 500004)')
    parser.add_argument('--no_softmasking', action='store_true',
        help='Disables softmasking.')
    parser.add_argument('--clamsa', type=str, default='',
        help='')
    parser.add_argument('--learnMSA',  type=str, default='',
        #help='Path to the learnMSA repository (only required if it is not installed with pip)')
        help=argparse.SUPPRESS)                        
    parser.add_argument('--codingseq', type=str, default='',
        help='Ouputs the coding sequences of all predicted genes as a FASTA file.')
    parser.add_argument('--protseq', type=str, default='',
        help='Ouputs the amino acid sequences of all predicted genes as a FASTA file.')
    parser.add_argument('--strand', type=str,
        help='Either "+" or "-" or "+,-".', default='+,-')
    parser.add_argument('--seq_len', type=int,
        help='Length of sub-sequences used for parallelizing the prediction.', default=500004)
    parser.add_argument('--batch_size', type=int,
        help='Number of sub-sequences per batch.', default=16)
    parser.add_argument('--id_prefix', type=str,
        help='Prefix for gene and transcript IDs in output GTF file.', default='')
    parser.add_argument('--min_genome_seqlen', type=int,
        help='Minimum length of input sequences used for predictions.', default=0)
    parser.add_argument('--show_cfg', action='store_true',
        help='Print the model config file in a readable format. Does not run Tiberius.')
    parser.add_argument('--list_cfg', action='store_true',
        help='List every file in model_cfg/ with its target species.')
    return parser.parse_args()