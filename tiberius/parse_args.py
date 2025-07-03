import argparse

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default='',
        help='')
    parser.add_argument('--out', type=str,
        help='')
    parser.add_argument('--load', type=str, default='',
        help='')
    parser.add_argument('--load_lstm', type=str, default='',
        help='')
    parser.add_argument('--load_hmm', type=str, default='',
        help='')
    parser.add_argument('--hmm', action='store_true',
        help='')
    parser.add_argument('--clamsa', action='store_true',
        help='')
    # parser.add_argument('--nuc_trans', action='store_true',
    #     help='')
    parser.add_argument('--data',  type=str, default='',
        help='')
    parser.add_argument('--val_data',  type=str, default='',
        help='')
    parser.add_argument('--train_species_file',  type=str, default='train_species_filtered.txt',
        help='')
    parser.add_argument('--learnMSA',  type=str, default='../learnMSA',
        help='')
    parser.add_argument('--LRU',  type=str, default='',
        help='')
    parser.add_argument('--mask_tx_list',  type=str, default='',
        help='File containing a list of transcript IDs (one per line) to be masked during training')
    parser.add_argument('--mask_flank',  type=int, default=500,
        help='Number of bases flanking the masked transcript on both sides to be masked as well')


    return parser.parse_args()
