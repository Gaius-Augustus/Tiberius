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
    parser.add_argument('--nuc_trans', action='store_true',
        help='')
    parser.add_argument('--data',  type=str, default='',
        help='')
    parser.add_argument('--val_data',  type=str, default='',
        help='')
    parser.add_argument('--train_species_file',  type=str, default='train_species_filtered.txt',
        help='')
    parser.add_argument('--learnMSA',  type=str, default='../learnMSA',
        help='')
    parser.add_argument('--LRU',  type=str, default='',
        help='Path to the LRU repository')
    return parser.parse_args()
