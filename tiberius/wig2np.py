import os, csv , sys, argparse
import numpy as np
from wig_class import Wig_util

def main():
    args = parseCmd()
    in_dir = args.inp_dir    
    out_dir = args.outp_dir
    prefix = args.prefix
    seq_names = args.seq_names


    wig = Wig_util()

    seq = []
    with open(seq_names, 'r') as f:
        for line in f.readlines():
            seq.append(line.strip().split())

    
    for s1, s2 in zip(['+', '-'], ['plus', 'minus']):
        for phase in [0,1,2]:                
            wig.addWig2numpy(f'{in_dir}/{prefix}{phase}-{s2}.wig', seq, strand=s1)
            
    for s in seq:
        # array = wig.get_chunks(500004, [s[0]])
        np.save(f"{out_dir}/{prefix}{s[0]}.npy", wig.wig_arrays[s[0]])
        
def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(        
        description="""Converts ClaMSA sitewise predictions in wig format to numpy arrays for each sequence.""")
    parser.add_argument('--inp_dir', type=str, 
        help='Directory where the wig files are located. They have to be named like <prefix><phase>-<strand>.wig')
    parser.add_argument('--outp_dir', type=str, 
        help='The directory where the numpy arrays will be saved. They are saved as <prefix><seq_name>.npy.')
    parser.add_argument('--prefix', type=str, default='',
        help='Prefix for the output numpy arrays. Default is empty.')
    parser.add_argument('--seq_names', type=str, 
        help='List of sequence names to be processed.')    
    return parser.parse_args()

if __name__ == '__main__':
    main()
