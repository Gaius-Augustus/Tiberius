import sys, os, argparse
sys.path.append("../../programs/learnMSA")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess as sp
import numpy as np
import csv, json
from eval_model_class import PredictionGTF

def main():
    args = parseCmd()
    home_dir = '/home/gabriell/'
    home_dir = '/home/jovyan/brain/'
    genome_path = f'{home_dir}/deepl_data/genomes/'
    annot_path = f'{home_dir}/deepl_data/annot_longest_fixed/'
    
    with open(args.cfg, 'r') as f_h:
        cfg = json.load(f_h)
    
    out = args.out

    if not os.path.exists(out):
        os.mkdir(out)

#     s = args.species
    
    for strand in ['+', '-']:
        temp_dir = f'{out}/temp_{strand}'
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
            
        relevant_keys = ['seq_len', 'batch_size', 
                         'model_path_lstm', 'emb', 'num_hmm', 'trans_lstm']
        relevant_args = {key: cfg[key] for key in relevant_keys if key in cfg}
        pred_gtf = PredictionGTF(
            **relevant_args, temp_dir=temp_dir, hmm=True, hmm_factor=1)
        pred_gtf.load_model()        
        out_str = ''
        f_chunks, r_chunks, coords = pred_gtf.load_inp_data(
                genome_path=cfg['genome_path'], #f'{genome_path}/{s}.fa.combined.masked',
                annot_path=cfg['annot_path'], #f'{annot_path}/{s}.gtf', 
                overlap_size=0, strand=strand, chunk_coords=True,
                softmask=True
            )
        y_pred = pred_gtf.get_predictions(f_chunks)
        pred_gtf.create_gtf_2(y_pred, coords, f'{out}/deepfinder_{strand}.gtf', f_chunks, strand=strand)        
   
    cmd = f'{home_dir}/TSEBRA/bin/tsebra.py -k {out}/deepfinder_+.gtf,{out}/deepfinder_-.gtf -o {out}/deepfinder.gtf'
    sp.call(cmd, shell=True)

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--species', type=str,
#         help='')
    parser.add_argument('--out', type=str,
        help='')
    parser.add_argument('--cfg', type=str,
        help='')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
