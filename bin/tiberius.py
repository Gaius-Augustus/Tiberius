#!/usr/bin/env python3

# ==============================================================
# Authors: Lars Gabriel
#
# Running Tiberius for single genome prediction
# ==============================================================

import sys, os, sys, argparse, requests, time, logging, math, gzip, bz2
script_dir = os.path.dirname(os.path.realpath(__file__))
import subprocess as sp
from Bio import SeqIO
from Bio.Seq import Seq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MAX_TF_VERSION = '2.12'
# Function to compare TensorFlow version
def check_tf_version(tf_version):
    if tf_version > MAX_TF_VERSION:
        print(f"WARNING: You are using TensorFlow version {tf_version}, "
                      f"which is newer than the recommended maximum version {MAX_TF_VERSION}. "
                      "It will produce an error if you use a sequence length > 260.000 during inference!")
        return False 
    return True

def check_seq_len(seq_len):
    if not (seq_len % 9 == 0 and seq_len % 2 == 0):
        logging.error(f'ERROR: The argument "seq_len" has to be  divisable by 9 and by 2 for the model to work! Please change the value {seq_len} to a different value!')
        sys.exit(1)
    return True

def check_parallel_factor(parallel_factor, seq_len):    
    if parallel_factor < 2 or parallel_factor > seq_len-1:
        logging.info(f'WARNING: The parallel factor is poorely chosen, please choose a different --seq_len or specify a --parallel_factor, which has to be a divisor of --seq_len. Current value: {parallel_factor}.')
    if seq_len % parallel_factor != 0:
        logging.error(f'ERROR: The argument "parallel_factor" has to be a divisor of --seq_len! Please change the value {parallel_factor} to a different value or choose a different seq_len!')
        sys.exit(1)
    if parallel_factor < 1:
        logging.error(f'ERROR: The argument "parallel_factor" has to be >0! Please change the value {parallel_factor} to a different value!')
        sys.exit(1)

def compute_parallel_factor(seq_len):
    sqrt_n = int(math.sqrt(seq_len))    
    # Check for divisors 
    for i in range(0, seq_len - sqrt_n + 1):        
        if seq_len % (sqrt_n-i) == 0:
            return sqrt_n-i
        if seq_len % (sqrt_n+i) == 0:
            return sqrt_n+i  
    return sqrt_n

# Function to assemble transcript taking strand into account
def assemble_transcript(exons, sequence, strand):
    parts = []
    exons.sort(reverse=strand=='-')
    for exon in exons:
        exon_seq = sequence.seq[exon[0]-1:exon[1]]
        if strand == '-':
            exon_seq = exon_seq.reverse_complement()
        parts.append(str(exon_seq))  

    coding_seq = Seq("".join(parts))
    if len(coding_seq) > 0  and len(coding_seq)%3==0:
        prot_seq = coding_seq.translate()
        if prot_seq[-1] == '*':
            return coding_seq, prot_seq
    return None, None

# Check for in-frame stop codons
def check_in_frame_stop_codons(seq):
    return '*' in seq[:-1]

def check_file_exists(file_path):
    """
    Check if a file exists at the specified path.
    
    file_path: Path to the file to check.
    """
    if not os.path.exists(file_path):
        error_message = f"Error: The file '{file_path}' does not exist."
        logging.error(error_message)
        sys.exit(1)
    
def group_sequences(seq_names, seq_lens, t=50000400, chunk_size=500004):
    groups = []
    current_group = []
    current_sum = 0

    for s_n, s_l in zip(seq_names, seq_lens):
        current_sum += chunk_size if s_l < chunk_size else s_l        
        current_group.append(s_n)
        if current_sum > t:
            if current_group:
                groups.append(current_group)
            current_group = []
            current_sum = 0
    
    if current_group:
        groups.append(current_group)    
    return groups

def download_weigths(url, file_path):
    # print(url, file_path)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=10000):
                f.write(chunk)
    return file_path

def extract_tar_gz(file_path, dest_dir):
    sp.run(f'tar -xzf {file_path} -C {dest_dir}', shell=True)

def is_writable(file_path):
    return os.access(file_path, os.W_OK)
    
def load_genome(genome_path):
    if genome_path.endswith(".gz"):
        with gzip.open(genome_path, "rt") as file:
            genome = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    elif genome_path.endswith(".bz2"):
        with bz2.open(genome_path, "rt") as file:
            genome = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        with open(genome_path, "r") as file:
            genome = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    return genome

def main():    
    args = parseCmd()        

    import tensorflow as tf

    start_time = time.time()
    if check_tf_version(tf.__version__):
        url_weights = {
            'Tiberius_default': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_weights.tgz',
            'Tiberius_nosm': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_nosm_weights.tgz',
            'Tiberius_denovo': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models//tiberius_denovo_weights.tgz'
        }
    else:
        url_weights = {
            'Tiberius_default': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_weights_tf2_17.keras',
            'Tiberius_nosm': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_nosm_weights_tf2_17.keras',
            'Tiberius_denovo': 'https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_denovo_weights_tf2_17.keras'
        }
        if args.seq_len > 259992:
            logging.error(f"Error: The sequence length {args.seq_len} is too long for TensorFlow version {tf.__version__}. "
                          "Please use a sequence length <= 259992 (--seq_len).")
            sys.exit(1)
    
    if args.learnMSA:
        sys.path.insert(0, args.learnMSA)  
       
    from eval_model_class import PredictionGTF
    from models import make_weighted_cce_loss        
    from genome_anno import Anno
    
    model_path = os.path.abspath(args.model) if args.model else None
    if model_path:        
        check_file_exists(model_path)
        logging.info(f'Model path: {model_path}')
        
    model_path_lstm = os.path.abspath(args.model_lstm) if args.model_lstm else None
    if model_path_lstm:
        check_file_exists(model_path_lstm)
        logging.info(f'Model LSTM path: {model_path_lstm}')
        
    model_path_hmm = os.path.abspath(args.model_hmm) if args.model_hmm else None
    if model_path_hmm:
        check_file_exists(model_path_hmm)
        logging.info(f'Model HMM path: {model_path_hmm}')

            
    gtf_out = os.path.abspath(args.out)
    logging.info(f'Output file: {gtf_out}')
    batch_size = args.batch_size
    logging.info(f'Batch size: {batch_size}')
    seq_len = args.seq_len
    logging.info(f'Tile length: {seq_len}')
    check_seq_len(seq_len)    
    strand = [s for s in args.strand.split(',') if s in ['+', '-']]
    logging.info(f'Strand: {strand}')    
    if not strand:
        logging.error(f'ERROR: The argument "strand" has to be either "+" or "-" or "+,-". Current value: {args.strand}.')
        sys.exit(1)

    parallel_factor = compute_parallel_factor(seq_len) if args.parallel_factor == 0 else args.parallel_factor
    logging.info(f'HMM parallel factor: {parallel_factor}')    

    softmasking = False if args.no_softmasking else True
    logging.info(f'Softmasking: {softmasking}')    
    
    genome_path = os.path.abspath(args.genome)
    check_file_exists(genome_path)
    logging.info(f'Genome sequence path: {genome_path}')    
    
    clamsa_prefix = args.clamsa
    
    if not model_path and not model_path_lstm:
        model_weights_dir = f'{script_dir}/../model_weights'        
        if not os.path.exists(model_weights_dir):
            os.makedirs(model_weights_dir)
        if not is_writable(model_weights_dir):
            model_weights_dir = os.getcwd()
        if not is_writable(model_weights_dir):
            logging.error(f'No model weights provided, and candidate directorys for download are not writeable. Please download the model weigths manually (see README.md) and specify them with --model!')
            sys.exit(1)

        logging.info(f'Warning: No model weights provided, they will be downloaded into {model_weights_dir}.')
        
        if clamsa_prefix:
            if not softmasking:
                logging.error(f'ERROR: Clamsa input requires softmasking.')
                sys.exit(1)
            logging.info(f'Weights for Tiberius de novo model will be downloaded from {url_weights["Tiberius_denovo"]}')
            model_file_name = url_weights["Tiberius_denovo"].split('/')[-1]       
            download_weigths(url_weights["Tiberius_denovo"], f'{model_weights_dir}/{model_file_name}')  
        elif not softmasking:
            logging.info(f'Weights for Tiberius model without softmasking will be downloaded from {url_weights["Tiberius_nosm"]}')
            model_file_name = url_weights["Tiberius_nosm"].split('/')[-1]          
            download_weigths(url_weights["Tiberius_nosm"], f'{model_weights_dir}/{model_file_name}')  
        else:
            logging.info(f'Weights for Tiberius model will be downloaded from {url_weights["Tiberius_default"]}')
            model_file_name = url_weights["Tiberius_default"].split('/')[-1]
            download_weigths(url_weights["Tiberius_default"], f'{model_weights_dir}/{model_file_name}')
        if model_file_name[-3:] == 'tgz':
            logging.info(f'Extracting weights to {model_weights_dir}')
            extract_tar_gz(f'{model_weights_dir}/{model_file_name}', f'{model_weights_dir}')
            model_file_name = model_file_name[:-4]
        model_path = f'{model_weights_dir}/{model_file_name}'
        
        if not os.path.exists(model_path):
            logging.error(f'Error: The model weights could not be downloaded. Please download the model weights manually (see README.md) and specify them with --model!')
            sys.exit(1)

    anno = Anno(gtf_out, f'anno')     
    tx_id=0
    
    # load genome once completely into memory
    # TODO: this should eventually be done in streaming mode to save RAM
    genome = load_genome(genome_path)
    tf.keras.utils.get_custom_objects()["weighted_cce_loss"] = make_weighted_cce_loss()

    for j, s_ in enumerate(strand):
        pred_gtf = PredictionGTF( 
            model_path=model_path,
            model_path_lstm=model_path_lstm,
            model_path_hmm=model_path_hmm,
            seq_len=seq_len, 
            batch_size=batch_size, 
            hmm=True, 
            temp_dir=None,
            emb=args.emb, 
            num_hmm=1,
            hmm_factor=1,    
            genome=genome,
            softmask=not args.no_softmasking, strand=s_,
            parallel_factor=parallel_factor,
            # lstm_cfg=args.lstm_cfg,
        )
        
        pred_gtf.load_model(summary=j==0)
        
        genome_fasta = pred_gtf.init_fasta(chunk_len=seq_len)
        
        seq_groups = group_sequences(genome_fasta.sequence_names,
                                   [len(s) for s in genome_fasta.sequences],
                                    t=50000400, chunk_size=seq_len)
        
        for k, seq in enumerate(seq_groups):
            logging.info(f'Tiberius gene predicton {k+1+len(seq_groups)*j}/{len(strand)*len(seq_groups)} ')
            x_data, coords = pred_gtf.load_genome_data(genome_fasta, seq,
                                                       softmask=softmasking, strand=s_)
            # print(x_data.shape)
            clamsa=None
            if clamsa_prefix:
                clamsa = pred_gtf.load_clamsa_data(clamsa_prefix=clamsa_prefix, seq_names=seq, 
                                 strand=s_, chunk_len=seq_len, pad=True)
                
            hmm_pred = pred_gtf.get_predictions(x_data, hmm_filter=True, clamsa_inp=clamsa)
            anno, tx_id = pred_gtf.create_gtf(y_label=hmm_pred, coords=coords, f_chunks=x_data,
                                clamsa_inp=clamsa, strand=s_, anno=anno, tx_id=tx_id,
                                filt=False)
        
    # filter transcripts
    anno_outp = Anno('', f'anno')        
    out_tx = {}
    for tx_id, tx in anno.transcripts.items():
        exons = tx.get_type_coords('CDS', frame=False)
        filt=False

        # filter out tx with inframe stop codons
        coding_seq, prot_seq = assemble_transcript(exons, genome[tx.chr], tx.strand )
        if not coding_seq or check_in_frame_stop_codons(prot_seq):
            filt = True
        # filter out transcripts with cds len shorter than args.filter_short
        if not filt and tx.get_cds_len() < 201:
            filt = True

        if not filt:
            out_tx[tx_id] = tx

    anno_outp.add_transcripts(out_tx, f'anno')
    anno_outp.norm_tx_format()
    anno_outp.find_genes()
    anno_outp.rename_tx_ids(args.id_prefix) 
    anno_outp.write_anno(gtf_out)

    prot_seq_out = ""
    coding_seq_out = ""
    if args.protseq or args.codingseq:
        for tx_id, tx in anno_outp.transcripts.items():
            exons = tx.get_type_coords('CDS', frame=False)
            coding_seq, prot_seq = assemble_transcript(exons, genome[tx.chr], tx.strand)
            if args.codingseq:
                coding_seq_out +=f">{tx_id}\n{coding_seq}\n"
            if args.protseq:
                prot_seq_out +=f">{tx_id}\n{prot_seq}\n"

    if args.codingseq:
        with open(args.codingseq, 'w+') as f:
            f.write(coding_seq_out.strip())

    if args.protseq:
        with open(args.protseq, 'w+') as f:
            f.write(prot_seq_out.strip())

    end_time = time.time()    
    duration = end_time - start_time
    print(f"Tiberius took {duration/60:.4f} minutes to execute.")
    
def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(
        
        description="""Tiberius predicts gene structures from a nucleotide sequences that can have repeat softmasking.

    There are flexible configuration to load the model, including options to:
    - Load a complete LSTM+HMM model
    - Load only the LSTM model and use a default HMM layer

    Example usage:
        Load LSTM+HMM model:
        tiberius.py --genome genome.fa --model model_keras_save --out tiberius.gtf

        Load only LSTM model:
        tiberius.py --genome genome.fa --model_lstm lstm_keras_save

        Load LSTM model and custom HMM Layer:
        tiberius.py --genome genome.fa --model_lstm lstm_keras_save --model_hmm hmm_layer_keras_save

        Use Tiberius with softmasking disabled:
        tiberius.py --genome genome.fa --model model_keras_save --out tiberius.gtf --no_softmasking
    """)
    parser.add_argument('--model_lstm', type=str, default='',
        help='LSTM model file that can be used with --model_hmm to add a custom HMM layer, otherwise a default HMM layer is added.')
    parser.add_argument('--model_hmm', type=str, default='',
        help='HMM layer file that can be used with --model_lstm.')
    parser.add_argument('--model', type=str,
        help='LSTM model file with HMM Layer.', default='')
    parser.add_argument('--out', type=str,
        help='Output GTF file with Tiberius gene prediction.', default='tiberius.gtf')
    parser.add_argument('--genome',  type=str, required=True,
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
    # parser.add_argument('--temp_dir', type=str, default='',
    #     help='')
    parser.add_argument('--emb', action='store_true',
        help='Indicates if the HMM layer uses embedding input. Currently not supported')
    parser.add_argument('--strand', type=str,
        help='Either "+" or "-" or "+,-".', default='+,-')
    parser.add_argument('--seq_len', type=int,
        help='Length of sub-sequences used for parallelizing the prediction.', default=500004)
    parser.add_argument('--batch_size', type=int,
        help='Number of sub-sequences per batch.', default=16)
    parser.add_argument('--id_prefix', type=str,
        help='Prefix for gene and transcript IDs in output GTF file.', default='')
        
    return parser.parse_args()

if __name__ == '__main__':
    main()
