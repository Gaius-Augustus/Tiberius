#!/usr/bin/env python3

# ==============================================================
# Authors: Lars Gabriel
#
# Running Tiberius for single genome prediction
# ==============================================================

import sys, os, sys, requests, time, logging, math, gzip, bz2, yaml
script_dir = os.path.dirname(os.path.realpath(__file__))
import subprocess as sp
from Bio import SeqIO
from Bio.Seq import Seq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MAX_TF_VERSION = '2.12'
seqgroup_size = 50000400

class MissingConfigFieldError(RuntimeError):
    """Raised when the model-config file lacks one or more required fields."""

class InvalidArgumentCombinationError(RuntimeError):
    """
    Raised when the user supplies an illegal or inconsistent combination of
    command-line arguments (e.g. --model-config together with --nosm).
    """

def load_model_config(
    filepath,
    required = ("weights_url", "softmasking", "clamsa"),
):
    """
    Read a YAML model-config file and return its contents as a dict.

    Parameters
    ----------
    filepath : str
        Path to the YAML file.
    required : Sequence[str], optional
        Keys that *must* be present (and non-null) in the YAML.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Treat absent keys *or* keys explicitly set to null/None as missing
    missing = [k for k in required if data.get(k) is None]
    if missing:
        raise MissingConfigFieldError(
            f"{filepath} is missing required field(s): {', '.join(missing)}"
        )

    return data

# Function to compare TensorFlow version
def check_tf_version(tf_version):
    if tf_version > MAX_TF_VERSION:
        print(f"WARNING: You are using TensorFlow version {tf_version}, "
                      f"which is newer than the recommended maximum version {MAX_TF_VERSION}. "
                      "It will produce an error if you use a sequence length > 259,992 during inference!")
        return False
    return True

def check_seq_len(seq_len):
    if not (seq_len % 9 == 0 and seq_len % 2 == 0):
        logging.error(f'ERROR: The argument "seq_len" has to be  divisable by 9 and by 2 for the model to work! Please change the value {seq_len} to a different value!')
        sys.exit(1)
    return True

def check_parallel_factor(parallel_factor, seq_len):    
    if parallel_factor < 2 or 2 * parallel_factor > seq_len:
        logging.info(f'WARNING: The parallel factor is poorely chosen, please choose a different --seq_len or specify a --parallel_factor, which has to be a divisor of --seq_len. Current value: {parallel_factor}.')
    if seq_len % parallel_factor != 0:
        logging.error(f'ERROR: The argument "parallel_factor" has to be a divisor of --seq_len. Please change the value {parallel_factor} to a different value or choose a different seq_len!')
        sys.exit(1)
    if parallel_factor < 1:
        logging.error(f'ERROR: The argument "parallel_factor" has to be >0. Please change the value {parallel_factor} to a different value!')
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
    """ Group seguences into chunks of size t, groups are shown in the progress output.
    """
    # Sort sequences by increasing length, so that similar long sequences are grouped together
    # and adaptive chunk sizes are effective.
    sorted_seqs = sorted(zip(seq_names, seq_lens), key=lambda x: x[1], reverse=False)

    groups = []
    current_group = []
    current_sum = 0

    for s_n, s_l in sorted_seqs:
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
    if (file_path.endswith(".tgz") or file_path.endswith(".tar.gz")) \
        and os.path.exists(file_path[:-4]) and not os.path.getsize(file_path[:-4]) == 0:
        file_path = file_path[:-4]
        logging.info(f"Warning: No model weights provided. Using existing file at {file_path}.")
    elif os.path.exists(file_path) and not os.path.getsize(file_path) == 0:
        logging.info(f"Warning: No model weights provided. Using existing file at {file_path}.")
    else:
        logging.info(f'Warning: No model weights provided, they will be downloaded to {file_path}.')
        logging.info(f'Weights for Tiberius model will be downloaded from {url}')
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


def run_tiberius(args):
    if 'tensorflow' in sys.modules:
        tf = sys.modules['tensorflow']
    else:
        import tensorflow as tf

    if not check_tf_version(tf.__version__) and args.seq_len > 259992:
        logging.error(f"Error: The sequence length {args.seq_len} is too long for TensorFlow version {tf.__version__}. "
                        "Please use a sequence length <= 259992 (--seq_len).")
        sys.exit(1)

    if args.learnMSA:
        sys.path.insert(0, args.learnMSA)  
    if args.LRU:
        sys.path.insert(0, args.LRU)
        lru = True # for PredictionGTF
    else:
        lru = False
        
    from tiberius import Anno, make_weighted_cce_loss, PredictionGTF

    start_time = time.time()
    config = None
    model_path = None
    model_path_hmm = None
    if not args.model_old and not args.model_lstm_old:
        if args.model_cfg:
            config = load_model_config(args.model_cfg)
        elif args.model:
            model_path = os.path.abspath(args.model)
        else:
            if args.clamsa and args.no_softmasking:
                raise InvalidArgumentCombinationError("Use either --clamsa or --no_softmasking with the default mammalian models!")
            elif args.clamsa:
                config = load_model_config(f"{script_dir}/../model_cfg/mammalia_clamsa_v2.yaml")
            elif args.no_softmasking:
                config = load_model_config(f"{script_dir}/../model_cfg/mammalia_nosoftmasking_v2.yaml")
            else:
                config = load_model_config(f"{script_dir}/../model_cfg/mammalia_softmasking_v2.yaml")
        
        if model_path:        
            check_file_exists(model_path)
            logging.info(f'Model path: {model_path}')
            
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
    min_seq_len = args.min_genome_seqlen
    logging.info(f'Minimum sequence length: {min_seq_len}')
    if min_seq_len > 0:
        logging.info(f'Warning: Sequences shorter than {min_seq_len} will be ignored.')
    check_seq_len(seq_len)    
    strand = [s for s in args.strand.split(',') if s in ['+', '-']]
    logging.info(f'Strand: {strand}')    
    if not strand:
        logging.error(f'ERROR: The argument "strand" has to be either "+" or "-" or "+,-". Current value: {args.strand}.')
        sys.exit(1)

    parallel_factor = compute_parallel_factor(seq_len) if args.parallel_factor == 0 else args.parallel_factor
    logging.info(f'HMM parallel factor: {parallel_factor}')    

    softmasking = not args.no_softmasking if not config else config["softmasking"]
    logging.info(f'Softmasking: {softmasking}')    
    
    genome_path = os.path.abspath(args.genome)
    check_file_exists(genome_path)
    logging.info(f'Genome sequence path: {genome_path}')    
    
    clamsa_prefix = args.clamsa if not config else config["clamsa"]
    
    if config:
        model_weights_dir = f'{script_dir}/../model_weights'        
        if not os.path.exists(model_weights_dir):
            os.makedirs(model_weights_dir)
        if not is_writable(model_weights_dir):
            model_weights_dir = os.getcwd()
        if not is_writable(model_weights_dir):
            logging.error(f'No model weights provided, and candidate directorys for download are not writeable. Please download the model weigths manually (see README.md) and specify them with --model!')
            sys.exit(1)
        model_file_name = config["weights_url"].split('/')[-1]
        model_path = download_weigths(config["weights_url"], f'{model_weights_dir}/{model_file_name}') 
        if model_path and model_path[-3:] in ['tgz', ".gz"]:
            logging.info(f'Extracting weights to {model_weights_dir}')
            extract_tar_gz(f'{model_path}', f'{model_weights_dir}')
            model_path = model_path[:-4] if model_path.endswith(".tgz") else model_path[:-7]
    if (model_path and not os.path.exists(model_path)):
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
            model_path_lstm_old=args.model_lstm_old,
            model_path_old=args.model_old,
            model_path=model_path,
            model_path_hmm=model_path_hmm,
            seq_len=seq_len, 
            batch_size=batch_size, 
            hmm=True, 
            temp_dir=None,
            num_hmm=1,
            hmm_factor=1,    
            genome=genome,
            softmask=not args.no_softmasking, strand=s_,
            parallel_factor=parallel_factor,
            lru=lru
        )
        
        pred_gtf.load_model(summary=j==0)
        
        genome_fasta = pred_gtf.init_fasta(chunk_len=seq_len, min_seq_len=min_seq_len)
        genome_seq_dict = {s_n: len(s) for s_n, s in zip(genome_fasta.sequence_names, genome_fasta.sequences)}
        seq_groups = group_sequences(genome_fasta.sequence_names,
                                   [len(s) for s in genome_fasta.sequences],
                                    t=seqgroup_size, chunk_size=seq_len)

        for k, seq in enumerate(seq_groups):
            logging.info(f'Tiberius gene predicton {k+1+len(seq_groups)*j}/{len(strand)*len(seq_groups)} ')
            x_data, coords, adapted_seqlen = pred_gtf.load_genome_data(genome_fasta, seq,
                                                       softmask=softmasking, strand=s_)
            pred_gtf.adapt_batch_size(adapted_seqlen)

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
            
        if not filt and tx.start < 1 or tx.end > genome_seq_dict[tx.chr]:
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
    

def main():    
    from tiberius_args import parseCmd
    args = parseCmd()
    run_tiberius(args)

if __name__ == '__main__':
    main()
