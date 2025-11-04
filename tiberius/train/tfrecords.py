import os,sys
import warnings
import argparse
import numpy as np
# import bricks2marble.tf as b2tf
import tensorflow as tf
from bricks2marble.io.gtf import load_gtf
from bricks2marble.io.fasta import load_fasta
from typing import Iterable, Tuple

allowed_transitions = (
    (0,0), (0,7), (7,5),
    (5,14), (14,0), (4,5),
    (5,6), (6,4), (4,8), (5,9),
    (6,10), 
    (11,4), (12,5), (13,6), 
    (1,1), (2,2), (3,3),
    (1,11), (2,12),(3,13),  
    (8,1), (9,2), (10,3),
    (1,14),(2,14),(3,14),
    (11,14),(12,14),(13,14),
    (7,1),(7,2),(7,3),
    (4,14),(5,14),(6,14)
)
fasta_encoding = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1],
    ])

def build_allowed_matrix(pairs: Iterable[Tuple[int, int]], n_labels: int = 15) -> np.ndarray:
    """
    Build a boolean adjacency matrix A where A[i, j] = True iff i -> j is allowed.
    pairs: iterable of (src, dst) transitions that are allowed.
    n_labels: number of distinct labels (0..n_labels-1)
    """
    A = np.zeros((n_labels, n_labels), dtype=bool)
    for i, j in pairs:
        if not (0 <= i < n_labels and 0 <= j < n_labels):
            raise ValueError(f"Transition ({i}->{j}) out of range 0..{n_labels-1}")
        A[i, j] = True
    return A

def check_transitions(labels: np.ndarray, allowed: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    labels: 1D array of ints in [0, allowed.shape[0)-1]
    allowed: boolean adjacency matrix, allowed[i, j] == True if i->j is valid
    
    Returns:
      ok: True iff all adjacent transitions are allowed
      bad_idx: positions i where transition labels[i] -> labels[i+1] is NOT allowed
               (empty if ok=True)
    """
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    n_labels = allowed.shape[0]
    if allowed.shape != (n_labels, n_labels):
        raise ValueError("allowed must be square (n x n)")
    if labels.size <= 1:
        return True, np.array([], dtype=int)

    if np.any((labels < 0) | (labels >= n_labels)):
        raise ValueError(f"labels contain values outside 0..{n_labels-1}")

    src = labels[:-1]
    dst = labels[1:]
    ok_mask = allowed[src, dst]            # vectorized lookup
    bad_idx = np.nonzero(~ok_mask)[0]      # indices of invalid transitions (src positions)
    return bad_idx.size == 0, bad_idx

def main():
    args = parseCmd()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    fasta = load_fasta(args.fasta, T=args.wsize, drop_remainder=True)
    anno = load_gtf(args.gtf)
    anno.find_missing_features()

    sequence_dict = {seq.name: seq.nuc.size for seq in fasta}
    anno_labels = anno.class_labels(sequence_dict, verbose=args.verbose)

    if args.verbose == 2:
        for name, labels in anno_labels.items():
            for strand in ["+", "-"]:
                lab = labels[strand] if strand == "+" else labels[strand][::-1]
                indices = np.where(labels[strand] == 14)
                print(name, strand, indices[0].shape, indices[:50], file=sys.stderr)
                A = build_allowed_matrix(allowed_transitions)
                check, idx = check_transitions(lab, A)
                for i in idx:
                    if not 7 in lab[i-4:i+4] or 14 in lab[i-4:i+4]: 
                        message = f"""Incorrect transition in sequence {name} in:
                            {lab[i-10:i+10]}""".strip()
                        warnings.warn(message)

    nuc = fasta.nuc
    labels_plus = np.concatenate(
        [np.reshape(anno_labels[s]["+"], (-1,args.wsize)) for s in sequence_dict])
    labels_minus = np.concatenate(
        [np.reshape(anno_labels[s]["-"], (-1,args.wsize)) for s in sequence_dict])

    assert nuc.shape[0] == labels_plus.shape[0] == labels_minus.shape[0]

    indices = np.arange(nuc.shape[0])
    np.random.shuffle(indices)

    def create_example(i):
        feature_bytes_x = tf.io.serialize_tensor(
            fasta_encoding[nuc[i]].astype(np.float32)
        ).numpy()
        labels_out = np.concatenate([
            np.eye(15, dtype=np.float32)[labels_plus[i]],
            np.eye(15, dtype=np.float32)[labels_minus[i]]],
            axis=-1
        )
        
        feature_bytes_y = tf.io.serialize_tensor(labels_out).numpy()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'input': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
            'output': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
        }))
        serialized_example = example.SerializeToString()
        return serialized_example
    
    def write_split(k):
        if args.file_prefix:
            out_path = f"{args.out_dir}/{args.file_prefix}_{k}.tfrecords"
        else:
            out_path = f"{args.out_dir}/{k}.tfrecords"

        with tf.io.TFRecordWriter(out_path, 
            options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
            for i in indices[k::args.numb_split]:
                serialized_example = create_example(i)
                writer.write(serialized_example)

    for k in range(args.numb_split):
        write_split(k)

    print('Writing complete.')

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description="""
    USAGE: write_tfrecord_species.py --gtf annot.gtf --fasta genome.fa --wsize 9999 --out tfrecords/speciesName
    
    This script will write input and output data as 100 tfrecord files as tfrecords/speciesName_i.tfrecords""")
    parser.add_argument('--out_dir', type=str, default='',
        help='')
    parser.add_argument('--file_prefix', type=str, default='',
        help='')
    parser.add_argument('--gtf', type=str, default='', required=True,
        help='Annotation in GTF format.')
    parser.add_argument('--fasta', type=str, default='', required=True,
        help='Genome sequence in FASTA format.')
    parser.add_argument('--wsize', type=int,
        help='', required=True)
    parser.add_argument('--numb_split', type=int,
        help='', default=100)
    parser.add_argument('--min_seq_len', type=int,
        help='Minimum length of input sequences used for training', default=500004)  
    parser.add_argument('--verbose', type=int,
        help='', default=1)    
    
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
