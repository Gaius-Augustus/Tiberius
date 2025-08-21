#!/usr/bin/env python3

import sys, json, os, re, sys, csv, argparse
from tiberius.genome_fasta import GenomeSequences
from tiberius.annotation_gtf import Annotation
import subprocess as sp
import numpy as np
import tensorflow as tf
import numpy as np
import sys
from tiberius.wig_class import Wig_util
import h5py
import gc
import time


def get_clamsa_track(file_path, seq_len=500004, prefix=''):
    wig = Wig_util()
    seq = []
    with open(f'{file_path}/../{prefix}_seq_names.txt', 'r') as f:
        for line in f.readlines():
            seq.append(line.strip().split())
    for s1, s2 in zip(['+', '-'], ['plus', 'minus']):
        for phase in [0,1,2]:
            print(f'{file_path}/{prefix}_{phase}-{s2}.wig', file=sys.stderr)
            wig.addWig2numpy(f'{file_path}/{prefix}_{phase}-{s2}.wig', seq, strand=s1)
    chunks_plus = wig.get_chunks(chunk_len=seq_len, sequence_names=[s[0] for s in seq])
    return np.concatenate([chunks_plus[::-1,::-1, [1,0,3,2]], chunks_plus], axis=0)

def load_clamsa_data(clamsa_prefix, seq_names, seq_len=None):
        clamsa_chunks = []
        seq = []
        with open(seq_names, 'r') as f:
            for line in f.readlines():
                seq.append(line.strip().split())
        for s in seq:
            if not os.path.exists(f'{clamsa_prefix}{s}.npy'):
                print(f'CLAMSA PATH {clamsa_prefix}{s}.npy does not exist!')
            clamsa_array = np.load(f'{clamsa_prefix}{s}.npy')
            numb_chunks = clamsa_array.shape[0] // seq_len
            clamsa_array_new = clamsa_array[:numb_chunks*seq_len].reshape(numb_chunks, seq_len, 4)
            clamsa_chunks.append(clamsa_array_new)
           
        clamsa_chunks = np.concatenate(clamsa_chunks, axis=0)
        return np.concatenate([clamsa_chunks[::-1,::-1, [1,0,3,2]], clamsa_chunks], axis=0)

def get_species_data_hmm(genome_path='', annot_path='', species='', 
            seq_len=500004, overlap_size=0, transition=False,
            min_seq_len=500000):

    fasta = GenomeSequences(fasta_file=genome_path,
            chunksize=seq_len,
            overlap=overlap_size)
    fasta.encode_sequences()
    seq_names = [seq_n for seq, seq_n in zip(fasta.sequences, fasta.sequence_names) \
                    if len(seq)>min_seq_len]
    seqs = [len(seq) for seq in fasta.sequences \
                    if len(seq)>min_seq_len]

    fasta.prep_seq_chunks(min_seq_len=min_seq_len,)
    gc.collect()

    ref_anno = Annotation(annot_path,
                seq_names, seqs, seq_len, )
    ref_anno.read_inputfile()


    return fasta, ref_anno

def write_h5(fasta, ref, out, ref_phase=None, split=100, 
                    trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)          
    ref = ref.astype(np.int32)

    file_size = fasta.shape[0] // split
    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)
    for k in range(split):
        # Create a new HDF5 file with compression
        with h5py.File(f'{out}_{k}.h5', 'w') as f:
            # Create datasets with GZIP compression
            f.create_dataset('input', data=fasta[indices[k::split]], compression='gzip', compression_opts=9)  # Maximum compression
            f.create_dataset('output', data=ref[indices[k::split]], compression='gzip', compression_opts=9)

def write_numpy(fasta, ref, out, ref_phase=None, split=100, trans=False, clamsa=np.array([])):
    fasta = fasta.astype(np.int32)          

    file_size = fasta.shape[0] // split
    indices = np.arange(fasta.shape[0])
    np.random.shuffle(indices)
    if ref_phase:
        ref_phase = ref_phase.astype(np.int32)
    for k in range(split):
        print(f'Writing numpy split {k+1}/{split}')
        if clamsa.any():
            np.savez(f'{out}_{k}.npz', array1=fasta[indices[k::split],:,:], 
                     array2=ref[indices[k::split],:,:], array3=clamsa[indices[k::split],:,:] )
        else:
            ref_arr = []
            for i in indices[k::split]:                
                ref_arr.append(ref.get_onehot(i))

            np.savez(f'{out}_{k}.npz', array1=fasta[indices[k::split],:,:], 
                     array2=np.array(ref_arr), )


def write_tf_record(fasta, ref, out, split=100, clamsa=np.array([]),
                    store_txs=False):
    print(f'Writing TFRecords to {out} with split {split}', file=sys.stderr)
    indices = np.arange(len(fasta.chunks_seq))
    np.random.shuffle(indices)
    def create_example(i):
        feature_bytes_x = tf.io.serialize_tensor(fasta.get_onehot(i)).numpy()
        if store_txs:
            y, tx = ref.get_onehot(i, get_tx_ids=True)
            feature_bytes_y = tf.io.serialize_tensor(y).numpy()
            tx = tf.constant(tx, shape=[len(tx),3], dtype=tf.string)
            feature_bytes_tx = tf.io.serialize_tensor(tx).numpy()
        else:
            feature_bytes_y = tf.io.serialize_tensor(ref.get_onehot(i)).numpy()
        if clamsa.any():
            feature_bytes_clamsa = tf.io.serialize_tensor(clamsa[i,:,:]).numpy()
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                'clamsa': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_clamsa]))
            }))
            del feature_bytes_clamsa
        elif store_txs:
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                'tx_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_tx]))
            }))
            del feature_bytes_tx
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                'output': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
            }))
        serialized_example = example.SerializeToString()
        del example, feature_bytes_x, feature_bytes_y
        return serialized_example    
    
    def write_split(k):
        with tf.io.TFRecordWriter(f'{out}_{k}.tfrecords', 
            options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
            for i in indices[k::split]:
                serialized_example = create_example(i)
                writer.write(serialized_example)

    for k in range(split):
        write_split(k)

    print('Writing complete.')

def main():
    args = parseCmd()
    fasta, ref = get_species_data_hmm(genome_path=args.fasta, annot_path=args.gtf, 
                species=args.species, seq_len=args.wsize,
                overlap_size=0, transition=True,
                min_seq_len=args.min_seq_len)

    # print('Loaded FASTA and GTF', fasta.shape, ref.shape)
    if args.clamsa:
        # clamsa = get_clamsa_track('/home/gabriell/deepl_data/clamsa/wig/', seq_len=args.wsize, prefix=args.species)
        clamsa = load_clamsa_data(args.clamsa, seq_names=args.seq_names, seq_len=args.wsize)
        print('Loaded CLAMSA')
        if args.np:
            write_numpy(fasta, ref, args.out, clamsa=clamsa)
        else:
            write_tf_record(fasta, ref, args.out, clamsa=clamsa)
    else:
        if args.h5:
            write_h5(fasta, ref, args.out)
        elif args.np:
            write_numpy(fasta, ref, args.out)
        else:
            add_tx_ids = not args.no_tx_ids
            write_tf_record(fasta, ref, args.out, store_txs=add_tx_ids,
                        split=args.numb_split)

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description="""
    USAGE: write_tfrecord_species.py --gtf annot.gtf --fasta genome.fa --wsize 9999 --out tfrecords/speciesName
    
    This script will write input and output data as 100 tfrecord files as tfrecords/speciesName_i.tfrecords""")
    parser.add_argument('--species', type=str, default='',
        help='')
    parser.add_argument('--gtf', type=str, default='', required=True,
        help='Annotation in GTF format.')
    parser.add_argument('--fasta', type=str, default='', required=True,
        help='Genome sequence in FASTA format.')
    parser.add_argument('--out', type=str, required=True,
        help='Prefix of output files')
    parser.add_argument('--wsize', type=int,
        help='', required=True)
    parser.add_argument('--numb_split', type=int,
        help='', default=100)
    parser.add_argument('--min_seq_len', type=int,
        help='Minimum length of input sequences used for training', default=500004)
    parser.add_argument('--clamsa',  type=str, default='',
        help='')
    parser.add_argument('--seq_names',  type=str, default='',
        help='')
    parser.add_argument('--h5', action='store_true',
        help='') 
    parser.add_argument('--np', action='store_true',
        help='')
    parser.add_argument('--no_tx_ids', action='store_true',
        help='Dont store transcript IDs and the overlap range for each trainings example.')
    
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
