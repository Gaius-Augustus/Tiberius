from genome_fasta import GenomeSequences
from annotation_gtf import GeneStructure
# import tensorflow records
import tensorflow as tf
import numpy as np
import psutil
import sys
import zlib
from copy import deepcopy

genome_path = '/home/jovyan/brain/deepl_data/genomes/'
annot_path = '/home/jovyan/brain//deepl_data/annot_longest_fixed/'

genome_path = '/home/gabriell/deepl_data/genomes/'
annot_path = '/home/gabriell/deepl_data/annot_longest_fixed/'

def get_athal():
    species = 'Arabidopsis_thaliana'
    out_path = '/home/jovyan/brain//deepl_data/'
    batch_size=5000
    overlap_size=2000
    with tf.io.TFRecordWriter(f'{out_path}/{species}.tfrecords') as writer:
        fasta = GenomeSequences(fasta_file=f'/home/jovyan/brain//test_data/{species}/data/genome.fasta.masked',
                            chunksize=batch_size, 
                            overlap=overlap_size)
        fasta.encode_sequences()
        fasta.create_chunks()
        ref = GeneStructure(f'/home/jovyan/brain//test_data/{species}/annot/annot.gtf', 
                               chunksize=batch_size, 
                               overlap=overlap_size)
        ref.translate_to_one_hot(fasta.sequence_names, 
                                [len(s) for s in fasta.sequences])
        ref.create_chunks(fasta.sequence_names)
        fasta = fasta.chunks
        ref = ref.chunks
        #fasta, ref = get_data_for_species(species, 
                        #batch_size=batch_size, overlap_size=overlap_size)
        #print(ref.dtype, fasta.dtype)        
        
        for i_chr in range(len(fasta)):
            for x, y in zip(fasta[i_chr], ref[i_chr]):
                x = np.array(x).astype(np.float32)
                y = np.array(y).astype(np.float32)
                print(x.shape, y.shape)
                feature_bytes_x = tf.io.serialize_tensor(x).numpy()
                feature_bytes_y = tf.io.serialize_tensor(y).numpy()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'input': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                    'output': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
                }))
                writer.write(example.SerializeToString())

# function that creates the data for one species 
def get_data_for_species(species, batch_size=6000, overlap_size=1000, seq=False):
    """Creates the data for one species.
        Args:   species (str): name of the species
                batch_size (int): size of the chunks
                overlap_size (int): size of the overlap between chunks

        Returns: data (tuple(np.array, np.array)): tuple of one hot encoded input and output

    """    
    fasta = GenomeSequences(fasta_file=f'{genome_path}/{species}.fa.combined.masked',
                            chunksize=batch_size, 
                            overlap=overlap_size)    
    fasta.encode_sequences()    
    if seq:
        f_chunk = fasta.create_chunks_seq()
    else:
        f_chunk = fasta.create_chunks_one_hot()    
    seq_len = [len(s) for s in fasta.sequences]
    del fasta.sequences
    del fasta.one_hot_encoded
    ref_anno = GeneStructure(f'{annot_path}/{species}.gtf', 
                           chunksize=batch_size, 
                          overlap=overlap_size)    
        
    ref_anno.translate_to_one_hot(fasta.sequence_names, 
                            seq_len)
    ref_anno.create_chunks(fasta.sequence_names, seq)
    
    return f_chunk, ref_anno.chunks

def get_data_for_species_phase(species, batch_size=6000, overlap_size=1000):
    """Creates the data for one species.
        Args:   species (str): name of the species
                batch_size (int): size of the chunks
                overlap_size (int): size of the overlap between chunks

        Returns: data (tuple(np.array, np.array)): tuple of one hot encoded input and output

    """    
    fasta = GenomeSequences(fasta_file=f'{genome_path}/{species}.fa.combined.masked',
                            chunksize=batch_size, 
                            overlap=overlap_size)    
    fasta.encode_sequences()    
    f_chunk = fasta.get_flat_chunks(strand='+') 
    full_f_chunks = np.concatenate((f_chunk[::-1,::-1, [3,2,1,0,4]], 
                                    f_chunk), axis=0)
    
    seq_len = [len(s) for s in fasta.sequences]
    del fasta.sequences
    del fasta.one_hot_encoded
    ref_anno = GeneStructure(f'{annot_path}/{species}.gtf', 
                           chunksize=batch_size, 
                          overlap=overlap_size)    
        
    ref_anno.translate_to_one_hot(fasta.sequence_names, 
                            seq_len)
    r_chunk, r_phase = ref_anno.get_flat_chunks(fasta.sequence_names, strand='-')
    r_chunk2, r_phase2 = ref_anno.get_flat_chunks(fasta.sequence_names, strand='+')
    
    return full_f_chunks, np.concatenate((r_chunk, r_chunk2), axis=0), np.concatenate((r_phase, r_phase2), axis=0)

def print_mem_usage():    
    process = psutil.Process()
    memory_usage = process.memory_info().rss
    sys.stderr.write(f"Current memory usage: {memory_usage / 1024 / 1024:.2f} MB")

# get data for a list of species and store them to a file using tfrecords
def create_tf_records_species(species_list, out_path, batch_size=3000, 
                              overlap_size=300, name='data'):
    """Creates the data for a list of species and stores them to a file using tfrecords.
        Args:   species_list (list(str)): list of species
                batch_size (int): size of the chunks
                overlap_size (int): size of the overlap between chunks
    """
    print(species_list)   
    # create tfrecords
    for species in species_list:
        with tf.io.TFRecordWriter(f'{out_path}/{name}.{species}.tfrecords', \
                  options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
            print(species)
            fasta, ref, ref_phase = get_data_for_species_phase(species, 
                    batch_size=batch_size, overlap_size=overlap_size)  
            fasta = fasta.astype(np.int32)          
            ref = ref.astype(np.int32)
            ref_phase = ref_phase.astype(np.int32)
            for i in range(fasta.shape[0]):        
                if True or np.any(ref[i,:,1]):
                    feature_bytes_x = tf.io.serialize_tensor(fasta[i,:,:]).numpy()
                    feature_bytes_y = tf.io.serialize_tensor(ref[i,:,:]).numpy()
                    feature_bytes_y_phase = tf.io.serialize_tensor(ref_phase[i,:,:]).numpy()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'input': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                        'output': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y])),
                        'output_phase': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes_y_phase]))
                    }))
                    serialized_example = example.SerializeToString()
                    writer.write(serialized_example)                    
            del fasta
            del ref
            
def create_tf_records_species_seq(species_list, out_path, batch_size=3000, 
                              overlap_size=300, name='data'):
    for species in species_list:
        with tf.io.TFRecordWriter(f'{out_path}/{species}.tfrecords', \
                  options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
                                  #) as writer:        
            print(species)
            count = 0
            fasta, ref = get_data_for_species(species, 
                            batch_size=batch_size, overlap_size=overlap_size, seq=True)            
            for i_chr in range(len(fasta)):
                #sys.stderr.write(i_chr)
                #print_mem_usage()
                for x, y in zip(fasta[i_chr], ref[i_chr]):        
                    #print(y.shape, np.any(y))
                    #x = np.array(y).astype(np.int32)
                    y = np.array(y).astype(np.int32)  
                    if np.any(y == 1):
                        #y = np.split(y, 2, axis=-1)
                        count +=1
                        #print(x.shape, y.shape)
                        feature_bytes_x = tf.io.serialize_tensor(x).numpy()
                        feature_bytes_y = tf.io.serialize_tensor(y).numpy()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'input': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
                            'output': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
                        }))
                        serialized_example = example.SerializeToString()
                        #compressed_example = zlib.compress(serialized_example)
                        writer.write(serialized_example) 
            print(count)
            del fasta
            del ref