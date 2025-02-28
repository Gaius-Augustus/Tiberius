# ==============================================================
# Authors: Lars Gabriel
#
# Class loading tfrecords so that they fit different traing 
# scenarios
# 
# Transformers 4.31.0
# ==============================================================

import sys
import tensorflow as tf
import numpy as np
# from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM

class DataGenerator:
    """DataGenerator class for reading and processing TFRecord files 
    so that they can be used for training a deepfinder model.

    Args:
        file_path (list(str)): Paths to the TFRecord files.
        batch_size (int): Number of examples per batch.
        shuffle (bool): Whether to shuffle the data.
        repeat (bool): Whether to repeat the data set.
        output_size (int): Number of class labels in traings examples.
        trans (bool): Whether the data should fit the transformer only model. (deprecated!!)
        trans_lstm (bool): Whether the data should fit the transformer-LSTM hybrid model. (deprecated!!)
        seq_weights (int): Weight of positons around exon borders. They aren't used if 0.
        softmasking (bool): Whether softmasking track should be added to input.
        clamsa (bool): Whether Clamsa track should be prepared as additional input,
        oracle (bool): Whether the input data should include the labels.
    """

    def __init__(self, file_path, 
                 batch_size, shuffle=True, 
                 repeat=True,                  
                 filter=False,
                 output_size=5,
                hmm_factor=None, 
                 seq_weights=0, softmasking=True,
                clamsa=False,
                oracle=False,
                tx_filter=None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter = filter
        self.repeat = repeat
        self.seq_weights = seq_weights
        self.output_size = output_size
        self.hmm_factor = hmm_factor
        self.softmasking=softmasking
        self.clamsa = clamsa
        self.oracle = oracle
        self.tx_filter = set(tx_filter)
        
        self.dataset = self._read_tfrecord_file(repeat=repeat)
        self.iterator = iter(self.dataset)
    
    def _parse_fn(self, example):
        """Parse function for decoding TFRecord examples.

        Args:
            example (tf.Tensor): Example in serialized TFRecord format.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Parsed input and output tensors.
        """
        features = {
            'input': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string),
            'tx_ids': tf.io.FixedLenFeature([], tf.string),
            #'output_phase': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example, features)
        x = tf.io.parse_tensor(parsed_features['input'], out_type=tf.int32)
        y = tf.io.parse_tensor(parsed_features['output'], out_type=tf.int32)
        t = tf.io.parse_tensor(parsed_features['tx_ids'], out_type=tf.string)
        #y_phase = tf.io.parse_tensor(parsed_features['output_phase'], out_type=tf.int32)
        return x, y, t#, y_phase
    
    def _parse_fn_clamsa(self, example):
        """Parse function for decoding TFRecord examples including clamsa data.

        Args:
            example (tf.Tensor): Example in serialized TFRecord format.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Parsed input and output tensors.
        """
        features = {
            'input': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string),
            'clamsa': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example, features)
        x = tf.io.parse_tensor(parsed_features['input'], out_type=tf.int32)
        y = tf.io.parse_tensor(parsed_features['output'], out_type=tf.int32)
        clamsa_track = tf.io.parse_tensor(parsed_features['clamsa'], out_type=tf.double)
        return x, y, clamsa_track
    
    def _read_tfrecord_file(self, repeat=True):
        """Read and preprocess the TFRecord file.

        Args:
            repeat (bool): Whether to repeat the dataset.

        Returns:
            tf.data.Dataset: Processed dataset containing input and output tensors.
        """       
        # dataset = tf.data.TFRecordDataset(self.file_path, compression_type='GZIP')        
        filepath_dataset = tf.data.Dataset.list_files(self.file_path, shuffle=True)

        threads = 96
        # Interleave the reading of these files, parsing tfrecord files in parallel.
        dataset = filepath_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
            cycle_length=min(threads,len(self.file_path)), 
            num_parallel_calls=min(threads,len(self.file_path)),#10,#tf.data.AUTOTUNE,
            deterministic=False
        )
        options = tf.data.Options()
        options.experimental_deterministic = False    
        options.threading.private_threadpool_size = threads
        
        dataset = dataset.with_options(options)
        if self.clamsa:
            dataset = dataset.map(self._parse_fn_clamsa,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self._parse_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.filter:
            dataset = dataset.filter(
#                 lambda x, y: tf.greater(tf.size(tf.unique(tf.argmax(y, axis=-1))[0]), 1)
                lambda x, y, t: tf.greater(tf.reduce_max(tf.argmax(y, axis=-1)), 0)
                )
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=50)

        if repeat:
            dataset = dataset.repeat()

        #dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
    
    def __iter__(self):
        """Return the iterator object for iteration.

        Returns:
            DataGenerator: Iterator object.
        """
        return self

    def __getitem__(self, _index):
        """Get the item at the given index.

        Args:
            _index: Index of the item.

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]: Batch of input and output tensors.
        """
        return self.__next__()
    
    def decode_one_hot(self, encoded_seq):
        # Define the mapping from index to nucleotide
        index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'A'])
        # Use np.argmax to find the index of the maximum value in each row
        nucleotide_indices = np.argmax(encoded_seq, axis=-1)
        # Map indices to nucleotides
        decoded_seq = index_to_nucleotide[nucleotide_indices]
        # Convert from array of characters to string for each sequence
        decoded_seq_str = [''.join(seq) for seq in decoded_seq]
        return decoded_seq_str
    
    def get_seq_weights(self, array, r=250, w=100):
        """
            Get weight matrix where the weights are w around label transitions from non coding to coding.

            Args:
            - labels: A numpy array of shape (batch_size, seq_len, label_size) containing one-hot encoded labels

            Returns:
            - A list of numpy arrays, one per batch, indicating the positions of transitions.
        """
        simp_array = array.argmax(-1)
        seq_weights = np.ones(simp_array.shape, float)
        if array.shape[-1] == 5:
            simp_array = np.where(simp_array < 2, 0, 1)
        elif array.shape[-1] == 7:
            simp_array = np.where(simp_array < 4, 0, 1)
        elif array.shape[-1] == 3:
            simp_array = labels
            
        for i in range(array.shape[0]):
            # Detect changes for the current sequence
            changes = np.diff(simp_array[i])
            transition_points = np.where(changes != 0)[0] + 1  # +1 because np.diff shifts indices to the left
            for t in transition_points:
                seq_weights[i, max(0, t-r):t+r] = w
        return seq_weights
    
    def __next__(self):
        """Get the next batch of data.

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]: Batch of input and output tensors.
        """       
        x_batch = []
        y_batch = []

        while len(x_batch) < self.batch_size:
            x, y, t = next(self.iterator)
            # print(t, tf.shape(t))
            set_t = set([i[0].numpy().decode('utf-8') for i in t]) if len(tf.shape(t)) > 1 else set([t[0].numpy().decode('utf-8')])
            if set_t.intersection(self.tx_filter):
                continue
            x_batch.append(x)
            y_batch.append(y)
            
        # if self.clamsa:
        #     # expect an additional clamsa track
        #     x_batch, y_batch, clamsa_track = next(self.iterator)
        #     clamsa_track = np.array(clamsa_track)                
        # else:
        #     x_batch, y_batch, t_batch = next(self.iterator)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
         
        if not self.softmasking:
            # remove softmasking track
            x_batch = x_batch[:,:,:5]
            
        if not y_batch.shape[-1] == self.output_size:
            # reformat labels so that they fit the output size
            if y_batch.shape[-1] == 7:
                if self.output_size == 5:
                    # reduce intron labels
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 5), np.float32)
                    y_new[:,:,0] = y_batch[:,:,0]
                    y_new[:,:,1] = np.sum(y_batch[:,:,1:4], axis=-1)            
                    y_new[:,:,2:] = y_batch[:,:,4:]
                elif self.output_size == 3:
                    # reduce intron and exon labels
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 3), np.float32)
                    y_new[:,:,0] = y_batch[:,:,0]
                    y_new[:,:,1] = np.sum(y_batch[:,:,1:4], axis=-1)            
                    y_new[:,:,2] = np.sum(y_batch[:,:,4:], axis=-1)  
                y_batch = y_new
            elif y_batch.shape[-1] == 15:
                if self.output_size == 3:
                    # reduce intron and exon labels
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 3), np.float32)
                    y_new[:,:,0] = y_batch[:,:,0]
                    y_new[:,:,1] = np.sum(y_batch[:,:,1:4], axis=-1)            
                    y_new[:,:,2] = np.sum(y_batch[:,:,4:], axis=-1) 
                elif self.output_size == 5:
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 5), np.float32)
                    y_new[:,:,0] = y_batch[:,:,0]
                    y_new[:,:,1] = np.sum(y_batch[:,:,1:4], axis=-1)            
                    y_new[:,:,2] = np.sum(y_batch[:,:,[4, 7, 10, 12]], axis=-1)   
                    y_new[:,:,3] = np.sum(y_batch[:,:,[5, 8, 13]], axis=-1)
                    y_new[:,:,4] = np.sum(y_batch[:,:,[6, 9, 11, 14]], axis=-1)
                elif self.output_size == 7:
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 7), np.float32)
                    y_new[:,:,:4] = y_batch[:,:,:4]       
                    y_new[:,:,4] = np.sum(y_batch[:,:,[4, 7, 10, 12]], axis=-1)   
                    y_new[:,:,5] = np.sum(y_batch[:,:,[5, 8, 13]], axis=-1)
                    y_new[:,:,6] = np.sum(y_batch[:,:,[6, 9, 11, 14]], axis=-1)
                elif self.output_size == 15:
                    y_new = y_batch.astype(np.float32)
                elif self.output_size == 2:
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 2), np.float64)
                    y_new[:,:,0] = np.sum(y_batch[:,:,:4], axis=-1) 
                    y_new[:,:,1] = np.sum(y_batch[:,:,4:], axis=-1) 
                elif self.output_size == 4:
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 4), np.float32)
                    y_new[:,:,0] = np.sum(y_batch[:,:,:4], axis=-1)            
                    y_new[:,:,1] = np.sum(y_batch[:,:,[4, 7, 10, 12]], axis=-1)   
                    y_new[:,:,2] = np.sum(y_batch[:,:,[5, 8, 13]], axis=-1)
                    y_new[:,:,3] = np.sum(y_batch[:,:,[6, 9, 11, 14]], axis=-1)
                y_batch = y_new
            elif y_batch.shape[-1] == 20:
                if self.output_size == 20:
                    y_new = y_batch.astype(np.float32)
                elif self.output_size == 15:
                    y_new = np.zeros((y_batch.shape[0], y_batch.shape[1], 15), np.float32)
                    y_new = y_batch[:,:,:15]
                    y_new[:,:,4] = np.sum(y_batch[:,:,[4, 16]], axis=-1)
                    y_new[:,:,5] = np.sum(y_batch[:,:,[5, 17]], axis=-1)
                    y_new[:,:,6] = np.sum(y_batch[:,:,[6, 18]], axis=-1)
                    y_new[:,:,7] = np.sum(y_batch[:,:,[7, 15]], axis=-1)
                    y_new[:,:,14] = np.sum(y_batch[:,:,[14, 19]], axis=-1)
                y_batch = y_new
        
        if self.hmm_factor:
            # deprecated by the parallelization of the HMM
            step_width = y_batch.shape[1] // self.hmm_factor
            start = y_batch[:,::step_width,:] # shape (batch_size, hmm_factor, 5)
            end = y_batch[:,step_width-1::step_width,:] # shape (batch_size, hmm_factor, 5)
            hints = np.concatenate([start[:,:,tf.newaxis,:], end[:,:,tf.newaxis,:]],-2)
            X = [x_batch, hints]
            Y = [y_batch, y_batch]
        elif self.clamsa:
            X = [x_batch, clamsa_track]
            #X = clamsa_track
            Y = y_batch
        elif self.oracle:
            X = [x_batch, y_batch]
            Y = y_batch
        else:
            X = x_batch
            Y = y_batch
            
        if self.seq_weights:
            return X, Y, self.get_seq_weights(y_batch, w=self.seq_weights)
        else:
            return X, Y

