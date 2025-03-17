# ==============================================================
# Authors: Lars Gabriel
#
# Class loading tfrecords so that they fit different training 
# scenarios
# 
# ==============================================================

import sys
import tensorflow as tf
import numpy as np

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
                # trans=False, trans_lstm=False, 
                 seq_weights=0, softmasking=True,
                clamsa=False,
                oracle=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter = filter
        self.repeat = repeat
        self.seq_weights = seq_weights
        self.output_size = output_size
        self.hmm_factor = hmm_factor
        # self.trans=trans
        # self.trans_lstm=trans_lstm
        self.softmasking=softmasking
        self.clamsa = clamsa
        self.oracle = oracle
        
        self.dataset = self._read_tfrecord_file(repeat=repeat)
    
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
            #'output_phase': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example, features)
        x = tf.io.parse_tensor(parsed_features['input'], out_type=tf.int32)
        y = tf.io.parse_tensor(parsed_features['output'], out_type=tf.int32)
        #y_phase = tf.io.parse_tensor(parsed_features['output_phase'], out_type=tf.int32)
        return x, y#, y_phase
    
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
                #  lambda x, y: tf.greater(tf.size(tf.unique(tf.argmax(y, axis=-1))[0]), 1)
                lambda x, y, t: tf.greater(tf.reduce_max(tf.argmax(y, axis=-1)), 0)
                )
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=50)

        if repeat:
            dataset = dataset.repeat()

        def preprocess(x, y):
            x = tf.ensure_shape(x, [None, None])  # Adjust shape as needed
            y = tf.ensure_shape(y, [None, self.output_size])

            if not self.softmasking:
                x = x[:, :, :5]

            if y.shape[-1] != self.output_size:
                y = self._reformat_labels(y)

            if self.hmm_factor:
                step_width = y.shape[1] // self.hmm_factor
                start = y[::step_width, :]
                end = y[step_width-1::step_width, :]
                hints = tf.stack([start, end], axis=-2)
                X = (x, hints)
                Y = (y, y)
            elif self.clamsa:
                X = (x, clamsa_track)  # clamsa_track from parse_fn_clamsa
                Y = y
            elif self.oracle:
                X = (x, y)
                Y = y
            else:
                X = x
                Y = y

            if self.seq_weights:
                weights = self._get_seq_weights(y, r=250, w=100)
                return X, Y, weights
            return X, Y

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _reformat_labels(self, y):
        """Reformat label tensor to match the desired output_size using TensorFlow operations.

        Args:
            y: A TensorFlow tensor of shape (batch_size, seq_len, input_classes) containing one-hot encoded labels.

        Returns:
            A TensorFlow tensor of shape (batch_size, seq_len, output_size) with reformatted labels.
        """
        batch_size = tf.shape(y)[0]
        seq_len = tf.shape(y)[1]
        input_classes = tf.shape(y)[-1]
    
        # Case 1: Input has 7 classes
        if input_classes == 7:
            if self.output_size == 5:
                # Reduce intron labels (1,2,3 -> 1)
                y_new = tf.concat([
                    y[:, :, :1],                    # 0 (non-coding)
                    tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),  # 1-3 (introns)
                    y[:, :, 4:]                     # 4-6 (exons)
                ], axis=-1)
            elif self.output_size == 3:
                # Reduce intron (1-3) and exon (4-6) labels
                y_new = tf.concat([
                    y[:, :, :1],                    # 0 (non-coding)
                    tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),  # 1-3 (introns)
                    tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)    # 4-6 (exons)
                ], axis=-1)
            else:
                # For other output_sizes, return unchanged or handle separately
                y_new = y

                # Case 2: Input has 15 classes
        elif input_classes == 15:
            if self.output_size == 3:
                # Reduce intron (1-3) and exon (4-14) labels
                y_new = tf.concat([
                    y[:, :, :1],                    # 0 (non-coding)
                    tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),  # 1-3 (introns)
                    tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)    # 4-14 (exons)
                ], axis=-1)
            elif self.output_size == 5:
                # Reduce to 5 classes: 0, 1-3, 4/7/10/12, 5/8/13, 6/9/11/14
                y_new = tf.concat([
                    y[:, :, :1],                    # 0 (non-coding)
                    tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),  # 1-3 (introns)
                    tf.reduce_sum(tf.gather(y, [4, 7, 10, 12], axis=-1), axis=-1, keepdims=True),  # Exon 1
                    tf.reduce_sum(tf.gather(y, [5, 8, 13], axis=-1), axis=-1, keepdims=True),      # Exon 2
                    tf.reduce_sum(tf.gather(y, [6, 9, 11, 14], axis=-1), axis=-1, keepdims=True)   # Exon 3
                ], axis=-1)
            elif self.output_size == 7:
                # Reduce to 7 classes: 0-3 unchanged, 4/7/10/12, 5/8/13, 6/9/11/14
                y_new = tf.concat([
                    y[:, :, :4],                    # 0-3 (non-coding, introns)
                    tf.reduce_sum(tf.gather(y, [4, 7, 10, 12], axis=-1), axis=-1, keepdims=True),  # Exon 1
                    tf.reduce_sum(tf.gather(y, [5, 8, 13], axis=-1), axis=-1, keepdims=True),      # Exon 2
                    tf.reduce_sum(tf.gather(y, [6, 9, 11, 14], axis=-1), axis=-1, keepdims=True)   # Exon 3
                ], axis=-1)
            elif self.output_size == 15:
                # No reformatting needed
                y_new = tf.cast(y, tf.float32)
            elif self.output_size == 2:
                # Binary: non-coding (0-3) vs. coding (4-14)
                y_new = tf.concat([
                    tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),   # 0-3 (non-coding)
                    tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)    # 4-14 (coding)
                ], axis=-1)
            elif self.output_size == 4:
                # 4 classes: 0-3, 4/7/10/12, 5/8/13, 6/9/11/14
                y_new = tf.concat([
                    tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),   # 0-3 (non-coding/introns)
                    tf.reduce_sum(tf.gather(y, [4, 7, 10, 12], axis=-1), axis=-1, keepdims=True),  # Exon 1
                    tf.reduce_sum(tf.gather(y, [5, 8, 13], axis=-1), axis=-1, keepdims=True),      # Exon 2
                    tf.reduce_sum(tf.gather(y, [6, 9, 11, 14], axis=-1), axis=-1, keepdims=True)   # Exon 3
                ], axis=-1)
            else:
                y_new = y
                
                # If input_classes already matches output_size, no change needed
        else:
            y_new = y

        return y_new

    def _get_seq_weights(self, y, r=250, w=100):
            """Get weight matrix where weights are `w` around label transitions from non-coding to coding.
            Args:
                y: A TensorFlow tensor of shape (batch_size, seq_len, label_size) containing one-hot encoded labels.
                r: Range around transitions (default: 250).
                w: Weight value to apply around transitions (default: 100).

            Returns:
                A TensorFlow tensor of shape (batch_size, seq_len) with weights, dtype tf.float32.
            """
            # Simplify one-hot labels to binary (0 for non-coding, 1 for coding)
            simp_array = tf.argmax(y, axis=-1)  # Shape: (batch_size, seq_len)

            # Adjust based on output_size (from your original logic)
            if self.output_size == 5:
                simp_array = tf.where(simp_array < 2, 0, 1)  # 0,1 -> 0 (non-coding); 2,3,4 -> 1 (coding)
            elif self.output_size == 7:
                simp_array = tf.where(simp_array < 4, 0, 1)  # 0-3 -> 0; 4-6 -> 1
            elif self.output_size == 3:
                simp_array = simp_array  # Already binary-ish, assuming 0=non-coding, 1=coding, 2=other

            # Initialize weights as ones
            seq_weights = tf.ones_like(simp_array, dtype=tf.float32)  # Shape: (batch_size, seq_len)

            # Compute differences to detect transitions
            changes = tf.concat([
                tf.zeros([tf.shape(simp_array)[0], 1], dtype=simp_array.dtype),  # Pad start
                tf.experimental.numpy.diff(simp_array, axis=1)  # Diff along seq_len
            ], axis=1)  # Shape: (batch_size, seq_len)

            # Find transition points (where changes != 0)
            transition_points = tf.where(changes != 0)  # Shape: (num_transitions, 2) [batch_idx, seq_idx]

            # Use self.seq_weights if set, otherwise fall back to default w
            weight_value = tf.cast(self.seq_weights if self.seq_weights != 0 else w, tf.float32)

            # Function to update weights for each transition
            def update_weights(weights, transition):
                batch_idx, t = transition[0], transition[1]
                start = tf.maximum(0, t - r)
                end = t + r  # End is exclusive in range, handled by scatter
                indices = tf.range(start, end, dtype=tf.int64)
                indices = tf.clip_by_value(indices, 0, tf.shape(weights)[1] - 1)  # Ensure within bounds
                indices = tf.stack([tf.fill([tf.shape(indices)[0]], batch_idx), indices], axis=1)
                updates = tf.fill([tf.shape(indices)[0]], weight_value)
                return tf.tensor_scatter_nd_update(weights, indices, updates)

            # Apply updates for all transitions
            if tf.size(transition_points) > 0:  # Only if there are transitions
                seq_weights = tf.foldl(
                    lambda acc, t: update_weights(acc, t),
                    transition_points,
                initializer=seq_weights
                )

            return seq_weights

    def get_dataset(self):
        """Return the tf.data.Dataset for use in model.fit."""
        return self.dataset
