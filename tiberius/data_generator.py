# ==============================================================
# Authors: Lars Gabriel
#
# Class loading tfrecords so that they fit different training 
# scenarios
# ==============================================================

import sys
import tensorflow as tf
import numpy as np

_empty_serial = tf.io.serialize_tensor(
    tf.constant([], shape=[0,3], dtype=tf.string)
).numpy()

class DataGenerator:
    """DataGenerator class for reading and processing TFRecord files 
    so that they can be used for training a deepfinder model.

    Args:
        file_path (list(str)): Paths to the TFRecord files.
        batch_size (int): Number of examples per batch.
        shuffle (bool): Whether to shuffle the data.
        repeat (bool): Whether to repeat the data set.
        output_size (int): Number of class labels in traings examples.
        seq_weights (int): Weight of positons around exon borders. They aren't used if 0.
        softmasking (bool): Whether softmasking track should be added to input.
        clamsa (bool): Whether Clamsa track should be prepared as additional input,
        oracle (bool): Whether the input data should include the labels.
        tx_filter (list): List of IDs for transcript which will be removed from
                         training by setting training weights in their region to 0.
        tx_filter_region (int): Region around the transcript IDs where the weights are set to 0 as well.
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
                threads=96,
                tx_filter=[],
                tx_filter_region=1000,
                unsupervised_loss=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter = filter
        self.repeat = repeat
        self.seq_weights = seq_weights
        self.output_size = output_size
        self.hmm_factor = hmm_factor
        self.softmasking=softmasking
        if self.softmasking:
            self.input_shape = 6
        else:
            self.input_shape = 5
        self.clamsa = clamsa
        self.oracle = oracle
        self.unsupervised_loss = unsupervised_loss
        self.threads = threads
        self.tx_filter = tf.constant(tx_filter, dtype=tf.string)
        self.tx_filter_region = tx_filter_region

        self.dataset = self._read_tfrecord_file(repeat=repeat)
        self.iterator = iter(self.dataset)
    
    def _parse_fn(self, example):
        features = {
            'input':  tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string),
            # Give tx_ids a default of the empty 0×3 serialization:
            'tx_ids': tf.io.FixedLenFeature([], tf.string,
                                            default_value=_empty_serial),
        }
        parsed = tf.io.parse_single_example(example, features)

        x = tf.io.parse_tensor(parsed['input'],  out_type=tf.int32)
        y = tf.io.parse_tensor(parsed['output'], out_type=tf.int32)
        t = tf.io.parse_tensor(parsed['tx_ids'], out_type=tf.string)

        return x, y, t
    
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
        filepath_dataset = tf.data.Dataset.list_files(self.file_path, shuffle=True)

        # Interleave the reading of these files, parsing tfrecord files in parallel.
        dataset = filepath_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
            cycle_length=min(self.threads,len(self.file_path)), 
            num_parallel_calls=min(self.threads,len(self.file_path)),
            deterministic=False
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = self.threads
        
        dataset = dataset.with_options(options)
        if self.clamsa:
            dataset = dataset.map(self._parse_fn_clamsa,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(self._parse_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.filter:
            dataset = dataset.filter(
                lambda x, y, t: tf.greater(tf.reduce_max(tf.argmax(y, axis=-1)), 0)
                )
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=50)
            options.experimental_deterministic = False 
        else: 
            options.experimental_deterministic = True

        if repeat:
            dataset = dataset.repeat()

        def preprocess(x, y, t=tf.constant([], dtype=tf.string)):            
            tf.debugging.assert_rank(y, 2, message="y must be [seq_len, output_size]")
            x = tf.reshape(x, [-1, tf.shape(x)[-1]]) 
            y = tf.reshape(y, [-1, 15]) 
            x.set_shape([None, self.input_shape])
            y.set_shape([None, 15])

            # TODO
            #x = tf.reshape(x, [-1, tf.shape(x)[-1]]) 
            #y = tf.reshape(y, [-1, self.output_size])
            #x.set_shape([None, self.input_shape])
            #y.set_shape([None, self.output_size])
            if tf.greater(tf.size(t), 0):
                t = tf.reshape(t, [-1, 3])
            if not self.softmasking:
                x = x[:, :5]
            
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
            seq_len = tf.shape(y)[0]

            real_w = self.get_seq_mask(seq_len, transcripts=t,
                                    tx_filter=self.tx_filter,
                                    r_u=self.tx_filter_region,
                                    r_f=self.tx_filter_region//2) 

            default_w = tf.ones([seq_len], dtype=tf.float32)         

            has_tx = tf.size(t) > 0
            w = tf.where(has_tx, real_w, default_w)                 
            
            w = tf.expand_dims(w, axis=-1)                           
            w.set_shape([None, 1])
            
            """
            Description ...
            """
            if self.unsupervised_loss:
                intergenic_regions = tf.cast(tf.equal(Y[:, 0], 1), tf.float32)
                transcript_regions = 1.0 - intergenic_regions
                transcript_regions = 1.0 - intergenic_regions
                w_squeezed = tf.squeeze(w, axis=-1)
                possible_mask_positions = 1.0 - (transcript_regions * w_squeezed)
                random_mask = tf.random.uniform([seq_len]) < 0.15   # bool array
                mask_positions = tf.cast(random_mask, tf.float32) * possible_mask_positions
                
                mask_bool = tf.cast(mask_positions, tf.bool)  # (seq_len,)
                mask_expanded = tf.expand_dims(mask_bool, axis=-1)  # (seq_len, 1)
                mask_4d = tf.tile(mask_expanded, [1, 4])  # (seq_len, 4) - für erste 4 dims

                nucleotides = x[:, :4]
                rest = x[:, 4:]

                nucleotides_masked = tf.where(mask_4d, 
                                               tf.zeros_like(nucleotides), 
                                               nucleotides)

                X_masked = tf.concat([nucleotides_masked, rest], axis=-1)
                w_unsupervised = tf.expand_dims(mask_positions, axis=-1)
                
                Y = tf.concat([
                    tf.cast(Y[:, :15], tf.float32),
                    tf.cast(x[:, :4], tf.float32),
                    tf.cast(w_unsupervised, tf.float32)
                ], axis=-1)
                
                return X_masked, Y, w
                
            return X, Y, w

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

    def get_seq_mask(
        self,
        seq_len,
        transcripts,
        tx_filter,
        r_f,
        r_u
    ) :
        """
        Build a position-wise mask of shape [seq_len], dtype float32, where:
        • Any base in the exact filtered transcripts (no radius) is 0.0 always.
        • Flanks around filtered transcripts (radius r_f) are 0.0, unless protected by a keep-flank.
        • Flanks around unfiltered transcripts (radius r_u) are 1.0, overriding filtered flanks.
        • All other positions are 1.0.

        | Region                    | hit_core | hit_u | hit_f | final_mask |
        | ------------------------- | :-------: | :----: | :----: | :---------: |
        | inside filtered (core)    |   True    |   —    |   —    |    0.0     |
        | keep-flank ∩ filter-flank |   False   |  True  |  True  |    1.0     |
        | filter-flank only         |   False   |  False |  True  |    0.0     |
        | keep-flank only           |   False   |  True  |  False |    1.0     |
        | outside everything        |   False   |  False |  False |    1.0     |

        Args:
            seq_len:     int, total sequence length.
            transcripts: tf.Tensor [num_tx,3], rows = [tx_id, start, end] (strings).
            tx_filter:   list of IDs to filter (mask) or tf.Tensor of shape [k].
            r_f:         int, radius around filtered transcripts to mask.
            r_u:         int, radius around unfiltered transcripts to protect.
        Returns:
            tf.Tensor [seq_len], float32 mask.
        """
        # parse out fields
        tx_ids = transcripts[:, 0]
        starts = tf.strings.to_number(transcripts[:, 1], out_type=tf.int32)
        ends   = tf.strings.to_number(transcripts[:, 2], out_type=tf.int32)

        # build filter-ID tensor
        if not isinstance(tx_filter, tf.Tensor):
            filter_ids = tf.constant(tx_filter, dtype=tf.string)
        else:
            filter_ids = tf.cast(tx_filter, tf.string)

        # boolean mask of filtered transcripts
        is_f = tf.reduce_any(
            tf.equal(
                tf.expand_dims(tx_ids, 1),    # [num_tx,1]
                tf.expand_dims(filter_ids, 0)  # [1, n_filters]
            ),
            axis=1                           # [num_tx]
        )
        is_f = tf.ensure_shape(is_f, [None])

        # split filtered vs unfiltered intervals
        st_f = tf.boolean_mask(starts, is_f)   # filtered starts
        en_f = tf.boolean_mask(ends, is_f)   # filtered ends
        st_u = tf.boolean_mask(starts, ~is_f)     # keep starts
        en_u = tf.boolean_mask(ends, ~is_f)     # keep ends

        # apply radii and clip to [0, seq_len]
        r_f = tf.convert_to_tensor(r_f, tf.int32)
        r_u = tf.convert_to_tensor(r_u, tf.int32)
        st_f_exp = tf.clip_by_value(st_f - r_f, 0, seq_len)
        en_f_exp = tf.clip_by_value(en_f + r_f, 0, seq_len)
        st_u_exp = tf.clip_by_value(st_u - r_u, 0, seq_len)
        en_u_exp = tf.clip_by_value(en_u + r_u, 0, seq_len)

        # position vector [seq_len,1]
        pos = tf.range(seq_len, dtype=tf.int32)[:, None]

        # hits in filtered+r and unfiltered+r intervals
        hit_f = tf.reduce_any(
            tf.logical_and(
                pos >= tf.expand_dims(st_f_exp, 0),
                pos <  tf.expand_dims(en_f_exp, 0)
            ),
            axis=1
        )  # [seq_len]
        hit_u = tf.reduce_any(
            tf.logical_and(
                pos >= tf.expand_dims(st_u_exp, 0),
                pos <  tf.expand_dims(en_u_exp, 0)
            ),
            axis=1
        )  # [seq_len]

        # core-hit: exact filtered intervals (no radius)
        hit_core = tf.reduce_any(
            tf.logical_and(
                pos >= tf.expand_dims(st_f, 0),
                pos <  tf.expand_dims(en_f, 0)
            ),
            axis=1
        )  # [seq_len]

        # build the dual-radius “unfiltered wins” mask
        mask_dual = tf.where(
            hit_u,
            tf.ones_like(hit_u, dtype=tf.float32),
            tf.where(
                hit_f,
                tf.zeros_like(hit_f, dtype=tf.float32),
                tf.ones_like(hit_f, dtype=tf.float32)
            )
        )

        # override cores to zero
        final_mask = tf.where(
            hit_core,
            tf.zeros_like(hit_core, dtype=tf.float32),
            mask_dual
        )

        return final_mask


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
        elif self.output_size == 4:
            simp_array = tf.where(simp_array == 0, 0, 1)    # TODO: not sure?

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
