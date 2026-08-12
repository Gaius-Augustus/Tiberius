import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, LSTM,
                                Dense, Bidirectional, Dropout, Activation, Input,
                                Reshape, LayerNormalization)
from tensorflow import keras
from tensorflow.keras import backend as K
from tiberius.hmm import HMMBlock, TrainableHMMHead, default_trainable_hmm_config
from tiberius.mixers import (DiagonalSSMBlock, SlidingWindowAttentionBlock,
                              BorderGatedSelfAttention,
                              BorderTopKMessagePassing)
from hidten import HMMMode

class Cast(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x[0][..., :5] if isinstance(x, list) else x[..., :5], tf.float32)


class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, peak=0.1, warmup=0, min_lr=0.0001):
        super(BatchLearningRateScheduler, self).__init__()
        self.peak = peak
        self.warmup = warmup
        self.total_batches = 0
        self.min_lr = min_lr

    def on_batch_end(self, batch, logs=None):
        self.total_batches += 1
        if self.total_batches <= self.warmup:
            new_lr = self.total_batches * self.peak / self.warmup
        if self.total_batches > self.warmup:
            new_lr = (self.total_batches-self.warmup)**(-1/2) * self.peak
        if new_lr > self.min_lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
        # tf.print("\n",new_lr, self.min_lr,"\n")


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, save_path):
        super(ValidationCallback, self).__init__()
        self.best_val_loss = float('inf')
        self.save_path = save_path
        self.val_gen = val_gen

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % 3000 == 0:
            #val_loss, loss1, loss2, acc1, acc2 = self.model.evaluate(self.val_gen, verbose=1)
            loss, acc = self.model.evaluate(self.val_gen, verbose=1)
            print(loss, acc)
            with open(self.save_path + '_log.txt', 'a+') as f_h:
                f_h.write(f'{loss}, {acc}\n')
            if loss < self.best_val_loss:
                print("Saving the model as the validation loss improved.")
                self.best_val_loss = loss
                self.model.save(self.save_path, save_traces=False)
                #self.model.save_weights(self.save_path)

class BatchSave(tf.keras.callbacks.Callback):
    def __init__(self, save_path, batch_number):
        super(BatchSave, self).__init__()
        self.save_path = save_path
        self.batch_number = batch_number
        self.prev_batch_numb = 0
        self.tf_old = tf.__version__ < "2.12.0"

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.batch_number == 0:
            self.prev_batch_numb += batch
            if self.tf_old:
                self.model.save(self.save_path.format(self.prev_batch_numb), save_traces=False)
            else:
                self.model.save(self.save_path.format(self.prev_batch_numb) +".keras")

def custom_cce_f1_loss(f1_factor, batch_size,
                    include_reading_frame=True, use_cce=True, from_logits=False,
                    exon_count_factor=0.0, exon_count_per_class=False,
                    exon_count_threshold=0.0, label_smoothing=0.0):
    # 15-class label layout (see gene_pred_hmm_emitter.py):
    # (Ir, I0, I1, I2, E0, E1, E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP)
    # Exon borders: starts = START(7), IE0-IE2(11-13); ends = EI0-EI2(8-10), STOP(14).
    exon_border_channels = [7, 8, 9, 10, 11, 12, 13, 14]

    @tf.function
    def loss_(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if use_cce:
            # Compute the categorical cross-entropy loss
            cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing)
            cce_loss = tf.reduce_mean(cce_loss, -1) #mean over sequence length
            cce_loss = tf.reduce_sum(cce_loss) / batch_size #mean over batch with global batch size
        else:
            cce_loss = 0
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        # Compute the f1 loss
        if tf.shape(y_true)[-1] == 5:
            if include_reading_frame:
                cds_pred = y_pred[:, :, -3:]
                cds_true = y_true[:, :, -3:]
            else:
                cds_pred = tf.reduce_sum(y_pred[:, :, -3:], axis=-1, keepdims=True)
                cds_true = tf.reduce_sum(y_true[:, :, -3:], axis=-1, keepdims=True)
        else:
            if include_reading_frame:
                cds_pred = y_pred[:, :, 4:]
                cds_true = y_true[:, :, 4:]
            else:
                cds_pred = tf.reduce_sum(y_pred[:, :, 4:], axis=-1, keepdims=True)
                cds_true = tf.reduce_sum(y_true[:, :, 4:], axis=-1, keepdims=True)

        # Compute precision and recall for the specified class
        true_positives = tf.reduce_sum(cds_pred * cds_true, axis=1)
        predicted_positives = tf.reduce_sum(cds_pred, axis=1)
        possible_positives = tf.reduce_sum(cds_true, axis=1)
        any_positives = tf.cast(possible_positives > 0, possible_positives.dtype)

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives  / (possible_positives + K.epsilon())

        # For the examples with positive class, maximize the F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon()) #f1 score per sequence
        f1_loss = tf.reduce_sum((1 - f1_score) * any_positives) / batch_size #mean over batch with global batch size

        # For the examples with no positive class, minimize the false positive rate
        L = tf.cast(tf.shape(cds_pred)[1], cds_pred.dtype)
        fpr = tf.reduce_sum(cds_pred * (1-any_positives)[:,tf.newaxis]) / (L * batch_size)

        # Soft exon-count penalty: L1 between the number of true exon-start
        # positions in the window and the summed predicted probability mass on
        # the exon-start channels. Only defined for the 15-class label space,
        # where exon-start markers exist as separate channels.
        n_classes = y_true.shape[-1]
        if exon_count_factor > 0 and n_classes == 15:
            gathered_pred = tf.gather(y_pred, exon_border_channels, axis=-1)  # (B, L, 8)
            gathered_true = tf.gather(y_true, exon_border_channels, axis=-1)  # (B, L, 8)
            if exon_count_threshold > 0.0:
                # zero out per-class positions below threshold before counting
                mask = tf.cast(gathered_pred > exon_count_threshold, gathered_pred.dtype)
                gathered_pred = gathered_pred * mask
            if exon_count_per_class:
                # L1 per channel (B, 8) -> mean over 8 border classes -> scalar
                count_pred = tf.reduce_sum(gathered_pred, axis=1)  # (B, 8)
                count_true = tf.reduce_sum(gathered_true, axis=1)  # (B, 8)
                per_class = tf.reduce_sum(tf.abs(count_pred - count_true), axis=0) / batch_size  # (8,)
                exon_count_loss = tf.reduce_mean(per_class)
            else:
                # original: merge channels first, then L1
                starts_pred = tf.reduce_sum(gathered_pred, axis=-1)  # (B, L)
                starts_true = tf.reduce_sum(gathered_true, axis=-1)  # (B, L)
                count_pred = tf.reduce_sum(starts_pred, axis=1)
                count_true = tf.reduce_sum(starts_true, axis=1)
                exon_count_loss = tf.reduce_sum(tf.abs(count_pred - count_true)) / batch_size
        else:
            exon_count_loss = tf.cast(0.0, cce_loss.dtype if use_cce else f1_loss.dtype)

        # Combine CCE loss, F1 score and soft exon-count penalty
        combined_loss = cce_loss + f1_factor * (f1_loss + fpr) + exon_count_factor * exon_count_loss
        return combined_loss
    return loss_

class StrandAccuracy(tf.keras.metrics.Metric):
    """Categorical accuracy for one strand of a dual-strand (B,T,2C) output.

    Parameters
    ----------
    name       : metric name shown in training logs
    slice_start: first class index of this strand (0 for fwd, C for rev)
    slice_end  : one-past-last class index (C for fwd, 2C for rev)
    """

    def __init__(self, name: str, slice_start: int, slice_end: int, **kwargs):
        super().__init__(name=name, **kwargs)
        self._s = slice_start
        self._e = slice_end
        self._correct = self.add_weight(name="correct", initializer="zeros", shape=())
        self._total = self.add_weight(name="total", initializer="zeros", shape=())

    def update_state(self, y_true, y_pred, sample_weight=None):  # noqa: ARG002
        yt = tf.cast(y_true[..., self._s:self._e], y_pred.dtype)
        yp = y_pred[..., self._s:self._e]
        match = tf.equal(tf.argmax(yt, axis=-1), tf.argmax(yp, axis=-1))
        self._correct.assign_add(
            tf.cast(tf.reduce_sum(tf.cast(match, tf.int32)), tf.float32)
        )
        self._total.assign_add(tf.cast(tf.size(match), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._correct, self._total)

    def reset_state(self):
        self._correct.assign(0.0)
        self._total.assign(0.0)


def custom_cce_f1_loss_dual_strand(f1_factor, batch_size,
                                   include_reading_frame=True, use_cce=True,
                                   from_logits=False,
                                   exon_count_factor=0.0,
                                   exon_count_per_class=False,
                                   exon_count_threshold=0.0,
                                   label_smoothing=0.0):
    """Dual-strand wrapper around custom_cce_f1_loss.

    Expects y_pred / y_true of shape (B, T, 2*C) where the first C channels
    are the forward strand and the last C are the reverse strand.  Loss is
    computed independently per strand and averaged.
    """
    per_strand = custom_cce_f1_loss(
        f1_factor, batch_size,
        include_reading_frame=include_reading_frame,
        use_cce=use_cce,
        from_logits=from_logits,
        exon_count_factor=exon_count_factor,
        exon_count_per_class=exon_count_per_class,
        exon_count_threshold=exon_count_threshold,
        label_smoothing=label_smoothing,
    )

    @tf.function
    def loss_(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        n = tf.shape(y_pred)[-1] // 2
        loss_fwd = per_strand(y_true[:, :, :n], y_pred[:, :, :n])
        loss_rev = per_strand(y_true[:, :, n:], y_pred[:, :, n:])
        return (loss_fwd + loss_rev) / tf.cast(2, loss_fwd.dtype)

    return loss_


def lstm_model(units=372, filter_size=128,
              kernel_size=9, numb_conv=3,
               numb_lstm=2, dropout_rate=0.0,
               pool_size=9,
               lstm_mask=False, output_size=15,
               multi_loss=False, residual_conv=True,
               clamsa=False, clamsa_kernel=6, softmasking=True, lru_layer=False,
               final_activation='softmax',
              ):
    """
    Constructs a hybrid model that combines CNNs and bLSTM layers for gene prediction.

    Parameters:
        units (int): The number of units in each LSTM layer.
        filter_size (int): The number of filters in the convolutional layers.
        kernel_size (int): The kernel size for convolutional operations.
        numb_conv (int): The total number of convolutional layers in the model.
        numb_lstm (int): The total number of bidirectional LSTM layers.
        dropout_rate (float): Dropout rate applied to LSTM layers for regularization.
        pool_size (int): The size of the pooling operation to reduce dimensionality.
        lstm_mask (bool): If True, applies a masking mechanism to LSTM layers.
        output_size (int): The dimensionality of the output layer, often equal to the number of classes.
        multi_loss (bool): If True, utilizes intermediate outputs for multi-loss training.
        residual_conv (bool): If True, adds a residual connection from the convolutional layers to the final output.
        clamsa (bool): If True, clamsa track is added to class labels of the LSTM.

    Returns:
        tf.keras.Model: A compiled Keras model that is ready for training, featuring a mix of
                        convolutional layers and bidirectional LSTM layers.

    The model takes sequence data as input, processes it through convolutional and LSTM layers,
    and outputs a prediction vector. The architecture supports customization through various
    parameters, enabling it to adapt to different types of sequential data and learning tasks.
    """
    if lru_layer:
        import LRU_tf as lru

    # Input
    outputs = []
    if softmasking:
        inp_size=6
    else:
        inp_size=5

    input_shape = (None, inp_size)
    main_input = Input(shape=input_shape)
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]

        # main_input = tf.concat([main_input, inp_clamsa], axis=-1)
        main_input = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
        inp_size+=4
    else:
        inp = main_input


    # First convolution
    inp_embedding = main_input
    x = Conv1D(filter_size, 3, padding='same',
                    activation="relu", name='initial_conv')(main_input)

    # Convolutional layers
    for i in range(numb_conv-1):
        x = LayerNormalization(name=f'layer_normalization{i+1}')(x)
        x = Conv1D(filter_size, kernel_size, padding='same',
                       activation="relu", name=f'conv_{i+1}')(x)

    # Add input to the convolutions
    cnn_out = x
    # x = tf.concat([inp_embedding, cnn_out], axis=-1)
    x = keras.layers.Concatenate(axis=-1)([inp_embedding, cnn_out])

    if multi_loss:
        y_cnn = Dense(output_size, activation='relu', name='cnn_dense')(x)
        y_cnn = Activation('softmax', name='out_cnn')(y_cnn)
        outputs.append(y_cnn)

    # Reshape layer
    if pool_size > 1:
        x = Reshape((-1, pool_size * (filter_size+inp_size)), name='R1')(x)

    # Dense layer to match unit size of LSTM
    x = Dense(2*units, name='pre_lstm_dense')(x)
    pi = 3.141
    # Bidirectional LSTM layers
    for i in range(numb_lstm):
        if lru_layer:
            # period >=3 in first layer, >=20 in deeper layers
            mp = 2*pi/2 if i<1 else 2*pi/50
            rmin = 0 if i<1 else 0.9
            lru_block = lru.LRU_Block(
                  N=2*units, # hidden dim
                  H=2*units, # output dim
                  bidirectional=True,
                  max_tree_depth=17,
                  r_min=rmin,
                  # return_sequences=True,
                   max_phase=mp)
            # lru_block.build(input_shape=x.shape)
            x_next = lru_block(x)
        else:
            x_next = Bidirectional(LSTM(units, return_sequences=True),
                    name=f'biLSTM_{i+1}')(x)
        if dropout_rate:
            x_next = Dropout(dropout_rate, name=f'dropout_{i+1}')(x_next)
            x = LayerNormalization(name=f'layer_normalization_lstm{i+1}')(x_next + x)
        else:
            x = x_next

        if multi_loss and i < numb_lstm-1:
            x_loss = Dense(pool_size * output_size, activation='relu', name=f'dense_lstm_{i+1}')(x)
            if pool_size > 1:
                x_loss = Reshape((-1, output_size), name=f'Reshape_loss_{i+1}')(x_loss)
            outputs.append(Activation('softmax', name=f'out_lstm_{i+1}')(x_loss))

    if lstm_mask:
        mask = Dense(1, activation='sigmoid', name='mask')(x)
        mask = tf.greater(mask[:, :, 0], 0.5)
        for i in range(2):
            x = Bidirectional(LSTM(units, return_sequences=True),
                name=f'biLSTM_mask_{i+1}')(inputs=x, mask=mask)

    if residual_conv:
        x = Dense(pool_size * 30, activation='relu', name='dense')(x)
        x = Reshape((-1, 30), name='Reshape2')(x)
        # x = tf.concat([x, cnn_out], axis=-1)
        x = keras.layers.Concatenate(axis=-1)([x, cnn_out])
        x = Dense(output_size, name='out_dense')(x)
    else:
        x = Dense(pool_size * output_size, activation='relu', name='out_dense')(x)

        if pool_size > 1:
            x = Reshape((-1, output_size), name='Reshape2')(x)

    y_end = Activation(final_activation, name='out')(x)

    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def border_gated_attn_lstm_model(units=372, filter_size=128,
                                  kernel_size=9, numb_conv=3,
                                  numb_lstm=2, dropout_rate=0.0,
                                  pool_size=9, output_size=15,
                                  clamsa=False, softmasking=True,
                                  num_heads=8, attn_dropout=0.0,
                                  attn_position='between',
                                  border_aux_output=False,
                                  attn_zero_init=True,
                                  attn_chunk_size=None):
    """
    CNN + biLSTM backbone (same structure as `lstm_model`) with a
    `BorderGatedSelfAttention` block inserted at pooled resolution.

    The attention block is a global multi-head self-attention whose KEY
    logits are biased by a per-position "exon-border" score predicted by a
    tiny internal head. High-scoring positions become message-passing hubs
    that every query attends to; the border head is learned end-to-end
    from the main loss (optionally with an auxiliary supervision output).

    Placement is controlled by `attn_position`:
        'between'     -- one block between the two biLSTMs (default, cheap).
        'before_each' -- one block before every biLSTM (2x on defaults).
        'before_first'/'before_last' -- one block before only that biLSTM.

    Cost at Tiberius defaults (w_size=9999, pool_size=9 -> L=1111,
    2*units=744): one attention block adds ~40% of one biLSTM's FLOPs and
    ~1.3 GB attention-matrix activations at batch=32. `'between'` adds
    ~15% training time and ~1.3 GB VRAM; `'before_each'` roughly doubles
    that.

    The attention output is added residually (via a zero-init projection
    inside `BorderGatedSelfAttention.o_proj`), so at step 0 the block is
    a no-op and the base CNN+biLSTM pathway is preserved.

    Args:
        units, filter_size, kernel_size, numb_conv, numb_lstm,
        dropout_rate, pool_size, output_size, clamsa, softmasking:
            same as `lstm_model`.
        num_heads: attention heads (must divide 2*units).
        attn_dropout: dropout on the attention softmax.
        attn_position: where to insert attention blocks (see above).
        border_aux_output: if True, expose per-position border logits
            as an additional named output ('border_logits_{i}') per
            attention block. Aux loss/target must be attached by the
            caller.
        attn_zero_init: if True, zero-init the attention output
            projection so the block is identity at step 0.
        attn_chunk_size: if not None, attention runs on non-overlapping
            chunks of that length (memory O(L*chunk) instead of O(L^2)).
            For inference on long sequences with a model trained at
            `chunk_size=None`, prefer patching post-load via
            `enable_chunked_border_attention(model, chunk_size)` from
            `tiberius.mixers`, so training-time weights are preserved
            without rebuilding.

    Returns:
        tf.keras.Model with softmax output at nucleotide resolution
        (plus any auxiliary border-logit outputs if enabled).
    """
    valid_positions = {'between', 'before_each', 'before_first', 'before_last'}
    if attn_position not in valid_positions:
        raise ValueError(
            f"attn_position must be one of {sorted(valid_positions)}, "
            f"got {attn_position!r}."
        )
    if numb_lstm < 1:
        raise ValueError("numb_lstm must be >= 1.")

    d_stream = 2 * units  # width of the pooled recurrent stream

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        main_input = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
        inp_size += 4
    else:
        inp = main_input

    # Conv stem (identical to lstm_model).
    inp_embedding = main_input
    x = Conv1D(filter_size, 3, padding='same',
               activation='relu', name='initial_conv')(main_input)
    for i in range(numb_conv - 1):
        x = LayerNormalization(name=f'layer_normalization{i+1}')(x)
        x = Conv1D(filter_size, kernel_size, padding='same',
                   activation='relu', name=f'conv_{i+1}')(x)
    cnn_out = x
    x = keras.layers.Concatenate(axis=-1)([inp_embedding, cnn_out])

    # Pool + project to recurrent-stream width (identical to lstm_model).
    if pool_size > 1:
        x = Reshape((-1, pool_size * (filter_size + inp_size)), name='R1')(x)
    x = Dense(d_stream, name='pre_lstm_dense')(x)

    def _insert_border_attn(x_in, name_prefix, block_idx):
        """Border-gated attention with residual add."""
        h = LayerNormalization(name=f'{name_prefix}_ln')(x_in)
        attn = BorderGatedSelfAttention(
            d_model=d_stream,
            num_heads=num_heads,
            dropout=attn_dropout,
            zero_init_proj=attn_zero_init,
            chunk_size=attn_chunk_size,
            name=f'{name_prefix}_attn',
        )
        out, border_logits = attn(h, return_border=True)
        if border_aux_output:
            aux = Activation('linear',
                             name=f'border_logits_{block_idx}')(border_logits)
            outputs.append(aux)
        return keras.layers.Add(name=f'{name_prefix}_add')([x_in, out])

    # Decide placement.
    positions_before_lstm_i = set()  # 0-indexed lstm block indices
    if attn_position == 'before_each':
        positions_before_lstm_i.update(range(numb_lstm))
    elif attn_position == 'before_first':
        positions_before_lstm_i.add(0)
    elif attn_position == 'before_last':
        positions_before_lstm_i.add(numb_lstm - 1)
    elif attn_position == 'between':
        if numb_lstm >= 2:
            positions_before_lstm_i.add(1)
        # else no-op: 'between' with a single biLSTM is meaningless.

    block_idx = 0
    for i in range(numb_lstm):
        if i in positions_before_lstm_i:
            block_idx += 1
            x = _insert_border_attn(x, f'attn_blk{block_idx}', block_idx)
        x_next = Bidirectional(LSTM(units, return_sequences=True),
                               name=f'biLSTM_{i+1}')(x)
        if dropout_rate:
            x_next = Dropout(dropout_rate, name=f'dropout_{i+1}')(x_next)
            x = LayerNormalization(
                name=f'layer_normalization_lstm{i+1}')(x_next + x)
        else:
            x = x_next

    # Output head (identical to lstm_model's residual_conv=True branch).
    x = Dense(pool_size * 30, activation='relu', name='dense')(x)
    x = Reshape((-1, 30), name='Reshape2')(x)
    x = keras.layers.Concatenate(axis=-1)([x, cnn_out])
    x = Dense(output_size, name='out_dense')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def border_topk_lstm_model(units=372, filter_size=128,
                            kernel_size=9, numb_conv=3,
                            numb_lstm=2, dropout_rate=0.0,
                            pool_size=9, output_size=15,
                            clamsa=False, softmasking=True,
                            num_heads=8, attn_dropout=0.0,
                            attn_position='between',
                            border_aux_output=False,
                            attn_zero_init=True,
                            topk_K=32,
                            straight_through=True):
    """
    CNN + biLSTM backbone (same structure as `lstm_model`) with a
    `BorderTopKMessagePassing` block inserted at pooled resolution.

    Instead of dense self-attention, the block picks the K positions
    with the highest predicted border score as "hub" nodes and lets
    every position attend only over those K hubs. Cost is O(L*K), so
    the block scales linearly to arbitrary sequence lengths.

    Training vs inference K
    -----------------------
    `topk_K` is length-independent -- Q/K/V/O projections and the border
    head are all K-agnostic in parameter count -- so the SAME trained
    weights can be reloaded into a model built with a different `topk_K`
    at inference. Typical workflow:

        # training
        arch: border_topk_lstm
        topk_K: 10          # <- pick something matched to 10k windows

        # inference: edit model_config.json (or override in eval)
        topk_K: 50          # <- more hubs to cover 500k sequences

    Same weights.h5 loads into both. Any softmax-distribution shift is
    controlled solely by the K ratio (`K_infer / K_train`), not by L.

    Args:
        units, filter_size, kernel_size, numb_conv, numb_lstm,
        dropout_rate, pool_size, output_size, clamsa, softmasking,
        num_heads, attn_dropout, attn_position, border_aux_output,
        attn_zero_init: same as `border_gated_attn_lstm_model`.
        topk_K: number of hubs picked by Top-K per sample. Keep small
            enough to be well below sequence length. Can be changed
            between training and inference without retraining.
        straight_through: if True, hub features are multiplied by
            `sigmoid(border_logit)` so the border-score head gets
            gradient through the discrete selection. If False, only
            the Q/K/V/O projections train and the border head learns
            only via an auxiliary supervision loss (needs
            `border_aux_output=True` plus an outer BCE loss).

    Returns:
        tf.keras.Model with softmax output at nucleotide resolution
        (plus per-block border-logit outputs if `border_aux_output=True`).
    """
    valid_positions = {'between', 'before_each', 'before_first', 'before_last'}
    if attn_position not in valid_positions:
        raise ValueError(
            f"attn_position must be one of {sorted(valid_positions)}, "
            f"got {attn_position!r}."
        )
    if numb_lstm < 1:
        raise ValueError("numb_lstm must be >= 1.")

    d_stream = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        main_input = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
        inp_size += 4
    else:
        inp = main_input

    # Conv stem (identical to lstm_model).
    inp_embedding = main_input
    x = Conv1D(filter_size, 3, padding='same',
               activation='relu', name='initial_conv')(main_input)
    for i in range(numb_conv - 1):
        x = LayerNormalization(name=f'layer_normalization{i+1}')(x)
        x = Conv1D(filter_size, kernel_size, padding='same',
                   activation='relu', name=f'conv_{i+1}')(x)
    cnn_out = x
    x = keras.layers.Concatenate(axis=-1)([inp_embedding, cnn_out])

    # Pool + project to recurrent-stream width.
    if pool_size > 1:
        x = Reshape((-1, pool_size * (filter_size + inp_size)), name='R1')(x)
    x = Dense(d_stream, name='pre_lstm_dense')(x)

    def _insert_topk_block(x_in, name_prefix, block_idx):
        h = LayerNormalization(name=f'{name_prefix}_ln')(x_in)
        block = BorderTopKMessagePassing(
            d_model=d_stream,
            K=topk_K,
            num_heads=num_heads,
            dropout=attn_dropout,
            zero_init_proj=attn_zero_init,
            straight_through=straight_through,
            name=f'{name_prefix}_topk',
        )
        out, border_logits = block(h, return_border=True)
        if border_aux_output:
            aux = Activation('linear',
                             name=f'border_logits_{block_idx}')(border_logits)
            outputs.append(aux)
        return keras.layers.Add(name=f'{name_prefix}_add')([x_in, out])

    positions_before_lstm_i = set()
    if attn_position == 'before_each':
        positions_before_lstm_i.update(range(numb_lstm))
    elif attn_position == 'before_first':
        positions_before_lstm_i.add(0)
    elif attn_position == 'before_last':
        positions_before_lstm_i.add(numb_lstm - 1)
    elif attn_position == 'between':
        if numb_lstm >= 2:
            positions_before_lstm_i.add(1)

    block_idx = 0
    for i in range(numb_lstm):
        if i in positions_before_lstm_i:
            block_idx += 1
            x = _insert_topk_block(x, f'topk_blk{block_idx}', block_idx)
        x_next = Bidirectional(LSTM(units, return_sequences=True),
                               name=f'biLSTM_{i+1}')(x)
        if dropout_rate:
            x_next = Dropout(dropout_rate, name=f'dropout_{i+1}')(x_next)
            x = LayerNormalization(
                name=f'layer_normalization_lstm{i+1}')(x_next + x)
        else:
            x = x_next

    # Output head (identical to lstm_model's residual_conv=True branch).
    x = Dense(pool_size * 30, activation='relu', name='dense')(x)
    x = Reshape((-1, 30), name='Reshape2')(x)
    x = keras.layers.Concatenate(axis=-1)([x, cnn_out])
    x = Dense(output_size, name='out_dense')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def lstm_residual_stream_model(units=372, d_model=None, filter_size=128,
                                kernel_size=9, numb_conv=3,
                                numb_lstm=2, dropout_rate=0.0,
                                pool_size=9, output_size=15,
                                multi_loss=False, clamsa=False,
                                softmasking=True, lru_layer=False,
                                zero_init_residual=True):
    """
    Residual-stream variant of `lstm_model`.

    A single tensor of fixed width `d_model` flows from the input embedding
    to the output head. Every block (conv, LSTM/LRU) is pre-norm and adds
    its update back to the stream:

        x = x + Block(LayerNorm(x))

    The stream is pooled once (by `pool_size`) between the conv stack and
    the recurrent stack: conv blocks operate at nucleotide resolution,
    recurrent blocks at pooled resolution.

    With `zero_init_residual=True`, the last projection of each block is
    zero-initialized, so at step 0 every block outputs 0 and the stream
    carries the input embedding unchanged. This makes depth non-destructive
    at init and matches the "identity is the default" property of ResNets,
    Transformers, and modern SSMs.

    Parameters:
        units (int): LSTM units. Bidirectional LSTM output width is 2*units.
        d_model (int|None): Residual stream width. Defaults to 2*units so
            the biLSTM output can flow into the stream with a light 1:1 proj.
        filter_size (int): Hidden width inside each conv block (before the
            projection back to d_model).
        kernel_size (int): Conv kernel size (the first block uses 3).
        numb_conv (int): Number of residual conv blocks.
        numb_lstm (int): Number of residual LSTM/LRU blocks.
        dropout_rate (float): Dropout inside each recurrent block, applied
            to the block output before the residual add.
        pool_size (int): Factor by which the stream is compressed along the
            sequence axis before the recurrent stack. `pool_size=1` disables
            pooling. Final head unpools back to nucleotide resolution.
        output_size (int): Number of output classes.
        multi_loss (bool): If True, attach intermediate softmax heads after
            the conv stack and after every non-final recurrent block.
        clamsa (bool): If True, add the clamsa input track.
        softmasking (bool): If True, input width is 6 instead of 5.
        lru_layer (bool): If True, replace biLSTM blocks with LRU blocks.
        zero_init_residual (bool): If True, zero-init the last projection
            in each residual block (recommended).

    Returns:
        tf.keras.Model: Model with softmax output(s) at nucleotide resolution.
    """
    if lru_layer:
        import LRU_tf as lru

    if d_model is None:
        d_model = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5

    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        x_in = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
    else:
        inp = main_input
        x_in = main_input

    zero_init = 'zeros' if zero_init_residual else 'glorot_uniform'

    # Lift the input to the residual stream width.
    x = Dense(d_model, name='input_embed')(x_in)

    # Residual conv blocks at nucleotide resolution.
    # Each block: x = x + Conv1x1(Conv_k(LN(x))).
    for i in range(numb_conv):
        k = 3 if i == 0 else kernel_size
        h = LayerNormalization(name=f'ln_conv_{i+1}')(x)
        h = Conv1D(filter_size, k, padding='same',
                   activation='relu', name=f'conv_{i+1}')(h)
        h = Conv1D(d_model, 1, padding='same',
                   kernel_initializer=zero_init,
                   name=f'conv_proj_{i+1}')(h)
        x = keras.layers.Add(name=f'add_conv_{i+1}')([x, h])

    # Intermediate head at nucleotide resolution.
    if multi_loss:
        h_cnn = LayerNormalization(name='ln_cnn_head')(x)
        y_cnn = Dense(output_size, name='cnn_head')(h_cnn)
        outputs.append(Activation('softmax', name='out_cnn')(y_cnn))

    # Pool: [B, L, d] -> [B, L/pool, pool*d] -> [B, L/pool, d].
    if pool_size > 1:
        x = Reshape((-1, pool_size * d_model), name='pool_reshape')(x)
        x = Dense(d_model, name='post_pool_proj')(x)

    # Residual recurrent blocks at pooled resolution.
    # Each block: x = x + Proj(Dropout(Rec(LN(x)))).
    pi = 3.141
    for i in range(numb_lstm):
        h = LayerNormalization(name=f'ln_rec_{i+1}')(x)
        if lru_layer:
            mp = 2*pi/2 if i < 1 else 2*pi/50
            rmin = 0 if i < 1 else 0.9
            h = lru.LRU_Block(N=d_model, H=d_model,
                              bidirectional=True,
                              max_tree_depth=17,
                              r_min=rmin,
                              max_phase=mp)(h)
        else:
            h = Bidirectional(LSTM(units, return_sequences=True),
                              name=f'biLSTM_{i+1}')(h)
        h = Dense(d_model, kernel_initializer=zero_init,
                  name=f'rec_proj_{i+1}')(h)
        if dropout_rate:
            h = Dropout(dropout_rate, name=f'dropout_{i+1}')(h)
        x = keras.layers.Add(name=f'add_rec_{i+1}')([x, h])

        # Intermediate head at nucleotide resolution (unpool).
        if multi_loss and i < numb_lstm - 1:
            h_head = LayerNormalization(name=f'ln_rec_head_{i+1}')(x)
            y_head = Dense(pool_size * output_size,
                           name=f'rec_head_{i+1}')(h_head)
            if pool_size > 1:
                y_head = Reshape((-1, output_size),
                                 name=f'rec_head_reshape_{i+1}')(y_head)
            outputs.append(Activation('softmax',
                                       name=f'out_lstm_{i+1}')(y_head))

    # Final head: pre-norm on the stream, linear, unpool, softmax.
    x = LayerNormalization(name='ln_final')(x)
    x = Dense(pool_size * output_size, name='out_dense')(x)
    if pool_size > 1:
        x = Reshape((-1, output_size), name='out_reshape')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def hmm_middle_residual_stream_model(units=160, d_model=None,
                                      pool_size=9,
                                      numb_pre_hmm_lstm=2,
                                      numb_post_hmm_lstm=2,
                                      numb_conv=3, filter_size=128,
                                      kernel_size=9, dropout_rate=0.0,
                                      output_size=15,
                                      hmm_parallel=9,
                                      hmm_config=None,
                                      hmm_embed=160,
                                      hmm_embed_norm='layer',
                                      hmm_embed_activation='softmax',
                                      hmm_readout_type='conv',
                                      hmm_readout_conv_kernel=9,
                                      aux_hmm_loss=False,
                                      clamsa=False, softmasking=True,
                                      zero_init_residual=True):
    """
    Residual-stream backbone with the new trainable HMM head
    (`TrainableHMMHead`) inserted between two biLSTM stacks. Shape:

        Input
          -> conv stem (nt, d_model)
          -> pool -> [numb_pre_hmm_lstm biLSTM residual blocks at pooled res]
          -> unpool -> LN -> Dense(output_size) -> softmax  (HMM input stream)
          -> TrainableHMMHead(x, nuc) at nt
             (LayerNorm + Dense(embed, softmax) -> AnnotationHMM
              -> reshape posterior -> Conv1D readout(-> output_size))
          -> Dense(d_model, zero-init) -> additive residual with pre-HMM stream
          -> pool -> [numb_post_hmm_lstm biLSTM residual blocks]
          -> unpool -> final softmax at nt

    Uses `TrainableHMMHead` (bricks2marble AnnotationHMM with a chained
    intron topology, default `intron_state_chain=2` -> 18 states,
    trainable emitter/transitions/start distribution). The head's readout
    projects the 18-state posterior back to `output_size`, so
    `output_size` no longer needs to match the HMM's state count.

    The HMM operates at nucleotide resolution on the pre-HMM class
    probabilities; its readout output is projected to `d_model` and
    additively fused with the pre-HMM residual stream. With
    `zero_init_residual=True`, the fused HMM contribution is 0 at step 0
    and grows as the projection is learned.

    Do NOT wrap this model with `add_hmm_layer(..., head='hmm')` or
    `--hmm_new` -- the HMM is baked into the backbone, so use `head='none'`
    (or `--` no head flag) when training.

    Notes:
      - Requires `L % pool_size == 0` at both training and inference (same
        rule as the other residual-stream variants).
      - `hmm_parallel` controls the AnnotationHMM parallel_factor. Users
        running long inference sequences may need to override this to
        stay within GPU memory (typically ~sqrt(L) is a good default).
      - The full trainable model (backbone + inline HMM) is saved and
        reloaded together because the HMM parameters are trainable.

    Args:
        units, d_model:               biLSTM hidden and stream width.
        pool_size:                    pool factor used before and after HMM.
        numb_pre_hmm_lstm:            biLSTM residual blocks before the HMM.
        numb_post_hmm_lstm:           biLSTM residual blocks after the HMM.
        numb_conv, filter_size,
        kernel_size:                  conv stem.
        dropout_rate:                 dropout inside each biLSTM block.
        output_size:                  number of output classes.
        hmm_parallel:                 HMM parallel_factor.
        hmm_config:                   dict merged into
                                      `default_trainable_hmm_config()` to
                                      override AnnotationHMM options
                                      (e.g. `intron_state_chain`,
                                      start/stop codon distributions,
                                      train_emitter, etc.).
        hmm_embed, hmm_embed_norm,
        hmm_embed_activation:         embedding config for the HMM head.
        hmm_readout_type,
        hmm_readout_conv_kernel:      readout config for the HMM head.
        aux_hmm_loss:                 if True, expose the HMM's
                                      safe-clipped-log posteriors as an
                                      additional output for a mid-model
                                      loss.
        clamsa, softmasking,
        zero_init_residual:           see other builders.
    """
    if d_model is None:
        d_model = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        x_in = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
    else:
        inp = main_input
        x_in = main_input

    zero_init = 'zeros' if zero_init_residual else 'glorot_uniform'
    x = Dense(d_model, name='input_embed')(x_in)

    # Residual conv stem at nucleotide resolution.
    for i in range(numb_conv):
        k = 3 if i == 0 else kernel_size
        h = LayerNormalization(name=f'ln_conv_{i+1}')(x)
        h = Conv1D(filter_size, k, padding='same', activation='relu',
                   name=f'conv_{i+1}')(h)
        h = Conv1D(d_model, 1, padding='same', kernel_initializer=zero_init,
                   name=f'conv_proj_{i+1}')(h)
        x = keras.layers.Add(name=f'add_conv_{i+1}')([x, h])

    def _bilstm_residual_block(x_in, name_prefix):
        h = LayerNormalization(name=f'{name_prefix}_ln')(x_in)
        h = Bidirectional(LSTM(units, return_sequences=True),
                          name=f'{name_prefix}_biLSTM')(h)
        h = Dense(d_model, kernel_initializer=zero_init,
                  name=f'{name_prefix}_proj')(h)
        if dropout_rate:
            h = Dropout(dropout_rate, name=f'{name_prefix}_drop')(h)
        return keras.layers.Add(name=f'{name_prefix}_add')([x_in, h])

    # Pre-HMM: pool -> biLSTM stack.
    x_pooled = Reshape((-1, pool_size * d_model), name='pre_hmm_pool')(x)
    x_pooled = Dense(d_model, name='pre_hmm_pool_proj')(x_pooled)
    for i in range(numb_pre_hmm_lstm):
        x_pooled = _bilstm_residual_block(x_pooled, f'pre_hmm_blk{i+1}')

    # Unpool back to nucleotide resolution for the HMM.
    x_nt = Dense(pool_size * d_model, name='pre_hmm_unpool_proj')(x_pooled)
    x_nt = Reshape((-1, d_model), name='pre_hmm_unpool')(x_nt)

    # Softmax head that produces class probabilities as the HMM input stream.
    h_hmm_in = LayerNormalization(name='ln_hmm_in')(x_nt)
    class_probs = Dense(output_size, name='hmm_in_dense')(h_hmm_in)
    class_probs = Activation('softmax', name='hmm_in_softmax')(class_probs)

    # Trainable HMM head at nucleotide resolution.
    nuc = Cast()(inp)
    hmm_layer = TrainableHMMHead(
        hmm_config=hmm_config,
        embed=hmm_embed,
        embed_norm=hmm_embed_norm,
        embed_activation=hmm_embed_activation,
        readout_units=output_size,
        readout_type=hmm_readout_type,
        readout_conv_kernel=hmm_readout_conv_kernel,
        parallel_factor=hmm_parallel,
        name='hmm_middle_head',
    )
    hmm_out = hmm_layer(class_probs, nuc, training=True)
    hmm_out = Reshape((-1, output_size), name='hmm_out_reshape')(hmm_out)

    if aux_hmm_loss:
        # hmm_out carries logit-like values (readout + log(class_probs) residual).
        # Softmax converts to class probabilities for the standard base_loss
        # (CategoricalCrossentropy with from_logits=False). Output is named
        # 'hmm_aux_out' so the inference loader can build the sub-model
        # without touching this tensor.
        y_hmm = Activation('softmax', name='hmm_aux_out')(hmm_out)
        outputs.append(y_hmm)

    # Project HMM output to d_model and additively fuse with the pre-HMM stream.
    hmm_proj = Dense(d_model, kernel_initializer=zero_init,
                     name='hmm_out_proj')(hmm_out)
    x_fused = keras.layers.Add(name='hmm_residual_add')([x_nt, hmm_proj])

    # Post-HMM: pool -> biLSTM stack.
    x_post = Reshape((-1, pool_size * d_model), name='post_hmm_pool')(x_fused)
    x_post = Dense(d_model, name='post_hmm_pool_proj')(x_post)
    for i in range(numb_post_hmm_lstm):
        x_post = _bilstm_residual_block(x_post, f'post_hmm_blk{i+1}')

    # Final head: unpool and softmax at nucleotide resolution.
    x_final = LayerNormalization(name='ln_final')(x_post)
    x_final = Dense(pool_size * output_size, name='out_dense')(x_final)
    x_final = Reshape((-1, output_size), name='out_reshape')(x_final)
    y_end = Activation('softmax', name='out')(x_final)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def multires_lstm_residual_stream_model(units=160, d_model=None,
                                         pool_schedule=None,
                                         numb_blocks_per_level=2,
                                         filter_size=128, kernel_size=9,
                                         numb_conv=3, dropout_rate=0.0,
                                         output_size=15, clamsa=False,
                                         softmasking=True,
                                         zero_init_residual=True):
    """
    Multi-resolution U-Net-style residual-stream backbone with biLSTM mixers.

    The stream is progressively pooled through `pool_schedule` in the encoder;
    at each level, `numb_blocks_per_level` biLSTM residual blocks run and
    their output is saved as a skip. The bottleneck is the deepest level.
    The decoder mirrors the schedule in reverse: unpool -> additive skip ->
    `numb_blocks_per_level` biLSTM blocks per level. The final head unpools
    once more by `pool_schedule[0]` to reach nucleotide resolution, so the
    output length matches the input length just like the other
    residual-stream variants.

    Reduces exactly to `lstm_residual_stream_model` with numb_lstm equal to
    `numb_blocks_per_level` when `pool_schedule=[9]` (single level, no
    skips, no decoder).

    Parameter budget:
      units=160, d_model=320, pool_schedule=[3, 3], numb_blocks_per_level=2
      -> ~6.2M params.  Bumping units to 192 (d_model=384) -> ~8.7M.

      The dominant costs are:
        - biLSTM blocks:    ~(4*(2*d+u+1)*u + d*d) per block, 2*len(schedule) blocks total
        - pool projections: ~pool*d*d per pool op (schedule ops down + schedule[1:] ops up)

      Using `pool_schedule=[3, 3]` instead of `[9]` shrinks each pool proj
      by a factor of 3 (3*d^2 vs 9*d^2), which is the main reason the
      multi-res variant can afford several extra blocks and still stay
      well under the 10M budget.

    Notes:
      - Input length must be divisible by prod(pool_schedule) at training
        AND inference (same divisibility rule the single-level variant
        already enforces via seq_len % 9 == 0).
      - Encoder / decoder are symmetric only in structure; weights are
        independent.
      - biLSTM blocks are only placed at pooled resolutions (level >= 1);
        the nucleotide-level features are shaped by the conv stem alone,
        as in the other residual-stream variants.

    Parameters:
        units:                 biLSTM units. Default 160 for the ~6M budget.
        d_model:               residual stream width. Defaults to 2*units.
        pool_schedule:         list of int pool factors per encoder level.
                               Default [3, 3] (total pool 9, matches the
                               single-level variant's downsampling).
        numb_blocks_per_level: biLSTM residual blocks at each level, both
                               encoder and decoder.
        filter_size, kernel_size, numb_conv:  conv stem, unchanged.
        dropout_rate:          dropout inside each biLSTM block.
        output_size:           number of output classes.
        clamsa:                add clamsa input track.
        softmasking:           input width is 6 instead of 5.
        zero_init_residual:    zero-init the last projection in each block
                               so at step 0 each block is identity.
    """
    if pool_schedule is None:
        pool_schedule = [3, 3]
    if d_model is None:
        d_model = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        x_in = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
    else:
        inp = main_input
        x_in = main_input

    zero_init = 'zeros' if zero_init_residual else 'glorot_uniform'
    x = Dense(d_model, name='input_embed')(x_in)

    # Residual conv stem at nucleotide resolution.
    for i in range(numb_conv):
        k = 3 if i == 0 else kernel_size
        h = LayerNormalization(name=f'ln_conv_{i+1}')(x)
        h = Conv1D(filter_size, k, padding='same', activation='relu',
                   name=f'conv_{i+1}')(h)
        h = Conv1D(d_model, 1, padding='same', kernel_initializer=zero_init,
                   name=f'conv_proj_{i+1}')(h)
        x = keras.layers.Add(name=f'add_conv_{i+1}')([x, h])

    def _bilstm_residual_block(x_in, name_prefix):
        h = LayerNormalization(name=f'{name_prefix}_ln')(x_in)
        h = Bidirectional(LSTM(units, return_sequences=True),
                          name=f'{name_prefix}_biLSTM')(h)
        h = Dense(d_model, kernel_initializer=zero_init,
                  name=f'{name_prefix}_proj')(h)
        if dropout_rate:
            h = Dropout(dropout_rate, name=f'{name_prefix}_drop')(h)
        return keras.layers.Add(name=f'{name_prefix}_add')([x_in, h])

    # Encoder: pool, then blocks, then save skip (except at the bottleneck).
    skips = []
    for lvl, pool in enumerate(pool_schedule, start=1):
        x = Reshape((-1, pool * d_model), name=f'enc_pool_{lvl}')(x)
        x = Dense(d_model, name=f'enc_pool_proj_{lvl}')(x)
        for i in range(numb_blocks_per_level):
            x = _bilstm_residual_block(x, f'enc_lvl{lvl}_blk{i+1}')
        if lvl < len(pool_schedule):
            skips.append(x)

    # Decoder: unpool, add skip, then blocks. Iterate schedule in reverse,
    # skipping the last (bottleneck) pool. The decoder ends at level 1.
    for lvl_from_bottom, pool in enumerate(reversed(pool_schedule[1:]),
                                            start=1):
        lvl_to = len(pool_schedule) - lvl_from_bottom
        # Unpool: (B, L, d) -> (B, L*pool, d) via channel expansion + reshape.
        x = Dense(pool * d_model, name=f'dec_unpool_proj_{lvl_to}')(x)
        x = Reshape((-1, d_model), name=f'dec_unpool_{lvl_to}')(x)
        skip = skips[lvl_to - 1]
        x = keras.layers.Add(name=f'dec_skip_add_{lvl_to}')([x, skip])
        for i in range(numb_blocks_per_level):
            x = _bilstm_residual_block(x, f'dec_lvl{lvl_to}_blk{i+1}')

    # Final head: unpool once more by pool_schedule[0] to reach nucleotide.
    x = LayerNormalization(name='ln_final')(x)
    x = Dense(pool_schedule[0] * output_size, name='out_dense')(x)
    x = Reshape((-1, output_size), name='out_reshape')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def swa_residual_stream_model(units=372, d_model=None,
                               num_heads=8, block_size=64, halo_blocks=1,
                               ffn_mult=4, use_alibi=True,
                               filter_size=128, kernel_size=9, numb_conv=3,
                               numb_lstm=2, dropout_rate=0.0,
                               pool_size=9, output_size=15,
                               multi_loss=False, clamsa=False,
                               softmasking=True, zero_init_residual=True):
    """
    Residual-stream backbone using SlidingWindowAttentionBlock as the mixer.

    Identical structure to `lstm_residual_stream_model` -- conv stem, pool,
    stacked mixer, final head -- with the biLSTM/LRU stack replaced by
    stacked bidirectional block-local attention with ALiBi. `numb_lstm` is
    reused as "number of mixer blocks" so switching `arch` between the
    residual-stream variants needs no other config changes.

    See `SlidingWindowAttentionBlock` for mixer-specific parameters.
    """
    if d_model is None:
        d_model = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        x_in = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
    else:
        inp = main_input
        x_in = main_input

    zero_init = 'zeros' if zero_init_residual else 'glorot_uniform'
    x = Dense(d_model, name='input_embed')(x_in)

    # Residual conv stem at nucleotide resolution.
    for i in range(numb_conv):
        k = 3 if i == 0 else kernel_size
        h = LayerNormalization(name=f'ln_conv_{i+1}')(x)
        h = Conv1D(filter_size, k, padding='same', activation='relu',
                   name=f'conv_{i+1}')(h)
        h = Conv1D(d_model, 1, padding='same', kernel_initializer=zero_init,
                   name=f'conv_proj_{i+1}')(h)
        x = keras.layers.Add(name=f'add_conv_{i+1}')([x, h])

    if multi_loss:
        h_cnn = LayerNormalization(name='ln_cnn_head')(x)
        y_cnn = Dense(output_size, name='cnn_head')(h_cnn)
        outputs.append(Activation('softmax', name='out_cnn')(y_cnn))

    if pool_size > 1:
        x = Reshape((-1, pool_size * d_model), name='pool_reshape')(x)
        x = Dense(d_model, name='post_pool_proj')(x)

    # Stacked sliding-window attention blocks.
    for i in range(numb_lstm):
        x = SlidingWindowAttentionBlock(
            d_model=d_model, num_heads=num_heads, block_size=block_size,
            halo_blocks=halo_blocks, ffn_mult=ffn_mult, dropout=dropout_rate,
            use_alibi=use_alibi, zero_init_residual=zero_init_residual,
            name=f'swa_block_{i+1}')(x)

        if multi_loss and i < numb_lstm - 1:
            h_head = LayerNormalization(name=f'ln_head_{i+1}')(x)
            y_head = Dense(pool_size * output_size,
                           name=f'head_{i+1}')(h_head)
            if pool_size > 1:
                y_head = Reshape((-1, output_size),
                                 name=f'head_reshape_{i+1}')(y_head)
            outputs.append(Activation('softmax',
                                       name=f'out_swa_{i+1}')(y_head))

    x = LayerNormalization(name='ln_final')(x)
    x = Dense(pool_size * output_size, name='out_dense')(x)
    if pool_size > 1:
        x = Reshape((-1, output_size), name='out_reshape')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


def ssm_residual_stream_model(units=372, d_model=None,
                               state_size=32, ssm_bidirectional=True,
                               dt_min=0.001, dt_max=0.1, ssm_mode='conv',
                               ffn_mult=4,
                               filter_size=128, kernel_size=9, numb_conv=3,
                               numb_lstm=2, dropout_rate=0.0,
                               pool_size=9, output_size=15,
                               multi_loss=False, clamsa=False,
                               softmasking=True, zero_init_residual=True):
    """
    Residual-stream backbone using DiagonalSSMBlock (S4D-style) as the mixer.

    Set `ssm_mode='recurrent'` for constant-memory inference on very long
    sequences; conv mode's kernel materialization needs O(D * N * L) complex
    and does not fit for full-chromosome L. `numb_lstm` counts SSM blocks.

    See `DiagonalSSMBlock` for mixer-specific parameters.
    """
    if d_model is None:
        d_model = 2 * units

    outputs = []
    inp_size = 6 if softmasking else 5
    main_input = Input(shape=(None, inp_size))
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        x_in = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
    else:
        inp = main_input
        x_in = main_input

    zero_init = 'zeros' if zero_init_residual else 'glorot_uniform'
    x = Dense(d_model, name='input_embed')(x_in)

    for i in range(numb_conv):
        k = 3 if i == 0 else kernel_size
        h = LayerNormalization(name=f'ln_conv_{i+1}')(x)
        h = Conv1D(filter_size, k, padding='same', activation='relu',
                   name=f'conv_{i+1}')(h)
        h = Conv1D(d_model, 1, padding='same', kernel_initializer=zero_init,
                   name=f'conv_proj_{i+1}')(h)
        x = keras.layers.Add(name=f'add_conv_{i+1}')([x, h])

    if multi_loss:
        h_cnn = LayerNormalization(name='ln_cnn_head')(x)
        y_cnn = Dense(output_size, name='cnn_head')(h_cnn)
        outputs.append(Activation('softmax', name='out_cnn')(y_cnn))

    if pool_size > 1:
        x = Reshape((-1, pool_size * d_model), name='pool_reshape')(x)
        x = Dense(d_model, name='post_pool_proj')(x)

    for i in range(numb_lstm):
        x = DiagonalSSMBlock(
            d_model=d_model, state_size=state_size,
            bidirectional=ssm_bidirectional, dt_min=dt_min, dt_max=dt_max,
            mode=ssm_mode, ffn_mult=ffn_mult, dropout=dropout_rate,
            zero_init_residual=zero_init_residual,
            name=f'ssm_block_{i+1}')(x)

        if multi_loss and i < numb_lstm - 1:
            h_head = LayerNormalization(name=f'ln_head_{i+1}')(x)
            y_head = Dense(pool_size * output_size,
                           name=f'head_{i+1}')(h_head)
            if pool_size > 1:
                y_head = Reshape((-1, output_size),
                                 name=f'head_reshape_{i+1}')(y_head)
            outputs.append(Activation('softmax',
                                       name=f'out_ssm_{i+1}')(y_head))

    x = LayerNormalization(name='ln_final')(x)
    x = Dense(pool_size * output_size, name='out_dense')(x)
    if pool_size > 1:
        x = Reshape((-1, output_size), name='out_reshape')(x)
    y_end = Activation('softmax', name='out')(x)
    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)


# ------------------------------------------------------------------
# Backbone registry & factory
# ------------------------------------------------------------------
# Central switch used by train.py and eval_model_class.py so that a new
# architecture only has to be registered here (plus a config key `arch`).
# `keys` lists the config entries that should be forwarded as kwargs to
# the builder; anything else in the config is ignored.
BACKBONE_REGISTRY = {
    "lstm": {
        "builder": lstm_model,
        "keys": [
            "units", "filter_size", "kernel_size",
            "numb_conv", "numb_lstm", "dropout_rate",
            "pool_size", "lstm_mask", "clamsa",
            "output_size", "residual_conv", "softmasking",
            "clamsa_kernel", "lru_layer", "final_activation",
        ],
    },
    "border_gated_attn_lstm": {
        "builder": border_gated_attn_lstm_model,
        "keys": [
            "units", "filter_size", "kernel_size",
            "numb_conv", "numb_lstm", "dropout_rate",
            "pool_size", "output_size",
            "clamsa", "softmasking",
            "num_heads", "attn_dropout", "attn_position",
            "border_aux_output", "attn_zero_init", "attn_chunk_size",
        ],
    },
    "border_topk_lstm": {
        "builder": border_topk_lstm_model,
        "keys": [
            "units", "filter_size", "kernel_size",
            "numb_conv", "numb_lstm", "dropout_rate",
            "pool_size", "output_size",
            "clamsa", "softmasking",
            "num_heads", "attn_dropout", "attn_position",
            "border_aux_output", "attn_zero_init",
            "topk_K", "straight_through",
        ],
    },
    "residual_stream": {
        "builder": lstm_residual_stream_model,
        "keys": [
            "units", "d_model", "filter_size", "kernel_size",
            "numb_conv", "numb_lstm", "dropout_rate",
            "pool_size", "output_size", "multi_loss",
            "clamsa", "softmasking", "lru_layer",
            "zero_init_residual",
        ],
    },
    "swa_residual_stream": {
        "builder": swa_residual_stream_model,
        "keys": [
            "units", "d_model",
            "num_heads", "block_size", "halo_blocks", "ffn_mult",
            "use_alibi",
            "filter_size", "kernel_size", "numb_conv", "numb_lstm",
            "dropout_rate", "pool_size", "output_size", "multi_loss",
            "clamsa", "softmasking", "zero_init_residual",
        ],
    },
    "ssm_residual_stream": {
        "builder": ssm_residual_stream_model,
        "keys": [
            "units", "d_model",
            "state_size", "ssm_bidirectional", "dt_min", "dt_max",
            "ssm_mode", "ffn_mult",
            "filter_size", "kernel_size", "numb_conv", "numb_lstm",
            "dropout_rate", "pool_size", "output_size", "multi_loss",
            "clamsa", "softmasking", "zero_init_residual",
        ],
    },
    "multires_residual_stream": {
        "builder": multires_lstm_residual_stream_model,
        "keys": [
            "units", "d_model",
            "pool_schedule", "numb_blocks_per_level",
            "filter_size", "kernel_size", "numb_conv",
            "dropout_rate", "output_size",
            "clamsa", "softmasking", "zero_init_residual",
        ],
    },
    "hmm_middle_residual_stream": {
        "builder": hmm_middle_residual_stream_model,
        "keys": [
            "units", "d_model", "pool_size",
            "numb_pre_hmm_lstm", "numb_post_hmm_lstm",
            "numb_conv", "filter_size", "kernel_size",
            "dropout_rate", "output_size",
            "hmm_parallel", "hmm_config",
            "hmm_embed", "hmm_embed_norm", "hmm_embed_activation",
            "hmm_readout_type", "hmm_readout_conv_kernel",
            "aux_hmm_loss",
            "clamsa", "softmasking", "zero_init_residual",
        ],
    },
}


def build_backbone_from_config(config, softmasking=None):
    """
    Build a backbone whose architecture is chosen by `config['arch']`.

    Defaults to `'lstm'` when the key is absent, so existing configs
    continue to work unchanged. Pass `softmasking` to override whatever
    the config says (used by inference, which derives softmask from
    `inp_size`).
    """
    arch = config.get("arch", "lstm")
    if arch not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone arch {arch!r}. "
            f"Choose from: {sorted(BACKBONE_REGISTRY)}."
        )
    spec = BACKBONE_REGISTRY[arch]
    kwargs = {k: config[k] for k in spec["keys"] if k in config}
    if softmasking is not None:
        kwargs["softmasking"] = softmasking
    return spec["builder"](**kwargs)


def reduce_lstm_output_7(x, new_size=5):
    """Reduces the output a legacy LSTM that was trained with 7 output classes."""
    assert(x.shape[-1] == 7)
    if new_size == 5:
        x_out = tf.concat([
                x[:,:,0:1],
                tf.reduce_sum(x[:, :, 1:4], axis=-1,
                              keepdims=True, name='reduce_inp_introns'),
                x[:,:,4:]
                ],
                axis=-1)
    elif new_size == 3:
        x_out = tf.concat([x[:,:,0:1],
                tf.reduce_sum(x[:,:,1:4], axis=-1, keepdims=True, name='reduce_output_introns'),
                tf.reduce_sum(x[:,:,4:], axis=-1, keepdims=True, name='reduce_output_exons')],
                axis=-1)
    elif new_size ==2:
        x_out = tf.concat([tf.reduce_sum(x[:,:,:4], axis=-1, keepdims=True, name='reduce_output_non_coding'),
                tf.reduce_sum(x[:,:,4:], axis=-1, keepdims=True, name='reduce_output_coding')],
                axis=-1)
    else:
        raise ValueError("Invalid new_size")
    return x_out

def reduce_lstm_output_5(x, new_size=3):
    """Reduces the output a legacy LSTM that was trained with 5 output classes."""
    assert(x.shape[-1] == 5)
    if new_size == 3:
        x_out = tf.concat([x[:,:,0:2],
                tf.reduce_sum(x[:,:,2:], axis=-1, keepdims=True, name='reduce_output_exons')],
                axis=-1)
    elif new_size ==2:
        x_out = tf.concat([tf.reduce_sum(x[:,:,:2], axis=-1, keepdims=True, name='reduce_output_non_coding'),
                tf.reduce_sum(x[:,:,2:], axis=-1, keepdims=True, name='reduce_output_coding')],
                axis=-1)
    else:
        raise ValueError("Invalid new_size")
    return x_out

def add_hmm_layer(model,
                gene_pred_layer=None,
                output_size=5,
                num_hmm=1,
                hmm_factor=9,
                include_lstm_in_output=False,
                emitter_epsilon: float = 0.0,
                initial_exon_len: int = 200,
                initial_intron_len: int = 4500,
                initial_ir_len: int = 10000,
                old_hmm=True
                ):
    """Add trainable HMM layer to existing hel_model.

    Parameters:
        model (tf.keras.Model): The initial model to which the layers will be added.
        gene_pred_layer (GenePredHMMLayer): The Gene Prediction HMM layer to add to the model.
                                            If None, a new layer will be created with the parameters passed to this method.
        output_size (int): The size of the output layer of the model. Will try to adapt if the model has 7 or 5 outputs but output_size is smaller.
        num_hmm (int): Number of semi-independent HMMs (see GenePredHMMLayer for more details).
        hmm_factor (int): Downsampling factor for sequence length processing in the HMM layer.
        batch_size (int): Batch size for the model's training (used for shaping inputs).
        share_intron_parameters (bool): If True, the HMM layer will share parameters for intron states for the emissions.
        include_lstm_in_output (bool): If True, the LSTM output will be included in the model's output (for multi-loss training).

    Returns:
        tf.keras.Model: The enhanced model with an added Dense layer and a custom Gene Prediction HMM layer.
    """
    inputs = model.input
    x = model.output
    x = x[0] if isinstance(x, list) else x

    if x.shape[-1] > output_size:
        if x.shape[-1] == 7:
            x = reduce_lstm_output_7(x, new_size=output_size)
        elif x.shape[-1] == 5:
            x = reduce_lstm_output_5(x, new_size=output_size)
        else:
            #if this happens, implement more reduction functions
            raise ValueError(f"Invalid combination of loaded output size ({x.shape[-1]}) and requested output size ({output_size}).")
    # x = Identity(name='lstm_out')(x)
    nuc = Cast()(inputs)

    if old_hmm:
        from tiberius.old_hmm.gene_pred_hmm import (GenePredHMMLayer, make_15_class_emission_kernel)
        from learnMSA.msa_hmm.Initializers import ConstantInitializer
        emitter_init = make_15_class_emission_kernel(smoothing=1e-2, num_copies=1,
                                                     noise_strength=0.001)
        emitter_init = np.repeat(emitter_init, num_hmm, axis=0)
        gene_pred_layer = GenePredHMMLayer(
            num_models=num_hmm,
            num_copies=1,
            emitter_init=ConstantInitializer(emitter_init),
            share_intron_parameters=False,
            trainable_emissions=False,
            trainable_transitions=False,
            trainable_starting_distribution=False,
            trainable_nucleotides_at_exons=False,
            parallel_factor=hmm_factor
        )
        y_hmm = gene_pred_layer(x, nuc)
    else:
        gene_pred_layer = HMMBlock(
            parallel=hmm_factor,
            mode=HMMMode.POSTERIOR,
            training=True,
            emitter_epsilon=emitter_epsilon,
            initial_exon_len=initial_exon_len,
            initial_intron_len=initial_intron_len,
            initial_ir_len=initial_ir_len,
        )
        y_hmm = gene_pred_layer(x, nuc, training=True)

    y = Reshape((-1, output_size) if num_hmm == 1 else (-1, num_hmm, output_size),
                name='hmm_out')(y_hmm) #make sure the last dimension is not None

    y = Activation(safe_clipped_log, name='hmm_out_safe_clipped_log')(y)

    model_hmm = Model(
        inputs=inputs,
        outputs=[x, y] if include_lstm_in_output else y,
    )

    return model_hmm


def safe_clipped_log(x):
    from hidten.tf.util import safe_log
    y = safe_log(x)
    y = tf.clip_by_value(y, -30, 0)
    return y


def add_hmm_new_layer(model,
                      output_size: int = 15,
                      hmm_config: dict | None = None,
                      embed: int = 160,
                      embed_norm: str | None = "layer",
                      embed_activation: str | None = "softmax",
                      readout_type: str = "conv",
                      readout_conv_kernel: int = 9,
                      parallel_factor: int = 125,
                      residual_from_input: bool = True,
                      include_lstm_in_output: bool = False,
                      both_strands: bool = False):
    """Attach the trainable HMM head (vipsania-style) to a backbone.

    Unlike `add_hmm_layer`, all HMM parameters here are trainable and
    the state topology uses an intron chain (default `intron_state_chain=2`,
    giving 18 states), which is projected back to `output_size` classes by
    the head's readout so the loss can stay the standard categorical CE
    with `from_logits=True`.

    When both_strands=True the HMM processes both the forward sequence and its
    reverse complement in a single pass (use_reverse_strand=True is set in
    hmm_config automatically).  The readout output size is doubled to
    2 * output_size; the first output_size channels are the forward-strand
    predictions and the last output_size are the reverse-strand predictions
    (in genomic, i.e. forward-coordinate, orientation).
    residual_from_input is disabled in this mode because the backbone output
    dim (output_size) does not match the readout dim (2 * output_size).
    """
    inputs = model.input
    x = model.output
    x = x[0] if isinstance(x, list) else x

    if both_strands:
        hmm_config = dict(hmm_config or {})
        hmm_config["use_reverse_strand"] = True
        readout_units = output_size * 2
        # residual_from_input requires backbone_dim == readout_dim; disable here.
        residual_from_input = False
    else:
        readout_units = output_size

    if x.shape[-1] > output_size:
        if both_strands and x.shape[-1] == readout_units:
            pass  # backbone already outputs 2*output_size; no reduction needed
        elif x.shape[-1] == 7:
            x = reduce_lstm_output_7(x, new_size=output_size)
        elif x.shape[-1] == 5:
            x = reduce_lstm_output_5(x, new_size=output_size)
        else:
            raise ValueError(
                f"Invalid combination of backbone output size ({x.shape[-1]})"
                f" and requested output_size ({output_size})."
            )

    nuc = Cast()(inputs)  # (B, T, 5) A,C,G,T,N -- Cast already slices to [:5]

    head = TrainableHMMHead(
        hmm_config=hmm_config,
        embed=embed,
        embed_norm=embed_norm,
        embed_activation=embed_activation,
        readout_units=readout_units,
        readout_type=readout_type,
        readout_conv_kernel=readout_conv_kernel,
        parallel_factor=parallel_factor,
        residual_from_input=residual_from_input,
        name="trainable_hmm_head",
    )
    y_hmm = head(x, nuc, training=True)

    y = Reshape((-1, readout_units), name="hmm_out")(y_hmm)

    return Model(
        inputs=inputs,
        outputs=[x, y] if include_lstm_in_output else y,
    )
