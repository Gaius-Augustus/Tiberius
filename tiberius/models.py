import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, LSTM,
                                Dense, Bidirectional, Dropout, Activation, Input,
                                Reshape, LayerNormalization)
from tensorflow import keras
from tensorflow.keras import backend as K
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.Training import Identity
from tiberius.hmm import HMMBlock
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
                    include_reading_frame=True, use_cce=True, from_logits=False):
    @tf.function
    def loss_(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if use_cce:
            # Compute the categorical cross-entropy loss
            cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
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

        # Combine CCE loss and F1 score
        combined_loss = cce_loss + f1_factor * (f1_loss + fpr)
        return combined_loss
    return loss_

def lstm_model(units=372, filter_size=128,
              kernel_size=9, numb_conv=3,
               numb_lstm=2, dropout_rate=0.0,
               pool_size=9,
               lstm_mask=False, output_size=15,
               multi_loss=False, residual_conv=True,
               clamsa=False, clamsa_kernel=6, softmasking=True, lru_layer=False
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

    y_end = Activation('softmax', name='out')(x)

    outputs.append(y_end)

    return Model(inputs=inp, outputs=outputs)

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
                  include_lstm_in_output=False):
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
    x = Identity(name='lstm_out')(x)
    nuc = Cast()(inputs)

    gene_pred_layer = HMMBlock(
        parallel=hmm_factor,
        mode=HMMMode.POSTERIOR,
        training=True,
        emitter_epsilon=0.01,
    )
    y_hmm = gene_pred_layer(x, nuc, training=True)

    y = Reshape((-1, output_size) if num_hmm == 1 else (-1, num_hmm, output_size),
                name='hmm_out')(y_hmm) #make sure the last dimension is not None

    model_hmm = Model(
        inputs=inputs,
        outputs=[x, y] if include_lstm_in_output else y,
    )

    return model_hmm
