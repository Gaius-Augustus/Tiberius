import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, LSTM, 
                                Dense, Concatenate, Bidirectional, Dropout, Activation, Input, 
                                BatchNormalization, Reshape, Embedding, Add, LayerNormalization)
import sys 
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from gene_pred_hmm import (GenePredHMMLayer, make_5_class_emission_kernel, 
                            make_15_class_emission_kernel, ReduceOutputSize)
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.Training import Identity
from tensorflow.keras.callbacks import Callback

class Cast(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x[0][..., :5] if isinstance(x, list) else x[..., :5], tf.float32)
    
class EpochSave(tf.keras.callbacks.Callback):
    def __init__(self, model_save_dir):
        super(EpochSave, self).__init__()
        self.model_save_dir = model_save_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"{self.model_save_dir}/epoch_{epoch:02d}", save_traces=False)

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

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.batch_number == 0:   
            self.prev_batch_numb += batch           
            self.model.save(self.save_path.format(self.prev_batch_numb), save_traces=False)


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

       
def lstm_model(units=200, filter_size=64, 
              kernel_size=9, numb_conv=2, 
               numb_lstm=3, dropout_rate=0.0, 
               pool_size=10, stride=0, 
               lstm_mask=False, output_size=7,
               multi_loss=False, residual_conv=False,
               clamsa=False, clamsa_kernel=6, softmasking=True, lru_layer=False,
               lru_hidden_state_dim=200, lru_max_tree_depth=17, lru_init_bounds='',
               lru_scan_use_tf_while_loop=False, lru_scan_base_case_n=None,
               use_optimized_scan=False
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
        stride (int): The stride length for convolutional operations. Applies striding if > 1.
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
    main_input = Input(shape=input_shape, name='main_input')
    if clamsa:
        inp_clamsa = Input(shape=(None, 4), name='clamsa_input')
        inp = [main_input, inp_clamsa]
        
        # main_input = tf.concat([main_input, inp_clamsa], axis=-1)
        main_input = keras.layers.Concatenate(axis=-1)([main_input, inp_clamsa])
        inp_size+=4
    else:
        inp = main_input
    #inp_embedding = Dense(filter_size, activation="relu", name="inp_embed")(main_input)

    if stride > 1:
        x = Conv1D(filter_size, kernel_size, strides=stride, padding='valid',
            activation="relu", name='initial_conv')(main_input) 
        inp_embedding = x
    else:
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
    if not lru_init_bounds:
        lru_init_bounds = [[[0.,1.,0.,2*pi,1.]] for _ in range(numb_lstm)]
    print("LRU init bounds: ", lru_init_bounds)
    for i in range(numb_lstm):
        if lru_layer:
            lru_block = lru.LRU_Block(
                  H=2*units,
                  N=lru_hidden_state_dim, 
                  bidirectional=True, 
                  max_tree_depth=lru_max_tree_depth,   #TODO: set tree depth extra???
                  return_sequences=True,
                  use_skip_connection=True, 
                  residual=True,
                  use_nonlin=True, 
                  dropout=dropout_rate, #CHANGED: set dropout rate 
                  use_batch_norm=True, 
                  init_bounds=lru_init_bounds[i],
                  scan_use_tf_while_loop=lru_scan_use_tf_while_loop, 
                  scan_base_case_n=lru_scan_base_case_n,
                  use_optimized_scan=use_optimized_scan)
            # lru_block.build(input_shape=x.shape)
            x_next = lru_block(x)
            x = x_next   # no additional dropout for lru layer
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
        if stride > 1:
            x = Conv1DTranspose(output_size, kernel_size, strides=stride, padding='valid',
                activation="relu", name='transpose_conv')(x) 
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


def add_hmm_layer(model, gene_pred_layer=None, dense_size=128, pool_size=9, 
                  output_size=5, num_hmm=1, num_copy=1, l2_lambda=0.01, 
                  hmm_factor=9, batch_size=32, seq_len=99999,
                 emit_embeddings = False, share_intron_parameters=True, 
                  trainable_nucleotides_at_exons = True, trainable_emissions = True,
                  trainable_transitions = True, trainable_starting_distribution=True,
                  use_border_hints=False, temperature=100., initial_variance=0.05,
                  include_lstm_in_output=False,
                  neutral_hmm=False, emission_noise_strength=0.001):
    """Add trainable HMM layer to existing hel_model.

    Parameters:
        model (tf.keras.Model): The initial model to which the layers will be added.
        gene_pred_layer (GenePredHMMLayer): The Gene Prediction HMM layer to add to the model. 
                                            If None, a new layer will be created.
        dense_size (int): Number of neurons in the newly added Dense layer.
        pool_size (int): Downsampling factor applied before the HMM layer, affecting the Dense layer's output.
        output_size (int): The size of the output layer of the model. Will try to adapt if the model has 7 or 5 outputs but output_size is smaller.
        num_hmm (int): Number of semi-independent HMMs (see GenePredHMMLayer for more details).
        num_copy (int): The number of gene model copies in an HMM that share the IR state.
        l2_lambda (float): L2 regularization lambda for the variance parameters within the HMM layer.
        hmm_factor (int): Downsampling factor for sequence length processing in the HMM layer.
        batch_size (int): Batch size for the model's training (used for shaping inputs).
        seq_len (int): Maximum sequence length the model can handle.
        emit_embeddings (bool): If True, the HMM layer will additionally emit embeddings (experimental).
        share_intron_parameters (bool): If True, the HMM layer will share parameters for intron states for the emissions.
        trainable_nucleotides_at_exons (bool): If True, the HMM layer will train nucleotide distributions at exon states.
        use_border_hints (bool): If True, the HMM layer will use border hints as input. Set to false to use proper parallel HMM.
        temperature (float): Temperature parameter for the softmax calculation in the HMM layer.
        initial_variance (float): Initial variance for the Gaussian distributions used in the HMM layer.
        include_lstm_in_output (bool): If True, the LSTM output will be included in the model's output (for multi-loss training).
        emission_noise_strength: Strength of the noise added to the emissions of the HMM layer.

    Returns:
        tf.keras.Model: The enhanced model with an added Dense layer and a custom Gene Prediction HMM layer.
    """
    inputs = model.input
    if use_border_hints:
        input_hints = Input(shape=(hmm_factor,2,5), name='border_hints')
    
    if emit_embeddings:
        # create embeddings for hmm layer of size dense_size
        emb = model.get_layer('out_dense').input
        emb = Dense(4*dense_size, name='dense_hmm_1', activation="relu")(emb)
        emb = Dense(dense_size, name='dense_hmm')(emb)
        emb = LayerNormalization(name=f'layer_normalization_hmm')(emb)
    else:
        emb = None
    
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
    
    if output_size == 5:
        emitter_init = make_5_class_emission_kernel(smoothing=1e-6, introns_shared=share_intron_parameters, 
                                                    num_copies=num_copy, noise_strength=emission_noise_strength)
    elif output_size == 15:
        assert not share_intron_parameters, "Can not share intron parameters if output size is 15."
        emitter_init = make_15_class_emission_kernel(smoothing=1e-2, num_copies=num_copy, 
                                                     noise_strength=emission_noise_strength)
    #currently, the emissions of the semi-independent HMMs are initialized the same
    emitter_init = np.repeat(emitter_init, num_hmm, axis=0)
    if gene_pred_layer is None:
        gene_pred_layer = GenePredHMMLayer(
            num_models=num_hmm,
            num_copies=num_copy,
            initial_exon_len=200, 
            initial_intron_len=4500,
            initial_ir_len=10000,
            start_codons=[("ATG", 1.)],
            stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
            intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.01)],
            intron_end_pattern=[("AGN", 1.)],
            emitter_init=ConstantInitializer(emitter_init),
            starting_distribution_init="zeros",
            emit_embeddings = emit_embeddings,
            embedding_dim = dense_size if emit_embeddings else None,
            full_covariance=False,
            embedding_kernel_init="random_normal",
            initial_variance=initial_variance,
            temperature=temperature,
            share_intron_parameters=share_intron_parameters,
            trainable_emissions=trainable_emissions,
            trainable_transitions=trainable_transitions,
            trainable_starting_distribution=trainable_starting_distribution,
            trainable_nucleotides_at_exons=trainable_nucleotides_at_exons,
            variance_l2_lambda=l2_lambda,
            parallel_factor=1 if use_border_hints else hmm_factor,
            use_border_hints=use_border_hints
        )

    if use_border_hints:
        input_hints_hmm = tf.reshape(input_hints, (-1, 2,5), name='reshape_border_hints')
        input_hints_hmm = tf.matmul(input_hints_hmm, A, transpose_b=True, name='matmul_hint_labels')
        window_size = seq_len//hmm_factor
        factor_x = tf.reshape(x, (-1, window_size, 5))
        factor_nuc = tf.reshape(nuc, (-1, window_size, 5))
        factor_emb = tf.reshape(emb, (-1, window_size, dense_size))
        y_hmm = gene_pred_layer(factor_x, nucleotides=factor_nuc, embeddings=factor_emb, end_hints=input_hints_hmm)
        y_hmm = tf.reshape(y_hmm, (-1, seq_len, 14*num_copy+1))
    else:
        y_hmm = gene_pred_layer(x, nucleotides=nuc, embeddings=emb)

    if output_size < 15:
        y = ReduceOutputSize(output_size, num_copies=num_copy, name='hmm_out')(y_hmm)
    else:
        y = Reshape((-1, output_size) if num_hmm == 1 else (-1, num_hmm, output_size), 
                    name='hmm_out')(y_hmm) #make sure the last dimension is not None
        
    model_hmm = Model(inputs=[inputs, input_hints] if use_border_hints else inputs, 
                    outputs=[x, y] if include_lstm_in_output else y)
    
    return model_hmm

def add_constant_hmm(model, seq_len=9999, batch_size=450, output_size=3):
    """Extends a given model with a Hidden Markov Model (HMM) layer that has constant emission probabilities.
    The HMM layer is configured with fixed emission probabilities for three classes. The seven output labels of 
    the LSTM are reduced to the three emission labels

    Parameters:
    - model (tf.keras.Model): The original model to which the HMM layer will be added.
    - seq_len (int): The sequence length that the HMM layer will be able to process. Default is 9999.
    - batch_size (int): The batch size for processing sequences through the model. Default is 450.

    Returns:
    - model_hmm (tf.keras.Model): The new model with the added GenePredHMMLayer with constant emissions.
    """
    inputs = model.input
    
    emb = model.layers[-1].output
    emb = tf.concat([
                    emb[:,:,0:1], 
                    tf.reduce_sum(emb[:, :, 1:4], axis=-1, keepdims=True, name='reduce_inp_introns'), 
                    emb[:,:,4:]
                    ], 
                    axis=-1, name='concat_inps')
    
    nuc = tf.cast(inputs[:,:,:5], tf.float32, name='cast_inp')
    
    gene_pred_layer = GenePredHMMLayer(
                        emitter_init=ConstantInitializer(make_5_class_emission_kernel(smoothing=0.01)),
                        initial_exon_len=150, 
                        initial_intron_len=4000,
                        initial_ir_len=100000,
                        emit_embeddings=False,
                        start_codons=[("ATG", 1.)],
                        stop_codons=[("TAG", .34), ("TAA", .33), ("TGA", .33)],
                        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.01)],
                        intron_end_pattern=[("AGN", 1.)],
                        starting_distribution_init="zeros",
                        starting_distribution_trainable=True,
                        simple=False)
    
    gene_pred_layer.build(emb.shape)
    x = gene_pred_layer(emb, nuc)
    y = Activation('softmax', name='out_hmm')(x)
    #y.trainable = False
    if output_size == 3:
        y = tf.concat([y[:,:,0:1], 
                tf.reduce_sum(y[:,:,1:4], axis=-1, keepdims=True, name='reduce_output_introns'), 
                tf.reduce_sum(y[:,:,4:], axis=-1, keepdims=True, name='reduce_output_exons')], 
              axis=-1, name='concat_output')
    elif output_size == 5:
        y = tf.concat([
                y[:,:,0:1], 
                tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True, name='reduce_inp_introns'), 
                tf.reduce_sum(tf.gather(y, [4, 7, 10, 12], axis=-1, name='gather_inp_e0'),
                              axis=-1, keepdims=True, name='reduce_inp_e0'),
                tf.reduce_sum(tf.gather(y, [5, 8, 13], axis=-1, name='gather_inp_e1'),
                              axis=-1, keepdims=True, name='reduce_inp_e1'),
                tf.reduce_sum(tf.gather(y, [6, 9, 11, 14], axis=-1, name='gather_inp_e2'),
                              axis=-1, keepdims=True, name='reduce_inp_e2'),
                ], 
                axis=-1, name='concat_inps')
    
    model_hmm = Model(inputs=inputs, outputs=y)
    return model_hmm

def add_hmm_only(model):
    inputs = model.input
    x = model.layers[-1].output         
    y = GenePredHMMLayer(initial_exon_len=172, 
                        initial_intron_len=4648,
                        initial_ir_len=177657,)(x)
    model_hmm = Model(inputs=inputs, outputs=y)
    
    return model_hmm

def get_positional_encoding(seq_len, d_model):
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_terms = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
    sines = tf.math.sin(positions * div_terms)
    cosines = tf.math.cos(positions * div_terms)
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding


def weighted_categorical_crossentropy(class_weights, overall_weight):
    def loss(y_true, y_pred):        
        # Apply class weights to true labels
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        weighted_true = tf.argmax(tf.multiply(y_true, class_weights),-1)
        
        # Check if the true label has more than one class
        example_weight =  tf.constant([tf.unique(tf.argmax(y, axis=.1))[0].shape[0] \
                                       for y in y_true], dtype=tf.float32)
        example_weight = tf.maximum(tf.ones(example_weight.shape[0], dtype=tf.float32),
            (example_weight-1)*overall_weight)
        
        weighted_true = tf.transpose(tf.multiply(tf.transpose(weighted_true), example_weight))
        
        # Compute crossentropy loss
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Apply weights to the loss
        weighted_loss = tf.multiply(loss, weighted_true)
        
        # Calculate mean loss across all classes
        return tf.reduce_mean(weighted_loss)    
    return loss

#Felix: added to make my code run
#do you have your own weighted loss? lets merge/remove one later
def make_weighted_cce_loss(weights=[1.]*5, batch_size=32):
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, 
                                                        reduction=tf.keras.losses.Reduction.NONE)
    def weighted_cce_loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=y_pred.dtype)
            base_level_weights = tf.linalg.matvec(y_true, weights)
            L = cce_loss(y_true, y_pred)
            W = tf.reduce_sum(base_level_weights, axis=-1, keepdims=True)
            return tf.reduce_sum(L*base_level_weights / (W * batch_size))
    return weighted_cce_loss
