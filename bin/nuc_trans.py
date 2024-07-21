from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM
import tensorflow as tf
import argparse, sys, os, re, json, types, csv
sys.path.append("/home/gabriell/programs/learnMSA")
import subprocess as sp
import numpy as np
from data_generator import DataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, SimpleRNN, Conv1DTranspose, LSTM, GRU, Dense, Bidirectional, Dropout, Activation, Input, BatchNormalization, LSTM, Reshape, Embedding, Add, LayerNormalization,
                                    AveragePooling1D)

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from models import (weighted_categorical_crossentropy, BatchLearningRateScheduler, add_hmm_only,
                    add_hmm_layer, lm_model_phase, ValidationCallback, BatchSave,EpochSave, lstm_model, add_constant_hmm,
                   transformer_model)

strategy = tf.distribute.MirroredStrategy()

def read_species(file_name):
    species = []
    with open(file_name, 'r') as f_h:
        species = f_h.read().strip().split('\n')
    return [s for s in species if s and s[0] != '#']

def trans_model(output_size=5, filter_size=64, kernel_size=9):    
    input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')
    
    # Import the tokenizer and the TensorFlow version of the model
   
    transformer_model = TFEsmForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    
    transformer_model.trainable=False
    # Compute the embeddings
    outputs = transformer_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    x = outputs['hidden_states'][-1]    
    
    x = Dense(600, activation='relu')(x)    
    x = tf.reshape(x, (-1, 6000, 100))
    for i in range(2):
        x = Conv1D(filter_size, kernel_size, padding='same',
                       activation="relu", name=f'conv_{i+1}')(x)
    x = Dense(output_size, activation='relu')(x)
    y_end = Activation('softmax', name='out')(x)
        
    model = Model(inputs=[input_ids, attention_mask], outputs=y_end[:,6:])

    return model

def trans_lstm_model(output_size=5, units=200, 
                     filter_size=64, kernel_size=9, max_token_len=5502, 
                     pool_size=9, num_conv=3, numb_lstm=2, model_load_lstm=''):    
    main_input = Input(shape=(None, 6), name='main_input')
    input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')
#     trans_out = Input(shape=(None, 2500), name='trans_input')
        
    # run transformer
    transformer_model = TFEsmForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    transformer_model.trainable=False
    
    trans_out = transformer_model(input_ids, 
                                  attention_mask=attention_mask, 
                                  output_hidden_states=True)
    trans_out = trans_out['hidden_states'][-1][:,1:]   
    trans_out = Dense(600, activation='relu', name='dense_transformer')(trans_out)
    trans_out = tf.reshape(trans_out, (-1, tf.shape(trans_out)[1]*6, 100))
    trans_out = tf.reshape(trans_out, (-1, tf.shape(trans_out)[1]*18, 100))    
    cnn_out = main_input

    # CNNs
    for i in range(num_conv):
        cnn_out = Conv1D(filter_size, kernel_size, padding='same',
                       activation="relu", name=f'conv_{i+1}')(cnn_out)        
        cnn_out = LayerNormalization(name=f'layer_normalization{i+1}')(cnn_out)

    # Concat main_inp, trans_out, and CNN_out
    x = tf.concat([main_input, trans_out, cnn_out], axis=-1)

    if pool_size > 1:
        x = Reshape((-1, pool_size * (filter_size+106)), name='R1')(x)

    if model_load_lstm:
        loaded_model = tf.keras.models.load_model(model_load_lstm)
        lstm_model = Model(
                    inputs=loaded_model.get_layer('biLSTM_1').input, 
                    outputs=[loaded_model.get_layer('biLSTM_2').output]
                    )
        x = Dense(1206, activation='relu', name='dense_biLSTM')(x)
        x = lstm_model(x)
    else:
        # Bidirectional LSTM layers
        for i in range(numb_lstm):
            x = Bidirectional(LSTM(units, return_sequences=True), 
                    name=f'biLSTM_{i+1}')(x)

    x = Dense(pool_size * 100, activation='relu', name='dense_out1')(x)  
    x = Reshape((-1, 100), name='Reshape2')(x)
    x = Dense(output_size, activation='relu', name='dense_out2')(x)
    y_end = Activation('softmax', name='out')(x)
    
    model = Model(inputs=[main_input, input_ids, attention_mask], outputs=y_end)
#     model = Model(inputs=[main_input, trans_out], outputs=y_end)
    return model


def trans_lstm_load(model_load_lstm, max_token_len=5502):
    lstm_model = tf.keras.models.load_model(model_load_lstm)
    input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')# run transformer
    transformer_model = TFEsmForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    transformer_model.trainable=False

    trans_out = transformer_model(input_ids,
                                  attention_mask=attention_mask, 
                                  output_hidden_states=True)
    trans_out = trans_out['hidden_states'][-1][:,1:]
    trans_out = Dense(134*6, name='dense_transformer',kernel_initializer='zeros')(trans_out)
    trans_out = tf.reshape(trans_out, (-1, tf.shape(trans_out)[1]*6, 134))
    trans_out = tf.reshape(trans_out, (-1, tf.shape(trans_out)[1]*18, 134))
    
    x = lstm_model.input
    for layer in lstm_model.layers:     
        if layer.name == 'tf.concat':
            x = tf.concat([lstm_model.input, x], axis=-1)
            x = tf.keras.layers.Add()([x, trans_out])
        else:
            x = layer(x)
    x = tf.concat([
            x[:,:,0:1], 
            tf.reduce_sum(x[:,:,1:4], axis=-1, keepdims=True, name='reduce_inp_introns'), 
            x[:,:,4:]
            ], 
            axis=-1, name='concat_outp')
    return Model(inputs=[lstm_model.input, input_ids, attention_mask], outputs=x)
    
def main():
    global file_paths
    home_dir = 'gabriell/'
    if not os.path.exists('/home/gabriell'):        
        home_dir = 'jovyan/brain/'
    w_size = 99999
    if w_size == 99999:
        batch_size = 6*4#32#110
        batch_save_numb = 100000
    elif w_size == 5994:
        batch_size = 45#110
        batch_save_numb = 1000000
    elif w_size == 107892:
        batch_size = 6#110
        batch_save_numb = 1000000
    
    config_dict = {
        "num_epochs": 2000,
#         "loss_weights": [1.0, 3.0, 3.0, 3.0, 30.0, 30.0, 30.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
#         "loss_weights": [1.0, 1.0, 15.0, 15.0, 15.0],#[1., 1., 1., 1., 1.],
        #[1.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0],#[0.33, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0],#
        "stride": 0,
        "units": 64,
        "filter_size": 64,
        "numb_lstm": 1,
        "numb_conv": 2,
        "dropout_rate": 0.0,
        "lstm_mask": False,
        "hmm_dense": 32,
        "pool_size": 9,
        "lr": 1e-5,
        "warmup": 1,
        'max_token_len': 5502,
        "min_lr": 1e-5,
        "batch_size": batch_size,
        "w_size": w_size,
        "model_save_dir": f'/home/{home_dir}/deepl_data//exclude_primates/weights/train2/train81/', 
        "filter": False,
        "trainable_lstm": True,
        'output_size': 5,
        'reduce_intron_label': True,
        'multi_loss': False,
        'constant_hmm': False,
        'num_hmm_layers': 2,
        'l2_lambda': 0.01,
        'temperature': 100.,
        'initial_variance': 0.05,
        'hmm_factor': 1,
        'seq_weights': False,
        # 'model_load': f'/home/{home_dir}/deepl_data//exclude_primates/weights/train2/train79/1/epoch_54',
        'model_load_lstm': f'/home/{home_dir}/deepl_data//exclude_primates/weights/train2/train2/1/1/epoch_00'
    }
    
    with open(f'{config_dict["model_save_dir"]}/config.json', 'w+') as f:
        json.dump(config_dict, f)
    
    data_path = f'/home/{home_dir}/deepl_data/tfrecords/data/{config_dict["w_size"]}_hmm/train/'
        
    for d in [config_dict["model_save_dir"], data_path]:
        if not os.path.exists(d):
            os.mkdir(d)

    num_examples = 0
    class_freq = np.zeros(3, float)
    
    species_file = f'/home/{home_dir}//deepl_data/exclude_primates/train_species_filtered.txt'
    species = read_species(species_file)
#     file_paths = [f'{data_path}/{file}' for file in os.listdir(data_path) if file.endswith('.tfrecords') and 'Mus_musculus' in file]

    file_paths = [f'{data_path}/{s}_{i}.tfrecords' for s in species for i in range(99)]
    
    generator = DataGenerator(file_path=file_paths, 
          batch_size=config_dict['batch_size'], 
          shuffle=True, 
          repeat=True,
          pool_size=config_dict['pool_size'],
          filter=config_dict["filter"],
          reduce_intron_label=config_dict["reduce_intron_label"],
          output_size=config_dict["output_size"],
          hmm_factor=config_dict["hmm_factor"],
#           trans=True, 
          trans_lstm=True,
          seq_weights=config_dict["seq_weights"], 
      )
    
    # Setup the ModelCheckpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=f'{config_dict["model_save_dir"]}' + '/epoch_{epoch:02d}',  # Path where to save the model
        save_weights_only=False,  # Save the full model, not just weights
#         monitor='val_loss',  # Monitor the validation loss to determine the best model
        save_best_only=False,  # Save the model after every epoch
        verbose=1,  # Log a message for each time the model is saved
    )


    adam = Adam(learning_rate=config_dict["lr"])
    epoch_callback = EpochSave(config_dict["model_save_dir"])
    csv_logger = CSVLogger(f'{config_dict["model_save_dir"]}/training.log', 
                append=True, separator=';')
    with strategy.scope():
        if 'model_load' in config_dict:
            model = tf.keras.models.load_model(config_dict['model_load'], 
                          custom_objects={'TFEsmForMaskedLM': TFEsmForMaskedLM})
        else:
#             model = trans_model()
            
            relevant_keys = ['output_size', 'units', 'filter_size', 'kernel_size', 'max_token_len',
                     'pool_size', 'num_conv', 'numb_lstm', 'model_load_lstm']
            relevant_args = {key: config_dict[key] for key in relevant_keys if key in config_dict}
#             model = trans_lstm_model(**relevant_args)
            model = trans_lstm_load(model_load_lstm=config_dict['model_load_lstm'])
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=cce_loss, optimizer=adam, metrics=['accuracy'])
        model.summary()
        model.fit(generator, epochs=500,
                    callbacks=[model_checkpoint_callback, csv_logger],
                    steps_per_epoch=1000
                 )    
    
if __name__ == '__main__':
    main()
