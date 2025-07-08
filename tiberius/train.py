#!/usr/bin/env python3
# ==============================================================
# Authors: Lars Gabriel, Felix Becker
#
# Train variety of LSTM or HMM models for gene prediction 
# using tfrecords.
# 
# Tensorflow 2.10.1
# tensorflow_probability 0.18.0
# ==============================================================

import tiberius.parse_args as parse_args
args = parse_args.parseCmd()
import sys, os, re, json, sys, csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tiberius import DataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger
import tiberius.models as models
from tiberius.models import (weighted_categorical_crossentropy, custom_cce_f1_loss, BatchLearningRateScheduler, 
                    add_hmm_only, add_hmm_layer, ValidationCallback,
                    BatchSave, EpochSave, lstm_model, add_constant_hmm, 
                    make_weighted_cce_loss, Cast)
from tensorflow.keras.callbacks import LearningRateScheduler

gpus = tf.config.list_physical_devices('GPU')

strategy = tf.distribute.MirroredStrategy()

batch_save_numb = 1000


@tf.keras.utils.register_keras_serializable()
class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_epochs, decay_rate, min_lr, steps_per_epoch):
        super().__init__()
        self.peak_lr         = peak_lr
        self.warmup_epochs   = warmup_epochs
        self.decay_rate      = decay_rate
        self.min_lr          = min_lr
        self.steps_per_epoch = tf.constant(steps_per_epoch, dtype=tf.float32)

    def __call__(self, step):
        # step is a scalar int32/64 tensor: number of batches so far
        # convert to epoch index by integer division
        epoch_int = tf.cast(step, dtype=tf.float32) // self.steps_per_epoch
        epoch = tf.cast(epoch_int, tf.float32)
        lr = tf.cond(
            epoch < self.warmup_epochs,
            lambda: self.peak_lr * ((epoch + 1) / tf.cast(self.warmup_epochs, tf.float32)),
            lambda: tf.maximum(
                self.peak_lr * tf.pow(self.decay_rate, epoch - self.warmup_epochs + 1),
                self.min_lr
            )
        )
        return lr

    def get_config(self):
        return {
            "peak_lr":         self.peak_lr,
            "warmup_epochs":   self.warmup_epochs,
            "decay_rate":      self.decay_rate,
            "min_lr":          self.min_lr,
            "steps_per_epoch": int(self.steps_per_epoch.numpy()),
        }

class PrintLr(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs['lr'] will be set by LearningRateScheduler
        lr = logs.get('lr')
        if lr is not None:
            print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}")
        else:
            # fallback if you didn’t use a scheduler callback
            lr_t = self.model.optimizer.learning_rate
            # if it’s a schedule, call it on current iteration
            if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_t = lr_t(self.model.optimizer.iterations)
            print(f"\nEpoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr_t):.6f}")

def train_hmm_model(dataset, model_save_dir, config, val_data=None,
                  model_load=None, model_load_lstm=None, model_load_hmm=None, trainable=True, constant_hmm=False
                 ):  
    """Trains a hybrid HMM-LSTM model with trainings example from a tf.dataset 
    and configuration, model and weights a re saved to model_save_dir.

    Parameters:
        - dataset (tf.data.dataset): A dataset for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
        - model_load_lstm (optional): Path to a pre-trained LSTM model to be loaded, 
                                      it will be trained together with newly initialized HMM.
        - model_load_hmm (optional): Path to a pre-trained HMM model to be loaded,
        - trainable (bool): Flag indicating whether the LSTM model's layers are trainable. 
        - constant_hmm (bool): Flag to add a constant HMM layer to the model. 
    """
    epoch_callback = EpochSave(model_save_dir)

    csv_logger = CSVLogger(f'{model_save_dir}/training.log', 
                append=True, separator=';')


    with strategy.scope():
        if config['use_lr_scheduler']:
            warmup_epochs = config['warmup']
            peak_lr = config['lr']
            min_lr = config['min_lr']
            decay_rate = config['lr_decay_rate']
            schedule = WarmupExponentialDecay(peak_lr=peak_lr,
                                            warmup_epochs=warmup_epochs,
                                            decay_rate=decay_rate,
                                            min_lr=min_lr, 
                                            steps_per_epoch=config["steps_per_epoch"])
            adam = Adam(learning_rate=schedule)
        else:            
            adam = Adam(learning_rate=config['lr'])
        
        if model_load:
            # load the weights onto the raw model instead of using model.load to allow hyperparameter changes
            # i.e. you can change hmm_factor and still use checkpoint saved with a different hmm_factor
            model = keras.models.load_model(model_load, 
                    custom_objects={
                    'custom_cce_f1_loss': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                    'loss_': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                    "Cast": Cast}, 
                    compile=False,
                    )
            print("Loaded model:", model_load)
        else:
            if config['oracle']:
                inputs = tf.keras.layers.Input(shape=(None, 6 if config['softmasking'] else 5), name='main_input')
                oracle_inputs = tf.keras.layers.Input(shape=(None, config['output_size']), name='oracle_input')
                model = tf.keras.Model(inputs=[inputs, oracle_inputs], outputs=oracle_inputs) 
            elif model_load_lstm:
                model = keras.models.load_model(model_load_lstm, 
                        custom_objects={
                        'custom_cce_f1_loss': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                        'loss_': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                        "Cast": Cast}, 
                        compile=False,
                        )
            else:
                relevant_keys = ['units', 'filter_size', 'kernel_size', 
                                'numb_conv', 'numb_lstm', 'dropout_rate', 
                                'pool_size', 'stride', 'lstm_mask', 'co',
                                'output_size', 'residual_conv', 'softmasking',
                                'clamsa_kernel', 'lru_layer', 'clamsa', 'clamsa_kernel']
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                model = lstm_model(**relevant_args)
            for layer in model.layers:
                layer.trainable = trainable
            if constant_hmm:
                model = add_constant_hmm(model,seq_len=config['sample_size'], batch_size=config['batch_size'], output_size=config['output_size'])    
            else: 
                if model_load_hmm:
                    model_hmm = keras.models.load_model(model_load_hmm, 
                                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                                            'loss_': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size'])})
                    gene_pred_layer = model_hmm.layers[-3]
                else:
                    gene_pred_layer = None
                model = add_hmm_layer(model, 
                                        gene_pred_layer,
                                        output_size=config['output_size'], 
                                        num_hmm=config['num_hmm_layers'],
                                        hmm_factor=config['hmm_factor'], 
                                        share_intron_parameters=config['hmm_share_intron_parameters'],
                                        trainable_nucleotides_at_exons=config['hmm_nucleotides_at_exons'],
                                        trainable_emissions=config['hmm_trainable_emissions'],
                                        trainable_transitions=config['hmm_trainable_transitions'],
                                        trainable_starting_distribution=config['hmm_trainable_starting_distribution'],
                                        include_lstm_in_output=config['multi_loss'])
        
        if config["loss_f1_factor"]:
            print("using f1 loss")
            loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        elif config['loss_weights']:
            loss = make_weighted_cce_loss(config['loss_weights'], config['batch_size'])
        else:
            loss = tf.keras.losses.CategoricalCrossentropy()
        if config['multi_loss']: 
            hmm_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"], from_logits=True)
            loss = [loss, hmm_loss] 
            loss_weights = [1, config['hmm_loss_weight_mul']]
        else:
            loss_weights = None
            loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"], from_logits=True)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'], loss_weights=loss_weights)     
        model.summary()
        print_lr_cb = PrintLr()
        callbacks = [epoch_callback, csv_logger, print_lr_cb] \
            if config['use_lr_scheduler'] else [epoch_callback, csv_logger]
        model.save(model_save_dir+"/untrained.keras")
        model.fit(dataset, epochs=config["num_epochs"], validation_data=val_data,
                steps_per_epoch=config["steps_per_epoch"],
                validation_batch_size=config['batch_size'],
                callbacks=callbacks)

def read_species(file_name):
    """Reads a list of species from a given file, filtering out empty lines and comments.

    Parameters:
        - file_name (str): The path to the file containing species names.

        Returns:
        - list of str: A list of species names extracted from the file.
    """
    species = []
    with open(file_name, 'r') as f_h:
        species = f_h.read().strip().split('\n')
    return [s for s in species if s and s[0] != '#']

def train_clamsa(dataset, model_save_dir, config, val_data=None, model_load=None, model_load_lstm=None):
    """Train simple CNN model that uses only CLAMSA as input.

    Parameters:
        - dataset (tf.data.dataset): A dataset for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
        - model_load_lstm (optional): Path to a pre-trained LSTM model to be loaded.
    """
    
    epoch_callback = EpochSave(model_save_dir)

    adam = Adam(learning_rate=config['lr'])
    
    with strategy.scope():
        cce_loss = tf.keras.losses.CategoricalCrossentropy()
        custom_objects = {}
        if config["loss_f1_factor"]:
            cce_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
            custom_objects['custom_cce_f1_loss'] = cce_loss
            custom_objects['loss_'] = cce_loss
        else:
            cce_loss = tf.keras.losses.CategoricalCrossentropy()
        if config["output_size"] == 1:
            cce_loss = tf.keras.losses.BinaryCrossentropy()
        
        csv_logger = CSVLogger(f'{model_save_dir}/training.log', 
                    append=True, separator=';')
        if model_load:
            model = keras.models.load_model(model_load, custom_objects=custom_objects)
        else:
            if config['clamsa_with_lstm']:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                if model_load_lstm:
                    lstm_model = keras.models.load_model(model_load_lstm, custom_objects=custom_objects)
                    model = models.clamsa_lstm_model(lstm_model, **relevant_args)
                else:
                    relevant_keys = ['units', 'filter_size', 'kernel_size', 
                             'numb_conv', 'numb_lstm', 'dropout_rate', 
                             'pool_size', 'stride', 'lstm_mask', 'clamsa',
                             'output_size', 'residual_conv', 'softmasking',
                            'clamsa_kernel']
                    relevant_args = {key: config[key] for key in relevant_keys if key in config}
                    model = models.lstm_model(**relevant_args)
            elif config['use_hmm']:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size",
                                "num_hmm", "hmm_factor", "share_intron_parameters"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                model = models.clamsa_hmm_model(only_hmm_output=not config["multi_loss"], **relevant_args)  
            else:
                relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size"]
                relevant_args = {key: config[key] for key in relevant_keys if key in config}
                model = models.clamsa_only_model( **relevant_args)    

        if config["loss_weights"]:
            model.compile(loss=cce_loss, optimizer=adam, 
                metrics=['accuracy'], #sample_weight_mode='temporal', 
                loss_weights=config["loss_weights"]
                )
        else:
            model.compile(loss=cce_loss, optimizer=adam, 
                metrics=['accuracy'])        
        model.summary()

        model.fit(dataset, epochs=config["num_epochs"], 
                steps_per_epoch=config["steps_per_epoch"],
                callbacks=[epoch_callback, csv_logger])
    
def train_lstm_model(dataset, model_save_dir, config, val_data=None, model_load=None):  
    """Trains the LSTM model using data provided by a tf.dataset, while saving the 
    training checkpoints and logging progress. The model can be trained from scratch or from a 
    pre-loaded state.

    Parameters:
        - dataset (tf.data.dataset): A dataset for training data.
        - model_save_dir (str): Directory path to save model weights and logs.
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
        - model_load (optional): Path to a directory from which 
                                 to load a preexisting model that will be trained
    """

    epoch_callback = EpochSave(model_save_dir)
    csv_logger = CSVLogger(f'{model_save_dir}/training.log', append=True, separator=';')
    
  
    
    with strategy.scope():
        # add learning rate scheduler
        if config['use_lr_scheduler']:
            warmup_epochs = config['warmup']
            peak_lr = config['lr']
            min_lr = config['min_lr']
            decay_rate = config['lr_decay_rate']
            schedule = WarmupExponentialDecay(peak_lr=peak_lr,
                                            warmup_epochs=warmup_epochs,
                                            decay_rate=decay_rate,
                                            min_lr=min_lr, 
                                            steps_per_epoch=config["steps_per_epoch"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
        elif config['sgd']:
            optimizer = SGD(learning_rate=config['lr'])
        else:
            optimizer = Adam(learning_rate=config['lr'])
        custom_objects = {}
        if config["loss_f1_factor"]:
            cce_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
            custom_objects['loss_'] = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
        else:
            cce_loss = tf.keras.losses.CategoricalCrossentropy() 

        if config["output_size"] == 1:
            cce_loss = tf.keras.losses.BinaryCrossentropy()
        
        relevant_keys = ['units', 'filter_size', 'kernel_size', 
                         'numb_conv', 'numb_lstm', 'dropout_rate', 
                         'pool_size', 'stride', 'lstm_mask', 'clamsa',
                         'output_size', 'residual_conv', 'softmasking',
                        'clamsa_kernel', 'lru_layer']

        relevant_args = {key: config[key] for key in relevant_keys if key in config}        
        if model_load:
            model = keras.models.load_model(model_load, 
                    custom_objects={
                    'custom_cce_f1_loss': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                    'loss_': custom_cce_f1_loss(config['loss_f1_factor'], config['batch_size']),
                    "Cast": Cast}, 
                    compile=False,
                    )
            # model.load_weights(model_load + '/variables/variables')
        else:
            model = lstm_model(**relevant_args)
        if config["loss_weights"]:
            model.compile(loss=cce_loss, optimizer=optimizer, 
                metrics=['accuracy'], #sample_weight_mode='temporal', 
                loss_weights=config["loss_weights"]
                )
        else:
            model.compile(loss=cce_loss, optimizer=optimizer, 
                metrics=['accuracy'])        
        model.summary()
        print_lr_cb = PrintLr()
        callbacks = [epoch_callback, csv_logger, print_lr_cb] \
            if config['use_lr_scheduler'] else [epoch_callback, csv_logger]

        model.fit(dataset, epochs=config["num_epochs"], validation_data=val_data,
                steps_per_epoch=config["steps_per_epoch"],
                callbacks=callbacks)


def load_val_data(file, hmm_factor=1, output_size=7, clamsa=False, softmasking=True, oracle=False):
    """
    Loads validation data from a specified file, adjusts the output 
    size based on parameters, and optionally applies HMM factor processing.

    Parameters:
        - file (str): The path to the numpy file containing validation data.
        - hmm_factor (int, optional): The factor to determine the interval 
                        at which to create HMM hints. 
        - output_size (int, optional): The desired number of output classes.

    Returns:
    - A list containing the input features and labels ready for model validation. 
    If hmm_factor is applied, the list will include hints for HMM processing as well.
    """
    data = np.load(file)
    x_val = data["array1"]
    y_val = data["array2"]
    if clamsa:
        clamsa_track = data["array3"]
    data.close()
    if not softmasking:
        x_val = x_val[:,:,:5]
    
    if output_size==5:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 5), np.float32)
        y_new[:,:,0] = y_val[:,:,0]
        y_new[:,:,1] = np.sum(y_val[:,:,1:4], axis=-1)    
        y_new[:,:,2] = np.sum(y_val[:,:,[4, 7, 10, 12]], axis=-1)   
        y_new[:,:,3] = np.sum(y_val[:,:,[5, 8, 13]], axis=-1)
        y_new[:,:,4] = np.sum(y_val[:,:,[6, 9, 11, 14]], axis=-1)
        y_val = y_new
    elif output_size == 7:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 7), np.float32)
        y_new[:,:,:4] = y_val[:,:,:4]
        y_new[:,:,4] = np.sum(y_val[:,:,[4, 7, 10, 12]], axis=-1)   
        y_new[:,:,5] = np.sum(y_val[:,:,[5, 8, 13]], axis=-1)
        y_new[:,:,6] = np.sum(y_val[:,:,[6, 9, 11, 14]], axis=-1)
        y_val = y_new
    elif output_size==3:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 3), np.float32)
        y_new[:,:,0] = y_val[:,:,0]
        y_new[:,:,1] = np.sum(y_val[:,:,1:4], axis=-1)            
        y_new[:,:,2] = np.sum(y_val[:,:,4:], axis=-1) 
        y_val = y_new
    elif output_size==2:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 2), np.float32)
        y_new[:,:,0] = np.sum(y_val[:,:,:4], axis=-1) 
        y_new[:,:,1] = np.sum(y_val[:,:,4:], axis=-1) 
        y_val = y_new
    if hmm_factor:
        step_width = y_val.shape[1] // hmm_factor
        start = y_val[:,::step_width,:] # shape (batch_size, hmm_factor, 5)
        end = y_val[:,step_width-1::step_width,:] # shape (batch_size, hmm_factor, 5)
        hints = np.concatenate([start[:,:,tf.newaxis,:], end[:,:,tf.newaxis,:]],-2)
        return ([np.array(x_val), hints], np.array(y_val))
    if clamsa:
        return [[(x, c) for x,c in zip(x_val, clamsa_track)], y_val]
    return [[x_val, y_val], y_val.astype(np.float32)] if oracle else [x_val, y_val]

file_paths = []

def main():
    global file_paths
    # currently only w_size=9999 is used
    w_size = 9999
    if w_size == 99999:
        batch_size = 96
        batch_save_numb = 100000
    elif w_size == 50004:
        batch_size = 28
        batch_save_numb = 100000
    elif w_size == 9999:
        batch_size = 512
        batch_save_numb = 1000    
    elif w_size == 29997:
        batch_size = 120*4
        batch_save_numb = 1000  
    if args.cfg:
        with open(args.cfg, 'r') as f: 
            config_dict = json.load(f)
    else:
        config_dict = {
            "num_epochs": 2000,
            "steps_per_epoch": 5000,
            "threads": 96,
            'use_hmm': args.hmm,
            "loss_weights": False,
            #[1,1,1e3,1e3,1e3],
            # [ 0.24064536,  1.23309401, 89.06682408, 89.68105166, 89.5963385 ],<- computed from class frequencies in train data
            # "loss_weights": [1.0, 1.0, 100.0, 100.0, 100.0],#[1., 1., 1., 1., 1.],
            # [1.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0],#[0.33, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0],#
            # binary weights: [0.5033910039153116, 74.22447990141231]
            "stride": 0, # if > 0 reduces size of sequence CNN stride
            "units": 372, #192, #512, # output size of LSTMS
            "filter_size": 128, #192,#64, # filter size of CNNs
            "numb_lstm": 2, 
            "numb_conv": 3,
            "dropout_rate": 0.0,
            "lstm_mask": False,
            # pool size is the reduction factor for the sequence before the LSTM,
            # number of adjacent nucleotides that are one position for the LSTM
            "pool_size": 9,
            "lr": 1e-4, # starting lr
            "warmup": 1, # number of trainingssteps to warmup the learning rate
            "min_lr": 1e-6, # minimum learning rate
            "lr_decay_rate": 0.9, # decay rate of learning rate
            "use_lr_scheduler": False, # if True, uses a learning rate scheduler
            "batch_size": batch_size,
            "w_size": w_size, # sequence length
            "filter": False, # if True, filters all training examples out that are IR-only
            "trainable_lstm": True, # if False, LSTM is not trainable -> only HMM is trained
            # output_size determines the shape of all outputs and the labels
            # hmm code will try to adapt if output size of loaded lstm is different to this number
            'output_size': 15, 
            'multi_loss': False, #if both this and use_hmm are True, uses a additional LSTM loss during training
            'hmm_factor': 99, # parallelization factor of HMM, use the factor of w_size that is closest to sqrt(w_size) (271 works well for w_size=99999, 99 for w_size=9999)
            'seq_weights': False, # Adds 3d weights with higher weights around positions of exon borders
            'softmasking': True, # Adds softmasking track to input 
            'residual_conv': True, # Adds result of CNNs to the input to the last dense layer of the LSTM model
            'hmm_loss_weight_mul': 0.1,
            "hmm_dense": 32, # size of embedding for HMM input
            'hmm_share_intron_parameters': False,
            'hmm_nucleotides_at_exons': False,
            'hmm_trainable_transitions': False,
            'hmm_trainable_starting_distribution': False,
            'hmm_trainable_emissions': False, 
            'constant_hmm': False, # maybe not working anymore
            'num_hmm_layers': 1, # numb. of parallel HMMs, currently only 1 is used
            'clamsa': args.clamsa, # adds clamsa track to input
            'clamsa_kernel_size': 7, # kernel size of CNN layer used after clamsa Input
            'clamsa_emb_size': 32, # embedding size used in the clamsa model
            'clamsa_with_lstm': True, # combines LSTM and clamsa model
            'loss_f1_factor': 2.0,
            'sgd': False,
            'oracle': False, # if True, the correct labels will be used as input data. Can be used to debug the HMM.
            "lru_layer": False
        }
        
    config_dict['model_load'] = os.path.abspath(args.load) if args.load else None
    config_dict['model_save_dir'] = os.path.abspath(args.out)
    config_dict['model_load_lstm'] = os.path.abspath(args.load_lstm) if args.load_lstm else None
    config_dict['model_load_hmm'] = os.path.abspath(args.load_hmm) if args.load_hmm else None
    config_dict["mask_tx_list_file"] = os.path.abspath(args.mask_tx_list) if args.mask_tx_list else None
    config_dict["mask_flank"] = args.mask_flank if args.mask_flank else 100 

    mask_tx_list = read_species(config_dict["mask_tx_list_file"]) if config_dict["mask_tx_list_file"] else []

    data_path = args.data
                                                 
    # write config file
    with open(f'{config_dict["model_save_dir"]}/config.json', 'w+') as f:
        json.dump(config_dict, f)
        
    for d in [config_dict["model_save_dir"], data_path]:
        if not os.path.exists(d):
            os.mkdir(d)

    # get paths of tfrecord files
    species_file = f'{args.train_species_file}'
    species = read_species(species_file)
    file_paths = [f'{data_path}/{s}_{i}.tfrecords' for s in species for i in range(100)]

    # init tfrecord generator
    generator = DataGenerator(file_path=file_paths, 
          batch_size=config_dict['batch_size'], 
          shuffle=True,
          repeat=True,
          filter=config_dict["filter"],
          output_size=config_dict["output_size"],
          hmm_factor=0,
          seq_weights=config_dict["seq_weights"], 
          softmasking=config_dict["softmasking"],
          clamsa=False if not "clamsa" in config_dict else config_dict["clamsa"],
          oracle=False if 'oracle' not in config_dict else config_dict['oracle'],
          threads=config_dict["threads"],
          tx_filter=mask_tx_list,
          tx_filter_region=config_dict["mask_flank"]
      )
    
    dataset = generator.get_dataset()

    if args.val_data:
        val_data = load_val_data(args.val_data, 
                    hmm_factor=0, 
                    output_size=config_dict["output_size"],
                    clamsa=config_dict["clamsa"], softmasking=config_dict["softmasking"],
                    oracle=False if 'oracle' not in config_dict else config_dict['oracle']
                                )
    else:
        val_data = None
        
    if args.hmm:
        train_hmm_model(dataset=dataset, val_data=val_data,
            model_save_dir=config_dict["model_save_dir"], config=config_dict,
            model_load_lstm=config_dict["model_load_lstm"],
            model_load_hmm=config_dict["model_load_hmm"],
            model_load=config_dict["model_load"],
            trainable=config_dict["trainable_lstm"], 
                      constant_hmm=config_dict["constant_hmm"]
        )
    elif args.clamsa:
        train_clamsa(dataset=dataset,
                    model_save_dir=config_dict["model_save_dir"], 
                    config=config_dict, 
                    model_load=config_dict["model_load"], 
                    model_load_lstm=config_dict["model_load_lstm"])
    else:
        train_lstm_model(dataset=dataset, val_data=val_data,
            model_save_dir=config_dict["model_save_dir"], config=config_dict,
            model_load=config_dict["model_load"]
        )

if __name__ == '__main__':
    main()
