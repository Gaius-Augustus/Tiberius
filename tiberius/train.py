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
                    make_weighted_cce_loss,load_tiberius_model, WarmupExponentialDecay, 
                    PrintLr)
from tensorflow.keras.callbacks import LearningRateScheduler

gpus = tf.config.list_physical_devices('GPU')

strategy = tf.distribute.MirroredStrategy()

batch_save_numb = 1000


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


def train_model(dataset, model, config, val_data):
    """Trains a LSTM model using data provided by a tf.dataset, while saving the 
    training checkpoints and logging progress. The model can be trained from scratch or from a 
    pre-loaded state.

    Parameters:
        - dataset (tf.data.dataset): A dataset for training data.
        - model (tf.model): Tiberius model to be trained (with or without HMM)
        - config (dict): Configuration dictionary specifying model 
                         parameters and training settings.
        - val_data (optional): Validation data to evaluate the model. Default is None.
    """
    epoch_callback = EpochSave(config["model_save_dir"], config)
    csv_logger = CSVLogger(f'{config["model_save_dir"]}/training.log', append=True, separator=';')
    
    # add learning rate scheduler
    if config.get('use_lr_scheduler', False):
        schedule = WarmupExponentialDecay(peak_lr=config.get("lr", 1e-4),
                            warmup_epochs=config.get("warmup", 0),
                            decay_rate=config.get("lr_decay_rate", 0.9),
                            min_lr=config.get("min_lr", 1e-6),
                            steps_per_epoch=config.get("steps_per_epoch", 5000)
                            )
        optimizer_lr = schedule
    else:
        optimizer_lr = config.get("lr", 1e-4)
    
    if config.get("sgd", False):
        optimizer = SGD(learning_rate=optimizer_lr)
    else:
        optimizer = Adam(learning_rate=optimizer_lr)


    if config["loss_f1_factor"]:
        loss_func = custom_cce_f1_loss(config["loss_f1_factor"], 
                batch_size=config["batch_size"], from_logits=config.get("use_hmm", False))
    elif config["output_size"] == 1:
        loss_func = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_func = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(loss=loss_func, optimizer=optimizer, 
            metrics=['accuracy'], 
            loss_weights=config["loss_weights"] if config.get("loss_weights", False) else None,
            )

    print_lr_cb = PrintLr()
    callbacks = [epoch_callback, csv_logger, print_lr_cb] \
        if config.get('use_lr_scheduler', False) else [epoch_callback, csv_logger]

    model.fit(dataset, epochs=config["num_epochs"], validation_data=val_data,
            steps_per_epoch=config.get("steps_per_epoch", 5000),
            callbacks=callbacks)

def get_model(config, model_path=None):
    model = load_tiberius_model(model_path=model_path,
        add_hmm=config.get("use_hmm", False), loss_weight=config.get("loss_f1_factor", 1.0),
        batch_size=config.get("batch_size", 100), summary=True, config=config)

    # add hmm in case an old lstm model is loaded for HMM training
    if not any("gene_pred_hmm_layer" in layer.name for layer in model.layers) and config.get("use_hmm", False):
        relevant_keys = ['output_size', 'num_hmm', 'num_copy', 'hmm_factor', 
                        'share_intron_parameters', 'trainable_nucleotides_at_exons',
                        'trainable_emissions', 'trainable_transitions',
                        'trainable_starting_distribution', 'include_lstm_in_output',
                        'emission_noise_strength']        
        relevant_args = {key: config[key] for key in relevant_keys if key in config}
        model = add_hmm_layer(model,
                        **relevant_args)
    model.summary()
    return model

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

    if args.cfg:
        with open(args.cfg, 'r') as f: 
            config_dict = json.load(f)
    else:
        config_dict = {
            "num_epochs": 2000,
            "steps_per_epoch": 5000,
            "threads": 96,
            'use_hmm': args.hmm,
            "loss_weights": None,
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
    config_dict["mask_tx_list_file"] = os.path.abspath(args.mask_tx_list) if args.mask_tx_list else None
    config_dict["mask_flank"] = args.mask_flank if args.mask_flank else 100 
    config_dict["use_hmm"] = True if args.hmm else False

    mask_tx_list = read_species(config_dict["mask_tx_list_file"]) if \
                config_dict.get("mask_tx_list_file",False) else []

    data_path = args.data
                                                 
    for d in [config_dict["model_save_dir"], data_path]:
        if not os.path.exists(d):
            os.mkdir(d)

    # write config file
    with open(f'{config_dict["model_save_dir"]}/config.json', 'w+') as f:
        json.dump(config_dict, f)

    # get paths of tfrecord files
    species_file = f'{args.train_species_file}'
    species = read_species(species_file)
    file_paths = [f'{data_path}/{s}_{i}.tfrecords' for s in species for i in range(100)]

    # init tfrecord generator
    generator = DataGenerator(file_path=file_paths, 
          batch_size=config_dict.get("batch_size", 100),
          shuffle=True,
          repeat=True,
          filter=config_dict.get("filter", False),
          output_size=config_dict.get("output_size", 15),
          hmm_factor=0,
          seq_weights=config_dict.get("seq_weights", False),
          softmasking=config_dict.get("softmasking", True),
          clamsa=False if not "clamsa" in config_dict else config_dict["clamsa"],
          oracle=False if 'oracle' not in config_dict else config_dict['oracle'],
          threads=config_dict.get("threads", 48),
          tx_filter=mask_tx_list,
          tx_filter_region=config_dict.get("mask_flank", 500),
      )
    
    dataset = generator.get_dataset()

    if args.val_data:
        val_data = load_val_data(args.val_data, 
                    hmm_factor=0, 
                    output_size=config_dict.get("output_size", 15),
                    clamsa=config_dict.get("clamsa", False), 
                    softmasking=config_dict.get("softmasking", True),
                    # oracle=False if 'oracle' not in config_dict else config_dict['oracle']
                                )
    else:
        val_data = None

    with strategy.scope():
        model = get_model(config_dict, model_path=args.load if args.load else None)
        train_model(dataset=dataset, model=model, config=config_dict, val_data=val_data)
        
if __name__ == '__main__':
    main()
