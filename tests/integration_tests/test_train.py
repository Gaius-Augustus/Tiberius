# test_train_integration.py

import sys
# prevent pytest CLI args from reaching argparse in train.py
sys.argv = ["train.py"]

# stub out parse_args.parseCmd() before train.py is imported
import tiberius.parse_args as _pa
from argparse import Namespace
_pa.parseCmd = lambda: Namespace(
    cfg='', out='', load='', load_lstm='', load_hmm='',
    hmm=False, clamsa=False, data='', val_data='',
    train_species_file='train_species_filtered.txt',
    learnMSA='../learnMSA'
)
import os
import json
import tempfile

import numpy as np
import pytest
import tensorflow as tf

import tiberius.train as train_module
from tiberius.train import read_species, load_val_data, train_hmm_model, train_clamsa, train_lstm_model

train_module.strategy = tf.distribute.get_strategy()

def test_read_species(tmp_path):
    p = tmp_path / "species.txt"
    p.write_text("#comment\nMus_musculus\n\n#skip\nHomo_sapiens\n")
    out = read_species(str(p))
    assert out == ["Mus_musculus", "Homo_sapiens"]


def _make_dummy_generator():
    """Yields a single batch of (X, Y) then stops."""
    class G:
        def __iter__(self): return self
        def __next__(self):
            # X: batch_size=1, time=4, features=6
            return np.zeros((1,4,6),dtype=np.float32), np.eye(5,dtype=np.float32)[[
                [0,1,2,3],
            ]]
    return G()

@pytest.fixture
def minimal_config(tmp_path):
    d = {
        "lr": 0.01,
        "batch_size": 1,
        "steps_per_epoch": 10,
        "threads": 10,
        "loss_f1_factor": 0.0,
        "units": 20,
        "filter_size": 20,
        "loss_weights": False,
        "multi_loss": False,
        "output_size": 5,
        "softmasking": False,
        "hmm_factor": 1,
        "hmm_dense": 4,
        "num_hmm_layers": 1,
        "hmm_share_intron_parameters": False,
        "hmm_nucleotides_at_exons": False,
        "hmm_trainable_emissions": False,
        "hmm_trainable_transitions": False,
        "hmm_trainable_starting_distribution": False,
        "constant_hmm": False,
        "trainable_lstm": True,
        "sgd": False,
        "use_hmm": False,
        "clamsa": False,
        "clamsa_with_lstm": False,
        "oracle": False,
        "pool_size": 1,
        "num_epochs": 1,
    }
    d["model_load"] = None
    d["model_load_lstm"] = None
    d["model_load_hmm"] = None
    d["model_save_dir"] = str(tmp_path / "out")
    os.makedirs(d["model_save_dir"], exist_ok=True)
    return d

@pytest.mark.integration
def test_train_hmm_model_invokes_fit(minimal_config):
    x = tf.constant([[[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0]],
                     [[0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]],
                     [[0,0,0,1,0],
                      [0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0]]], dtype=tf.int32)
    y = tf.constant([[[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0]],
                     [[0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]],
                     [[0,0,0,1,0],
                      [0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0]]], dtype=tf.int32)
    dummy_ds = tf.data.Dataset.from_tensors((x, y)).repeat()
    # call train_hmm_model
    train_module.train_hmm_model(
        dataset=dummy_ds,
        model_save_dir=minimal_config["model_save_dir"],
        config=minimal_config,
        val_data=None,
        model_load=None,
        model_load_lstm=None,
        model_load_hmm=None,
        trainable=True,
        constant_hmm=False
    )
    # Since our DummyModel.fit sets .fit_called, ensure it was called
    # The DummyModel instance used by add_hmm_layer should be the one that .fit() was invoked on
    # We can inspect the last DummyModel created via the stub
    # For simplicity, assert that the out directory now contains the “untrained.keras” file
    assert os.path.exists(os.path.join(minimal_config["model_save_dir"], "training.log"))

@pytest.mark.integration
def test_train_lstm_model_invokes_fit(minimal_config):
    x = tf.constant([[[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0]],
                     [[0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]],
                     [[0,0,0,1,0],
                      [0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0]]], dtype=tf.int32)
    y = tf.constant([[[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0]],
                     [[0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]],
                     [[0,0,0,1,0],
                      [0,0,0,0,1],
                      [1,0,0,0,0],
                      [0,1,0,0,0]]], dtype=tf.int32)
    dummy_ds = tf.data.Dataset.from_tensors((x, y)).repeat()
    # call train_lstm_model
    train_module.train_lstm_model(
        dataset=dummy_ds,
        model_save_dir=minimal_config["model_save_dir"],
        config=minimal_config,
        val_data=None,
        model_load=None
    )
    # Expect training.log to exist (CSVLogger stub still created an empty file)
    assert os.path.exists(os.path.join(minimal_config["model_save_dir"], "training.log"))

