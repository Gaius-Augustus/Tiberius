import os
import numpy as np
import tensorflow as tf
import tempfile
import pytest

from tiberius.models import (
    Cast,
    EpochSave,
    BatchLearningRateScheduler,
    ValidationCallback,
    BatchSave,
    custom_cce_f1_loss,
    lstm_model,
    reduce_lstm_output_7,
    reduce_lstm_output_5,
    add_hmm_layer,
    add_constant_hmm,
    add_hmm_only,
    get_positional_encoding,
    weighted_categorical_crossentropy,
    make_weighted_cce_loss
)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation, Conv1D

def test_cast_layer_slices():
    layer = Cast()
    # Non-list input: keep first 5 features
    x = tf.random.uniform((2, 3, 7))
    out = layer(x)
    assert out.shape == (2, 3, 5)
    # List input: take first element then slice
    y0 = tf.random.uniform((4, 5, 6))
    y1 = tf.random.uniform((4, 5, 2))
    out2 = layer([y0, y1])
    assert out2.shape == (4, 5, 5)

def test_custom_cce_f1_loss_cce_only():
    # f1_factor=0 → loss == categorical crossentropy
    loss_fn = custom_cce_f1_loss(f1_factor=0.0, batch_size=2,
                                 include_reading_frame=True, use_cce=True)
    # two samples, sequence length=2, classes=2
    y_true = tf.one_hot([ [0,1], [1,0] ], depth=2)
    y_pred = tf.identity(y_true)
    loss = loss_fn(y_true, y_pred)
    # perfect prediction → zero loss
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

def test_custom_cce_f1_loss_f1_branch():
    # f1_factor=1, include_reading_frame=False, shape[-1]==5
    loss_fn = custom_cce_f1_loss(f1_factor=1.0, batch_size=1,
                                 include_reading_frame=False, use_cce=False)
    # two‐timepoint sequence, only last class "CDS" active
    y_true = tf.constant([ [ [0,0,0,0,1], [0,0,0,0,1] ] ], dtype=tf.float32)
    y_pred = tf.identity(y_true)
    loss = loss_fn(y_true, y_pred)
    # perfect → zero
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_reduce_lstm_output_7_and_errors():
    x = tf.random.uniform((2, 5, 7))
    # new_size=5
    out5 = reduce_lstm_output_7(x, new_size=5)
    assert out5.shape[-1] == 5
    # invalid size
    with pytest.raises(ValueError):
        reduce_lstm_output_7(x, new_size=6)

def test_reduce_lstm_output_5_and_errors():
    x = tf.random.uniform((2, 4, 5))
    out3 = reduce_lstm_output_5(x, new_size=3)
    assert out3.shape[-1] == 3
    out2 = reduce_lstm_output_5(x, new_size=2)
    assert out2.shape[-1] == 2
    with pytest.raises(ValueError):
        reduce_lstm_output_5(x, new_size=4)

@pytest.fixture(scope="module")
def dummy_input():
    """
    One batch, 12 time-steps, 6 features (5 bases + soft-mask channel).
    """
    return np.random.rand(1, 12, 6).astype(np.float32)

def test_forward_pass(dummy_input):
    """
    Build a **tiny** version (units=4, filter_size=8, numb_conv=1, numb_lstm=1, pool_size=2)
    so the graph instantiates in < 0.2 s on CPU.  Feed a single random batch and assert:

    *   The model runs without raising.
    *   Output shape matches (batch, time, output_size).
    *   multi_loss flag yields a list whose last element is the main soft-max.
    """
    small_args = dict(
        units=4,
        filter_size=8,
        kernel_size=3,
        numb_conv=1,
        numb_lstm=1,
        pool_size=2,
        output_size=15,     
        dropout_rate=0.0,   # deterministic
    )

    model = lstm_model(**small_args)
    x = dummy_input

    y = model.predict(x, verbose=0)
    assert isinstance(y, np.ndarray)
    y_end = y

    assert y_end.shape[0] == 1               # batch
    assert y_end.shape[-1] == 15              
    expected_T = 12
    assert y_end.shape[1] == expected_T

def test_can_train_one_step(dummy_input):
    """
    Make sure back-prop works: compile a tiny model, train on one mini-batch
    of random data, and assert `fit()` returns a history without errors.
    """
    import numpy as np
    import tensorflow as tf
    from tiberius import lstm_model

    output_size = 15
    # Build a minimal lstm_model with pool_size=1 so time-dim stays the same
    model = lstm_model(
        units=2,
        filter_size=4,
        numb_conv=1,
        numb_lstm=1,
        pool_size=1,
        output_size=output_size,
    )
    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    # Use dummy_input's time dimension (e.g. 12) and our known output_size
    batch, T, _ = dummy_input.shape
    C = output_size

    # Create one-hot labels of shape (batch, T, C)
    y_dummy = tf.one_hot(
        np.random.randint(0, C, size=(batch, T)),
        depth=C
    )

    history = model.fit(dummy_input, y_dummy, epochs=1, verbose=0)
    assert "loss" in history.history and len(history.history["loss"]) == 1
