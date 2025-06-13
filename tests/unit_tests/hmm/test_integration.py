import numpy as np
import tensorflow as tf
from learnMSA.msa_hmm.Initializers import ConstantInitializer

import tiberius

from test_predictions import get_inputs, assert_outputs


def test_tf_model_functional() -> None:

    # test the tensorflow layer as part of a functional-style tf model
    hmm_inputs = tf.keras.Input(shape=(None, 15))
    nuc = tf.keras.Input(shape=(None, 5))

    # generate some random hmm inputs simulating previous layers

    # create the hmm layer
    hmm_outputs = tiberius.GenePredHMMLayer(initial_exon_len=10,
                                    initial_intron_len=10,
                                    initial_ir_len=40, 
                                    starting_distribution_init=ConstantInitializer([0] + [-100.]*14))(hmm_inputs, nuc)

    # create a functional model
    model = tf.keras.Model(inputs=[hmm_inputs, nuc], outputs=hmm_outputs)

    # compute one forward pass
    outputs = model(get_inputs())

    # in this example, argmax on the posterior probabilities equals Viterbi
    assert_outputs(np.argmax(outputs, axis=-1))

    # compute one backward pass
    with tf.GradientTape() as tape:
        y = model(get_inputs())
        loss = tf.reduce_mean(y**2) # useless, just to get a gradient

    grads = tape.gradient(loss, model.trainable_variables)

    assert all([tf.reduce_all(tf.math.is_finite(g)).numpy() for g in grads]), "Gradient is not finite"