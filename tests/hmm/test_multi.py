import numpy as np
import tensorflow as tf
from learnMSA.msa_hmm.Initializers import ConstantInitializer

import tiberius


def test_multi_copy_transitions() -> None:
    trans = tiberius.GenePredMultiHMMTransitioner(k=2)
    indices = trans.make_transition_indices()
    ref_indices = [
        [0, 0],
        [0, 13], [0, 14], [13, 9], [14, 10], #START in / out
        [27, 0], [28,0], [9, 27], [10, 28], #STOP in / out
        [9, 11], [10, 12], [9, 17], [10, 18], [17, 3], [18, 4],
        [3, 3], [4, 4], [3, 23], [4, 24], [23, 9], [24, 10], #E1 + Intron 1

        [11, 7], [12, 8], [11, 19], [12, 20], [19, 5], [20, 6],
        [5, 5], [6, 6], [5, 25], [6, 26], [25, 11], [26, 12], #E2 + Intron 2

        [7, 9], [8, 10], [7, 15], [8, 16], [15, 1], [16, 2],
        [1, 1], [2, 2], [1, 21], [2, 22], [21, 7], [22, 8], #E0 + Intron 0
    ]
    #check existence of bijection between ref_indices and indices
    assert len(indices) == len(ref_indices)
    for pair in ref_indices:
        print(f"{[0]+pair=}")
        print(f"{indices=}")
        assert np.any(
            np.all(indices == [0]+pair, -1)
        ), f"{pair} for in {indices[:, 1:]}"


def test_model_transitions() -> None:
    trans = tiberius.GenePredHMMTransitioner()

    # test if the transition indices are correct
    indices = trans.make_transition_indices()
    ref_indices = [
        [0, 0],
        [0, 7], [7, 5], #START in / out
        [14, 0], [5, 14], #STOP in / out
        [5, 6], [5, 9], [9, 2], [2, 2], [2, 12], [12, 5], #E1 + Intron 1
        [6, 4],[6, 10], [10, 3], [3, 3], [3, 13], [13, 6], #E2 + Intron 2
        [4, 5], [4, 8], [8, 1], [1, 1], [1, 11], [11, 4] #E0 + Intron 0
    ]
    num_transitions = len(ref_indices)
    # check existence of bijection between num_models copies
    # of ref_indices and indices
    assert len(indices) == num_transitions
    i = 0
    for pair in ref_indices:
        assert np.any(
            np.all(indices == [0]+pair, -1)
        ), f"{pair} not in {indices[:, 1:]}"

    # test the final matrix A
    trans.build()
    A = trans.make_A()
    logA = trans.make_log_A()
    np.testing.assert_almost_equal(np.sum(A, -1), 1.)
    np.testing.assert_almost_equal(tf.reduce_logsumexp(logA, -1).numpy(), 0.)
    for i in range(trans.num_states):
        for j in range(trans.num_states):
            if [i, j] in ref_indices:
                assert A[0, i, j] > 0, f"transition from {i} to {j} not > 0"
            else:
                assert A[0, i, j] == 0, f"transition from {i} to {j} not 0"



def test_expected_lengths() -> None:
    initial_ir_len = 300
    initial_intron_len = 200
    initial_exon_len = 100
    trans = tiberius.GenePredHMMTransitioner(
        initial_exon_len=initial_exon_len,
        initial_intron_len=initial_intron_len,
        initial_ir_len=initial_ir_len,
    )
    trans.build()
    A = trans.make_A()

    # compute the mean of geometric distribution, i.e. the number of
    # Bernoulli trials to get one success (= leaving a state)

    #intergenic:
    expected_IR = 1 / (1 - A[0, 0, 0])
    np.testing.assert_almost_equal(
        expected_IR.numpy(),
        initial_ir_len,
        decimal=3,
    )

    # introns
    for i in range(1, 4):
        expected_intron = 1 / (1 - A[0, i, i])
        np.testing.assert_almost_equal(
            expected_intron.numpy(),
            initial_intron_len,
            decimal=3,
        )

    # exons
    for i in range(4, 7):
        expected_exon = 1 / (1-A[0, i, (i-3) % 3 + 4])
        np.testing.assert_almost_equal(
            expected_exon.numpy(),
            initial_exon_len,
            decimal=3,
        )


def test_multi_model_layer() -> None:
    for num_models in [2, 3, 5]:
        hmm_layer = tiberius.GenePredHMMLayer(num_models=num_models)
        hmm_layer.build([None, None, 15])
        #check if the number of parameters is correct
        assert hmm_layer.cell.transitioner.transition_kernel.shape == (1, 23)
        assert (hmm_layer.cell.transitioner
                .starting_distribution_kernel.shape == (1, 1, 15))
        assert (hmm_layer.cell.emitter[0]
                .emission_kernel.shape == (num_models, 15, 15))


def test_multi_model_parameter_noise() -> None:
    n = 2
    kernel_init = tiberius.make_15_class_emission_kernel(
        smoothing=0.01,
        num_models=n,
        noise_strength=0.001,
    )
    hmm_layer = tiberius.GenePredHMMLayer(
        num_models=n,
        emitter_init=ConstantInitializer(kernel_init),
    )

    np.random.seed(77) #make the test deterministic
    hmm_layer.build((None, None, 15))
    # check if emission matrices are different...
    emission_probs = tf.nn.softmax(
        hmm_layer.cell.emitter[0].emission_kernel
    ).numpy()
    # kernels are different...
    assert np.any(emission_probs[0] != emission_probs[1])
    #...but not too different
    # note that difference could be larger than noise_strength due to rescaling
    assert np.max(np.abs(emission_probs[0] - emission_probs[1])) < 0.01


def test_multi_model_algorithms() -> None:
    for num_models in [1, 2]:
        hmm_layer = tiberius.GenePredHMMLayer(num_models=num_models)
        hmm_layer.build([None, None, 15])
        inputs = np.random.rand(5, 100, 15).astype(np.float32)
        inputs /= np.sum(inputs, -1, keepdims=True)
        nuc = np.random.rand(5, 100, 5).astype(np.float32)
        posterior = hmm_layer(inputs, nuc)
        viterbi = hmm_layer.viterbi(inputs, nuc)
        #check shapes, omit model dimension if num_models = 1 for compatibility
        assert posterior.shape == (
            (5, 100, 15) if num_models == 1 else (5, 100, num_models, 15)
        )
        assert viterbi.shape == (
            (5, 100) if num_models == 1 else (5, 100, num_models)
        )
    
