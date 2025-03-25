import numpy as np

import tiberius


def test_output_aggregation_single_model() -> None:
    # 15 probabilities that sum to 1
    probs = np.array([
        0.2, #IR
        0.0, 0.04, 0.06, #introns
        0.15, 0.15, 0.1, #exons
        0.1, #start
        0.0,0.05,0.05, #EI
        0.05,0.0,0.05, #IE
        0.0,
    ]) #stop
    probs_agg_ref = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    A = tiberius.gene_pred_hmm.make_aggregation_matrix(k=1)
    probs_agg = np.matmul(probs[np.newaxis], A)
    np.testing.assert_almost_equal(np.sum(probs_agg, -1), 1.)
    np.testing.assert_almost_equal(probs_agg[0], probs_agg_ref)


def test_output_aggregation_multi_model() -> None:
    # 1 + 28 (k=2) probabilities that sum to 1
    probs = np.array([
        0.2, #IR
        0, 0.01, 0.02, 0.01, 0.04, 0.02, #introns
        0.08, 0.07, 0.14, 0.01, 0.0, 0.1, #exons
        0.08, 0.02, #start
        0, 0, 0.01, 0.04, 0.02, 0.03, #EI
        0.05, 0, 0, 0, 0.02, 0.03, #IE
        0.0, 0.0,
    ]) #stop
    probs_agg_ref = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    A = tiberius.gene_pred_hmm.make_aggregation_matrix(k=2)
    probs_agg = np.matmul(probs[np.newaxis], A)
    np.testing.assert_almost_equal(np.sum(probs_agg, -1), 1.)
    np.testing.assert_almost_equal(probs_agg[0], probs_agg_ref)
