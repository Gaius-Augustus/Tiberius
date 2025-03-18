import numpy as np

import tiberius


def test_hmm_init() -> None:
    trans = tiberius.gene_pred_hmm_transitioner.GenePredHMMTransitioner(
        initial_ir_len=4,
        initial_intron_len=5,
        initial_exon_len=6,
    )
    expected = {
        (0,0): 1-1/4,
        (0,7): 1/4,
        (1,1): 1-1/5,
        (1,11): 1/5,
        (2,2): 1-1/5,
        (2,12): 1/5,
        (3,3): 1-1/5,
        (3,13): 1/5,
        (4,5): 1-1/6,
        (4,8): 1/6,
        (5,6): 1-1/6,
        (5,9): 1/12,
        (5,14): 1/12,
        (6,4): 1-1/6,
        (6,10): 1/6,
        (7,5): 1,
        (8,1): 1,
        (9,2): 1,
        (10,3): 1,
        (11,4): 1,
        (12,5): 1,
        (13,6): 1,
        (14,0): 1,
    }
    trans.build()
    A = trans.make_A()[0]
    for _, u, v in trans.indices:
        np.testing.assert_almost_equal(
            A[u, v],
            expected[(u,v)],
            err_msg=f"edge {(u,v)}",
        )

    trans2 = tiberius.gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(
        k=1,
        initial_ir_len=4,
        initial_intron_len=5,
        initial_exon_len=6,
        init_component_sd=0.,
    )
    trans2.build()
    A2 = trans2.make_A()[0]
    for _, u, v in trans2.indices:
        np.testing.assert_almost_equal(
            A2[u, v],
            expected[(u, v)],
            err_msg=f"edge {(u, v)}",
        )


def test_multi_hmm_init() -> None:
    trans = tiberius.gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(
        k=2,
        initial_ir_len=4,
        initial_intron_len=5,
        initial_exon_len=6,
        init_component_sd=0.,
    )
    expected = {
        (0,0): 1-1/4,
        (0,13): 1/8,
        (0,14): 1/8,
        (1,1): 1-1/5,
        (1,21): 1/5,
        (2,2): 1-1/5,
        (2,22): 1/5,
        (3,3): 1-1/5,
        (3,23): 1/5,
        (4,4): 1-1/5,
        (4,24): 1/5,
        (5,5): 1-1/5,
        (5,25): 1/5,
        (6,6): 1-1/5,
        (6,26): 1/5,
        (7,9): 1-1/6,
        (7,15): 1/6,
        (8,10): 1-1/6,
        (8,16): 1/6,
        (9,11): 1-1/6,
        (9,17): 1/12,
        (9,27): 1/12,
        (10,12): 1-1/6,
        (10,18): 1/12,
        (10,28): 1/12,
        (11,7): 1-1/6,
        (11,19): 1/6,
        (12,8): 1-1/6,
        (12,20): 1/6,
        (13,9): 1,
        (14,10): 1,
        (15,1): 1,
        (16,2): 1,
        (17,3): 1,
        (18,4): 1,
        (19,5): 1,
        (20,6): 1,
        (21,7): 1,
        (22,8): 1,
        (23,9): 1,
        (24,10): 1,
        (25,11): 1,
        (26,12): 1,
        (27,0): 1,
        (28,0): 1
    }
    trans.build()
    A = trans.make_A()[0]
    for _, u, v in trans.indices:
        np.testing.assert_almost_equal(
            A[u, v],
            expected[(u, v)],
            err_msg=f"edge {(u, v)}",
        )
