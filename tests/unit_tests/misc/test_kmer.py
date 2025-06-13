import numpy as np
import tensorflow as tf

import tiberius


def test_encode_kmer() -> None:
    a = tiberius.kmer.encode_kmer_string("AAA")
    assert a.shape == (16, 4)
    assert np.sum(a) == 1
    assert a[0, 0] == 1
    a2 = tiberius.kmer.encode_kmer_string("AAA", pivot_left=False)
    assert a2.shape == (16, 4)
    assert np.sum(a2) == 1
    assert a2[0, 0] == 1

    b = tiberius.kmer.encode_kmer_string("ACG")
    assert b.shape == (16, 4)
    assert np.sum(b) == 1
    assert b[6, 0] == 1
    b2 = tiberius.kmer.encode_kmer_string("ACG", pivot_left=False)
    assert b2.shape == (16, 4)
    assert np.sum(b2) == 1
    assert b2[4, 2] == 1

    c = tiberius.kmer.encode_kmer_string("NAA")
    assert c.shape == (16, 4)
    assert np.sum(c) == 1
    for i in range(4):
        assert c[0, i] == 0.25
    c2 = tiberius.kmer.encode_kmer_string("NAA", pivot_left=False)
    assert c2.shape == (16, 4)
    assert np.sum(c2) == 1
    for i in range(4):
        assert c2[i, 0] == 0.25

    d = tiberius.kmer.encode_kmer_string("ANA")
    assert d.shape == (16, 4)
    assert np.sum(d) == 1
    for i in range(4):
        assert d[i*4, 0] == 0.25
    d2 = tiberius.kmer.encode_kmer_string("ANA", pivot_left=False)
    assert d2.shape == (16, 4)
    assert np.sum(d2) == 1
    for i in range(4):
        assert d2[i*4, 0] == 0.25


def test_3mers() -> None:
    # ACGT
    example_sequence = tf.constant(
        [[[1., 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]]
    )

    expected_3mers_left = ["ACG", "CGT", "GTN", "TNN"]
    k_mers_left = tiberius.kmer.make_k_mers(
        example_sequence,
        k=3,
        pivot_left=True,
    )
    for i, k in enumerate(expected_3mers_left):
        np.testing.assert_equal(
            k_mers_left[0, i].numpy(),
            tiberius.kmer.encode_kmer_string(k).numpy(),
        )

    expected_3mers_right = ["NNA", "NAC", "ACG", "CGT"]
    k_mers_right = tiberius.kmer.make_k_mers(
        example_sequence,
        k=3,
        pivot_left=False,
    )
    for i, k in enumerate(expected_3mers_right):
        np.testing.assert_equal(
            k_mers_right[0, i].numpy(),
            tiberius.kmer.encode_kmer_string(k, pivot_left=False).numpy(),
        )


def test_unknown_symbols() -> None:
    example_sequence = tf.constant(
        [[[1., 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]]
    )
    expected_3mers_left = ["ACN", "CNT", "NTN", "TNN"]
    k_mers_left = tiberius.kmer.make_k_mers(
        example_sequence,
        k=3,
        pivot_left=True,
    )
    for i, k in enumerate(expected_3mers_left):
        np.testing.assert_equal(
            k_mers_left[0, i].numpy(),
            tiberius.kmer.encode_kmer_string(k).numpy(),
        )

    expected_3mers_right = ["NNA", "NAC", "ACN", "CNT"]
    k_mers_right = tiberius.kmer.make_k_mers(
        example_sequence,
        k=3,
        pivot_left=False,
    )
    for i, k in enumerate(expected_3mers_right):
        np.testing.assert_equal(
            k_mers_right[0, i].numpy(),
            tiberius.kmer.encode_kmer_string(k, pivot_left=False).numpy(),
        )
