import os
import tempfile
import numpy as np
import tensorflow as tf
import pytest

from tiberius.data_generator import DataGenerator

def _write_tfrecord(path, records):
    """Helper to write a TFRecord file with given (x, y[, clamsa]) tuples."""
    with tf.io.TFRecordWriter(path, options="GZIP") as writer:
        for i in range(records[0].shape[0]):
            x, y = records[0][i], records[1][i]
            features = {
                'input': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(x).numpy()])),
                'output': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(y).numpy()])),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(ex.SerializeToString())

def test_parse_and_next(tmp_path):
    # Create one record: x shape (3,4), y shape (3,5)
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
                      [0,1,0,0,0]]], dtype=tf.int32)  # shape (3,4,5)
    rec_path = tmp_path / "data.tfrecord"
    _write_tfrecord(str(rec_path), [x, y])

    # Instantiate generator with no repeat/shuffle
    dg = DataGenerator(
        file_path=[str(rec_path)],
        batch_size=1,
        shuffle=False,
        repeat=False,
        output_size=5,
        softmasking=True,
        clamsa=False,
        oracle=False
    )
    dataset_iter = iter(dg.get_dataset())
    # __next__ should return (X, Y)
    X, Y = next(dataset_iter)
    # X should equal x with batch dim
    np.testing.assert_array_equal(X, x.numpy()[np.newaxis, 0,...])
    # Y should equal y with batch dim
    np.testing.assert_array_equal(Y, y.numpy()[np.newaxis,0, ...])

    X, Y = next(dataset_iter)
    # X should equal x with batch dim
    np.testing.assert_array_equal(X, x.numpy()[np.newaxis,1, ...])
    # Y should equal y with batch dim
    np.testing.assert_array_equal(Y, y.numpy()[np.newaxis, 1,...])

