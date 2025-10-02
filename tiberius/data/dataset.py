from pathlib import Path
from typing import Any, Sequence

import tensorflow as tf
from pydantic import BaseModel

from .util import _parse_batch, _preprocess_batched, expand_path


class DatasetConfig(BaseModel):

    train_paths: Sequence[str]
    validation_paths: Sequence[str] | None = None

    batch_size: int = 500
    shuffle: bool = True
    repeat: bool = True
    output_size: int = 15
    hmm_factor: int | None = None
    input_size: int = 6
    clamsa: bool = False
    oracle: bool = False
    tx_filter: Sequence[bytes] = ()
    tx_filter_region: int = 1000
    seq_weights_window: int = 250                # r
    seq_weights_value: float = 100.0             # w
    softmasking: bool = True

    compression: str | None = "GZIP"
    shuffle_buffer: int = 100
    num_parallel_calls: Any = tf.data.AUTOTUNE

    # default value: serialized empty [0,3] string tensor
    @property
    def empty_tx_serial(self) -> tf.Tensor:
        return tf.io.serialize_tensor(
            tf.constant([], shape=[0, 3], dtype=tf.string),
        )


def build_dataset(paths: Sequence[str], cfg: DatasetConfig) -> tf.data.Dataset:
    files = []
    for file in paths: files.extend(expand_path(Path(file)))

    files_ds = tf.data.Dataset.from_tensor_slices(files)
    if len(paths) > 1: files_ds = files_ds.shuffle(len(paths))

    ds = files_ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=cfg.compression),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=cfg.num_parallel_calls,
        deterministic=not cfg.shuffle,
    )

    if cfg.shuffle:
        ds = ds.shuffle(cfg.shuffle_buffer)

    if cfg.repeat:
        ds = ds.repeat()

    # Batch BEFORE parse to use parse_example
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    ds = ds.map(
        lambda batch: _parse_batch(batch, cfg),
        num_parallel_calls=cfg.num_parallel_calls,
    )
    ds = ds.map(
        lambda ex: _preprocess_batched(ex, cfg),
        num_parallel_calls=cfg.num_parallel_calls,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
