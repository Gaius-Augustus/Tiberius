"""Indexed data generator for both-strand gene prediction training.

Replaces the TFRecord-based DataGenerator for dual-strand runs.  Data is
generated on-the-fly from GenomeSequences + Annotation objects, so no
TFRecord pre-generation step is needed.

For both_strands=True each example yields:
  x : (seq_len, inp_size)  forward-strand one-hot sequence
  y : (seq_len, 30)        forward labels ([:15]) | reverse labels ([15:])

For both_strands=False:
  x : (seq_len, inp_size)  forward-strand one-hot sequence
  y : (seq_len, 15)        forward-strand labels only (same as TFRecord mode)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tiberius.genome_fasta import GenomeSequences
from tiberius.annotation_gtf import Annotation


class IndexedDataGenerator:
    """Data generator that reads directly from GenomeSequences + Annotation.

    Parameters
    ----------
    species_data:
        List of (GenomeSequences, Annotation) pairs, one per training species.
        Both objects must be fully initialised (encode_sequences /
        prep_seq_chunks / read_inputfile already called).
    batch_size:
        Training batch size.
    shuffle:
        Shuffle example order each epoch.
    repeat:
        Repeat the dataset indefinitely (set True for model.fit).
    both_strands:
        If True, return paired (fwd, rev) labels as a single (seq_len, 30)
        array.  The model's HMM head must have use_reverse_strand=True.
    softmasking:
        If False, strip the 6th softmask channel from the input.
    """

    def __init__(
        self,
        species_data: List[Tuple[GenomeSequences, Annotation]],
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = True,
        both_strands: bool = True,
        softmasking: bool = True,
    ) -> None:
        self.species_data = species_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.both_strands = both_strands
        self.softmasking = softmasking

        # Build a flat index: list of (species_idx, genomic_window_idx).
        # chunks_seq = [minus_chunks... , plus_chunks...], so N = len // 2.
        index: list[tuple[int, int]] = []
        for sp_idx, (fasta, _) in enumerate(species_data):
            N = len(fasta.chunks_seq) // 2
            for k in range(N):
                index.append((sp_idx, k))
        self._index = np.array(index, dtype=np.int32)  # (total_windows, 2)

    @property
    def n_examples(self) -> int:
        return len(self._index)

    def _fetch(self, sp_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) for species sp_idx, genomic window k."""
        fasta, annotation = self.species_data[sp_idx]
        N = len(fasta.chunks_seq) // 2
        # Forward-strand one-hot sequence
        x = fasta.get_onehot(N + k).astype(np.float32)
        if not self.softmasking and x.shape[-1] > 5:
            x = x[:, :5]
        # Labels
        if self.both_strands:
            fwd_oh, rev_oh = annotation.get_onehot_dual_strand(k)
            y = np.concatenate([fwd_oh, rev_oh], axis=-1).astype(np.float32)
        else:
            y = annotation.get_onehot(N + k).astype(np.float32)
        return x, y

    def get_dataset(self) -> tf.data.Dataset:
        """Return a tf.data.Dataset ready to pass to model.fit()."""
        # Infer shapes from first example
        sp0, k0 = int(self._index[0, 0]), int(self._index[0, 1])
        x0, y0 = self._fetch(sp0, k0)
        inp_size = x0.shape[-1]
        out_size = y0.shape[-1]

        index = self._index
        fetch = self._fetch

        def generator():
            idx = index.copy()
            if self.shuffle:
                np.random.shuffle(idx)
            for row in idx:
                x, y = fetch(int(row[0]), int(row[1]))
                yield x, y

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, inp_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None, out_size), dtype=tf.float32),
            ),
        )

        # Shuffle buffer gives cross-species mixing within a batch.
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=min(2048, len(index)))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if self.repeat:
            dataset = dataset.repeat()

        return dataset
