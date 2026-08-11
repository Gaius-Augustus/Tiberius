"""Indexed data generator for both-strand gene prediction training.

Uses bricks2marble.struct.index.IndexedBGZipFasta for memory-mapped sequence
access on bgzip-compressed FASTA files (only the requested window bytes are
read from disk) and keeps Tiberius's Annotation in memory (transcript
coordinates only, ~10-100 MB per genome).

Requires:
  - bgzip-compressed genome: species.fa.gz  (bgzip, NOT gzip)
  - pyfaidx index alongside it: species.fa.gz.fai  (samtools faidx species.fa.gz)

For both_strands=True each batch yields:
  x : (batch, seq_len, inp_size)   forward-strand one-hot sequence
  y : (batch, seq_len, 30)         fwd labels ([:15]) | rev labels ([15:])

For both_strands=False:
  x : (batch, seq_len, inp_size)   forward-strand one-hot sequence
  y : (batch, seq_len, 15)         forward-strand labels only
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from bricks2marble.struct.index import IndexedBGZipFasta
from tiberius.annotation_gtf import Annotation

# ---------------------------------------------------------------------------
# One-hot lookup: bricks2marble int8 encoding → Tiberius one-hot rows
# bricks2marble translation_table maps ASCII bytes:
#   A→0  C→1  G→2  T→3  N→4  n→4  a→5  c→6  g→7  t→8
# Tiberius one_hot_table (genome_fasta.py) columns: [A, C, G, T, N, softmask]
# ---------------------------------------------------------------------------
_B2M_INT_TO_ONEHOT = np.array([
    [1, 0, 0, 0, 0, 0],  # 0 = A
    [0, 1, 0, 0, 0, 0],  # 1 = C
    [0, 0, 1, 0, 0, 0],  # 2 = G
    [0, 0, 0, 1, 0, 0],  # 3 = T
    [0, 0, 0, 0, 1, 0],  # 4 = N / n
    [1, 0, 0, 0, 0, 1],  # 5 = a  (softmasked)
    [0, 1, 0, 0, 0, 1],  # 6 = c
    [0, 0, 1, 0, 0, 1],  # 7 = g
    [0, 0, 0, 1, 0, 1],  # 8 = t
], dtype=np.float32)


def _sequence_to_onehot(seq, softmasking: bool = True) -> np.ndarray:
    """Convert a bricks2marble Sequence to a Tiberius (L, 5-or-6) float32 array."""
    arr = np.clip(seq.flat, 0, 8)       # guard against unexpected values
    x = _B2M_INT_TO_ONEHOT[arr]         # (L, 6)
    if not softmasking:
        x = x[:, :5]
    return x


def _build_window_index(
    annotation: Annotation,
) -> list[tuple[str, int, int, int]]:
    """Return (seq_name, start_bp, end_bp, genomic_k) for every chunk window.

    Iterates through sequences in annotation.seqnames order, matching the
    layout that Annotation.seq2chunk_pos["-"] uses.
    """
    windows: list[tuple[str, int, int, int]] = []
    genomic_k = 0
    for seq_name, seq_len in zip(annotation.seqnames, annotation.seq_lens):
        n_chunks = seq_len // annotation.chunk_len
        for local_k in range(n_chunks):
            start = local_k * annotation.chunk_len
            end = start + annotation.chunk_len
            windows.append((seq_name, start, end, genomic_k))
            genomic_k += 1
    return windows


def get_species_data_indexed(
    genome_path: str | Path,
    annot_path: str | Path,
    seq_len: int,
    min_seq_len: int | None = None,
) -> tuple[IndexedBGZipFasta, Annotation]:
    """Build (IndexedBGZipFasta, Annotation) for one species.

    Parameters
    ----------
    genome_path:
        Path to a bgzip-compressed FASTA (.fa.gz).  A pyfaidx index
        (.fa.gz.fai) must exist alongside it; create one with:
            samtools faidx species.fa.gz
    annot_path:
        Path to a GTF or GFF file with CDS/intron features.
    seq_len:
        Training window length (chunk_len for Annotation).
    min_seq_len:
        Sequences shorter than this are skipped.  Defaults to seq_len.
    """
    if min_seq_len is None:
        min_seq_len = seq_len

    ifasta = IndexedBGZipFasta(genome_path)

    seq_names = [
        s for s in ifasta.sequence_names()
        if ifasta.length(s) >= min_seq_len
    ]
    seq_lens = [ifasta.length(s) for s in seq_names]

    annot = Annotation(str(annot_path), seq_names, seq_lens, seq_len)
    annot.read_inputfile()

    return ifasta, annot


class IndexedDataGenerator:
    """Data generator backed by bricks2marble IndexedFasta (memory-mapped).

    Only the window bytes requested per training step are read from disk;
    genome sequences are never fully loaded into RAM.  The Annotation
    (transcript coordinates) is held in memory as before.

    Parameters
    ----------
    species_data:
        List of (IndexedBGZipFasta, Annotation) pairs, one per training species.
        Use get_species_data_indexed() to construct each pair.
    batch_size, shuffle, repeat, both_strands, softmasking:
        Same semantics as the previous TFRecord-based DataGenerator.
    """

    def __init__(
        self,
        species_data: List[Tuple[IndexedBGZipFasta, Annotation]],
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

        # Flat index: (species_idx, seq_name, start_bp, end_bp, genomic_k)
        self._index: list[tuple[int, str, int, int, int]] = []
        for sp_idx, (_, annot) in enumerate(species_data):
            for seq_name, start, end, gk in _build_window_index(annot):
                self._index.append((sp_idx, seq_name, start, end, gk))

    @property
    def n_examples(self) -> int:
        return len(self._index)

    def _fetch(
        self,
        sp_idx: int,
        seq_name: str,
        start: int,
        end: int,
        genomic_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        ifasta, annot = self.species_data[sp_idx]

        # Forward-strand sequence (memory-mapped read, window bytes only)
        seq = ifasta.fetch(seq_name, (start, end))
        x = _sequence_to_onehot(seq, softmasking=self.softmasking)

        # Labels
        if self.both_strands:
            fwd_oh, rev_oh = annot.get_onehot_dual_strand(genomic_k)
            y = np.concatenate([fwd_oh, rev_oh], axis=-1).astype(np.float32)
        else:
            y = annot.get_onehot(annot.n_genomic_chunks + genomic_k).astype(np.float32)

        return x, y

    def get_dataset(self) -> tf.data.Dataset:
        """Return a tf.data.Dataset ready for model.fit()."""
        # Infer output shapes from first example
        sp0, sn0, s0, e0, gk0 = self._index[0]
        x0, y0 = self._fetch(sp0, sn0, s0, e0, gk0)
        inp_size = x0.shape[-1]
        out_size = y0.shape[-1]

        index = list(self._index)
        fetch = self._fetch
        shuffle = self.shuffle

        def generator():
            idx = list(index)
            if shuffle:
                import random
                random.shuffle(idx)
            for sp_idx, seq_name, start, end, gk in idx:
                x, y = fetch(sp_idx, seq_name, start, end, gk)
                yield x, y

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, inp_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None, out_size), dtype=tf.float32),
            ),
        )

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=min(2048, len(index)))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if self.repeat:
            dataset = dataset.repeat()

        return dataset
