import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

import bricks2marble as b2m
import numpy as np
import tensorflow as tf

ALLOWED_TRANSITIONS = (
    (0, 0), (0, 7), (7, 5),
    (5, 14), (14, 0), (4, 5),
    (5, 6), (6, 4), (4, 8), (5, 9),
    (6, 10),
    (11, 4), (12, 5), (13, 6),
    (1, 1), (2, 2), (3, 3),
    (1, 11), (2, 12), (3, 13),
    (8, 1), (9, 2), (10, 3),
    (1, 14), (2, 14), (3, 14),
    (11, 14), (12, 14), (13, 14),
    (7, 1), (7, 2), (7, 3),
    (4, 14), (5, 14), (6,14),
)


def build_allowed_matrix(
    edges: Iterable[tuple[int, int]],
    n_labels: int = 15,
) -> np.ndarray:
    """Builds a boolean adjacency matrix ``A`` where ``A[i, j] = True``
    iff `i -> j` is allowed.

    Args:
        edges (sequence of tuple[int, int]): Iterable of ``(src, dst)``
            transitions that are allowed.
        n_labels (int, optional): Number of distinct labels, i.e. rows
            and columns of the resulting matrix. Defaults to 15.
    """
    A = np.zeros((n_labels, n_labels), dtype=bool)
    A[tuple(zip(*edges))] = True
    return A


def check_transitions(
    labels: np.ndarray,
    allowed: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Checks if the given label sequence can be achieved by only
    allowing neighboring labels to be sampled from the allowed
    transitions.

    Args:
        labels (np.ndarray): 1D array of integers in range ``[0,
            allowed.shape[0]-1]``.
        allowed (np.ndarray): Boolean adjacency matrix, where
            ``allowed[i, j] == True`` if `i -> j` is valid.

    Returns:
        (bool, np.ndarray): A boolean indicating if all adjacent
            transitions are allowed and an array of positions ``i``,
            where ``labels[i] -> labels[i+1]`` is NOT allowed.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    n_labels = allowed.shape[0]
    if allowed.shape != (n_labels, n_labels):
        raise ValueError("allowed must be square (n x n)")
    if labels.size <= 1:
        return True, np.array([], dtype=int)

    if np.any((labels < 0) | (labels >= n_labels)):
        raise ValueError(f"labels contain values outside 0..{n_labels-1}")

    src = labels[:-1]
    dst = labels[1:]
    ok_mask = allowed[src, dst]
    bad_idx = np.nonzero(~ok_mask)[0]
    return bad_idx.size == 0, bad_idx


def create_tfrecords(
    fasta: Path | str,
    gtf: Path | str,
    T: int,
    out_path: Path | str,
    out_prefix: str | None = None,
    splits: int = 100,
    verbose: int = 1,
) -> None:
    """Create a number of `.tfrecords` files based on the given data.

    Args:
        fasta (Path | str): Path to the fasta file.
        gtf (Path | str): Path to the corresponding gtf file.
        T (int): Size of the chunks generated.
        out_path (Path | str): Target directory of where to put the
            final records.
        out_prefix (str, optional): A string that is prepended to an
            enumeration of the records, naming them
            `prefix_*.tfrecords`. Defaults to no prefix.
        splits (int, optional): Number of `.tfrecords` files to split
            the given data into. Defaults to 100.
        verbose (bool, optional): Verbosity level for
            progress information and warnings. Defaults to 1, which
            means only warnings are enabled. Can be set to 0 for no
            verbosity or 2 for extended checks on the created records.
    """
    fasta = Path(fasta).expanduser()
    gtf = Path(gtf).expanduser()
    out_path = Path(out_path).expanduser()
    out_path.mkdir(exist_ok=True)

    fasta_ = b2m.io.load_fasta(fasta, T=T, drop_remainder=True)
    fasta_.rename(lambda name: name.split(" ")[0])
    anno = b2m.io.load_gtf(gtf, cds_only=True)
    anno.finalize()

    sequence_dict = {seq.name: seq.nuc.size for seq in fasta_}
    anno_labels = anno.encode(sequence_dict, warn=verbose > 0)

    if verbose == 2:
        for name, labels in anno_labels.items():
            for strand in ["+", "-"]:
                lab = labels[strand] if strand == "+" else labels[strand][::-1]
                indices = np.where(labels[strand] == 14)
                print(
                    name,
                    strand,
                    indices[0].shape,
                    indices[:50],
                    file=sys.stderr,
                )
                A = build_allowed_matrix(ALLOWED_TRANSITIONS)
                check, idx = check_transitions(lab, A)
                for i in idx:
                    if 7 not in lab[i-4:i+4] or 14 in lab[i-4:i+4]:
                        warnings.warn(
                            f"Incorrect transition in sequence {name} in:"
                            f" {lab[i-10:i+10]}"
                        )

    nuc = fasta_.nuc
    labels_plus = np.concatenate(
        [np.reshape(anno_labels[s]["+"], (-1, T)) for s in sequence_dict])
    labels_minus = np.concatenate(
        [np.reshape(anno_labels[s]["-"], (-1, T)) for s in sequence_dict])

    if not (nuc.shape[0] == labels_plus.shape[0] == labels_minus.shape[0]):
        raise RuntimeError("Noticed shape mismatch for labels.")

    indices = np.arange(nuc.shape[0])
    np.random.shuffle(indices)

    def create_example(i):
        feature_bytes_x = tf.io.serialize_tensor(
            b2m.struct.fasta.one_hot(nuc[i], dtype=np.float32)
        ).numpy()
        labels_out = np.concatenate([
            np.eye(15, dtype=np.float32)[labels_plus[i]],
            np.eye(15, dtype=np.float32)[labels_minus[i]]],
            axis=-1
        )

        feature_bytes_y = tf.io.serialize_tensor(labels_out).numpy()

        example = tf.train.Example(features=tf.train.Features(feature={
            'input': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature_bytes_x])),
            'output': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[feature_bytes_y]))
        }))
        serialized_example = example.SerializeToString()
        return serialized_example

    def write_split(k):
        if out_prefix is not None:
            out_dir = out_path / f"{out_prefix}_{k}.tfrecords"
        else:
            out_dir = out_path / f"{k}.tfrecords"

        with tf.io.TFRecordWriter(
            str(out_dir),
            options=tf.io.TFRecordOptions(compression_type='GZIP')
        ) as writer:
            for i in indices[k::splits]:
                serialized_example = create_example(i)
                writer.write(serialized_example)

    for k in range(splits): write_split(k)
    if verbose > 0: print("Writing complete.")
