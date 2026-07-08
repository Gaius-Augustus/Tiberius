"""Optional per-feature and BigWig outputs of the preHMM (LSTM) class
probabilities. Used by ``tiberius.main`` when ``--cds_probs`` or
``--bigwig_out`` are set on the CLI.

Class layout (see ``annotation_gtf.py``):

- 0            : intergenic (IR)
- 1..3         : intron classes (I0, I1, I2)
- 4..14        : coding classes (E0-E2, START, EI0-EI2, IE0-IE2, STOP)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

CDS_CLASS_SLICE = slice(4, 15)
INTRON_CLASS_SLICE = slice(1, 4)
IR_CLASS_INDEX = 0

CDS_PROBS_TSV_HEADER = (
    "sequence\tstrand\ttranscript_id\tcds_index\tstart\tend\t"
    "min_cds_prob\tmax_cds_prob\tmean_cds_prob\n"
)


def cds_probs_tsv_path(gtf_out: str) -> str:
    """Return the sidecar TSV path for ``--cds_probs`` given the GTF output."""
    return f"{gtf_out}.cds_probs.tsv"


def init_cds_probs_tsv(gtf_out: str) -> str:
    path = cds_probs_tsv_path(gtf_out)
    with open(path, "w") as fh:
        fh.write(CDS_PROBS_TSV_HEADER)
    return path


def parse_seq_filter(bigwig_seqs: str) -> set[str] | None:
    if not bigwig_seqs:
        return None
    return {s.strip() for s in bigwig_seqs.split(",") if s.strip()}


def sequence_lengths_from_fasta(fasta_path: str) -> list[tuple[str, int]]:
    """Return ``[(name, length), ...]`` for each record in ``fasta_path``.

    Uses the same name-splitting convention as bricks2marble's
    ``annotate_genome`` (name is taken up to the first whitespace).
    Handles gzip-compressed input transparently.
    """
    import gzip

    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    lengths: list[tuple[str, int]] = []
    current_name: str | None = None
    current_len = 0
    with opener(fasta_path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if current_name is not None:
                    lengths.append((current_name, current_len))
                current_name = line[1:].split()[0] if len(line) > 1 else ""
                current_len = 0
            else:
                current_len += len(line.strip())
    if current_name is not None:
        lengths.append((current_name, current_len))
    return lengths


class BigWigBuffer:
    """Buffers per-sequence CDS/intron/IR tracks until close.

    pyBigWig requires that ``addEntries`` calls appear in the same
    chromosome order as the header. Sequences arrive in whatever order
    bricks2marble's grouping picks, so we accumulate the three float32
    tracks per sequence in a dict and flush them alphabetically on
    ``close`` right before opening the three BigWig files.
    """

    def __init__(
        self,
        bigwig_prefix: str,
        seq_lengths: Iterable[tuple[str, int]],
        seq_filter: set[str] | None,
    ) -> None:
        header = sorted(
            [
                (name, length)
                for name, length in seq_lengths
                if seq_filter is None or name in seq_filter
            ],
            key=lambda x: x[0],
        )
        if not header:
            raise ValueError(
                "No sequences selected for BigWig output. Check --bigwig_seqs "
                "against the sequence names in the input FASTA."
            )
        self._prefix = bigwig_prefix
        self._header = header
        self._expected: dict[str, int] = dict(header)
        self._buffered: dict[str, dict[str, np.ndarray]] = {}
        Path(bigwig_prefix).parent.mkdir(parents=True, exist_ok=True)

    def has(self, seq_name: str) -> bool:
        return seq_name in self._expected

    def add(self, seq_name: str, tracks: dict[str, np.ndarray]) -> None:
        if seq_name not in self._expected:
            return
        self._buffered[seq_name] = tracks

    def close(self) -> None:
        import pyBigWig
        for key in ("cds", "intron", "ir"):
            bw = pyBigWig.open(f"{self._prefix}.{key}.bw", "w")
            try:
                bw.addHeader(self._header)
                for name, _length in self._header:
                    tracks = self._buffered.get(name)
                    if tracks is None:
                        continue
                    values = tracks[key]
                    if values.size == 0:
                        continue
                    bw.addEntries(
                        name,
                        0,
                        values=values.astype(np.float64),
                        span=1,
                        step=1,
                    )
            finally:
                bw.close()


def open_bigwig_writers(
    bigwig_prefix: str,
    seq_lengths: Iterable[tuple[str, int]],
    seq_filter: set[str] | None,
) -> BigWigBuffer:
    return BigWigBuffer(bigwig_prefix, seq_lengths, seq_filter)


def close_bigwig_writers(writers: BigWigBuffer) -> None:
    try:
        writers.close()
    except Exception:
        pass


def _softmax_for_sequence(
    softmax_all: np.ndarray,
    chunk_shift: int,
    n_chunks: int,
    seq_size: int,
) -> np.ndarray:
    """Extract per-position softmax for a single sequence from the group-
    level softmax cache. Shape returned: ``(seq_size, n_classes)``.
    """
    chunk_slice = softmax_all[chunk_shift:chunk_shift + n_chunks]
    n_classes = chunk_slice.shape[-1]
    return chunk_slice.reshape(-1, n_classes)[:seq_size]


def compute_class_group_tracks(
    softmax_fwd: np.ndarray,
    softmax_bwd: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Reduce per-position class probabilities to CDS/intron/IR groups.

    Both strands are collapsed via elementwise max so a single track
    reflects the maximum probability across strands. This is the "easy
    solution" from the design; revisit if it turns out too coarse.
    """
    p_cds = softmax_fwd[:, CDS_CLASS_SLICE].sum(axis=-1)
    p_intron = softmax_fwd[:, INTRON_CLASS_SLICE].sum(axis=-1)
    p_ir = softmax_fwd[:, IR_CLASS_INDEX]
    if softmax_bwd is not None:
        p_cds = np.maximum(p_cds, softmax_bwd[:, CDS_CLASS_SLICE].sum(axis=-1))
        p_intron = np.maximum(
            p_intron, softmax_bwd[:, INTRON_CLASS_SLICE].sum(axis=-1)
        )
        p_ir = np.maximum(p_ir, softmax_bwd[:, IR_CLASS_INDEX])
    return {"cds": p_cds.astype(np.float32),
            "intron": p_intron.astype(np.float32),
            "ir": p_ir.astype(np.float32)}


def write_bigwig_sequence(
    writers: "BigWigBuffer",
    seq_name: str,
    tracks: dict[str, np.ndarray],
) -> None:
    """Buffer one sequence's tracks for eventual BigWig flush."""
    writers.add(seq_name, tracks)


def append_cds_probs_for_annotation(
    tsv_path: str,
    annotation,
    softmax_by_seq: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """For every transcript in ``annotation``, compute min/max/mean of the
    summed CDS-class probability across each CDS block and append one row
    per CDS to the sidecar TSV.

    ``softmax_by_seq[seq_name]`` is ``(softmax_fwd, softmax_bwd)`` with
    each array shaped ``(seq_size, n_classes)`` in forward-strand
    coordinates.
    """
    with open(tsv_path, "a") as fh:
        for seq_ann in annotation:
            seq_name = seq_ann.sequence
            entry = softmax_by_seq.get(seq_name)
            if entry is None:
                continue
            sm_fwd, sm_bwd = entry
            for tx in seq_ann:
                sm = sm_fwd if tx.strand == "+" else sm_bwd
                p_cds_full = sm[:, CDS_CLASS_SLICE].sum(axis=-1)
                for i, cds in enumerate(tx.cds, start=1):
                    start = max(0, cds.start)
                    end = min(p_cds_full.shape[0], cds.end)
                    if end <= start:
                        continue
                    vals = p_cds_full[start:end]
                    fh.write(
                        f"{seq_name}\t{tx.strand}\t{tx.name}\t{i}\t"
                        f"{cds.start}\t{cds.end}\t"
                        f"{float(vals.min()):.6f}\t"
                        f"{float(vals.max()):.6f}\t"
                        f"{float(vals.mean()):.6f}\n"
                    )
