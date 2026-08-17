"""Indexed data generator for both-strand gene prediction training.

Uses bricks2marble.struct.index.IndexedBGZipFasta for memory-mapped sequence
access on bgzip-compressed FASTA files (only the requested window bytes are
read from disk).  Annotation is loaded via bricks2marble.io.load_annotation
from a GenePred file (.gp); isoform selection uses gene.longest().

Requires:
  - bgzip-compressed genome: species.fa.gz  (bgzip, NOT gzip)
  - pyfaidx index alongside it: species.fa.gz.fai  (samtools faidx species.fa.gz)
  - GenePred annotation: species.gp  (convert from GTF with:
        gtfToGenePred -genePredExt species.gtf species.gp)

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
from bricks2marble.struct.annotation import (
    Annotation as B2MAnnotation,
    CDS as B2MCDS,
    Transcript as B2MTranscript,
)

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


# ---------------------------------------------------------------------------
# GenePred loader with gene-level isoform grouping
# ---------------------------------------------------------------------------

def _load_genepred_grouped(path: str | Path) -> B2MAnnotation:
    """Load a GenePred (extended) file and add only the longest isoform per gene.

    Uses field 11 (name2, gene name) from gtfToGenePred -genePredExt output to
    group isoforms.  The transcript with the longest total CDS length is kept;
    all others are discarded before building the bricks2marble Annotation.

    Falls back to the transcript name (field 0) as the grouping key when
    field 11 is absent (plain GenePred without -genePredExt).

    Coordinate note: GenePred fields 8/9 are ALL exon intervals (including UTR
    exons).  Fields 5/6 give the CDS boundaries (cdsStart, cdsEnd; 0-based
    half-open).  This function intersects each exon interval with [cdsStart,
    cdsEnd] so that only coding sequence is included in tx.cds — matching the
    behaviour of the TFRecord pipeline which reads CDS-only features from the
    GTF.  Transcripts with cdsStart == cdsEnd (non-coding) are skipped.
    """
    # gene_key → list of Transcript
    groups: dict[str, list] = {}

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")

            # Skip non-coding transcripts (cdsStart == cdsEnd).
            cds_start = int(fields[5])   # 0-based inclusive
            cds_end   = int(fields[6])   # 0-based exclusive
            if cds_start >= cds_end:
                continue

            # Parse sequence/strand from fields; exon coords from fields 8/9.
            sequence = fields[1]
            strand   = fields[2]
            name     = fields[0]
            starts = [int(x) for x in fields[8].rstrip(",").split(",")]
            ends   = [int(x) for x in fields[9].rstrip(",").split(",")]

            # Intersect each exon with [cds_start, cds_end] to strip UTR exons.
            cds_blocks = []
            for s, e in zip(starts, ends):
                cs = max(s, cds_start)
                ce = min(e, cds_end)
                if cs < ce:
                    cds_blocks.append(B2MCDS(start=cs, end=ce))

            if not cds_blocks:
                continue

            tx = B2MTranscript(name=name, sequence=sequence, strand=strand,
                               cds=cds_blocks)

            # Field 11 is the gene name in -genePredExt format.
            gene_name = fields[11].strip() if len(fields) > 11 else fields[0].strip()
            # Include sequence + strand so same-name genes on different
            # sequences or strands are never merged.
            gene_key = f"{sequence}|{strand}|{gene_name}"
            groups.setdefault(gene_key, []).append(tx)

    ann = B2MAnnotation()
    for txs in groups.values():
        longest = max(txs, key=lambda t: sum(c.end - c.start for c in t.cds))
        ann.add(longest)   # ann.add(tx) wraps the transcript in a solo Gene internally
    return ann


# ---------------------------------------------------------------------------
# 15-class label generation from bricks2marble Annotation
# ---------------------------------------------------------------------------

def _b2m_transcript_to_genomic_labels(tx) -> np.ndarray:
    """Convert bricks2marble Transcript to 15-class integer labels in genomic order.

    Returns array of shape (tx_end - tx_start,) with values 0-14.
    Index 0 corresponds to genomic position tx.cds[0].start (0-based).

    Label scheme (Tiberius HMM):
      0=IR, 1-3=I0-I2, 4-6=E0-E2, 7=START, 8-10=EI0-EI2,
      11-13=IE0-IE2, 14=STOP
    """
    from tiberius.annotation_gtf import GeneFeature

    cds_blocks = tx.cds   # list of CDS(start, end) — 0-based, end exclusive
    strand = tx.strand

    if not cds_blocks:
        return np.array([], dtype=np.int32)

    tx_start = cds_blocks[0].start   # 0-based inclusive
    tx_end = cds_blocks[-1].end      # 0-based exclusive
    tx_len = tx_end - tx_start

    # Build GeneFeature list (1-based inclusive convention).
    # CDS block  0-based [s, e) → 1-based [s+1, e]
    # Intron gap 0-based [e1, s2) → 1-based [e1+1, s2]
    features = []
    for i, blk in enumerate(cds_blocks):
        features.append(GeneFeature('CDS', blk.start + 1, blk.end, strand, 0))
        if i < len(cds_blocks) - 1:
            next_blk = cds_blocks[i + 1]
            features.append(GeneFeature('intron', blk.end + 1, next_blk.start, strand, 0))

    features.sort(key=lambda f: f.start)

    # Compute reading-frame phases — same algorithm as Tiberius Transcript.read_input.
    # For + strand iterate left-to-right; for - strand right-to-left (5'→3').
    idx_order = (
        list(range(len(features))) if strand == '+'
        else list(range(len(features) - 1, -1, -1))
    )
    prev_cds_phase = 0
    for i in idx_order:
        f = features[i]
        if f.type == 'CDS':
            f.phase = prev_cds_phase
            prev_cds_phase = (3 - (f.end - f.start + 1 - prev_cds_phase) % 3) % 3

    for k, i in enumerate(idx_order):
        f = features[i]
        if f.type == 'intron':
            try:
                f.phase = features[idx_order[k + 1]].phase
            except IndexError:
                f.phase = 0

    # Fill label array (index 0 = genomic position tx_start).
    tx_labels = np.zeros(tx_len, dtype=np.int32)
    for f in features:
        # Convert 1-based inclusive → 0-based relative to tx_start.
        rs = f.start - 1 - tx_start   # 0-based relative start (inclusive)
        re = f.end - tx_start          # 0-based relative end (exclusive)
        rs_c = max(0, rs)
        re_c = min(tx_len, re)
        if rs_c >= re_c:
            continue
        feat_labels = f.to_class_labels()   # length = f.end - f.start + 1
        clip_s = rs_c - rs
        tx_labels[rs_c:re_c] = feat_labels[clip_s: clip_s + (re_c - rs_c)]

    # Mark START and STOP at transcript 5' and 3' ends.
    if strand == '+':
        tx_labels[0] = 7    # START
        tx_labels[-1] = 14  # STOP
    else:
        # - strand: 5' end is at the highest genomic coordinate.
        tx_labels[-1] = 7   # START
        tx_labels[0] = 14   # STOP

    return tx_labels


def _chunk_labels_from_b2m(
    b2m_annot,
    seq_name: str,
    chunk_start: int,   # 0-based inclusive
    chunk_end: int,     # 0-based exclusive
    strand: str,        # "+" or "-"
) -> np.ndarray:
    """Generate 15-class labels for one strand in a genomic window.

    For strand="+": labels[i] = label at genomic position chunk_start+i.
    For strand="-": labels[j] = label at RC position j, i.e. the - strand
        label at genomic position chunk_end-1-j  (5'→3' on - strand).
    """
    chunk_len = chunk_end - chunk_start
    labels = np.zeros(chunk_len, dtype=np.int32)

    if seq_name not in b2m_annot:
        return labels

    seq_ann = b2m_annot[seq_name]

    for gene in seq_ann.genes():
        if gene.strand != strand:
            continue
        tx = gene.longest()
        if not tx.cds:
            continue
        gene_start = tx.cds[0].start   # 0-based inclusive
        gene_end = tx.cds[-1].end       # 0-based exclusive
        if gene_end <= chunk_start or gene_start >= chunk_end:
            continue

        tx_labels = _b2m_transcript_to_genomic_labels(tx)  # genomic (left-to-right) order

        ovlp_start = max(chunk_start, gene_start)
        ovlp_end = min(chunk_end, gene_end)
        seg_len = ovlp_end - ovlp_start

        labels[ovlp_start - chunk_start: ovlp_start - chunk_start + seg_len] = \
            tx_labels[ovlp_start - gene_start: ovlp_start - gene_start + seg_len]

    if strand == '-':
        labels = labels[::-1].copy()   # 5'→3' on - strand = reversed genomic order

    return labels


def _build_window_index(
    seq_names: list[str],
    seq_lens: list[int],
    seq_len: int,
) -> list[tuple[str, int, int, int]]:
    """Return (seq_name, start_bp, end_bp, genomic_k) for every chunk window."""
    windows: list[tuple[str, int, int, int]] = []
    genomic_k = 0
    for seq_name, slen in zip(seq_names, seq_lens):
        n_chunks = slen // seq_len
        for local_k in range(n_chunks):
            start = local_k * seq_len
            end = start + seq_len
            windows.append((seq_name, start, end, genomic_k))
            genomic_k += 1
    return windows


def get_species_data_indexed(
    genome_path: str | Path,
    annot_path: str | Path,
    seq_len: int,
    min_seq_len: int | None = None,
) -> tuple:
    """Build (IndexedBGZipFasta, B2MAnnotation, seq_names, seq_lens) for one species.

    Parameters
    ----------
    genome_path:
        Path to a bgzip-compressed FASTA (.fa.gz).  A pyfaidx index
        (.fa.gz.fai) must exist alongside it; create with:
            samtools faidx species.fa.gz
    annot_path:
        Path to a GenePred file (.gp).  Convert from GTF with:
            gtfToGenePred -genePredExt species.gtf species.gp
    seq_len:
        Training window length.
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

    b2m_annot = _load_genepred_grouped(annot_path)

    return ifasta, b2m_annot, seq_names, seq_lens


class IndexedDataGenerator:
    """Data generator backed by bricks2marble IndexedBGZipFasta (memory-mapped).

    Only the window bytes requested per training step are read from disk;
    genome sequences are never fully loaded into RAM.  Annotation is loaded
    once via bricks2marble.io.load_annotation (GenePred format); the longest
    transcript per gene is used for label generation.

    Parameters
    ----------
    species_data:
        List of (IndexedBGZipFasta, B2MAnnotation, seq_names, seq_lens) tuples,
        one per training species.  Use get_species_data_indexed() to build each.
    batch_size, seq_len, shuffle, repeat, both_strands, softmasking:
        Same semantics as the TFRecord-based DataGenerator.
    """

    def __init__(
        self,
        species_data: List[Tuple],
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        repeat: bool = True,
        both_strands: bool = True,
        softmasking: bool = True,
        filter_empty: bool = False,
    ) -> None:
        self.species_data = species_data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.repeat = repeat
        self.both_strands = both_strands
        self.softmasking = softmasking
        self.filter_empty = filter_empty

        # Flat index: (species_idx, seq_name, start_bp, end_bp, genomic_k, strand)
        # For both_strands=True one '+' entry covers both strands in a single pass.
        # For both_strands=False both '+' and '-' entries are emitted so that
        # minus-strand genes receive a proper RC sequence and reversed labels
        # (matching the TFRecord pipeline which stores 2N chunks: N minus + N plus).
        self._index: list[tuple[int, str, int, int, int, str]] = []
        for sp_idx, (_ifasta, _annot, seq_names, seq_lens) in enumerate(species_data):
            for entry in _build_window_index(seq_names, seq_lens, seq_len):
                seq_name, start, end, gk = entry
                self._index.append((sp_idx, seq_name, start, end, gk, '+'))
                if not self.both_strands:
                    self._index.append((sp_idx, seq_name, start, end, gk, '-'))

    @property
    def n_examples(self) -> int:
        return len(self._index)

    def _fetch(
        self,
        sp_idx: int,
        seq_name: str,
        start: int,
        end: int,
        genomic_k: int,  # unused; retained for index-tuple compatibility
        strand: str = '+',
    ) -> tuple[np.ndarray, np.ndarray]:
        ifasta, b2m_annot, _seq_names, _seq_lens = self.species_data[sp_idx]

        seq = ifasta.fetch(seq_name, (start, end))
        x_fwd = _sequence_to_onehot(seq, softmasking=self.softmasking)

        if strand == '-':
            # Reverse-complement: time-reverse then swap A↔T (col 0↔3) and C↔G (col 1↔2).
            # One-hot columns: [A, C, G, T, N] or [A, C, G, T, N, softmask].
            nc = x_fwd.shape[1]
            col = [3, 2, 1, 0, 4] if nc == 5 else [3, 2, 1, 0, 4, 5]
            x = x_fwd[::-1][:, col].copy()
            labels = _chunk_labels_from_b2m(b2m_annot, seq_name, start, end, '-')
            y = np.eye(15, dtype=np.float32)[labels]
        elif self.both_strands:
            x = x_fwd
            fwd_labels = _chunk_labels_from_b2m(b2m_annot, seq_name, start, end, '+')
            rev_labels = _chunk_labels_from_b2m(b2m_annot, seq_name, start, end, '-')
            fwd_oh = np.eye(15, dtype=np.float32)[fwd_labels]
            rev_oh = np.eye(15, dtype=np.float32)[rev_labels]
            y = np.concatenate([fwd_oh, rev_oh], axis=-1)
        else:
            x = x_fwd
            labels = _chunk_labels_from_b2m(b2m_annot, seq_name, start, end, '+')
            y = np.eye(15, dtype=np.float32)[labels]

        return x, y

    def get_dataset(self) -> tf.data.Dataset:
        """Return a tf.data.Dataset ready for model.fit()."""
        # Infer output shapes from first example
        sp0, sn0, s0, e0, gk0, st0 = self._index[0]
        x0, y0 = self._fetch(sp0, sn0, s0, e0, gk0, st0)
        inp_size = x0.shape[-1]
        out_size = y0.shape[-1]

        index = list(self._index)
        fetch = self._fetch
        shuffle = self.shuffle
        filter_empty = self.filter_empty

        def _has_gene(y: np.ndarray) -> bool:
            # y is (T, 15) for single-strand or (T, 30) for both_strands.
            # Returns True if any position carries a non-IR (non-zero) label.
            nc = y.shape[-1]
            if nc == 30:
                return (np.any(np.argmax(y[:, :15], axis=-1) != 0)
                        or np.any(np.argmax(y[:, 15:], axis=-1) != 0))
            return np.any(np.argmax(y, axis=-1) != 0)

        def generator():
            idx = list(index)
            if shuffle:
                import random
                random.shuffle(idx)
            for sp_idx, seq_name, start, end, gk, strand in idx:
                x, y = fetch(sp_idx, seq_name, start, end, gk, strand)
                if filter_empty and not _has_gene(y):
                    continue
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
