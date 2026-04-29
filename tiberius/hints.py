# ==============================================================
# Authors: Lars Gabriel
#
# Read hints (introns, start/stop codons) and weight LSTM output
# class probabilities at hint positions before the HMM stage.
# ==============================================================
"""Hint integration for Tiberius inference.

Hint file format (GFF-like, BRAKER/Augustus convention as produced by
``tiberius/scripts/aln2hints.pl``)::

    seqname \\t source \\t feature \\t start \\t end \\t score \\t strand \\t frame \\t attrs

where ``feature`` is one of ``intron``, ``start``/``start_codon``,
``stop``/``stop_codon`` and coordinates are 1-based inclusive.

Class layout (intron_state_chain=1, 15 classes; see
``bricks2marble.tf.hmm.tools.state_names``)::

    0  IR
    1-3   I0  I1  I2     (inner intron, 3 frames)
    4-6   E0  E1  E2     (exon, 3 frames)
    7  START
    8-10  EI0 EI1 EI2    (donor splice site)
    11-13 IE0 IE1 IE2    (acceptor splice site)
    14 STOP
"""

from collections import defaultdict

import numpy as np

I_INNER = (1, 2, 3)
E0, E1, E2 = 4, 5, 6
START = 7
EI = (8, 9, 10)
IE = (11, 12, 13)
STOP = 14

_FEATURE_ALIASES = {
    "intron": "intron",
    "start": "start",
    "start_codon": "start",
    "stop": "stop",
    "stop_codon": "stop",
}


def load_hints(path):
    """Read a GFF-like hints file.

    Returns:
        dict[str, list[tuple[str, int, int, str]]]: For each sequence
        name, a list of ``(feature, start, end, strand)`` tuples with
        coordinates converted to 0-based half-open ``[start, end)``.
    """
    hints: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            seqname, _src, feature, start, end, _score, strand = parts[:7]
            feature = _FEATURE_ALIASES.get(feature.lower())
            if feature is None or strand not in ("+", "-"):
                continue
            try:
                s = int(start) - 1
                e = int(end)
            except ValueError:
                continue
            if e <= s:
                continue
            hints[seqname].append((feature, s, e, strand))
    return dict(hints)


def _emit_intron(gpos, classes, start, end):
    L = end - start
    if L < 6:
        # too short to split into donor + body + acceptor, boost any
        # intron-related class along the whole region
        for p in range(start, end):
            gpos.append(p)
            classes.append(I_INNER + EI + IE)
        return
    for off in range(3):
        gpos.append(start + off)
        classes.append(EI)
    for off in range(3, L - 3):
        gpos.append(start + off)
        classes.append(I_INNER)
    for off in range(3):
        gpos.append(end - 3 + off)
        classes.append(IE)


def _emit_codon(gpos, classes, start, end, kind, strand):
    if end - start < 3:
        return
    if kind == "start":
        cls_seq = ((START,), (E1,), (E2,))
    else:
        cls_seq = ((E0,), (E1,), (STOP,))
    if strand == "+":
        # 5'->3' reads start, start+1, start+2
        seq_pos = (start, start + 1, start + 2)
    else:
        # 5'->3' on - strand reads from end-1 down to start
        seq_pos = (end - 1, end - 2, end - 3)
    for p, c in zip(seq_pos, cls_seq):
        gpos.append(p)
        classes.append(c)


def apply_hints(lstm_out, fasta, hints, weight, strand):
    """Multiply LSTM class probabilities at hint positions by ``weight``.

    Args:
        lstm_out (np.ndarray): Array of shape ``(N, T, C)`` containing
            post-softmax class probabilities. Mutated in place.
        fasta: A ``bricks2marble.struct.Fasta`` whose chunked layout
            ``lstm_out`` was generated from. ``seq.start`` provides the
            genome-coordinate offset of each sequence.
        hints (dict): Output of :func:`load_hints`.
        weight (float): Multiplicative factor applied to the boosted
            classes before per-position renormalization. ``weight <= 0``
            or ``weight == 1`` is a no-op.
        strand (str): ``"+"`` for forward LSTM output or ``"-"`` for
            backward (chunks reverse-complemented per-chunk). The genome
            position ``p`` maps to chunk position ``T - 1 - (p % T)``
            for ``"-"``.

    Returns:
        np.ndarray: ``lstm_out`` (same array, mutated).
    """
    if not hints or weight <= 0.0 or weight == 1.0:
        return lstm_out

    N, T, _C = lstm_out.shape
    chunk_offset = 0
    flat_chunk: list[int] = []
    flat_pos: list[int] = []
    flat_class: list[int] = []

    for seq in fasta:
        n_chunks = seq.N
        seq_start = seq.start
        seq_end = seq.start + seq.size
        gpos_buf: list[int] = []
        cls_buf: list[tuple[int, ...]] = []

        for (feature, hstart, hend, hstrand) in hints.get(seq.name, ()):
            if hstrand != strand:
                continue
            cs = max(hstart, seq_start)
            ce = min(hend, seq_end)
            if ce <= cs:
                continue
            if feature == "intron":
                _emit_intron(gpos_buf, cls_buf, cs, ce)
            else:
                if hstart < seq_start or hend > seq_end:
                    continue
                _emit_codon(gpos_buf, cls_buf, hstart, hend, feature, strand)

        if gpos_buf:
            gpos_arr = np.asarray(gpos_buf, dtype=np.int64) - seq_start
            chunk_local = gpos_arr // T
            pos_local = gpos_arr % T
            if strand == "-":
                pos_local = T - 1 - pos_local
            valid = (chunk_local >= 0) & (chunk_local < n_chunks)
            for i in np.flatnonzero(valid):
                ci = int(chunk_local[i]) + chunk_offset
                pi = int(pos_local[i])
                for c in cls_buf[i]:
                    flat_chunk.append(ci)
                    flat_pos.append(pi)
                    flat_class.append(c)

        chunk_offset += n_chunks

    if not flat_chunk:
        return lstm_out

    chunks = np.asarray(flat_chunk, dtype=np.int64)
    poss = np.asarray(flat_pos, dtype=np.int64)
    classes = np.asarray(flat_class, dtype=np.int64)
    lstm_out[chunks, poss, classes] = (
        lstm_out[chunks, poss, classes] * np.float32(weight)
    )

    flat_idx = chunks * T + poss
    unique_flat = np.unique(flat_idx)
    u_chunks = unique_flat // T
    u_pos = unique_flat % T
    sums = lstm_out[u_chunks, u_pos].sum(axis=-1, keepdims=True)
    lstm_out[u_chunks, u_pos] = (
        lstm_out[u_chunks, u_pos] / np.maximum(sums, np.float32(1e-12))
    )
    return lstm_out
