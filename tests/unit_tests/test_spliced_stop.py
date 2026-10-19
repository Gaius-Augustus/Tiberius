"""Tests for the transcript-centric spliced-stop detector and 24->15 remap.

The detector operates on 15-state Tiberius HMM output (flattened across
windows); the remap folds the 24-state restrictive-HMM output back to 15
states for downstream code.
"""

import numpy as np
import pytest

pytest.importorskip("bricks2marble")

from tiberius.spliced_stop import (
    STATE_REMAP_24_TO_15,
    STOP_CODON_CODES,
    TranscriptSpan,
    find_spliced_stop_transcripts,
    remap_labels_24_to_15,
)


# Nucleotide encoding: A=0, C=1, G=2, T=3 (ACGTN).
# Standard 15-state HMM: 0=IR, 1=I0, 4=E0, 7=Start, 14=Stop.
# _split_regions only distinguishes intergenic / intron / CDS, so we can
# use E0 (4) for every exonic base in the test fixtures.


def _row(nuc_str: str) -> np.ndarray:
    code = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    return np.array([code[c] for c in nuc_str], dtype=np.int8)


def test_remap_24_to_15_covers_all_states():
    assert STATE_REMAP_24_TO_15.shape == (24,)
    assert STATE_REMAP_24_TO_15.min() == 0
    assert STATE_REMAP_24_TO_15.max() == 14
    # IR, Start, Stop are fixed points by convention.
    assert STATE_REMAP_24_TO_15[0] == 0    # IR
    assert STATE_REMAP_24_TO_15[10] == 7   # Start
    assert STATE_REMAP_24_TO_15[23] == 14  # Stop


def test_remap_vectorized():
    # 24-state indices: 0=IR 1=I0 4=I3 7=E0 8=E1 10=Start 14=EI3 23=Stop
    labels_24 = np.array([[0, 10, 7, 23], [1, 4, 8, 14]], dtype=np.int32)
    out = remap_labels_24_to_15(labels_24)
    assert out.shape == labels_24.shape
    # row 0: IR, Start, E0, Stop -> 0, 7, 4, 14
    np.testing.assert_array_equal(out[0], [0, 7, 4, 14])
    # row 1: I0, I3->I0, E1, EI3->EI2 -> 1, 1, 5, 10
    np.testing.assert_array_equal(out[1], [1, 1, 5, 10])


def test_stop_codon_codes_match_kmer_encoding():
    from bricks2marble.struct.fasta import nucleotides_to_kmers
    # TAA, TAG, TGA
    for codon_str, expected in [("TAA", 192), ("TAG", 194), ("TGA", 208)]:
        arr = _row(codon_str).astype(np.int64)
        kmer = int(nucleotides_to_kmers(arr.copy())[0])
        assert kmer == expected
        assert expected in STOP_CODON_CODES.tolist()


def test_detects_spliced_stop_on_plus_strand():
    # Window layout (T=30):
    #   IR [0:5]  E0 [5:12]  I0 [12:15]  E0 [15:26]  IR [26:30]
    # Spliced CDS = nuc[5:12] + nuc[15:26] = 18 bases =
    #   ATGCGCT + AGCGCCGCTAG -> ATG CGC TAG CGC CGC TAG
    # Internal TAG at codon 3 should trip the detector.
    labels = np.array(
        [0]*5 + [4]*7 + [1]*3 + [4]*11 + [0]*4, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "NNNNN" + "ATGCGCT" + "GTA" + "AGCGCCGCTAG" + "NNNN"
    )
    assert labels.shape == (1, 30) and nuc_row.shape == (30,)
    spans = find_spliced_stop_transcripts(labels, nuc_row[None, :], strand="+")
    assert len(spans) == 1
    assert spans[0] == TranscriptSpan(start=5, end=26)


def test_terminal_stop_is_not_flagged():
    # Spliced CDS = ATGCGC + TACAGCTAG = ATG CGC TAC AGC TAG, terminal
    # TAG only.
    labels = np.array(
        [0]*5 + [4]*6 + [1]*3 + [4]*9 + [0]*7, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "NNNNN" + "ATGCGC" + "GTA" + "TACAGCTAG" + "NNNNNNN"
    )
    spans = find_spliced_stop_transcripts(labels, nuc_row[None, :], strand="+")
    assert spans == []


def test_detects_spliced_stop_on_minus_strand():
    # Reverse-complement of "ATGCGCTAGCGCCGCTAG" is "CTAGCGGCGCTAGCGCAT".
    labels = np.array(
        [0]*5 + [4]*7 + [1]*3 + [4]*11 + [0]*4, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "NNNNN" + "CTAGCGG" + "GTA" + "CGCTAGCGCAT" + "NNNN"
    )
    spans = find_spliced_stop_transcripts(labels, nuc_row[None, :], strand="-")
    assert len(spans) == 1
    assert spans[0] == TranscriptSpan(start=5, end=26)


def test_cross_window_spliced_stop_is_detected():
    # Two windows of T=20. Transcript crosses the boundary: its exon2
    # touches the window-0 end AND continues as the first 6 bases of
    # window 1. The transcript-centric detector sees it on the flattened
    # labels.
    #
    # Flat regions: IR[0:5] CDS[5:11] intron[11:14] CDS[14:26] IR[26:40]
    # Spliced CDS = ATG CGC + TAG CGC CGC CAG -> internal TAG at codon 3.
    T = 20
    labels = np.array([
        [0]*5 + [4]*6 + [1]*3 + [4]*6,   # window 0
        [4]*6 + [0]*14,                  # window 1
    ], dtype=np.int32)
    nuc = np.stack([
        _row("NNNNN" + "ATGCGC" + "GTA" + "TAGCGC"),
        _row("CGCCAG" + "N" * 14),
    ])
    spans = find_spliced_stop_transcripts(labels, nuc, strand="+")
    assert len(spans) == 1
    # Transcript span runs from forward coord 5 (CDS start) through 26
    # (CDS end), covering the window-0/1 boundary at 20.
    assert spans[0] == TranscriptSpan(start=5, end=26)


def test_cross_window_without_spliced_stop_not_flagged():
    T = 20
    labels = np.array([
        [0]*5 + [4]*6 + [1]*3 + [4]*6,
        [4]*6 + [0]*14,
    ], dtype=np.int32)
    nuc = np.stack([
        _row("NNNNN" + "ATGCGC" + "GTA" + "CGCCGC"),
        _row("CGCCAG" + "N" * 14),
    ])
    # Spliced = ATG CGC CGC CGC CGC CAG — no internal stops.
    spans = find_spliced_stop_transcripts(labels, nuc, strand="+")
    assert spans == []


def test_partial_transcript_at_edge_is_skipped():
    # Transcript starts at position 0 (no IR before it) and no IR after
    # the final CDS either. _transcripts_from_regions treats it as
    # partial and drops it.
    labels = np.array(
        [4]*7 + [1]*3 + [4]*11 + [0]*9, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "ATGCGCT" + "GTA" + "AGCGCCGCTAG" + "NNNNNNNNN"
    )
    spans = find_spliced_stop_transcripts(labels, nuc_row[None, :], strand="+")
    # The flattened scan still only reports complete IR-bounded transcripts.
    # The input lacks a leading IR, so the transcript is partial and skipped.
    assert spans == []
