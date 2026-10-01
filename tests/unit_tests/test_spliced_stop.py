"""Tests for the spliced-stop detector and the 24->15 state remap.

The detector operates on 15-state Tiberius HMM output; the remap folds the
24-state restrictive-HMM output back to 15 states for downstream code.
"""

import numpy as np
import pytest

pytest.importorskip("bricks2marble")

from tiberius.spliced_stop import (
    STATE_REMAP_24_TO_15,
    STOP_CODON_CODES,
    find_cross_window_spliced_stop_boundaries,
    find_spliced_stop_windows,
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
    bad = find_spliced_stop_windows(labels, nuc_row[None, :], strand="+")
    assert bad.tolist() == [0]


def test_terminal_stop_is_not_flagged():
    # Same layout, but the only stop codon lands as the terminal codon
    # (TAG at the final 3 bases of the spliced CDS). The detector drops
    # the terminal codon before scanning, so no false positive.
    # Spliced CDS = ATGCGCT + AGCCGCTAG -> ATG CGC TAG CGC... wait, 9 ->
    # use 7+8=15, codons = ATG CGC TAC AGC CTAG... careful.
    # Use 6+9 = 15 nt, codons = ATGCGC + TACAGCTAG = ATG CGC TAC AGC TAG,
    # terminal TAG only.
    labels = np.array(
        [0]*5 + [4]*6 + [1]*3 + [4]*9 + [0]*7, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "NNNNN" + "ATGCGC" + "GTA" + "TACAGCTAG" + "NNNNNNN"
    )
    bad = find_spliced_stop_windows(labels, nuc_row[None, :], strand="+")
    assert bad.tolist() == []


def test_detects_spliced_stop_on_minus_strand():
    # Build a window whose forward-strand bases, when reverse-complemented,
    # yield a transcript ATG CGC TAG CGC CGC TAG with an internal stop.
    # Reverse-complement of "ATGCGCTAGCGCCGCTAG" is "CTAGCGGCGCTAGCGCAT".
    # Split that 18 bp back into exon1 (7 bp) + intron (3 bp) + exon2 (11 bp):
    #   forward bases 5..12  = first 7 of RC   = "CTAGCGG"
    #   forward bases 12..15 = intron (any)    = "GTA"
    #   forward bases 15..26 = last 11 of RC   = "CGCTAGCGCAT"
    labels = np.array(
        [0]*5 + [4]*7 + [1]*3 + [4]*11 + [0]*4, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "NNNNN" + "CTAGCGG" + "GTA" + "CGCTAGCGCAT" + "NNNN"
    )
    bad = find_spliced_stop_windows(labels, nuc_row[None, :], strand="-")
    assert bad.tolist() == [0]


def test_cross_window_spliced_stop_detected():
    # Two windows of T=20. Transcript crosses the boundary (its exon2
    # touches the window-0 end AND continues as the first 6 bases of
    # window 1), so per-window _transcripts_from_regions drops it in
    # both windows. The cross-window scanner, operating on the flattened
    # labels, catches the transcript.
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
    assert labels.shape == (2, T) and nuc.shape == (2, T)
    # Per-window scan sees nothing.
    assert find_spliced_stop_windows(labels, nuc, strand="+").size == 0
    # Cross-window scan catches boundary 0 (between windows 0 and 1).
    bad = find_cross_window_spliced_stop_boundaries(labels, nuc, strand="+")
    assert bad.tolist() == [0]


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
    bad = find_cross_window_spliced_stop_boundaries(labels, nuc, strand="+")
    assert bad.size == 0


def test_cross_window_single_window_group_short_circuits():
    labels = np.array([[0]*5 + [4]*25], dtype=np.int32)
    nuc = np.stack([_row("N" * 30)])
    assert find_cross_window_spliced_stop_boundaries(
        labels, nuc, strand="+",
    ).size == 0


def test_partial_transcript_at_edge_is_skipped():
    # Transcript starts at position 0 (no IR before it). The decoder treats
    # it as a partial transcript and _transcripts_from_regions drops it.
    labels = np.array(
        [4]*7 + [1]*3 + [4]*11 + [0]*9, dtype=np.int32,
    )[None, :]
    nuc_row = _row(
        "ATGCGCT" + "GTA" + "AGCGCCGCTAG" + "NNNNNNNNN"
    )
    bad = find_spliced_stop_windows(labels, nuc_row[None, :], strand="+")
    # Partial transcript at the window edge is intentionally skipped by
    # the per-window detector; the full-annotation post-filter catches
    # these cross-boundary cases.
    assert bad.tolist() == []
