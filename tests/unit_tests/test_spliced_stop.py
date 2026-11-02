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
    repredicted_boundaries,
    sequence_segments,
    window_segments,
)
from bricks2marble.struct.fasta import Fasta, Sequence
from bricks2marble.tools.annotate import _find_mismatches


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


# ---------------------------------------------------------------------------
# Scan segments, padding and reprediction boundaries.
#
# Every fixture below carries an in-frame stop (TAG) in its CDS, so the old
# behaviour (whole flattened array, no `repredicted`) flags it. The tests
# check that the old call flags it and the new arguments do not.
# ---------------------------------------------------------------------------


def _padded(nuc_str: str, n_pad: int) -> np.ndarray:
    return np.concatenate([_row(nuc_str), np.full(n_pad, -1, dtype=np.int8)])


def _joined_rows():
    # T=20. One CDS runs over flat [5:26], across the row 0 / row 1 edge:
    # ATG TAG CGC CGC CGC | CGC CAG -> internal TAG at codon 1.
    labels = np.array([[0]*5 + [4]*15, [4]*6 + [0]*14], dtype=np.int32)
    nuc = np.stack([
        _row("NNNNN" + "ATGTAGCGCCGCCGC"),
        _row("CGCCAG" + "N"*14),
    ])
    return labels, nuc


def _ends_at_boundary_rows():
    # T=20. CDS is flat [5:20]; it ends exactly at the row edge (flat 20).
    labels = np.array([[0]*5 + [4]*15, [0]*20], dtype=np.int32)
    nuc = np.stack([_row("NNNNN" + "ATGTAGCGCCGCCGC"), _row("N"*20)])
    return labels, nuc


def _starts_at_boundary_rows():
    # T=20. CDS is flat [20:35]; it starts exactly at the row edge (flat 20).
    labels = np.array([[0]*20, [4]*15 + [0]*5], dtype=np.int32)
    nuc = np.stack([_row("N"*20), _row("ATGTAGCGCCGCCGC" + "N"*5)])
    return labels, nuc


def test_partial_transcript_at_sequence_start_is_not_flagged():
    # T=24. Row 0 is sequence A (20 bases + 4 padding), all intergenic.
    # Row 1 is sequence B: CDS from offset 0 (cut off at the sequence start),
    # flat [24:42]: ATG TAG CGC CGC CGC CAG. Flattened, the IR of row 0 makes
    # it look complete; as a segment of its own it is partial and dropped.
    labels = np.array([[0]*24, [4]*18 + [0]*6], dtype=np.int32)
    nuc = np.stack([
        _padded("N"*20, 4),
        _row("ATGTAGCGCCGCCGCCAG" + "N"*6),
    ])
    assert find_spliced_stop_transcripts(labels, nuc, "+") == [
        TranscriptSpan(start=24, end=42)
    ]
    spans = find_spliced_stop_transcripts(
        labels, nuc, "+", segments=[(0, 20), (24, 48)],
    )
    assert spans == []


def test_transcript_joined_across_sequences_is_not_flagged():
    # Row 0 ends in CDS and row 1 starts in CDS, but they are two sequences.
    labels, nuc = _joined_rows()
    assert find_spliced_stop_transcripts(labels, nuc, "+") == [
        TranscriptSpan(start=5, end=26)
    ]
    spans = find_spliced_stop_transcripts(
        labels, nuc, "+", segments=[(0, 20), (20, 40)],
    )
    assert spans == []


def test_cds_running_into_padding_is_not_flagged():
    # T=30, sequence of 24 bases, 6 padding (-1). The labels stay CDS for 3
    # padding bases and only then return to IR: CDS flat [3:27] =
    # ATG TAG CGC CGC CGC CGC CGC + (-1 -1 -1). The last codon is dropped as
    # the terminal one, so the TAG at codon 1 is what trips the old scan.
    labels = np.array([[0]*3 + [4]*24 + [0]*3], dtype=np.int32)
    nuc = _padded("NNN" + "ATGTAGCGCCGCCGCCGCCGC", 6)[None, :]
    assert nuc.shape == labels.shape
    assert find_spliced_stop_transcripts(labels, nuc, "+") == [
        TranscriptSpan(start=3, end=27)
    ]
    spans = find_spliced_stop_transcripts(labels, nuc, "+", segments=[(0, 24)])
    assert spans == []


def test_reprediction_rows_are_not_joined_but_real_transcripts_are_kept():
    # Rows 0/1 are the joined pair from above. Row 2 holds a genuine complete
    # transcript, CDS flat [43:55] = ATG TAG CGC CAG; it must keep its
    # flat coordinates (offset 40) under `window_segments`.
    joined_labels, joined_nuc = _joined_rows()
    labels = np.vstack([joined_labels, [0]*3 + [4]*12 + [0]*5])
    nuc = np.vstack([joined_nuc, _row("NNN" + "ATGTAGCGCCAG" + "N"*5)])
    assert labels.shape == nuc.shape == (3, 20)
    assert window_segments(nuc) == [(0, 20), (20, 40), (40, 60)]
    assert find_spliced_stop_transcripts(labels, nuc, "+") == [
        TranscriptSpan(start=5, end=26), TranscriptSpan(start=43, end=55),
    ]
    spans = find_spliced_stop_transcripts(
        labels, nuc, "+", segments=window_segments(nuc),
    )
    assert spans == [TranscriptSpan(start=43, end=55)]


@pytest.mark.parametrize(
    "build, span, outside",
    [
        (_joined_rows, TranscriptSpan(5, 26), 4),            # crosses 20
        (_ends_at_boundary_rows, TranscriptSpan(5, 20), 21),  # ends at 20
        (_starts_at_boundary_rows, TranscriptSpan(20, 35), 19),  # starts at 20
    ],
    ids=["crosses", "ends_at", "starts_at"],
)
def test_transcript_touching_repredicted_boundary_is_skipped(
    build, span, outside,
):
    labels, nuc = build()

    def scan(boundaries):
        return find_spliced_stop_transcripts(
            labels, nuc, "+", repredicted=np.array(boundaries, dtype=np.int64),
        )

    assert scan([20]) == []              # touches or crosses -> skipped
    assert scan([]) == [span]            # nothing repredicted
    assert scan([40]) == [span]          # boundary elsewhere
    assert scan([outside]) == [span]     # one base outside the span
    assert find_spliced_stop_transcripts(labels, nuc, "+") == [span]


def test_cross_window_transcript_with_unflagged_boundary_is_detected():
    # T=20, one sequence. Row 0 ends in I0 and row 1 starts in I0, so the two
    # windows agree and the boundary is not repredicted. Flat regions:
    # IR[0:3] CDS[3:9] intron[9:24] CDS[24:36] IR[36:40]; spliced CDS =
    # ATGTAG + CGCCGCCGCCAG -> ATG TAG CGC CGC CGC CAG, internal TAG.
    labels = np.array([
        [0]*3 + [4]*6 + [1]*11,
        [1]*4 + [4]*12 + [0]*4,
    ], dtype=np.int32)
    nuc = np.stack([
        _row("NNN" + "ATGTAG" + "N"*11),
        _row("N"*4 + "CGCCGCCGCCAG" + "N"*4),
    ])
    assert _find_mismatches(labels).size == 0
    spans = find_spliced_stop_transcripts(
        labels, nuc, "+", segments=[(0, 40)],
        repredicted=np.array([], dtype=np.int64),
    )
    assert spans == [TranscriptSpan(start=3, end=36)]


def test_window_segments_strip_trailing_padding():
    nuc = np.array([
        [0, 1, 2, 3, 0, 1],       # full row
        [0, 1, 2, -1, -1, -1],    # 3 bases, 3 padding
        [0, 1, 2, 3, 0, -1],      # 5 bases, 1 padding
        [-1] * 6,                 # empty
    ], dtype=np.int8)
    assert window_segments(nuc) == [(0, 6), (6, 9), (12, 17), (18, 18)]


def _toy_fasta(sizes, T):
    # Same resampling as bricks2marble.io.fasta.iterate_sequences: every
    # sequence is cut into windows of T, the last one padded with -1.
    seqs = [
        Sequence(np.zeros(n, dtype=np.int8), name=f"s{i}")
        for i, n in enumerate(sizes)
    ]
    return Fasta(seqs).resample(T)


def test_sequence_segments_exclude_padding():
    # T=10: a 7-base sequence (1 window, 3 padding) followed by a 16-base
    # sequence (2 windows, 4 padding) -> rows 0 | 1, 2.
    fasta = _toy_fasta([7, 16], 10)
    nuc = fasta.nuc
    assert nuc.shape == (3, 10)
    assert (nuc[0, 7:] == -1).all() and (nuc[2, 6:] == -1).all()
    assert sequence_segments(fasta) == [(0, 7), (10, 26)]


@pytest.mark.parametrize(
    "sizes, first_row, expected",
    [
        ([7, 16], 1, 20),   # sequence 2 starts at row 1: (0 + 1 + 1) * 10
        ([16, 7], 0, 10),   # sequence 1 starts at row 0: (0 + 0 + 1) * 10
    ],
)
def test_repredicted_boundaries_only_within_a_sequence(
    sizes, first_row, expected,
):
    # The 16-base sequence occupies rows first_row, first_row + 1 and has a
    # genuine mismatch between them (I0 vs I1). The edge between the two
    # sequences also mismatches (IR vs state 3) but must not be reported.
    fasta = _toy_fasta(sizes, 10)
    labels = np.zeros((3, 10), dtype=np.int32)
    labels[first_row, -1] = 1
    labels[first_row + 1, 0] = 2
    labels[2 - first_row, 0] = 3
    assert _find_mismatches(labels).size == 2  # naive scan sees both edges
    boundaries = repredicted_boundaries(labels, fasta)
    np.testing.assert_array_equal(boundaries, [expected])


def test_repredicted_boundaries_empty_when_windows_agree():
    fasta = _toy_fasta([7, 16], 10)
    labels = np.zeros((3, 10), dtype=np.int32)
    boundaries = repredicted_boundaries(labels, fasta)
    assert boundaries.size == 0
