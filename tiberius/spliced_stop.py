# ==============================================================
# Spliced-stop detection and 24->15 state remapping.
#
# Standard Tiberius HMM has 15 states (bricks2marble no_spliced_stop=False).
# The restrictive HMM (no_spliced_stop=True) has 24 states and structurally
# forbids any path that would splice across a stop codon. We use the
# restrictive HMM to re-predict windows whose standard prediction contained
# a spliced stop, instead of discarding those transcripts.
# ==============================================================

import numpy as np

from bricks2marble.struct.fasta import complement, nucleotides_to_kmers
from bricks2marble.tools.annotate import (
    _split_regions,
    _transcripts_from_regions,
)


# Stop codons for translation table 1, encoded as nucleotides_to_kmers(k=3)
# would encode them: (n0 << 6) | (n1 << 3) | n2 with A=0, C=1, G=2, T=3.
# TAA = (3<<6) | 0 | 0 = 192
# TAG = (3<<6) | 0 | 2 = 194
# TGA = (3<<6) | (2<<3) | 0 = 208
STOP_CODON_CODES = np.array([192, 194, 208], dtype=np.uint16)


# Collapse the 24-state restrictive HMM output back to the 15-state standard
# ordering so downstream code (_split_regions, GTF writing, inframe-stop
# post-filter) sees a uniform encoding. Within-intron frame is not meaningful
# at region level: I3/I4/I5 fold onto I0/I1/I2; EI3-5 and IE3-5 fold onto the
# existing EI/IE state that shares their aggregation column.
#
# Standard 15-state order (no_spliced_stop=False, isc=1):
#   0:IR 1:I0 2:I1 3:I2 4:E0 5:E1 6:E2 7:Start
#   8:EI0 9:EI1 10:EI2 11:IE0 12:IE1 13:IE2 14:Stop
#
# Restrictive 24-state order (no_spliced_stop=True, isc=1):
#   0:IR 1:I0 2:I1 3:I2 4:I3 5:I4 6:I5 7:E0 8:E1 9:E2 10:Start
#   11:EI0 12:EI1 13:EI2 14:EI3 15:EI4 16:EI5
#   17:IE0 18:IE1 19:IE2 20:IE3 21:IE4 22:IE5 23:Stop
STATE_REMAP_24_TO_15 = np.array([
    0,   # 0  IR    -> IR
    1,   # 1  I0    -> I0
    2,   # 2  I1    -> I1
    3,   # 3  I2    -> I2
    1,   # 4  I3    -> I0 (any intron frame)
    2,   # 5  I4    -> I1
    3,   # 6  I5    -> I2
    4,   # 7  E0    -> E0
    5,   # 8  E1    -> E1
    6,   # 9  E2    -> E2
    7,   # 10 Start -> Start
    8,   # 11 EI0   -> EI0  (f1)
    9,   # 12 EI1   -> EI1  (f2)
    10,  # 13 EI2   -> EI2  (f0)
    10,  # 14 EI3   -> EI2  (f0)
    8,   # 15 EI4   -> EI0  (f1)
    8,   # 16 EI5   -> EI0  (f1)
    11,  # 17 IE0   -> IE0  (f2)
    12,  # 18 IE1   -> IE1  (f0)
    13,  # 19 IE2   -> IE2  (f1)
    13,  # 20 IE3   -> IE2  (f1)
    11,  # 21 IE4   -> IE0  (f2)
    11,  # 22 IE5   -> IE0  (f2)
    14,  # 23 Stop  -> Stop
], dtype=np.int32)


def remap_labels_24_to_15(labels_24: np.ndarray) -> np.ndarray:
    return STATE_REMAP_24_TO_15[labels_24]


def _window_has_spliced_stop(
    labels_row: np.ndarray,
    nuc_row: np.ndarray,
    strand: str,
) -> bool:
    regions = _split_regions(labels_row, strand=strand)
    for tx_regions in _transcripts_from_regions(regions):
        cds_regions = [r for r in tx_regions if r.name == "CDS"]
        if not cds_regions:
            continue
        cds_nuc = np.concatenate([nuc_row[r.start:r.end] for r in cds_regions])
        if strand == "-":
            cds_nuc = complement(cds_nuc.copy(), reverse=True)
        if cds_nuc.size < 6:
            continue
        # Exclude the terminal codon which is the legitimate stop.
        codons = nucleotides_to_kmers(cds_nuc.copy())[:-1]
        if np.any(np.isin(codons, STOP_CODON_CODES)):
            return True
    return False


def find_spliced_stop_windows(
    labels: np.ndarray,
    nuc: np.ndarray,
    strand: str,
) -> np.ndarray:
    """Return window indices whose decoded standard-HMM labels contain at
    least one transcript with an inframe stop after splicing.

    Partial transcripts touching window edges are skipped by
    _transcripts_from_regions; spliced stops that live in a cross-window
    transcript are caught by `find_cross_window_spliced_stop_boundaries`.

    Args:
        labels: (N_windows, T) int, 15-state HMM output.
        nuc:    (N_windows, T) int, nucleotide encoding (0=A,1=C,2=G,3=T,
                4=N, 5-8 lower-case for repeat-masked).
        strand: '+' or '-'. For '-' the labels row is already aligned to
                forward coordinates but encodes a reverse-strand gene, so
                CDS bases are reverse-complemented before translation.
    """
    bad = [
        i for i in range(labels.shape[0])
        if _window_has_spliced_stop(labels[i], nuc[i], strand)
    ]
    return np.array(bad, dtype=np.int64)


def find_cross_window_spliced_stop_boundaries(
    labels: np.ndarray,
    nuc: np.ndarray,
    strand: str,
) -> np.ndarray:
    """Scan a group's full labels for transcripts that span one or more
    window boundaries and contain an inframe stop after splicing. Return
    the sorted unique boundary indices to re-predict with the restrictive
    HMM — boundary `b` means between `labels[b]` and `labels[b+1]`.

    Complements `find_spliced_stop_windows`: that one only sees transcripts
    bounded by IR on both sides within a single window; this one sees
    transcripts that cross window edges and are therefore invisible to the
    per-window scan.
    """
    N, T = labels.shape
    if N < 2:
        return np.array([], dtype=np.int64)
    flat_labels = labels.reshape(-1)
    flat_nuc = nuc.reshape(-1)
    regions = _split_regions(flat_labels, strand=strand)
    bad: set[int] = set()
    for tx_regions in _transcripts_from_regions(regions):
        cds_regions = [r for r in tx_regions if r.name == "CDS"]
        if not cds_regions:
            continue
        start_win = cds_regions[0].start // T
        end_win = (cds_regions[-1].end - 1) // T
        if start_win == end_win:
            continue
        cds_nuc = np.concatenate([flat_nuc[r.start:r.end] for r in cds_regions])
        if strand == "-":
            cds_nuc = complement(cds_nuc.copy(), reverse=True)
        if cds_nuc.size < 6:
            continue
        codons = nucleotides_to_kmers(cds_nuc.copy())[:-1]
        if np.any(np.isin(codons, STOP_CODON_CODES)):
            bad.update(range(start_win, end_win))
    return np.array(sorted(bad), dtype=np.int64)
