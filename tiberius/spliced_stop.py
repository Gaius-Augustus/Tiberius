# ==============================================================
# Spliced-stop detection and 24->15 state remapping.
#
# Standard Tiberius HMM has 15 states (bricks2marble no_spliced_stop=False).
# The restrictive HMM (no_spliced_stop=True) has 24 states and structurally
# forbids any path that would splice across a stop codon. We use the
# restrictive HMM to re-predict transcripts whose standard prediction
# contained a spliced stop, instead of discarding those transcripts.
#
# Detection is transcript-centric and operates on the group's flattened
# labels (concatenation of all windows), so it catches transcripts that
# straddle a window boundary as naturally as single-window ones.
# ==============================================================

from dataclasses import dataclass

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


@dataclass(frozen=True)
class TranscriptSpan:
    """A transcript flagged as containing a spliced stop. Coordinates are
    on the flattened labels array (concatenation of all windows per group).
    `start` and `end` cover the full transcript span including surrounding
    non-CDS regions; the restrictive HMM will re-decode a chunk containing
    this range and we overwrite labels only within [start, end)."""
    start: int
    end: int


def _transcript_has_spliced_stop(
    cds_regions,
    flat_nuc: np.ndarray,
    strand: str,
) -> bool:
    if not cds_regions:
        return False
    cds_nuc = np.concatenate([flat_nuc[r.start:r.end] for r in cds_regions])
    if strand == "-":
        cds_nuc = complement(cds_nuc.copy(), reverse=True)
    if cds_nuc.size < 6:
        return False
    # Exclude the terminal codon which is the legitimate stop.
    codons = nucleotides_to_kmers(cds_nuc.copy())[:-1]
    return bool(np.any(np.isin(codons, STOP_CODON_CODES)))


def find_spliced_stop_transcripts(
    labels: np.ndarray,
    nuc: np.ndarray,
    strand: str,
) -> list[TranscriptSpan]:
    """Scan a group's full labels for every transcript whose CDS contains
    an inframe stop after splicing. Operates on the flattened labels so
    cross-window transcripts are caught as naturally as single-window ones.

    Returns a list of (start, end) spans (half-open) in flattened
    coordinates. The caller re-decodes a chunk around each span with the
    restrictive HMM and overwrites labels in [start, end).

    Args:
        labels: (N_windows, T) int, 15-state HMM output.
        nuc:    (N_windows, T) int, nucleotide encoding (0=A,1=C,2=G,3=T,
                4=N, 5-8 lower-case for repeat-masked).
        strand: '+' or '-'. For '-' the labels row is already aligned to
                forward coordinates but encodes a reverse-strand gene, so
                CDS bases are reverse-complemented before translation.
    """
    if labels.size == 0:
        return []
    flat_labels = labels.reshape(-1)
    flat_nuc = nuc.reshape(-1)
    regions = _split_regions(flat_labels, strand=strand)
    spans: list[TranscriptSpan] = []
    for tx_regions in _transcripts_from_regions(regions):
        cds_regions = [r for r in tx_regions if r.name == "CDS"]
        if not _transcript_has_spliced_stop(cds_regions, flat_nuc, strand):
            continue
        spans.append(TranscriptSpan(
            start=int(tx_regions[0].start),
            end=int(tx_regions[-1].end),
        ))
    return spans
