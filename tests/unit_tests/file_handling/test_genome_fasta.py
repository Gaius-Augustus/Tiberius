import os
import gzip
import bz2
import math
import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from tiberius import GenomeSequences

def test_extract_seqarray_filters_by_length():
    # Prepare a fake genome dict
    genome = {
        "chr1": SeqRecord(Seq("ATGC"), id="chr1"),
        "chr2": SeqRecord(Seq("AT")),    # too short
        "chr3": SeqRecord(Seq("AAAAA")),
    }
    # min_seq_len = 4 => chr2 dropped
    gs = GenomeSequences(genome=genome, min_seq_len=4)
    assert gs.sequence_names == ["chr1", "chr3"]
    assert gs.sequences == ["ATGC", "AAAAA"]

@pytest.mark.parametrize("suffix, open_func", [
    ("",        open),
    (".gz",     gzip.open),
    (".bz2",    bz2.open),
])
def test_read_fasta_various_compressions(tmp_path, suffix, open_func):
    # Create a small FASTA
    fasta = tmp_path / f"test.fa{suffix}"
    content = ">seq1\nACGT\n>seq2\nnnNNX\n"
    # Write compressed or plain
    if suffix == "":
        (fasta).write_text(content)
    else:
        # binary write with the appropriate module
        with open(fasta, "wb") as f:
            comp = gzip.compress if suffix == ".gz" else bz2.compress
            f.write(comp(content.encode("utf8")))
    # Read it in
    gs = GenomeSequences(fasta_file=str(fasta), min_seq_len=0)
    assert gs.sequence_names == ["seq1", "seq2"]
    assert gs.sequences == ["ACGT", "nnNNX"]

def test_encode_sequences_upper_lower_and_default():
    # Create GS with a single sequence
    gs = object.__new__(GenomeSequences)
    gs.sequences = ["AaCxTNB"]
    gs.sequence_names = ["chrTest"]
    gs.one_hot_encoded = None

    gs.encode_sequences(seq=["chrTest"])
    arr = gs.one_hot_encoded["chrTest"]
    # Should be shape (7,6)
    assert arr.shape == (7,6)
    # Row 0: 'A' → [1,0,0,0,0,0]
    assert np.array_equal(arr[0], [1,0,0,0,0,0])
    # Row 1: 'a' → [1,0,0,0,0,1]
    assert np.array_equal(arr[1], [1,0,0,0,0,1])
    # Row 2: 'C' → [0,1,0,0,0,0]
    assert np.array_equal(arr[2], [0,1,0,0,0,0])
    # Row 3: 'x' → default → [0,0,0,0,1,0]
    assert np.array_equal(arr[3], [0,0,0,0,1,0])
    # Row 4: 'T' → [0,0,0,1,0,0]
    assert np.array_equal(arr[4], [0,0,0,1,0,0])
    # Row 5: 'N' → default → [0,0,0,0,1,0]
    assert np.array_equal(arr[5], [0,0,0,0,1,0])
    # Row 6: 'B' → default → [0,0,0,0,1,0]
    assert np.array_equal(arr[6], [0,0,0,0,1,0])

def _make_gs_for_chunks(one_hot_array, names, chunksize, overlap):
    """Helper to produce a GS stub with one_hot_encoded filled."""
    gs = object.__new__(GenomeSequences)
    gs.chunksize = chunksize
    gs.overlap = overlap
    gs.one_hot_encoded = { name: one_hot_array for name in names }
    gs.sequence_names = names[:]
    return gs

def test_get_flat_chunks_no_pad_default():
    # length 9, chunksize=4, overlap=1 → num_chunks = floor(8/3)+1=2+1=3
    length = 9
    base = np.zeros((length,6),dtype=np.uint8)
    base[:,4] = 1   # default channel
    gs = _make_gs_for_chunks(base, ["chr"], chunksize=4, overlap=0)

    chunks, coords, cs_out = gs.get_flat_chunks(
        sequence_names=["chr"], strand='+',
        coords=False, pad=False, adapt_chunksize=False
    )
    # Expect (num_chunks-1)=2 full chunks, no pad → shape (2,4,6)
    assert chunks.shape == (2,4,6)
    # coords should be None, chunksize unchanged
    assert coords is None
    assert cs_out == 4

def test_get_flat_chunks_with_pad_and_coords():
    # length 9, chunksize=4, overlap=1 → num_chunks=3 → 2 full + 1 pad
    length = 9
    base = np.zeros((length,6),dtype=np.uint8)
    base[:,4] = 1
    gs = _make_gs_for_chunks(base, ["chr"], chunksize=4, overlap=0)

    chunks, coords, cs_out = gs.get_flat_chunks(
        sequence_names=["chr"], strand='+',
        coords=True, pad=True, adapt_chunksize=False
    )
    # 2 full + 1 padded → 3 chunks
    assert chunks.shape == (3,4,6)
    # coords length == num_chunks = 3
    assert len(coords) == 3
    # first coord covers 1–4
    assert coords[0] == ["chr", "+", 1, 4]
    # last coord covers the padded chunk start
    assert coords[-1][2] == 9

    assert cs_out == 4

def test_get_flat_chunks_minus_strand_channel_flip():
    # Reuse pad+coords case but strand='-'
    length = 9
    base = np.zeros((length,6),dtype=np.uint8)
    base[:,4] = 1
    gs = _make_gs_for_chunks(base, ["chr"], chunksize=4, overlap=1)

    chunks_p, _, _ = gs.get_flat_chunks(
        sequence_names=["chr"], strand='+',
        coords=False, pad=True, adapt_chunksize=False
    )
    chunks_m, _, _ = gs.get_flat_chunks(
        sequence_names=["chr"], strand='-',
        coords=False, pad=True, adapt_chunksize=False
    )
    # Negative strand should reverse chunk order, reverse positions, and remap channels
    # channel order should become [T,G,C,A,default,softmask] i.e. indices [3,2,1,0,4,5]
    remapped = chunks_p[..., [3,2,1,0,4,5]][::-1, ::-1, :]
    assert np.array_equal(chunks_m, remapped)

def test_get_flat_chunks_adapt_chunksize_small_seq():
    # Sequence shorter than chunksize → adapt_chunksize=True
    # use length=3, chunksize=10, overlap=2
    length = 3
    base = np.zeros((length,6),dtype=np.uint8)
    base[:,4] = 1
    names = ["chrS"]
    gs = _make_gs_for_chunks(base, names, chunksize=10, overlap=0)
    # no coords
    chunks, coords, cs_out = gs.get_flat_chunks(
        sequence_names=names, strand='+',
        coords=False, pad=True, adapt_chunksize=True
    )
    # Adapted chunksize should be rounded up to divisor*(1+…), see logic → 18
    assert cs_out == 18
    # Only one padded chunk
    assert chunks.shape == (1, 18, 6)
    # coords still None
    assert coords is None
    # The first row of the padded chunk equals the last base of original sequence
    # original sequence = three rows of default → last row = default [0,0,0,0,1,0]
    assert np.array_equal(chunks[0,0], base[-1])

