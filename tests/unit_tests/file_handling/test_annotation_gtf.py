import os
import numpy as np
import pytest
from tiberius import GeneStructure

def test_read_gtf(tmp_path):
    # Create a small GTF with comments, track lines, out‐of‐order entries
    gtf = tmp_path / "test.gtf"
    gtf.write_text(
        "# a comment line\n"
        "chr2\tsrc\tCDS\t50\t60\t.\t-\t.\t; transcript_id \"txA\"; foo;\n"
        "chr1\tsrc\tCDS\t10\t20\t.\t+\t0\t; transcript_id \"txB\"; foo;\n"
        "chr1\tsrc\tintron\t21\t39\t.\t+\t.\t; transcript_id \"txB\"; foo;\n"
        "chr1\tsrc\tCDS\t40\t54\t.\t+\t2\t; transcript_id \"txB\"; foo;\n"
    )
    gs = GeneStructure(filename=str(gtf))
    # Expect sorted by chromosome then end coordinate
    assert gs.gene_structures == [        
        ("chr2", "CDS", "-", ".", 50, 60, "txA"),
        ("chr1", "CDS", "+", "0", 10, 20, "txB"),
        ("chr1", "intron", "+", ".", 21, 39, "txB"),
        ("chr1", "CDS", "+", "2", 40, 54, "txB"),
    ]
    print(gs.tx_ranges)
    assert gs.tx_ranges == {'txA': [50, 60], 'txB': [10, 54]}

def _make_gs(structs, tx_ranges=None, chunksize=20, overlap=5):
    # Helper to bypass __init__ completely
    gs = object.__new__(GeneStructure)
    gs.filename = ""
    gs.np_file = ""
    gs.chunksize = chunksize
    gs.overlap = overlap
    gs.gene_structures = structs
    gs.tx_ranges = tx_ranges or {}
    gs.one_hot = None
    gs.one_hot_phase = None
    gs.chunks = None
    gs.chunks_phase = None
    return gs

def test_translate_to_one_hot():
    # One exon on + strand, phase=0, from pos2–4
    encoded_labels = np.eye(15, dtype=int)[[0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0]]
    gs = _make_gs([
        ("chrX", "CDS", "+", "0", 4, 11, "txA"),
        ("chrX", "intron", "+", "0", 12, 15, "txA"),
        ("chrX", "CDS", "+", "2", 16, 22, "txA"),        
    ])

    seqs = ["chrX"]; lengths = [23]
    gs.translate_to_one_hot_hmm(seqs, lengths, transition=True)
    arr = gs.one_hot["+"]["chrX"]
    print(arr.argmax(1))
    print(encoded_labels.argmax(1))
    np.testing.assert_array_equal(arr, encoded_labels)


def test_translate_to_one_hot_minus():
    expected_idx = [0, 0, 0, 14, 5, 4, 6, 5, 4, 6, 13, 3, 3, 3, 3, 10, 6, 5, 4, 6, 5, 7, 0]
    encoded_minus = np.eye(15, dtype=int)[expected_idx]

    gs = _make_gs([
        ("chrX", "CDS", "-", "0", 4, 11, "txA"),
        ("chrX", "intron", "-", "0", 12, 15, "txA"),  # this entry is ignored by the new logic
        ("chrX", "CDS", "-", "2", 16, 22, "txA"),
    ])

    seqs = ["chrX"]
    lengths = [23]
    gs.translate_to_one_hot_hmm(seqs, lengths, transition=True)
    arr = gs.one_hot["-"]["chrX"]
    np.testing.assert_array_equal(arr, encoded_minus)
    
def test_get_flat_chunks_hmm_coords_and_minus():
    # Build an oh map manually and assign to gs.one_hot
    seq = "chrZ"
    length = 10
    # Create a dummy one-hot with shape (10,7)
    base = np.zeros((length,7),dtype=int)
    base[:,0] = 1
    gs = _make_gs([], chunksize=4, overlap=1)
    gs.one_hot = {"+": {seq: base.copy()}, "-": {seq: base.copy()}}
    # coords=True
    plus_chunks, coords = gs.get_flat_chunks_hmm([seq], strand="+", coords=True)
    # num_chunks = floor((10-1)/(4-1))+1 = floor(9/3)+1 = 3+1=4 → 3 actual chunks returned
    assert plus_chunks.shape == (3,4,7)
    # coords length == num_chunks == 4
    assert len(coords) == 4
    assert coords[0] == [seq, "+", 1, 4]
    # minus strand flips
    minus_chunks = gs.get_flat_chunks_hmm([seq], strand="-")
    assert np.array_equal(minus_chunks, plus_chunks[::-1, ::-1, :])

def test_get_flat_chunks_hmm_with_transcript_ids():
    # Simulate one transcript covering bases 3–8 on chr1
    seq = "chr1"
    length = 12
    base = np.zeros((length,7),dtype=int); base[:,0]=1
    gs = _make_gs([], chunksize=5, overlap=2, tx_ranges={"txA":[3,8]})
    gs.one_hot = {"+": {seq: base.copy()}, "-": {seq: base.copy()}}
    # transcript_ids=True
    chunks, tx_ids = gs.get_flat_chunks_hmm([seq], strand="+", transcript_ids=True)
    # num_chunks = floor((12-2)/(5-2))+1 = floor(10/3)+1 = 3+1=4→3 chunks
    assert chunks.shape[0] == tx_ids.shape[0] == 3
    # For each chunk, ensure tx_ids has three columns
    assert tx_ids.shape[1] == 3
    # The middle chunk (i=1) covers bases [ (1*(5-2)+1=4) … (1*(5-2)+5=8) ]
    # It overlaps txA [3–8], so tx_ids[1] should be ['txA','4','8']
    assert list(tx_ids[1]) == ["txA","4","8"]
