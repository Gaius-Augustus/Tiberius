import os
import numpy as np
import pytest
from tiberius import GeneStructure, Annotation

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
    gs = Annotation(file_path=str(gtf), 
        seqnames=["chr1", "chr2"], seq_lens=[70, 60],
        chunk_len=10)
    gs.read_inputfile()

    transcripts = {}
    transcripts_result = {'txA': [[50, 60, 'CDS', '-', 0]], 
        'txB': [[10, 20, 'CDS', '+', 0], [21, 39, 'intron', '+', 1], [40, 54, 'CDS', '+', 1]]}
    seq2_chunk_pos_result = {'-': {'chr1': 0, 'chr2': 7}, '+': {'chr1': 13, 'chr2': 20}}
    chunk2transcripts_result = {12: [0], 13: [0], 14: [1], 15: [1], 16: [1], 17: [1], 18: [1]}

    for tx in gs.transcripts:
        transcripts.update({tx.id : []})
        for f in tx.features:
            transcripts[tx.id].append([f.start, f.end, f.type, f.strand, f.phase])
    
    assert gs.seq2chunk_pos == seq2_chunk_pos_result
    assert gs.chunk2transcripts == chunk2transcripts_result
    assert transcripts == transcripts_result

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

def test_translate_to_one_hot(tmp_path):
    # One exon on + strand, phase=0, from pos2–4
    encoded_labels = np.eye(15, dtype=int)[[0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0]]
    gtf = tmp_path / "test.gtf"
    gtf.write_text(
        "# a comment line\n"
        "chrX\tsrc\tCDS\t4\t11\t.\t+\t0\t; transcript_id \"txA\"; foo;\n"
        "chrX\tsrc\tintron\t12\t15\t.\t+\t.\t; transcript_id \"txA\"; foo;\n"
        "chrX\tsrc\tCDS\t16\t22\t.\t+\t2\t; transcript_id \"txA\"; foo;\n"
    )
    gs = Annotation(file_path=str(gtf), 
        seqnames=["chrX"], seq_lens=[23],
        chunk_len=23)    
    gs.read_inputfile()

    arr = gs.get_onehot(1)
    np.testing.assert_array_equal(arr, encoded_labels)


def test_translate_to_one_hot_minus(tmp_path):
    expected_idx = [0, 7, 5, 6, 4, 5, 6, 10, 3, 3, 3, 3, 13, 6, 4, 5, 6, 4, 5, 14, 0, 0, 0]
    encoded_minus = np.eye(15, dtype=int)[expected_idx]
    gtf = tmp_path / "test.gtf"
    gtf.write_text(
        "# a comment line\n"
        "chrX\tsrc\tCDS\t4\t11\t.\t-\t0\t; transcript_id \"txA\"; foo;\n"
        "chrX\tsrc\tCDS\t16\t22\t.\t-\t2\t; transcript_id \"txA\"; foo;\n"

    )
    gs = Annotation(file_path=str(gtf), 
        seqnames=["chrX"], seq_lens=[23],
        chunk_len=23)    
    gs.read_inputfile()

    arr = gs.get_onehot(0)
    np.testing.assert_array_equal(arr, encoded_minus)
    
