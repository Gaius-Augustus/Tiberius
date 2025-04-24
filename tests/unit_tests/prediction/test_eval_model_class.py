import tiberius
import numpy as np


def test_reduce_labels() -> None:
    state_list = [0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0]
    state_reduced_list = [0,0,0,2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,0]

    dummy_predictionGTF = tiberius.PredictionGTF()
    np.testing.assert_array_equal(dummy_predictionGTF.reduce_label(state_list, num_hmm=1), state_reduced_list)


def test_get_ranges() -> None:
    encoded_labels = [0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0]
    
    # Expected output is derived as follows:
    # - Indices 0 to 2: label 0 → "intergenic" → [0, 2]
    # - Indices 3 to 5: label 1 → "intron" → [3, 5]
    # - Indices 6 to 8: label 2 → "CDS"      → [6, 8]
    # - Indices 9 to 10: label 1 → "intron"  → [9, 10]
    # - Indices 11 to 12: label 0 → "intergenic" → [11, 12]
    expected_ranges = [
        ["intergenic", 35, 37],
        ["CDS", 38, 45],
        ["intron", 46, 49],
        ["CDS", 50, 56],
        ["intergenic", 57, 57]
    ]
    
    # Create an instance of your PredictionGTF class.
    dummy_predictionGTF = tiberius.PredictionGTF()
    
    # Call get_ranges with the sample encoded labels (using the default offset of 0)
    result_ranges = dummy_predictionGTF.get_ranges(encoded_labels, offset=35)
    # Assert that the computed ranges exactly match the expected ranges.
    assert result_ranges == expected_ranges

def test_merge_empty_all_tx() -> None:
    """When there are no existing transcripts, the method should return only the new predictions."""
    pred = tiberius.PredictionGTF()
    all_tx = []
    new_tx = [[['CDS', 100, 200]]]
    bpoint = 150
    expected = new_tx
    assert pred.merge_re_prediction(all_tx, new_tx, bpoint) == expected


def test_merge_empty_new_tx() -> None:
    """When there are no new transcripts, the method should return only the existing predictions."""
    pred = tiberius.PredictionGTF()
    all_tx = [[['CDS', 300, 400]]]
    new_tx = []
    bpoint = 350
    expected = all_tx
    assert pred.merge_re_prediction(all_tx, new_tx, bpoint) == expected

def test_merge_no_overlap() -> None:
    """Test when the transcript from all_tx ends before the transcript from new_tx starts.
    
    In this situation, the method should concatenate all transcripts.
    """
    pred = tiberius.PredictionGTF()
    all_tx = [[['CDS', 100, 150]]]
    new_tx = [[['CDS', 160, 200]]]
    bpoint = 120  # falls within the all_tx transcript
    expected = [[['CDS', 100, 150]], [['CDS', 160, 200]]]
    assert pred.merge_re_prediction(all_tx, new_tx, bpoint) == expected


def test_merge_overlap_new_tx_larger() -> None:
    """Test when transcripts overlap and the new_tx version is longer.
    
    In this case, the new_tx transcript should be retained.
    """
    pred = tiberius.PredictionGTF()
    all_tx = [[['CDS', 100, 150]]]   # length = 50
    new_tx = [[['CDS', 100, 160]]]      # length = 60 (larger)
    bpoint = 120
    expected = [[['CDS', 100, 160]]]
    assert pred.merge_re_prediction(all_tx, new_tx, bpoint) == expected


def test_merge_multiple_transcripts() -> None:
    """Test a scenario with multiple transcripts where some transcripts overlap and some do not."""
    pred = tiberius.PredictionGTF()
    all_tx = [
        [['CDS', 100, 150]],
        [['CDS', 200, 250]]
    ]
    new_tx = [
        [['CDS', 100, 150]],
        [['CDS', 200, 245]],  # overlaps with all_tx[1] (length 45 vs. 50)
        [['CDS', 260, 300]]
    ]
    bpoint = 210  # falls in the region of the second transcript
    # For all_tx, overlap index becomes 1; for new_tx, overlap index becomes 1.
    # Since all_tx[1] is larger than new_tx[1], we keep the all_tx transcript.
    # The returned merged list should include both transcripts from all_tx plus the remaining transcript from new_tx.
    expected = [
        [['CDS', 100, 150]],
        [['CDS', 200, 250]],
        [['CDS', 260, 300]]
    ]
    assert pred.merge_re_prediction(all_tx, new_tx, bpoint) == expected


def test_create_gtf() -> None:
    hmm_label = np.array([[0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0],
                 [0,0,0,7,5,6,4,5,14,0,0,0,0,0,0,0,0,0,7,5,6,4,5]])
    coords = [["chr1", '+', 250, 272], ["chr2", '+', 1, 22]]
    dummy_nuc = np.zeros((2,22,6))    
    cds_coords_out = [[[253, 260], [265, 271]], [[4, 9]]]

    dummy_predictionGTF = tiberius.PredictionGTF()
    anno, tx_id = dummy_predictionGTF.create_gtf(y_label=hmm_label, 
            coords=coords, f_chunks=dummy_nuc, out_file='out.gtf', filt=False,
            strand='+')

    txs = []
    for tx in anno.transcripts.values():
        txs.append(tx.get_type_coords('CDS', frame=False))            
    assert txs == cds_coords_out

def test_create_gtf_rev() -> None:
    hmm_label = np.array([[0,0,0,7,5,6,4,5,6,4,8,1,1,1,1,11,4,5,6,4,5,14,0],
                 [0,0,0,7,5,6,4,5,14,0,0,0,0,0,0,0,0,0,7,5,6,4,5]])
    coords = [["chr1", '+', 250, 272], ["chr2", '+', 1, 22]]
    dummy_nuc = np.zeros((2,22,6))    
    cds_coords_out = [[[15, 20]],  [[251, 257], [262, 269]]]

    dummy_predictionGTF = tiberius.PredictionGTF()
    anno, tx_id = dummy_predictionGTF.create_gtf(y_label=hmm_label, 
            coords=coords, f_chunks=dummy_nuc, out_file='out.gtf', filt=False,
            strand='-')

    txs = []
    for tx in anno.transcripts.values():
        txs.append(tx.get_type_coords('CDS', frame=False))
    assert txs == cds_coords_out