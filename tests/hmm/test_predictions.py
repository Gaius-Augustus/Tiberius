import tiberius
import numpy as np
from learnMSA.msa_hmm.Initializers import ConstantInitializer



def get_inputs() -> tuple[np.ndarray, np.ndarray]:
    
    np.random.seed(5) #make the test deterministic

    nuc = np.zeros((3, 100), dtype=np.int32)
    # single exon
    nuc[0, 17:20] = [0, 3, 2] # ATG
    nuc[0, 23:26] = [3, 0, 2] # TAG
    # exon with intron
    nuc[1, 17:20] = [0, 3, 2] # ATG
    nuc[1, 20:23] = [4, 2, 3] # NGT
    nuc[1, 23:26] = [0, 2, 4] # AGN
    nuc[1, 33:36] = [3, 0, 2] # TAG
    # wrong reading frame
    nuc[2, 17:20] = [0, 3, 2] # ATG
    nuc[2, 22:25] = [3, 0, 2] # TAG

    # one hot encode
    nuc = np.eye(5)[nuc].astype(np.float32)

    # simple input probabilities that slightly hint the expected gene structure
    p = 0.3
    inputs = np.zeros((3, 100, 15), dtype=np.float32) + (1.-p) / 14.

    inputs[0,:17, 0] = p
    inputs[0, 17:26, :4] = 0.1/4
    inputs[0, 17:26, 4:] = 0.9/11
    inputs[0, 26:, 0] = p

    inputs[1,:17, 0] = p
    inputs[1, 17:21, :4] = 0.1/4
    inputs[1, 17:21, 4:] = 0.9/11
    inputs[1, 21:25, 1:4] = 0.1
    inputs[1, 21:25, 4:] = 0.7/10
    inputs[1, 25:36, :4] = 0.1/4
    inputs[1, 25:36, 4:] = 0.9/11
    inputs[1, 36:, 0] = p
    
    inputs[2, :, 0] = p

    return inputs, nuc


def assert_outputs(x : np.ndarray) -> None:
    # make sure the gene structures have been found
    assert np.all(x[0, :17] == 0)
    assert np.all(x[0, 17] == 7)
    assert np.all(x[0, 18:25] == [5,6,4,5,6,4,5])
    assert np.all(x[0, 25] == 14)
    assert np.all(x[0, 26:] == 0)

    assert np.all(x[1, :17] == 0)
    assert np.all(x[1, 17] == 7)
    assert np.all(x[1, 18:20] == [5,6])
    assert np.all(x[1, 20] == 10)
    assert np.all(x[1, 21:25] == [3]*4)
    assert np.all(x[1, 25] == 13)
    assert np.all(x[1, 26:35] == [6,4,5,6,4,5,6,4,5])
    assert np.all(x[1, 35] == 14)
    assert np.all(x[1, 36:] == 0)

    assert np.all(x[2, :] == 0)


def test_nuc_pattern_detection() -> None:
    hmm_layer = tiberius.GenePredHMMLayer( #default parameters for convenience
                                        start_codons=[("ATG", 1.)],
                                        stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
                                        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
                                        intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
                                        # non-default parameters
                                        initial_exon_len=10,
                                        initial_intron_len=10,
                                        initial_ir_len=40, 
                                        # always start intergenic
                                        starting_distribution_init=ConstantInitializer([0] + [-100.]*14))
    
    most_likely_states = hmm_layer.viterbi(*get_inputs())

    assert_outputs(most_likely_states)
