import os
import sys
sys.path.insert(0, "../../learnMSA")
sys.path.insert(0, "../../programs/learnMSA")
# don't use GPU for tests
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import numpy as np
import tensorflow as tf
import kmer
import gene_pred_hmm
import gene_pred_hmm_transitioner
from learnMSA.msa_hmm.Initializers import ConstantInitializer


class TestKmer(unittest.TestCase):


    def test_encode_kmer(self):
        a = kmer.encode_kmer_string("AAA")
        self.assertEqual(a.shape, (16, 4))
        self.assertEqual(np.sum(a), 1)
        self.assertEqual(a[0,0], 1)
        a2 = kmer.encode_kmer_string("AAA", pivot_left=False)
        self.assertEqual(a2.shape, (16, 4))
        self.assertEqual(np.sum(a2), 1)
        self.assertEqual(a2[0,0], 1)

        b = kmer.encode_kmer_string("ACG")
        self.assertEqual(b.shape, (16, 4))
        self.assertEqual(np.sum(b), 1)
        self.assertEqual(b[6,0], 1)
        b2 = kmer.encode_kmer_string("ACG", pivot_left=False)
        self.assertEqual(b2.shape, (16, 4))
        self.assertEqual(np.sum(b2), 1)
        self.assertEqual(b2[4,2], 1)

        c = kmer.encode_kmer_string("NAA")
        self.assertEqual(c.shape, (16, 4))
        self.assertEqual(np.sum(c), 1)
        for i in range(4):
            self.assertEqual(c[0,i], 0.25)
        c2 = kmer.encode_kmer_string("NAA", pivot_left=False)
        self.assertEqual(c2.shape, (16, 4))
        self.assertEqual(np.sum(c2), 1)
        for i in range(4):
            self.assertEqual(c2[i,0], 0.25)

        d = kmer.encode_kmer_string("ANA")
        self.assertEqual(d.shape, (16, 4))
        self.assertEqual(np.sum(d), 1)
        for i in range(4):
            self.assertEqual(d[i*4,0], 0.25)
        d2 = kmer.encode_kmer_string("ANA", pivot_left=False)
        self.assertEqual(d2.shape, (16, 4))
        self.assertEqual(np.sum(d2), 1)
        for i in range(4):
            self.assertEqual(d2[i*4,0], 0.25)



    def test_3mers(self):
        
        # ACGT
        example_sequence = tf.constant([[[1., 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]])

        expected_3mers_left = ["ACG", "CGT", "GTN", "TNN"]
        k_mers_left = kmer.make_k_mers(example_sequence, k=3, pivot_left=True)
        for i, k in enumerate(expected_3mers_left):
            np.testing.assert_equal(k_mers_left[0,i].numpy(), 
                                    kmer.encode_kmer_string(k).numpy())

        expected_3mers_right = ["NNA", "NAC", "ACG", "CGT"]
        k_mers_right = kmer.make_k_mers(example_sequence, k=3, pivot_left=False)
        for i, k in enumerate(expected_3mers_right):
            np.testing.assert_equal(k_mers_right[0,i].numpy(), 
                                    kmer.encode_kmer_string(k, pivot_left=False).numpy())


    def test_unknown_symbols(self):
        # ACNT
        example_sequence = tf.constant([[[1., 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]])

        expected_3mers_left = ["ACN", "CNT", "NTN", "TNN"]
        k_mers_left = kmer.make_k_mers(example_sequence, k=3, pivot_left=True)
        for i, k in enumerate(expected_3mers_left):
            np.testing.assert_equal(k_mers_left[0,i].numpy(), 
                                    kmer.encode_kmer_string(k).numpy())

        expected_3mers_right = ["NNA", "NAC", "ACN", "CNT"]
        k_mers_right = kmer.make_k_mers(example_sequence, k=3, pivot_left=False)
        for i, k in enumerate(expected_3mers_right):
            np.testing.assert_equal(k_mers_right[0,i].numpy(), 
                                    kmer.encode_kmer_string(k, pivot_left=False).numpy())


# missing model file, need to fix this test
# class TestInputs(unittest.TestCase):
#     def test_inputs(self):
#         from eval_model_class import PredictionGTF

#         batch_size = 12
#         seq_len = 99999
#         strand = '+'

#         inp_data_dir = '../test_data/Panthera_pardus/inp/'
#         out_dir = '../test_data/unittest_workdir/'
#         if not os.path.exists(out_dir):
#             os.mkdir(out_dir)

#         model_path = f'../test_data/model_lstm.h5'
#         genome_path = f'{inp_data_dir}/genome.fa'
#         annot_path= f'{inp_data_dir}/annot.gtf'
#         temp_dir = f'{out_dir}/temp_model/'

#         # load input data and model
#         pred_gtf = PredictionGTF( 
#         #     model_path='/home/jovyan/brain/deepl_data//exclude_primates/weights/train2/train6/1/epoch_02',
#             model_path_lstm=model_path, 
#             seq_len=seq_len, batch_size=batch_size, hmm=True, temp_dir=temp_dir)
#         pred_gtf.load_model()

#         f_chunks, r_chunks, coords = pred_gtf.load_inp_data(
#             genome_path=genome_path,
#             annot_path=annot_path, 
#             overlap_size=0, strand=strand, chunk_coords=True
#         )

#         np.testing.assert_equal(np.sum(f_chunks[..., :5], -1), 1.)


        
class TestInitialization(unittest.TestCase):

    def test_hmm_init(self):
        trans = gene_pred_hmm_transitioner.GenePredHMMTransitioner(
                initial_ir_len=4,
                initial_intron_len=5,
                initial_exon_len=6,
                init_component_sd=0.)
        expected = {
            (0,0): 1-1/4,
            (0,7): 1/4,
            (1,1): 1-1/5,
            (1,11): 1/5,
            (2,2): 1-1/5,
            (2,12): 1/5,
            (3,3): 1-1/5,
            (3,13): 1/5,
            (4,5): 1-1/6,
            (4,8): 1/6,
            (5,6): 1-1/6,
            (5,9): 1/12,
            (5,14): 1/12,
            (6,4): 1-1/6,
            (6,10): 1/6,
            (7,5): 1,
            (8,1): 1,
            (9,2): 1,
            (10,3): 1,
            (11,4): 1,
            (12,5): 1,
            (13,6): 1,
            (14,0): 1,
        }
        trans.build()
        A = trans.make_A()[0]
        for i,(_,u,v) in enumerate(trans.indices):
            np.testing.assert_almost_equal(A[u,v], expected[(u,v)], err_msg=f"edge {(u,v)}")
        #the same as above
        trans2 = gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(
                k=1,
                initial_ir_len=4,
                initial_intron_len=5,
                initial_exon_len=6,
                init_component_sd=0.)
        trans2.build()
        A2 = trans2.make_A()[0]
        for i,(_,u,v) in enumerate(trans2.indices):
            np.testing.assert_almost_equal(A2[u,v], expected[(u,v)], err_msg=f"edge {(u,v)}")


    def test_multi_hmm_init(self):
        trans = gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(
                k=2,
                initial_ir_len=4,
                initial_intron_len=5,
                initial_exon_len=6,
                init_component_sd=0.)
        expected = {
            (0,0): 1-1/4,
            (0,13): 1/8,
            (0,14): 1/8,
            (1,1): 1-1/5,
            (1,21): 1/5,
            (2,2): 1-1/5,
            (2,22): 1/5,
            (3,3): 1-1/5,
            (3,23): 1/5,
            (4,4): 1-1/5,
            (4,24): 1/5,
            (5,5): 1-1/5,
            (5,25): 1/5,
            (6,6): 1-1/5,
            (6,26): 1/5,
            (7,9): 1-1/6,
            (7,15): 1/6,
            (8,10): 1-1/6,
            (8,16): 1/6,
            (9,11): 1-1/6,
            (9,17): 1/12,
            (9,27): 1/12,
            (10,12): 1-1/6,
            (10,18): 1/12,
            (10,28): 1/12,
            (11,7): 1-1/6,
            (11,19): 1/6,
            (12,8): 1-1/6,
            (12,20): 1/6,
            (13,9): 1,
            (14,10): 1,
            (15,1): 1,
            (16,2): 1,
            (17,3): 1,
            (18,4): 1,
            (19,5): 1,
            (20,6): 1,
            (21,7): 1,
            (22,8): 1,
            (23,9): 1,
            (24,10): 1,
            (25,11): 1,
            (26,12): 1,
            (27,0): 1,
            (28,0): 1
        }
        trans.build()
        A = trans.make_A()[0]
        for i,(_,u,v) in enumerate(trans.indices):
            np.testing.assert_almost_equal(A[u,v], expected[(u,v)], err_msg=f"edge {(u,v)}")


class TestMultiHMM(unittest.TestCase):

    def test_multi_copy_transitions(self):
        trans = gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(k = 2)
        indices = trans.make_transition_indices()
        ref_indices = [ [0,0], 
                        [0, 13], [0, 14], [13, 9], [14, 10], #START in / out
                        [27, 0], [28,0], [9, 27], [10, 28], #STOP in / out
                        [9, 11], [10, 12], [9, 17], [10, 18], [17, 3], [18, 4],
                        [3,3], [4,4], [3,23], [4,24], [23,9], [24,10], #E1 + Intron 1

                        [11, 7], [12, 8], [11, 19], [12, 20], [19, 5], [20, 6],
                        [5,5], [6,6], [5,25], [6,26], [25,11], [26,12], #E2 + Intron 2

                        [7, 9], [8, 10], [7, 15], [8, 16], [15, 1], [16, 2],
                        [1,1], [2,2], [1,21], [2,22], [21,7], [22,8], #E0 + Intron 0
                         ]
        #check existence of bijection between ref_indices and indices
        self.assertEqual(len(indices), len(ref_indices))
        for pair in ref_indices:
            self.assertTrue(np.any(np.all(indices == [0]+pair, -1)), msg=f"{pair} for in {indices[:,1:]}")


    # def test_multi_model_transitions(self):
    #     for num_models in [2,3,5]: 
    #         trans = gene_pred_hmm_transitioner.GenePredMultiHMMTransitioner(num_models = num_models)
    #         indices = trans.make_transition_indices()
    #         ref_indices = [ [0,0], 
    #                         [0, 7], [7, 5], #START in / out
    #                         [14, 0], [5, 14], #STOP in / out
    #                         [5, 6], [5,9], [9,2], [2,2], [2,12], [12,5], #E1 + Intron 1
    #                         [6,4],[6,10], [10,3], [3,3], [3,13], [13,6], #E2 + Intron 2
    #                         [4,5], [7,11], [8,1], [1,1], [1,11], [11,4] #E0 + Intron 0
    #                         ]
    #         num_transitions = len(ref_indices)
    #         #check existence of bijection between num_models copies of ref_indices and indices
    #         self.assertEqual(len(indices), num_transitions*num_models)
    #         i = 0
    #         for j in range(num_models):
    #             for pair in ref_indices:
    #                 self.assertTrue(np.any(np.all(indices == [j]+pair, -1)), msg=f"{pair} for in {indices[:,1:]}")


    def test_multi_model_layer(self):
        for num_models in [2,3,5]:
            hmm_layer = gene_pred_hmm.GenePredHMMLayer(num_models = num_models)
            hmm_layer.build([None, None, 15])
            #check if the number of parameters is correct
            self.assertEqual(hmm_layer.cell.transitioner.transition_kernel.shape, (1, 23))
            self.assertEqual(hmm_layer.cell.transitioner.starting_distribution_kernel.shape, (1, 1, 15))
            self.assertEqual(hmm_layer.cell.emitter[0].emission_kernel.shape, (num_models, 15, 15))


    def test_multi_model_parameter_noise(self):
        n = 2
        kernel_init = gene_pred_hmm.make_15_class_emission_kernel(smoothing=0.01, num_models=n, noise_strength=0.001)
        hmm_layer = gene_pred_hmm.GenePredHMMLayer(num_models=n, emitter_init=ConstantInitializer(kernel_init))
        np.random.seed(77) #make the test deterministic
        hmm_layer.build([None, None, 15])
        #check if emission matrices are different...
        emission_probs = tf.nn.softmax(hmm_layer.cell.emitter[0].emission_kernel).numpy()
        #kernels are different...
        self.assertFalse(np.all(emission_probs[0] == emission_probs[1]))
        #...but not too different
        # note that difference could be larger than noise_strength due to rescaling
        self.assertLess(np.max(np.abs(emission_probs[0] - emission_probs[1])), 0.01)


    def test_multi_model_algorithms(self):
        for num_models in [1,2]:
            hmm_layer = gene_pred_hmm.GenePredHMMLayer(num_models = num_models)
            hmm_layer.build([None, None, 15])
            inputs = np.random.rand(5, 100, 15).astype(np.float32)
            inputs /= np.sum(inputs, -1, keepdims=True)
            nuc = np.random.rand(5, 100, 5).astype(np.float32)
            posterior = hmm_layer(inputs, nuc)
            viterbi = hmm_layer.viterbi(inputs, nuc)
            #check shapes, omit model dimension if num_models = 1 for compatibility 
            self.assertEqual(posterior.shape, (5, 100, 15) if num_models == 1 else (5, 100, num_models, 15))
            self.assertEqual(viterbi.shape, (5, 100) if num_models == 1 else (5, 100, num_models))


class TestUtility(unittest.TestCase):

    def test_output_aggregation_single_model(self):
        # 15 probabilities that sum to 1
        probs = np.array([0.2, #IR
                          0.0, 0.04, 0.06, #introns
                          0.15, 0.15, 0.1, #exons
                          0.1, #start
                          0.0,0.05,0.05, #EI
                          0.05,0.0,0.05, #IE
                          0.0]) #stop
        probs_agg_ref = np.array([0.2, 0.1, 0.3, 0.2, 0.2])  
        A = gene_pred_hmm.make_aggregation_matrix(k=1)
        probs_agg = np.matmul(probs[np.newaxis], A)
        np.testing.assert_almost_equal(np.sum(probs_agg, -1), 1.)
        np.testing.assert_almost_equal(probs_agg[0], probs_agg_ref)

    def test_output_aggregation_multi_model(self):
        # 1 + 28 (k=2) probabilities that sum to 1
        probs = np.array([0.2, #IR
                            0, 0.01, 0.02, 0.01, 0.04, 0.02, #introns 
                            0.08, 0.07, 0.14, 0.01, 0.0, 0.1, #exons
                            0.08, 0.02, #start
                            0, 0, 0.01, 0.04, 0.02, 0.03, #EI
                            0.05, 0, 0, 0, 0.02, 0.03, #IE
                            0.0, 0.0]) #stop
        probs_agg_ref = np.array([0.2, 0.1, 0.3, 0.2, 0.2])  
        A = gene_pred_hmm.make_aggregation_matrix(k=2)
        probs_agg = np.matmul(probs[np.newaxis], A)
        np.testing.assert_almost_equal(np.sum(probs_agg, -1), 1.)
        np.testing.assert_almost_equal(probs_agg[0], probs_agg_ref)
