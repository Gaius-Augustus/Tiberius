import tensorflow as tf
import numpy as np
from learnMSA.msa_hmm.MsaHmmCell import HmmCell
# replace this import later, when learnMSA renames "MsaHmmLayer" properly to "HmmLayer"
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer as HmmLayer
from learnMSA.msa_hmm.Viterbi import viterbi
from learnMSA.msa_hmm.Utility import deserialize
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from gene_pred_hmm_emitter import SimpleGenePredHMMEmitter, GenePredHMMEmitter
from gene_pred_hmm_transitioner import SimpleGenePredHMMTransitioner, GenePredHMMTransitioner, GenePredMultiHMMTransitioner
    



class GenePredHMMLayer(HmmLayer):
    """A layer that implements a gene prediction HMM.
    Args:
        num_models: The number of semi-independent HMMs. Currently, the HMMs share architecture and transition parameters, 
                    but allow independent emission parameters for the gene structure classes output by previous layers.
        num_copies: The number of gene model copies that share an IR state.
        start_codons: The allowed start codons. A list of pairs. The first element of each pair is a string that is a triplet over the alphabet ACGTN. 
                    The second entry is the probability of that triplet. All probabilities must sum to 1.
        stop_codons: The allowed stop codons. Same format as `start_codons`.
        intron_begin_pattern: The allowed start patterns in introns. Same format as `start_codons`. 
                    Since only the first 2 nucleotides of an intron are relevant, give 3-mers of the form "N.." where N indicates 
                    that any nucleotide that comes last in the previous exon is allowed.
        intron_end_pattern: The allowed end patterns in introns. Same format as `start_codons`.
                    Since only the last 2 nucleotides of an intron are relevant, give 3-mers of the form "..N" where N indicates
                    that any nucleotide that comes first in the next exon is allowed.
        initial_exon_len: The initial expected length of exons.
        initial_intron_len: The initial expected length of introns.
        initial_ir_len: The initial expected length of intergenic regions.
        emitter_init: The initializer for the class emission parameters. 
        starting_distribution_init: The initializer for the starting distribution.
        starting_distribution_trainable: Whether the starting distribution should be trainable.
        emit_embeddings: Whether to emit embeddings of the inputs as output by a language model.
        embedding_dim: The dimensionality of the embeddings. Must be specified if and only if emit_embeddings=True.
        full_covariance: Whether to use a full covariance matrix for the emission distribution. False is recommended too keep the parameter count low.
        embedding_kernel_init: The initializer for the embedding kernel. Only relevant if emit_embeddings=True.
        initial_variance: The initial variance of each embedding dimension. Only relevant if emit_embeddings=True.
        temperature: A temperature parameter to regulate the mvn pdf.
        share_intron_parameters: Whether to share the emission parameters of the intron states. This does currently not affect transitions.
        simple: Whether to use a simple HMM with only 3 states (IR, I, E) instead of the full 7 state HMM (use only for backwards compatibility).
        disable_metrics: Disabled additional metrics added during training.
        parallel_factor: The number of chunks the input is split into to process them in parallel. Must devide the length of the input.
                        Increases speed and GPU utilization but also memory usage.
    """
    def __init__(self, 
                num_models=1,
                num_copies=1,
                start_codons=[("ATG", 1.)],
                stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
                intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
                intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
                initial_exon_len=100,
                initial_intron_len=10000,
                initial_ir_len=10000,
                emitter_init=ConstantInitializer(0.),
                starting_distribution_init="zeros",
                trainable_emissions=True,
                trainable_transitions=True,
                trainable_starting_distribution=True,
                trainable_nucleotides_at_exons=False,
                emit_embeddings = False,
                embedding_dim=None, 
                full_covariance=False,
                embedding_kernel_init="random_normal",
                initial_variance=1.,
                temperature=100.,
                share_intron_parameters=False,
                simple=False,
                variance_l2_lambda=0.01,
                disable_metrics=True,
                parallel_factor=1,
                use_border_hints=True,
                **kwargs):
        self.num_models = num_models
        self.num_copies = num_copies
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.emitter_init = emitter_init
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        self.starting_distribution_init = starting_distribution_init
        self.trainable_emissions = trainable_emissions
        self.trainable_transitions = trainable_transitions
        self.trainable_starting_distribution = trainable_starting_distribution
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        self.emit_embeddings = emit_embeddings
        self.embedding_dim = embedding_dim
        self.full_covariance = full_covariance
        self.embedding_kernel_init = embedding_kernel_init
        self.initial_variance = initial_variance
        self.temperature = temperature
        self.share_intron_parameters = share_intron_parameters
        self.simple = simple
        self.variance_l2_lambda = variance_l2_lambda
        self.disable_metrics = disable_metrics
        self.parallel_factor = parallel_factor
        self.use_border_hints = use_border_hints
        if use_border_hints:
            super(GenePredHMMLayer, self).__init__(cell=None, use_prior=False, num_seqs=1e6, **kwargs)
        else:
            super(GenePredHMMLayer, self).__init__(cell=None, use_prior=False, num_seqs=1e6, parallel_factor=parallel_factor, **kwargs)


    def build(self, input_shape):
        if self.built:
            return
        if self.simple:
            emitter = SimpleGenePredHMMEmitter(init=self.emitter_init,
                                                trainable_emissions=self.trainable_emissions,
                                                emit_embeddings=self.emit_embeddings,
                                                embedding_dim=self.embedding_dim,
                                                full_covariance=self.full_covariance,
                                                embedding_kernel_init=self.embedding_kernel_init,
                                                initial_variance=self.initial_variance,
                                                temperature=self.temperature,
                                                share_intron_parameters=self.share_intron_parameters),
            transitioner = SimpleGenePredHMMTransitioner(initial_exon_len=self.initial_exon_len,
                                                        initial_intron_len=self.initial_intron_len,
                                                        initial_ir_len=self.initial_ir_len,
                                                        starting_distribution_init=self.starting_distribution_init,
                                                        starting_distribution_trainable=self.trainable_starting_distribution,
                                                        transitions_trainable=self.trainable_transitions)
        else:
            emitter = GenePredHMMEmitter(start_codons=self.start_codons,
                                        stop_codons=self.stop_codons,
                                        intron_begin_pattern=self.intron_begin_pattern,
                                        intron_end_pattern=self.intron_end_pattern,
                                        l2_lambda=self.variance_l2_lambda,
                                        num_models=self.num_models,
                                        num_copies=self.num_copies,
                                        init=self.emitter_init,
                                        trainable_emissions=self.trainable_emissions,
                                        emit_embeddings=self.emit_embeddings,
                                        embedding_dim=self.embedding_dim,
                                        full_covariance=self.full_covariance,
                                        embedding_kernel_init=self.embedding_kernel_init,
                                        initial_variance=self.initial_variance,
                                        temperature=self.temperature,
                                        share_intron_parameters=self.share_intron_parameters,
                                        trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons)
            #note that for num_models>1, the transition parameters are shared between the models,
            #the argument num_models is currently just to get the correct shapes
            if self.num_copies == 1:
                transitioner = GenePredHMMTransitioner(num_models=self.num_models,
                                                        initial_exon_len=self.initial_exon_len,
                                                        initial_intron_len=self.initial_intron_len,
                                                        initial_ir_len=self.initial_ir_len,
                                                        starting_distribution_init=self.starting_distribution_init,
                                                        starting_distribution_trainable=self.trainable_starting_distribution,
                                                        transitions_trainable=self.trainable_transitions)
            else:
                transitioner = GenePredMultiHMMTransitioner(k=self.num_copies,
                                                        num_models=self.num_models,
                                                        initial_exon_len=self.initial_exon_len,
                                                        initial_intron_len=self.initial_intron_len,
                                                        initial_ir_len=self.initial_ir_len,
                                                        starting_distribution_init=self.starting_distribution_init,
                                                        starting_distribution_trainable=self.trainable_starting_distribution,
                                                        transitions_trainable=self.trainable_transitions)
        # configure the cell
        self.cell = HmmCell([emitter.num_states]*self.num_models,
                            dim=input_shape[-1],
                            emitter=emitter, 
                            transitioner=transitioner,
                            use_fake_step_counter=True,
                            name="gene_pred_hmm_cell")
        super(GenePredHMMLayer, self).build(input_shape)


    def concat_inputs(self, inputs, nucleotides, embeddings=None):
        assert nucleotides is not None
        inputs = tf.expand_dims(inputs, 0)
        nucleotides = tf.expand_dims(nucleotides, 0)
        input_list = [inputs, nucleotides]
        if self.emit_embeddings:
            assert embeddings is not None
            embeddings = tf.expand_dims(embeddings, 0)
            input_list.insert(1, embeddings)
        stacked_inputs = tf.concat(input_list, axis=-1)
        return stacked_inputs


    def call(self, inputs, nucleotides=None, embeddings=None, end_hints=None, training=False, use_loglik=True):
        """ 
        Computes the state posterior log-probabilities.
        Args: 
                inputs: Shape (batch, len, alphabet_size)
                nucleotides: Shape (batch, len, 5) one-hot encoded nucleotides with N in the last position.
                embeddings: Shape (batch, len, dim) embeddings of the inputs as output by a language model.
                end_hints: A tensor of shape (batch, 2, num_states) that contains the correct state for the left and right ends of each chunk.
        Returns:
                State posterior log-probabilities (without loglik if use_loglik is False). The order of the states is Ir, I0, I1, I2, E0, E1, E2.
                Shape (batch, len, number_of_states) if num_models=1 and (batch, len, num_models, number_of_states) if num_models>1.
        """ 
        #batch matmul of k inputs with k matricies
        if end_hints is not None:
            end_hints = tf.expand_dims(end_hints, 0)
        if self.simple:
            log_post, prior, _ = self.state_posterior_log_probs(tf.expand_dims(inputs, 0), return_prior=True, end_hints=end_hints, training=training, no_loglik=not use_loglik)
        else:
            stacked_inputs = self.concat_inputs(inputs, nucleotides, embeddings)
            log_post, prior, _ = self.state_posterior_log_probs(stacked_inputs, end_hints=end_hints, return_prior=True, training=training, no_loglik=not use_loglik) 
        if training:
            prior = tf.reduce_mean(prior)
            self.add_loss(prior)
            self.add_metric(prior, "prior") #deprecated in tf 2.17
        return log_post[0] if self.num_models == 1 else tf.transpose(log_post, [1,2,0,3])


    def viterbi(self, inputs, nucleotides=None, embeddings=None, end_hints=None):
        """ 
        Computes the most likely state sequence.
        Args: 
                inputs: Shape (batch, len, alphabet_size)
                nucleotides: Shape (batch, len, 5) one-hot encoded nucleotides with N in the last position.
                embeddings: Shape (batch, len, dim) embeddings of the inputs as output by a language model.
        Returns:
                Most likely state sequence of shape (batch, len) if num_models=1 and (batch, len, num_models) if num_models>1.
        """
        self.cell.recurrent_init()
        if self.simple:
            viterbi_seq = viterbi(tf.expand_dims(inputs, 0), self.cell, parallel_factor=self.parallel_factor)
        else:
            stacked_inputs = self.concat_inputs(inputs, nucleotides, embeddings)
            viterbi_seq = viterbi(stacked_inputs, self.cell, end_hints=end_hints, parallel_factor=self.parallel_factor)
        return viterbi_seq[0] if self.num_models == 1 else tf.transpose(viterbi_seq, [1,2,0])


    def get_config(self):
        return {"num_models": self.num_models,
                "num_copies": self.num_copies,
                "start_codons": self.start_codons,
                "stop_codons": self.stop_codons,
                "intron_begin_pattern": self.intron_begin_pattern,
                "intron_end_pattern": self.intron_end_pattern,
                "initial_exon_len": self.initial_exon_len,
                "initial_intron_len": self.initial_intron_len,
                "initial_ir_len": self.initial_ir_len,
                "emitter_init": self.emitter_init,
                "starting_distribution_init": self.starting_distribution_init,
                "trainable_emissions": self.trainable_emissions,
                "trainable_transitions": self.trainable_transitions,
                "trainable_starting_distribution": self.trainable_starting_distribution,
                "trainable_nucleotides_at_exons": self.trainable_nucleotides_at_exons,
                "emit_embeddings": self.emit_embeddings,
                "embedding_dim": self.embedding_dim,
                "full_covariance": self.full_covariance,
                "embedding_kernel_init": self.embedding_kernel_init,
                "initial_variance": self.initial_variance,
                "temperature": self.temperature,
                "share_intron_parameters": self.share_intron_parameters,
                "simple": self.simple,
                "variance_l2_lambda": self.variance_l2_lambda,
                "use_border_hints": self.use_border_hints,
                "parallel_factor" : self.parallel_factor}


    # tell tensorflow that some config items have been renamed
    # required to load older models
    @classmethod
    def from_config(cls, config):
        if "starting_distribution_trainable" in config:
            config["trainable_starting_distribution"] = config["starting_distribution_trainable"]
            del config["starting_distribution_trainable"]
        config["emitter_init"] = deserialize(config["emitter_init"])
        return cls(**config)


#inputs: IR, I, E
class3_emission_matrix_simple = np.array([[[1., 0., 0.]] + [[0., 1., 0.]]*3 + [[0., 0., 1.]]*3]) * 100. #shape (1,7,3)
class3_emission_matrix = np.array([[[1., 0., 0.]] + [[0., 1., 0.]]*3 + [[0., 0., 1.]]*11]) * 100. #shape (1,15,3)

def make_5_class_emission_kernel(smoothing=0.01, introns_shared=False, num_copies=1):
    # input classes: IR, I, E0, E1, E2
    # states: Ir, I0, I1, I2, E0, E1, E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    # Returns: shape (1,1 + num_copies*(14 - 2*introns_shared),5) 
    assert smoothing > 0, "Smoothing can not be exactly zero to prevent numerical issues."
    n = 5
    if introns_shared:
        expected_classes_per_state = np.array([0,1,2,3,4,2,3,4,2,4,2,3,4])
    else:
        expected_classes_per_state = np.array([0,1,1,1,2,3,4,2,3,4,2,4,2,3,4])
    probs = np.eye(n)[expected_classes_per_state]
    probs += -probs * smoothing + (1-probs)*smoothing/(n-1)
    if num_copies > 1:
        repeats = [1] + [num_copies]*(probs.shape[-2]-1)
        probs = np.repeat(probs, repeats, axis=-2)
    return np.log(probs[np.newaxis, ...])

def make_15_class_emission_kernel(smoothing=0.1, num_copies=1):
    # input classes: IR, I, E0, E1, E2
    # states: Ir, I0, I1, I2, E0, E1, E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    # Returns: shape (1,1 + num_copies*(14 - 2*introns_shared),15) 
    assert smoothing > 0, "Smoothing can not be exactly zero to prevent numerical issues."
    n = 15
    probs = np.eye(n)
    probs += -probs * smoothing + (1-probs)*smoothing/(n-1)
    if num_copies > 1:
        repeats = [1] + [num_copies]*(probs.shape[-2]-1)
        probs = np.repeat(probs, repeats, axis=-2)
    return np.log(probs[np.newaxis, ...])


#a matrix that can be multiplied to the state posterior probabilities of the full model 
# with 1+14*k many states
# input states Ir, I0*k, I1*k, I2*k, E0*k, E1*k, E2*k, START*k, EI0*k, EI1*k, EI2*k, IE0*k, IE1*k, IE2*k, STOP*k
# output states IR, I, E0, E1, E2
def make_aggregation_matrix(k=1):
    A = np.zeros((1+14*k, 5), dtype=np.float32)

    A[0, 0] = 1              #intergenic
    A[1:1+3*k, 1] = 1        #introns

    A[1+3*k:1+4*k, 2] = 1    #exon 0
    A[1+6*k:1+7*k, 2] = 1    #start
    A[1+9*k:1+10*k, 2] = 1   #EI2
    A[1+11*k:1+12*k, 2] = 1  #IE1

    A[1+4*k:1+5*k, 3] = 1    #exon 1
    A[1+7*k:1+8*k, 3] = 1    #EI0
    A[1+12*k:1+13*k, 3] = 1  #IE2

    A[1+5*k:1+6*k, 4] = 1    #exon 2
    A[1+13*k:1+14*k, 4] = 1    #stop
    A[1+8*k:1+9*k, 4] = 1    #EI1
    A[1+10*k:1+11*k, 4] = 1  #IE0

    return A


tf.keras.utils.get_custom_objects()["GenePredHMMLayer"] = GenePredHMMLayer
