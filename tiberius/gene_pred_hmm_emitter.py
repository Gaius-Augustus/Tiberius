import tensorflow as tf
import numpy as np
import sys
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.Utility import deserialize
from learnMSA.protein_language_models.MvnMixture import MvnMixture, DefaultDiagBijector
from tiberius import kmer



class SimpleGenePredHMMEmitter(tf.keras.layers.Layer):
    """ Defines the emission probabilities for a gene prediction HMM 
        with embeddings or class predictions as inputs.
        Args:
            num_models: The number of semi-independent gene models (see GenePredHMMLayer).
            num_copies: The number of gene model copies in one HMM that share the IR state.
            init: The initializer for the emission probabilities.
            share_intron_parameters: If True, the intron states share the same emission parameters.
    """
    def __init__(self, 
                num_models=1,
                num_copies=1,
                init=ConstantInitializer(0.), 
                trainable_emissions=True,
                share_intron_parameters=True,
                **kwargs):
        super(SimpleGenePredHMMEmitter, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_copies = num_copies
        self.num_states = 1+6*num_copies
        self.init = init
        self.trainable_emissions = trainable_emissions
        self.share_intron_parameters = share_intron_parameters


        
    def build(self, input_shape):
        if self.built:
            return
        s = input_shape[-1] 
        self.emission_kernel = self.add_weight(
                                        shape=[self.num_models, 
                                               self.num_states-2*self.num_copies*int(self.share_intron_parameters), 
                                               s], 
                                        initializer=self.init, 
                                        trainable=self.trainable_emissions,
                                        name="emission_kernel")
        self.built = True
        


    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.B = self.make_B()


        
    def make_B(self):
        """ Constructs the emission probabilities from the emission kernel.
        """
        return tf.nn.softmax(self.emission_kernel)
        


    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (num_models, batch_size, length, alphabet_size) 
                end_hints: DEPRECATED.
        Returns:
                A tensor with emission probabilities of shape (num_models, batch_size, length, num_states).
        """
        emit = tf.einsum("...s,kqs->k...q", inputs[0], self.B) #remove unsed model dimension of inputs 
        if self.share_intron_parameters:
            emit = tf.concat([emit[..., :1+self.num_copies]] + [emit[..., 1:1+self.num_copies]]*2 + [emit[..., 1+self.num_copies:]], axis=-1)
        return emit
    

    def get_prior_log_density(self):
        # Can be used for regularization in the future.
        return [[0.]]

    
    def get_aux_loss(self):
        # Can define an auxiliary loss here. Currently unused.
        return 0.


    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.emission_kernel.numpy())
        emitter_copy = SimpleGenePredHMMEmitter(num_models=self.num_models,
                                                num_copies=self.num_copies,
                                                init = init, 
                                                trainable_emissions=self.trainable_emissions,
                                                share_intron_parameters=self.share_intron_parameters) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.built = True
        return emitter_copy


    def get_config(self):
        return {"num_models": self.num_models,
                "num_copies": self.num_copies,
                "init": self.init, 
                "trainable_emissions": self.trainable_emissions,
                "share_intron_parameters": self.share_intron_parameters}
    

    @classmethod
    def from_config(cls, config):
        config["init"] = deserialize(config["init"])
        return cls(**config)



def assert_codons(codons):
    assert sum(p for _,p in codons) == 1, "Codon probabilities must sum to 1, got: " + str(codons) 
    for triplet, prob in codons:
        assert len(triplet) == 3, "Triplets must be of length 3, got: " + str(codons) 
        assert prob >= 0 and prob <= 1, "Probabilities must be between 0 and 1, got: " + str(codons) 


def make_codon_probs(codons, pivot_left):
    assert_codons(codons)
    codon_probs = sum(prob * kmer.encode_kmer_string(triplet, pivot_left) for triplet, prob in codons) #(16,4)
    codon_probs = tf.reshape(codon_probs, (64,))
    return codon_probs[tf.newaxis, tf.newaxis, :] #(1,1,64)




class GenePredHMMEmitter(SimpleGenePredHMMEmitter):
    """ Extends the simple HMM with start- and stop-states that enforce biological structure.
    """
    def __init__(self, 
                start_codons,
                stop_codons,
                intron_begin_pattern,
                intron_end_pattern,
                nucleotide_kernel_init=ConstantInitializer(0.),
                trainable_nucleotides_at_exons=False,
                **kwargs):
        """ Args:
                start_codons: The allowed start codons. A list of pairs. The first element of each pair is a string that is a triplet over the alphabet ACGTN. 
                            The second entry is the probability of that triplet. All probabilities must sum to 1.
                stop_codons: The allowed stop codons. Same format as `start_codons`.
                intron_begin_pattern: The allowed start patterns in introns. Same format as `start_codons`. 
                            Since only the first 2 nucleotides of an intron are relevant, give 3-mers of the form "N.." where N indicates 
                            that any nucleotide that comes last in the previous exon is allowed.
                intron_end_pattern: The allowed end patterns in introns. Same format as `start_codons`.
                            Since only the last 2 nucleotides of an intron are relevant, give 3-mers of the form "..N" where N indicates
                            that any nucleotide that comes first in the next exon is allowed.
                nucleotide_kernel_init: Initializer for the kernel closely related to the nucleotide distributions at exon states.
                trainable_nucleotides_at_exons: If true, nucleotides at exon states are trainable.
        """
        super(GenePredHMMEmitter, self).__init__(**kwargs)
        self.num_states = 1+14*self.num_copies
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.nucleotide_kernel_init = nucleotide_kernel_init
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        # emission probabilities of start and stop codons / begin and end patterns
        self.start_codon_probs = make_codon_probs(start_codons, pivot_left=True) #START state
        self.stop_codon_probs = make_codon_probs(stop_codons, pivot_left=False) #STOP state
        self.intron_begin_codon_probs = make_codon_probs(intron_begin_pattern, pivot_left=True) #intron begin state
        self.intron_end_codon_probs = make_codon_probs(intron_end_pattern, pivot_left=False) #intron end state
        self.any_codon_probs = make_codon_probs([("NNN", 1.)], pivot_left=False) #any other state except E2 and EI1
        self.not_stop_codon_probs = self.any_codon_probs * tf.cast(self.stop_codon_probs == 0, dtype=self.stop_codon_probs.dtype) #any other state 
        self.not_stop_codon_probs /= tf.reduce_sum(self.not_stop_codon_probs) #sum to 1 over all codons except stop codons
        # order of states: (Ir, I0, I1, I2, E0, E1), E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
        # (the first 6 states are omitted because they impose no codon restrictions)
        self.left_codon_probs = tf.concat([self.any_codon_probs]
                                        + [self.start_codon_probs]
                                        + [self.intron_begin_codon_probs]*3
                                        + [self.any_codon_probs]*4, axis=1)
        self.right_codon_probs = tf.concat([self.not_stop_codon_probs]
                                        + [self.any_codon_probs]*2
                                        + [self.not_stop_codon_probs]
                                        + [self.any_codon_probs]
                                        + [self.intron_end_codon_probs]*3
                                        + [self.stop_codon_probs], axis=1)
        self.codon_probs = tf.concat([self.left_codon_probs, self.right_codon_probs], axis=0) #(2,num_states,64)
        
        
    def build(self, input_shape):
        if self.built:
            return
        super(GenePredHMMEmitter, self).build(input_shape)
        s = input_shape[-1] 
        if self.trainable_nucleotides_at_exons:
            assert self.num_models == 1, "Trainable nucleotide emissions are currently only supported for one model."
            self.nuc_emission_kernel = self.add_weight(
                                            shape=[self.num_models, 3*self.num_copies, 4], 
                                            initializer=self.nucleotide_kernel_init, 
                                            name="nuc_emission_kernel")
        
        
    def get_nucleotide_probs(self):
        return tf.nn.softmax(self.nuc_emission_kernel)

        
    def call(self, inputs, end_hints=None, training=False):
        """ 
        Args: 
                inputs: A tensor of shape (num_models, batch, length, alphabet_size + 5) 
                end_hints: DEPRECATED.
        Returns:
                A tensor with emission probabilities of shape (num_models, batch, length, num_states).
        """
        nucleotides = inputs[..., -5:]
        inputs = inputs[..., :-5]
        emit = super(GenePredHMMEmitter, self).call(inputs, training=training)

        # compute probabilities to start the first exon or introns 
        # as well as the probabilities to end the last exon or introns
        num_models, batch, length = tf.unstack(tf.shape(nucleotides)[:3])
        nucleotides = tf.reshape(nucleotides, [-1, length, 5])
        left_3mers = kmer.make_k_mers(nucleotides, k=3, pivot_left=True)
        left_3mers = tf.reshape(left_3mers, [num_models, batch, length, 64]) 
        right_3mers = kmer.make_k_mers(nucleotides, k=3, pivot_left=False)
        right_3mers = tf.reshape(right_3mers, [num_models, batch, length, 64])
        input_3mers = tf.stack([left_3mers, right_3mers], axis=-2) #(num_models, batch, length, 2, 64)
        codon_emission_probs = tf.einsum("k...rs,rqs->k...rq", input_3mers, self.codon_probs)
        codon_emission_probs = tf.reduce_prod(codon_emission_probs, axis=-2)

        if self.num_copies > 1:
            repeats = [self.num_copies]*codon_emission_probs.shape[-1]
            codon_emission_probs = tf.repeat(codon_emission_probs, repeats=repeats, axis=-1)
        codon_emission_probs = tf.concat([tf.ones_like(codon_emission_probs[...,:(1+5*self.num_copies)])/4096., codon_emission_probs], axis=-1)
        
        if training:
            codon_emission_probs += 1e-7
        
        full_emission = emit * codon_emission_probs
        
        if self.trainable_nucleotides_at_exons:
            nucleotides = inputs[..., -5:]
            nucleotides_no_N = nucleotides[...,:4] + nucleotides[..., 4:]/4
            nuc_emission_probs = tf.einsum("k...s,kqs->k...q", nucleotides_no_N, self.get_nucleotide_probs())
            nuc_emission_probs = tf.concat([tf.ones_like(full_emission[..., :1+3*self.num_copies])/4., 
                                            nuc_emission_probs, 
                                            tf.ones_like(full_emission[..., 1+6*self.num_copies:])/4.], axis=-1)
            full_emission *= nuc_emission_probs

        return full_emission


    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.emission_kernel.numpy())
        if self.trainable_nucleotides_at_exons:
            nucleotide_kernel_init = ConstantInitializer(self.nuc_emission_kernel.numpy())
        else:
            nucleotide_kernel_init = "zeros"
        emitter_copy = GenePredHMMEmitter(self.start_codons,
                                         self.stop_codons,
                                         self.intron_begin_pattern,
                                         self.intron_end_pattern,
                                         num_models=self.num_models,
                                         num_copies=self.num_copies,
                                         init = init, 
                                         trainable_emissions=self.trainable_emissions,
                                         share_intron_parameters=self.share_intron_parameters,
                                         nucleotide_kernel_init=nucleotide_kernel_init,
                                         trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons) 
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            if self.trainable_nucleotides_at_exons:
                emitter_copy.nuc_emission_kernel = self.nuc_emission_kernel
            emitter_copy.built = True
        return emitter_copy


    def get_config(self):
        config = super(GenePredHMMEmitter, self).get_config()
        config.update(
            {"start_codons": self.start_codons, 
            "stop_codons": self.stop_codons, 
            "intron_begin_pattern": self.intron_begin_pattern, 
            "intron_end_pattern": self.intron_end_pattern,
            "nucleotide_kernel_init": self.nucleotide_kernel_init,
            "trainable_nucleotides_at_exons": self.trainable_nucleotides_at_exons})
        return config
    

    @classmethod
    def from_config(cls, config):
        config["init"] = deserialize(config["init"])
        config["nucleotide_kernel_init"] = deserialize(config["nucleotide_kernel_init"])
        return cls(**config)



tf.keras.utils.get_custom_objects()["GenePredHMMEmitter"] = GenePredHMMEmitter
tf.keras.utils.get_custom_objects()["SimpleGenePredHMMEmitter"] = SimpleGenePredHMMEmitter
