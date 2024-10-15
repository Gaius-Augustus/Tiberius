import tensorflow as tf
import numpy as np
import sys
from learnMSA.msa_hmm.Initializers import ConstantInitializer
from learnMSA.msa_hmm.Utility import deserialize



class SimpleGenePredHMMTransitioner(tf.keras.layers.Layer):
    """ Defines which transitions between HMM states are allowed and how they are initialized.
            Assumed order of states: Ir, I0, I1, I2, E0, E1, E2
    """
    def __init__(self, 
                initial_exon_len=100,
                initial_intron_len=10000,
                initial_ir_len=10000,
                init=None,
                starting_distribution_init="zeros",
                starting_distribution_trainable=True,
                transitions_trainable=True,
                init_component_sd=0.2,
                **kwargs):
        super(SimpleGenePredHMMTransitioner, self).__init__(**kwargs)
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        if not hasattr(self, "num_states"):
            self.num_states = 7
        self.indices = self.make_transition_indices()
        self.starting_distribution_init = starting_distribution_init
        self.starting_distribution_trainable = starting_distribution_trainable
        self.transitions_trainable = transitions_trainable  
        self.num_transitions = len(self.indices)
        self.reverse = False
        if init is None:
            self.init = ConstantInitializer(self.make_transition_init(1, init_component_sd))
        else:
            self.init = init


    def is_intergenic_loop(self, edge):
        return edge[1]==edge[2] and edge[1] == 0


    def is_intron_loop(self, edge, k=1):
        return edge[1]==edge[2] and edge[1] > 0 and edge[1] < 1+3*k


    def is_exon_transition(self, edge, k=1):
        found_any = False
        exon_offset = 1+3*k
        for i in range(k):
            found = edge[2]-exon_offset == (edge[1]-exon_offset+k)%(3*k) and edge[1] >= exon_offset and edge[1] < exon_offset+3*k
            found_any = found_any or found
        return found_any


    def is_exon_1_out_transition(self, edge, k=1):
        return edge[1] >= 1+4*k and edge[1] < 1+5*k and edge[1] != edge[2]


    def is_intergenic_out_transition(self, edge, k=1):
        return edge[1] == 0 and edge[2] != 0


    def cell_init(self, cell):
        """ Automatically called when the owner cell is created.
        """
        pass #nothing to do here in this case
        

    def build(self, input_shape=None):
        if self.built:
            return
        self.transition_kernel = self.add_weight(shape=[1, self.num_transitions],  #just 1 HMM for now
                                        initializer = self.init,
                                        trainable=self.transitions_trainable,
                                        name="transition_kernel")
        self.starting_distribution_kernel = self.add_weight(shape=[1,1,self.num_states], 
                                                           initializer=self.starting_distribution_init, 
                                                           name="starting_distribution_kernel",
                                                           trainable=self.starting_distribution_trainable)
        self.built = True
        

    def recurrent_init(self):
        """ Automatically called before each recurrent run. Should be used for setups that
            are only required once per application of the recurrent layer.
        """
        self.A = self.make_A()
        self.A_transposed = tf.transpose(self.A, (0,2,1))


    def make_A_sparse(self, values=None):
        """Computes state-wise transition probabilities from kernels. Applies a softmax to the kernel values of 
            all outgoing edges of a state.
        Args:
            values: If not None, the values of the sparse tensor are set to this. Otherwise, the kernel is used.
        Returns:
            A dictionary that maps transition types to probabilies. 
        """
        if values is None:
            values = tf.reshape(self.transition_kernel, [-1])
        sparse_kernel =  tf.sparse.SparseTensor(
                                        indices=self.indices,
                                        values=values, 
                                        dense_shape=[1, self.num_states, self.num_states])
        sparse_kernel = tf.sparse.reorder(sparse_kernel) #required for tf.sparse.softmax
        A_sparse = tf.sparse.softmax(sparse_kernel) #ignores implicit zeros
        return A_sparse

    
    def make_A(self):
        return tf.sparse.to_dense(self.make_A_sparse(), default_value=0.)


    def make_log_A(self):
        A_sparse = self.make_A_sparse()
        log_A_sparse = tf.sparse.map_values(tf.math.log, A_sparse)  
        return tf.sparse.to_dense(log_A_sparse, default_value=-1e3)


    def make_initial_distribution(self):
        return tf.nn.softmax(self.starting_distribution_kernel)
        
        
    def call(self, inputs):
        """ 
        Args: 
                inputs: Shape (k, b, q)
        Returns:
                Shape (k, b, q)
        """
        #batch matmul of k inputs with k matricies
        if self.reverse:
            return tf.matmul(inputs, self.A_transposed)
        else:
            return tf.matmul(inputs, self.A)
    

    def get_prior_log_densities(self):
         # Can be used for regularization in the future.
        return 0.
    
    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.transition_kernel.numpy())
        starting_distribution_init = ConstantInitializer(self.starting_distribution_kernel.numpy())
        transitioner_copy = SimpleGenePredHMMTransitioner(init = init,
                                                          initial_exon_len=self.initial_exon_len,
                                                          initial_intron_len=self.initial_intron_len,
                                                          initial_ir_len=self.initial_ir_len,
                                                          starting_distribution_init=starting_distribution_init,
                                                          starting_distribution_trainable=self.starting_distribution_trainable,
                                                          transitions_trainable=self.transitions_trainable) 
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.starting_distribution_kernel = self.starting_distribution_kernel
            transitioner_copy.built = True
        return transitioner_copy
    

    def make_transition_indices(self, model_index=0):
        """ Returns 3D indices (model_index, from_state, to_state) for the kernel of a sparse transition matrix.
            Assumed order of states: Ir, I0, I1, I2, E0, E1, E2
        """
        Ir = 0
        I = list(range(1,4))
        E = list(range(4,7))
        indices = [(Ir, Ir), (Ir, E[0]), (E[2], Ir)]
        for cds in range(3):
            indices.append((E[cds], E[(cds+1)%3]))
            indices.append((E[cds], I[cds]))
            indices.append((I[cds], I[cds]))
            indices.append((I[cds], E[(cds+1)%3]))
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices)==15
        return indices

    def make_transition_init(self, k=1, sd=0.05):
        # use roughly realistic initial length distributions
        init = []
        for edge in self.indices:
            #edge = (model, from, to), ingore model for now
            if self.is_intergenic_loop(edge):  
                p_loop = 1 - 1. / self.initial_ir_len
                init.append(-np.log(1/p_loop-1))
            elif self.is_intron_loop(edge, k):   
                p_loop = 1 - 1. / self.initial_intron_len
                init.append(-np.log(1/p_loop-1))
            elif self.is_exon_transition(edge, k): 
                p_next_exon = 1 - 1. / self.initial_exon_len
                init.append(-np.log(1/p_next_exon-1))
            elif self.is_exon_1_out_transition(edge, k):
                init.append(np.log(1./(2)))
            elif self.is_intergenic_out_transition(edge, k):
                init.append(np.log(1./k) + np.random.normal(0., sd))
            else:
                init.append(0)
        return np.array(init)


    def get_config(self):
        return {"init" : self.init,
                "initial_exon_len": self.initial_exon_len,
                "initial_intron_len": self.initial_intron_len,
                "initial_ir_len": self.initial_ir_len,
                "starting_distribution_init": self.starting_distribution_init,
                "starting_distribution_trainable": self.starting_distribution_trainable,
                "transitions_trainable": self.transitions_trainable}


    @classmethod
    def from_config(cls, config):
        config["init"] = deserialize(config["init"])
        return cls(**config)


                

class GenePredHMMTransitioner(SimpleGenePredHMMTransitioner):
    """ Extends the simple HMM with start- and stop-states that enforce biological structure.
            Assumed order of states: Ir, I0, I1, I2, E0, E1, E2, 
                                    START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    """
    def __init__(self, init_component_sd=0.2, **kwargs):
        if not hasattr(self, "num_states"):
            self.num_states = 15
        if not hasattr(self, "k"):
            self.k = 1
        super(GenePredHMMTransitioner, self).__init__(init_component_sd=init_component_sd, **kwargs)
        self.alpha = self.make_prior_alpha()
    

    def make_transition_indices(self, model_index=0):
        """ Returns 3D indices (model_index, from_state, to_state) for the kernel of a sparse transition matrix.
        """
        Ir = 0
        I = list(range(1,4))
        E = list(range(4,7))
        START = 7
        EI = list(range(8,11))
        IE = list(range(11,14))
        STOP = 14
        indices = [(Ir, Ir), (Ir, START), (STOP, Ir), (START, E[1]), (E[1], STOP)]
        for cds in range(3):
            indices.append((E[cds], E[(cds+1)%3]))
            indices.append((E[cds], EI[cds]))
            indices.append((EI[cds], I[cds]))
            indices.append((I[cds], I[cds]))
            indices.append((I[cds], IE[cds]))
            indices.append((IE[cds], E[cds]))
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices)==23
        return indices


    def gather_binary_probs_for_prior(self, A):
        """ Extracts binary distributions from a transition matrix that are used in the prior.
        """
        #self loop states
        m = 1 + 3*self.k
        diag = tf.linalg.diag_part(A[:m, :m])
        probs_ir_intron = tf.stack([diag, tf.reduce_sum(A[:m, :], axis=-1) - diag], axis=1)
        probs_exon = []
        #exons
        for i in range(3):
            for j in range(self.k):
                e = 1+(i+3)*self.k+j
                next_e = 1+3*self.k+((i+1)%3)*self.k+j
                probs_exon.extend([A[e, next_e], tf.reduce_sum(A[e, :]) - A[e, next_e]])
        probs_exon = tf.stack(probs_exon, axis=0)
        probs_exon = tf.reshape(probs_exon, (3*self.k, 2))
        probs = tf.concat([probs_ir_intron, probs_exon], axis=0)
        return probs


    def make_prior_alpha(self, n=1e6):
        #assume number of prior draws
        #we choose alpha according to the expect times we see each transition
        #higher values make the prior more strict
        p0 = self.init(shape=(1, self.num_transitions))
        A0_sparse = self.make_A_sparse(values=tf.reshape(p0, [-1]))
        A0 = tf.sparse.to_dense(A0_sparse, default_value=0.)[0]
        return self.gather_binary_probs_for_prior(A0) * n

    
    def get_prior_log_densities(self):
        # Regularizes the transition probabilities based on the values given as initial distribution.
        # The dirichlet parameters are chosen based on n prior draws from the initial distribution.
        self.binary_probs = self.gather_binary_probs_for_prior(self.A[0])
        log_p = tf.math.log(self.binary_probs)
        priors = tf.reduce_sum((self.alpha-1) * log_p, axis=-1)
        return {i : priors[i] for i in range(1+6*self.k)}
    

    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.transition_kernel.numpy())
        starting_distribution_init = ConstantInitializer(self.starting_distribution_kernel.numpy())
        transitioner_copy = GenePredHMMTransitioner(init = init,
                                                    initial_exon_len=self.initial_exon_len,
                                                    initial_intron_len=self.initial_intron_len,
                                                    initial_ir_len=self.initial_ir_len,
                                                    starting_distribution_init=starting_distribution_init,
                                                    starting_distribution_trainable=self.starting_distribution_trainable,
                                                    transitions_trainable=self.transitions_trainable)
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.starting_distribution_kernel = self.starting_distribution_kernel
            transitioner_copy.built = True
        return transitioner_copy


class GenePredMultiHMMTransitioner(GenePredHMMTransitioner):
    """ The same as GenePredHMMTransitioner, but with multiple (sub-)HMMs that share the same architecture, but 
        not parameters. 
        The same order of states as GenePredHMMTransitioner except that each state other than Ir is multiplied k times: 
        Ir, I0*k, I1*k, I2*k, E0*k, E1*k, E2*k, START*k, EI0*k, EI1*k, EI2*k, IE0*k, IE1*k, IE2*k, STOP*k
            Args:
                k: number of sub-HMMs.
                init_component_sd: standard deviation of the noise used to initialize the transition IR -> components
    """
    def __init__(self, k, init_component_sd=0.2, **kwargs):
        self.k = k
        self.num_states = 1 + 14 * k
        super(GenePredMultiHMMTransitioner, self).__init__(init_component_sd=init_component_sd, **kwargs)
        self.init = ConstantInitializer(self.make_transition_init(k, init_component_sd))
    

    def make_transition_indices(self, model_index=0):
        """ Returns 3D indices (model_index, from_state, to_state) for the kernel of a sparse transition matrix.
        """
        Ir = 0
        I = list(range(1,1+3*self.k))
        E = list(range(1+3*self.k,1+6*self.k))
        START = list(range(1+6*self.k, 1+7*self.k))
        EI = list(range(1+7*self.k, 1+10*self.k))
        IE = list(range(1+10*self.k, 1+13*self.k))
        STOP = list(range(1+13*self.k, 1+14*self.k))
        indices = [(Ir, Ir)]
        for hmm in range(self.k): 
            indices.extend([(Ir, START[hmm]), (STOP[hmm], Ir), 
                            (START[hmm], E[self.k+hmm]), (E[self.k+hmm], STOP[hmm])])
            for cds in range(3):
                indices.extend([(E[self.k*cds+hmm], E[self.k*((cds+1)%3)+hmm]), 
                                (E[self.k*cds+hmm], EI[self.k*cds+hmm]), 
                                (EI[self.k*cds+hmm], I[self.k*cds+hmm]), 
                                (I[self.k*cds+hmm], I[self.k*cds+hmm]), 
                                (I[self.k*cds+hmm], IE[self.k*cds+hmm]), 
                                (IE[self.k*cds+hmm], E[self.k*cds+hmm])])
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices)==1+22*self.k
        return indices
    

    def duplicate(self, model_indices=None, share_kernels=False):
        init = ConstantInitializer(self.transition_kernel.numpy())
        starting_distribution_init = ConstantInitializer(self.starting_distribution_kernel.numpy())
        transitioner_copy = GenePredMultiHMMTransitioner(k = self.k,
                                                    init = init,
                                                    initial_exon_len=self.initial_exon_len,
                                                    initial_intron_len=self.initial_intron_len,
                                                    initial_ir_len=self.initial_ir_len,
                                                    starting_distribution_init=self.starting_distribution_init,
                                                    starting_distribution_trainable=self.starting_distribution_trainable,
                                                    transitions_trainable=self.transitions_trainable) 
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.starting_distribution_kernel = self.starting_distribution_kernel
            transitioner_copy.built = True
        return transitioner_copy


    def get_config(self):
        config = super(GenePredMultiHMMTransitioner, self).get_config()
        config.update(
            {"k": self.k})
        return config

tf.keras.utils.get_custom_objects()["SimpleGenePredHMMTransitioner"] = SimpleGenePredHMMTransitioner
tf.keras.utils.get_custom_objects()["GenePredHMMTransitioner"] = GenePredHMMTransitioner
tf.keras.utils.get_custom_objects()["GenePredMultiHMMTransitioner"] = GenePredMultiHMMTransitioner