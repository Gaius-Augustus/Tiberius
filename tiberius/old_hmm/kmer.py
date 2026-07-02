import tensorflow as tf

def make_k_mers(sequences, k, pivot_left=True):
    """ Maps one hot encoded nucleotide sequences to a k-mer representation.
        Args:
            sequences: A tensor of shape (b, L, 5) representing sequences of length L.
                        Assumes that the last dimension is one-hot encoded with "N" corresponding to the last position.
            k: An integer specifying the length of the k-mer
            pivot_left: A boolean specifying whether to pivot the k-mer to the left or right.
        Returns:
            A tensor of shape (b, L, 4**k-1, 4). If pivot_left is True, the last dimension corresponds
            to the 4 possible nucleotides in the leftmost position of the k-mer.
            Otherwise, the last dimension corresponds to the rightmost position in the k-mer.
            If the k-mer contains N, this is expressed equiprobably among the regular 4 nucleotides possible
            at that position.
    """
    L = tf.shape(sequences)[-2]
    n = tf.shape(sequences)[-1]-1 #alphabet size is the number of characters minus 1 (N)
    n = tf.cast(n, dtype=sequences.dtype)
    # uniform distribution over alphabet in case of N
    sequences_no_N = sequences[..., :-1]
    N_pos = tf.cast(sequences[..., -1:] == 1, dtype=sequences.dtype)
    sequences_no_N += (1/n) * N_pos
    # compute a padding for kmers that range over the sequence boundaries
    pad = tf.ones_like(sequences_no_N[:, :k-1, :], dtype=sequences.dtype) / n
    if pivot_left:
        sequences_padded_no_N = tf.concat([sequences_no_N, pad], axis=-2)
        k_mers = sequences_padded_no_N[:, :L, tf.newaxis, :]
    else:
        sequences_padded_no_N = tf.concat([pad, sequences_no_N], axis=-2)
        k_mers = sequences_padded_no_N[:, k-1:L+k-1, tf.newaxis, :]
    for i in range(1, k) if pivot_left else range(k-2, -1, -1):
        shift_i = sequences_padded_no_N[:, i:L+i, tf.newaxis, :, tf.newaxis]
        k_mers = k_mers[..., tf.newaxis, :] * shift_i
        shape = [4**i, 4] if pivot_left else [4**(k-i-1), 4]
        k_mers = tf.reshape(k_mers, tf.concat([tf.shape(k_mers)[:-3], shape], axis=0))
    return k_mers



def encode_kmer_string(kmer : str, pivot_left=True, alphabet="ACGT"):
    """ Converts a k-mer to classes in the format (i,j) with i < n^{k-1} and j < n where n is the alphabet size.
        E.g. AAA -> (0,0), AAT -> (3,0), TAA -> (0,3) if pivot_left is True, otherwise
             AAA -> (0,0), AAT -> (0,3), TAA -> (12, 0)
        The output is a one-hot encoding of these classes in case of A,C,G,T.
        If the k-mer contains N, this is expressed equiprobably among the regular 4 nucleotides.
    """
    k = len(kmer)
    alphabet_with_unknown = alphabet + "N"
    kmer = [alphabet_with_unknown.index(x) for x in kmer]
    kmer = tf.constant(kmer)
    one_hot = tf.one_hot(kmer, len(alphabet_with_unknown))
    encoded_kmers = make_k_mers(one_hot[tf.newaxis, ...], k=k, pivot_left=pivot_left)
    if pivot_left:
        return tf.squeeze(encoded_kmers)[0]
    else:
        return tf.squeeze(encoded_kmers)[-1]


