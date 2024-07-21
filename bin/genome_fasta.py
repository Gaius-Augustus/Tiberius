import numpy as np

class GenomeSequences:
    def __init__(self, fasta_file='', np_file='', chunksize=20000, overlap=1000):
        """Initialize the GenomeSequences object.

        Arguments:
            fasta_file (str): Path to the FASTA file containing genome sequences.
            np_file (str): Path to the numpy file containing one-hot encoded sequences.
            chunksize (int): Size of each chunk for splitting sequences.
            overlap (int): Overlap size between consecutive chunks.
        """
        self.fasta_file = fasta_file
        self.np_file = np_file
        self.chunksize = chunksize
        self.overlap = overlap

        self.sequences = []
        self.sequence_names = []
        self.one_hot_encoded = None
        self.chunks_one_hot = None 
        self.chunks_seq = None
        if self.fasta_file:
            self.read_fasta()        
        else:
            self.load_np_array(self.np_file)
        #self.encode_sequences()

    def read_fasta(self):
        """Read genome sequences from the specified FASTA file.
        """
        
        with open(self.fasta_file, "r") as file:
            lines = file.readlines()
            current_sequence = ""
            for line in lines:
                if line.startswith(">"):
                    if current_sequence:                    
                        self.sequences.append(current_sequence)
                        current_sequence = ""
                    self.sequence_names.append(line[1:].strip())
                else:
                    current_sequence += line.strip()
            self.sequences.append(current_sequence)

    def encode_sequences(self, seq=None):
        """One-hot encode the sequences and store in self.one_hot_encoded.
        """
        if not seq:
            seq = self.sequence_names
    
        # One-hot encode the sequences
        self.one_hot_encoded = {}
        
        for s in seq:       
            sequence = self.sequences[self.sequence_names.index(s)]
            # Create combined lookup table
            table = np.zeros((256, 6), dtype=np.uint8)
            table[:, 4] = 1
            
            # Set specific labels for A, C, G, T
            table[ord('A'), :] = [1, 0, 0, 0, 0, 0]
            table[ord('C'), :] = [0, 1, 0, 0, 0, 0]
            table[ord('G'), :] = [0, 0, 1, 0, 0, 0]
            table[ord('T'), :] = [0, 0, 0, 1, 0, 0]
            # Set labels for a, c, g, t with softmasking indicator
            table[ord('a'), :] = [1, 0, 0, 0, 0, 1]
            table[ord('c'), :] = [0, 1, 0, 0, 0, 1]
            table[ord('g'), :] = [0, 0, 1, 0, 0, 1]
            table[ord('t'), :] = [0, 0, 0, 1, 0, 1]
            
            # Convert the sequence to integer sequence
            int_seq = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
            # Perform one-hot encoding
            self.one_hot_encoded[s] = table[int_seq]

#     def save_to_file(self, filename):
#         """Save the one-hot encoded sequences to a numpy file.

#         Arguments:
#             filename (str): Name of the numpy file to save the data.
#         """

#         self.np_file = filename
#         np.save(filename, self.one_hot_encoded)

#     def load_np_array(self):
#         """Load one-hot encoded sequences from a numpy file.
#         """
#         self.one_hot_encoded = np.load(self.np_file)

#     def create_chunks_one_hot(self):
#         """Create overlapping chunks of one-hot encoded sequences.

#         Returns:
#             List of Numpy array of one hot encoded chunks (one Numpy array per sequence)
#         """
#         self.chunks_one_hot = []
#         for sequence in self.one_hot_encoded:
#             self.chunks_one_hot.append([])
#             num_chunks = (len(sequence) - self.overlap) \
#                 // (self.chunksize - self.overlap) + 1
#             self.chunks_one_hot[-1] = np.array([sequence[i * (self.chunksize - self.overlap):\
#                 i * (self.chunksize - self.overlap) + self.chunksize,:] \
#                 for i in range(num_chunks-1)])
#         return self.chunks_one_hot 
    
#     def create_chunks_seq(self):
#         """Create overlapping chunks of original sequences.

#         Returns:
#             List of Numpy array of chunks (one Numpy array per sequence)
#         """
#         self.chunks_seq = []
#         for sequence in self.sequences:
#             self.chunks_seq.append([])
#             num_chunks = (len(sequence) - self.overlap) \
#                 // (self.chunksize - self.overlap) + 1
#             self.chunks_seq[-1] = np.array([sequence[i * (self.chunksize - self.overlap):\
#                 i * (self.chunksize - self.overlap) + self.chunksize] \
#                 for i in range(num_chunks-1)])
#         return self.chunks_seq

    def get_flat_chunks(self, sequence_name=None, strand='+', coords=False, pad=True):
        """Get flattened chunks of a specific sequence by name.

        Arguments:
            sequence_name (str): Name of the sequence to extract chunks from.
            strand (char): Strand direction ('+' for forward, '-' for reverse).
        
        Returns: 
            chunks_one_hot (np.array): Flattened chunks of the specified sequence.
        """

        if not sequence_name: 
            sequence_name = self.sequence_names
        sequences_i = [self.one_hot_encoded[i] for i in sequence_name]            
        
        chunks_one_hot = []        
        chunk_coords = []
        for seq_name, sequence in zip(sequence_name, sequences_i):
            num_chunks = (len(sequence) - self.overlap) \
                // (self.chunksize - self.overlap) + 1
            if num_chunks > 1:
                chunks_one_hot += [sequence[i * (self.chunksize - self.overlap):\
                    i * (self.chunksize - self.overlap) + self.chunksize,:] \
                    for i in range(num_chunks-1)]
            if coords:
                num = num_chunks if pad else num_chunks-1
                chunk_coords += [[
                        seq_name, strand,
                        i * (self.chunksize - self.overlap)+1, 
                        i * (self.chunksize - self.overlap) + self.chunksize] \
                        for i in range(num)]                
            
            last_chunksize = (len(sequence) - self.overlap)%(self.chunksize - self.overlap)
            if pad and last_chunksize > 0:
                padding = np.zeros((self.chunksize, 6),dtype=np.uint8)
                padding[:,4] = 1
                padding[0:last_chunksize] = sequence[-last_chunksize:]
                chunks_one_hot.append(padding)
            
        chunks_one_hot = np.array(chunks_one_hot)
        if strand == '-':
            chunks_one_hot = chunks_one_hot[::-1, ::-1, [3, 2, 1, 0, 4, 5]]
            chunk_coords.reverse()
        if coords:
            return chunks_one_hot, chunk_coords
        return chunks_one_hot

#     def get_flat_chunks_padding(self, strand='+'):
#         """Get flattened chunks for all sequences. Padd all chunks to the same size.
#         Chunks at the end of the sequence are padded with the end of the previous chunk.
#         Sequences smaller than the chunk size are padded with zeros.

#         Arguments:
#             strand (char): Strand direction ('+' for forward, '-' for reverse).
        
#         Returns: 
#             chunks_one_hot (np.array): Flattened chunks of all sequences.
#         """
#         chunks = []
#         chunk_size = self.chunksize
#         for sequence in self.one_hot_encoded:            
#             seq_length = len(sequence)
#             if chunk_size > seq_length:
                
#                 chunks.append(np.zeros((chunk_size, sequence.shape[-1]), sequence.dtype))
                
#                 chunks[-1][-seq_length:] = sequence
#             else:
#                 num_chunks = seq_length // chunk_size + 1
#                 chunks.extend([sequence[i * self.chunksize:(i+1) * self.chunksize, :] \
#                     for i in range(num_chunks-1)])

#                 if not seq_length % chunk_size == 0:
#                     chunks.append(sequence[seq_length - self.chunksize:])
#         chunks = np.array(chunks)
#         if strand == '-':
#             chunks = chunks[::-1, ::-1, [3, 2, 1, 0, 4, 5]]
#         return chunks

    def one_hot2seq(self, one_hot_encoding):
        """Translate one hot encoded sequence to str

        Arguments:
            one_hot_encoding (np array): One hot encoding of A,C,G,T, softmasking

        Returns:
            sequences (list(str)): List of sequences as strings
        """
        # Mapping for each position in one-hot encoding
        int_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        
        # Use numpy's argmax to get the position of the 1 in the one-hot encoding
        indices = np.argmax(one_hot_encoding, axis=-1)
        
        # Convert indices back to characters
        sequences = ''.join([int_to_char[i] for i in indices])

        return sequences

    # deprecated
#     def generate_auto_chunks_one_hot(self, max_chunksize, strand='+'):
#         """Generate one-hot encoded chunks for each sequence with variable chunk sizes.

#         max_chunksize: Maximum chunk size for each sequence.
#         return: List of lists of one-hot encoded chunks for each sequence.
#         """
#         auto_chunks_one_hot = []

#         for sequence in self.one_hot_encoded:
#             seq_length = len(sequence)

#             for chunk_size in range(max_chunksize+1, 9, -1):
#                 if chunk_size % 9 == 0  and seq_length % chunk_size == 0:
#                     break
#             print('CHUNK', chunk_size, seq_length)
                
#             num_chunks = seq_length // chunk_size
#             chunks = np.array_split(sequence, num_chunks)
#             auto_chunks_one_hot.append(chunks)
#         if strand == '-':
#             chunks_one_hot = chunks_one_hot[::-1, ::-1, [3, 2, 1, 0, 4, 5]]
#         return auto_chunks_one_hot

#     # deprecated
#     def get_chunks_in_range(self, seq, start, end):
#         # !!!! overlap is not implemented, doesn't work for overlap>0
#         chunks = [self.sequences[i * self.chunksize:(i+1) * self.chunksize] \
#             for i in range(int(start/self.chunksize),int(end/self.chunksize)+1)]
        
#         table = np.zeros((256, 5), dtype=np.uint8)
#         table[ord('A'), 0] = 1
#         table[ord('a'), 0] = 1
#         table[ord('C'), 1] = 1
#         table[ord('c'), 1] = 1
#         table[ord('G'), 2] = 1
#         table[ord('g'), 2] = 1
#         table[ord('T'), 3] = 1
#         table[ord('t'), 3] = 1
#         table[ord('a'):ord('z')+1, 4] = 1
#         return np.array([np.frombuffer(c.encode('ascii'), dtype=np.uint8) for c in chunks]), start%self.chunksize-1, end%self.chunksize-1
        