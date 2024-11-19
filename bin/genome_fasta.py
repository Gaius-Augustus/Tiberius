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
