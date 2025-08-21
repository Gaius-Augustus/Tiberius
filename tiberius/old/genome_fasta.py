import numpy as np
import gzip, bz2
import math
import time, sys

one_hot_table = np.zeros((256, 6), dtype=np.int32)
one_hot_table[:, 4] = 1

# Set specific labels for A, C, G, T
one_hot_table[ord('A'), :] = [1, 0, 0, 0, 0, 0]
one_hot_table[ord('C'), :] = [0, 1, 0, 0, 0, 0]
one_hot_table[ord('G'), :] = [0, 0, 1, 0, 0, 0]
one_hot_table[ord('T'), :] = [0, 0, 0, 1, 0, 0]
# Set labels for a, c, g, t with softmasking indicator
one_hot_table[ord('a'), :] = [1, 0, 0, 0, 0, 1]
one_hot_table[ord('c'), :] = [0, 1, 0, 0, 0, 1]
one_hot_table[ord('g'), :] = [0, 0, 1, 0, 0, 1]
one_hot_table[ord('t'), :] = [0, 0, 0, 1, 0, 1]

class GenomeSequences:
    def __init__(self, fasta_file='', genome=None, np_file='', 
                chunksize=20000, overlap=1000, min_seq_len=0):
        """Initialize the GenomeSequences object.

        Arguments:
            fasta_file (str): Path to the FASTA file containing genome sequences.
            np_file (str): Path to the numpy file containing one-hot encoded sequences.
            chunksize (int): Size of each chunk for splitting sequences.
            overlap (int): Overlap size between consecutive chunks.
        """
        self.fasta_file = fasta_file
        self.genome = genome
        self.np_file = np_file
        self.chunksize = chunksize
        self.overlap = overlap
        self.min_seq_len = min_seq_len
        self.sequences = []
        self.sequence_names = []
        self.one_hot_encoded = None
        self.chunks_one_hot = None 
        self.chunks_seq = None
        if self.genome:
            self.extract_seqarray()
        elif self.fasta_file:
            self.read_fasta()
        else:
            self.load_np_array(self.np_file)

    def extract_seqarray(self):
        """Extract the sequence array from the genome object.
        """
        for name, seqrec in self.genome.items():
            if len(seqrec.seq) < self.min_seq_len:
                continue
            self.sequences.append(str(seqrec.seq))
            self.sequence_names.append(name)

    def read_fasta(self):
        """Read genome sequences from a FASTA file, it can be compressed with gz or bz2.
        """
        if self.fasta_file.endswith('.gz'):
            with gzip.open(self.fasta_file, 'rt') as file:
                lines = file.readlines()
        elif self.fasta_file.endswith('.bz2'):
            with bz2.open(self.fasta_file, 'rt') as file:
                lines = file.readlines()
        else:
            with open(self.fasta_file, "r") as file:
                lines = file.readlines()
        current_sequence = ""
        for line in lines:
            if line.startswith(">"):
                if current_sequence:               
                    if len(current_sequence) >= self.min_seq_len:                             
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
            table = np.zeros((256, 6), dtype=np.int32)
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
    
    def prep_seq_chunks(self, min_seq_len=0,):
        seq_names = [seq_n for seq, seq_n in zip(self.sequences, self.sequence_names) \
                    if len(seq)>min_seq_len]
        seqs_lens = [len(seq) for seq in self.sequences \
                        if len(seq)>min_seq_len]
        chunks_seq_plus = []
        chunk_seq_minus = []
        for s in self.sequences:
            if len(s) < min_seq_len:
                continue
            num_chunks = len(s) // self.chunksize
            chunks = [s[i * self.chunksize: \
                        (i+1) * self.chunksize] \
                        for i in range(num_chunks)]
            chunks_seq_plus.extend(chunks)
            # get reverse complement of the chunks
            chunk_seq_minus.extend([self.reverse_complement(chunk) for chunk in chunks])
        self.chunks_seq = chunk_seq_minus + chunks_seq_plus        
    
    def get_onehot(self, i):
        """Get the one-hot encoded representation of a sequence by index.

        Arguments:
            i (int): Index of the sequence to retrieve.
        
        Returns:
            np.array: One-hot encoded representation of the sequence.
        """
        
        # one hot encode the i-th element of self.chunks_seq
        # Create combined lookup table
        
        # Convert the sequence to integer sequence
        int_seq = np.frombuffer(self.chunks_seq[i].encode('ascii'), dtype=np.uint8)
        # Perform one-hot encoding
        return one_hot_table[int_seq]
    
    def reverse_complement(self, sequence):
        """Get the reverse complement of a DNA sequence.

        Arguments:
            sequence (str): The DNA sequence to reverse complement.
        
        Returns:
            str: The reverse complement of the input sequence.
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
                      'N': 'N', 'n': 'n'}
        return ''.join(complement.get(base, base) for base in reversed(sequence))

    def get_flat_chunks(self, sequence_names=None, strand='+', coords=False, pad=True,
                        adapt_chunksize=False, parallel_factor=None):
        """Get flattened chunks of a specific sequence by name.

        Arguments:
            sequence_name (str): Name of the sequence to extract chunks from.
            strand (char): Strand direction ('+' for forward, '-' for reverse).
        
        Returns: 
            chunks_one_hot (np.array): Flattened chunks of the specified sequence.
            chunk_coords (list): List of coordinates of the chunks if coors is True.
            chunksize (int): Possibly reduced size of each chunk.
        """
        chunk_coords = None
        chunksize = self.chunksize

        if not sequence_names: 
            sequence_names = self.sequence_names
        sequences_i = [self.one_hot_encoded[i] for i in sequence_names]

        # if all sequences are shorter than chunksize, reduce the chunksize
        if adapt_chunksize:
            max_len = max([len(seq) for seq in sequences_i])
            if max_len < self.chunksize:
                chunksize = max_len
                if chunksize <= 2*self.overlap:
                    chunksize = min(2*self.overlap + 1, self.chunksize)
                if parallel_factor is None:
                    parallel_factor = 1
                if chunksize < 2 * parallel_factor:
                    chunksize = 2 * parallel_factor # hmm code requires at least 2 chunks
                # new chunksize must divide 2, 9 and parallel_factor
                # round chunksize up to smallest multiple
                divisor = 2 * 9 * parallel_factor // math.gcd(18, parallel_factor)
                chunksize = divisor * (1 + (chunksize - 1) // divisor)

        chunks_one_hot = []
        if coords:
            chunk_coords = []
        for seq_name, sequence in zip(sequence_names, sequences_i):
            num_chunks = (len(sequence) - self.overlap) \
                // (chunksize - self.overlap) + 1
            if num_chunks > 1:
                chunks_one_hot += [sequence[i * (chunksize - self.overlap):\
                    i * (chunksize - self.overlap) + chunksize,:] \
                    for i in range(num_chunks-1)]
            if coords:
                num = num_chunks if pad else num_chunks-1
                chunk_coords += [[
                        seq_name, strand,
                        i * (chunksize - self.overlap)+1, 
                        i * (chunksize - self.overlap) + chunksize] \
                        for i in range(num)]
            
            last_chunksize = (len(sequence) - self.overlap)%(chunksize - self.overlap)
            if pad and last_chunksize > 0:
                padding = np.zeros((chunksize, 6),dtype=np.uint8)
                padding[:,4] = 1
                padding[0:last_chunksize] = sequence[-last_chunksize:]
                chunks_one_hot.append(padding)
            
        chunks_one_hot = np.array(chunks_one_hot)
        if strand == '-':
            chunks_one_hot = chunks_one_hot[::-1, ::-1, [3, 2, 1, 0, 4, 5]]
            if chunk_coords:
                chunk_coords.reverse()

        return chunks_one_hot, chunk_coords, chunksize
