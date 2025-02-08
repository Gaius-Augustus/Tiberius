# ==============================================================
# Authors: Lars Gabriel
#
# Class handling GTF information to generate trainings examples
# ==============================================================

import numpy as np
import sys 

class GeneStructure:
    """Handles gene structure information from a gtf file, 
    prepares one-hot encoded trainings examples"""

    def __init__(self, filename='',  np_file='', chunksize=20000, overlap=1000):
        """Initialize GeneStructure.

        Arguments:
            filename (str): Path to GTF file.
            np_file (str): Path to save/load numpy array.
            chunksize (int): Size of each chunk.
            overlap (int): Overlapping bases in chunks."""
        self.filename = filename
        self.np_file = np_file
        self.chunksize = chunksize
        self.overlap = overlap

        self.gene_structures = []

        # one hot encoding for each sequence (intergenic [0], CDS [1], intron [2])
        self.one_hot = None
        # one hot encoding for phase of CDS (0 [0], 1 [1], 2 [2], non coding [3])
        self.one_hot_phase = None

        # chunks of one hot encoded numpy array fitted to genomic chunks
        self.chunks = None
        self.chunks_phase = None
        
        if self.filename:
            self.read_gtf(self.filename)
        else:
            self.load_np_array(self.np_file)

    def read_gtf(self, filename):
        """Read gene structure information from a GTF file.

        Arguments:
            filename (str): Path to GTF file."""
        with open(filename, 'r') as f:
            for line in f:
                # Skip comments and header lines
                if line.startswith('#') or line.startswith('track'):
                    continue
                
                # Parse the GTF line
                line_parts = line.strip().split('\t')
                
                # Extract the gene structure information
                chromosome = line_parts[0]
                feature = line_parts[2]
                strand =  line_parts[6]
                phase = line_parts[7]
                
                start = int(line_parts[3])
                end = int(line_parts[4])
                
                # Store the gene structure information
                self.gene_structures.append((chromosome, feature, strand, phase, start, end))
        
        # Sort the gene structures by chromosome and end position
        self.gene_structures.sort(key=lambda x: (x[0], x[5]))
    
    def save_to_file(self, filename):
        """Save one-hot encoding to a numpy file.

        Arguments:
            filename (str): Path to save numpy array."""
        self.np_file = filename
        np.save(filename, self.one_hot)

    def load_np_array(self):
        """Load one-hot encoding from a numpy file."""
        self.one_hot = np.load(self.np_file)
        
    def translate_to_one_hot_hmm(self, sequence_names, sequence_lengths, transition=False):
        """Translate gene structure information to one-hot encoding.
            7 classes IR, I0, I1, I2, E0, E1, E2

        Arguments:
            sequence_names (list): Names of sequences.
            sequence_lengths (list): Lengths of sequences."""
        
        self.one_hot = {}

        numb_labels = 7
        if transition:
            numb_labels = 15
            
        # Initialize a numpy array to store the one-hot encoded positions
        for strand in ['+', '-']:
            self.one_hot[strand] = {seq : np.zeros((seq_l, numb_labels), dtype=np.int8) \
                for seq, seq_l in zip(sequence_names, sequence_lengths)}
            for seq in sequence_names:
                self.one_hot[strand][seq][:, 0] =  1                

        # Set the one-hot encoded positions for each gene structure
        for chromosome, feature, strand, phase, start, end in self.gene_structures:    
            if chromosome not in sequence_names:
                continue
            if chromosome not in self.one_hot[strand]:
                raise KeyError(f"Sequence in fasta '{chromosome}' not found in GTF file.")                      
            if feature == 'CDS':
                exon_start = (3 - int(phase)) % 3
                self.one_hot[strand][chromosome][start-1:end, 0] = 0
                
                one_help = (np.linspace(0, end-start, end-start+1) + exon_start) % 3
                if strand == '-':
                    one_help = one_help[::-1]
                self.one_hot[strand][chromosome][start-1:end, 4:7] = \
                    np.eye(3)[one_help.astype(int)]
                    
        for chromosome, feature, strand, phase, start, end in self.gene_structures:  
            if feature == 'intron':                
                if strand == '+':
                    idx = start - 2
                    if transition:
                        idx = start - 3
                    exon_strand = np.argmax(self.one_hot[strand][chromosome][idx]) - 4
                else:
                    idx = end
                    if transition:
                        idx = end + 1 
                    exon_strand = np.argmax(self.one_hot[strand][chromosome][idx])-4
                self.one_hot[strand][chromosome][start-1:end, 1 + exon_strand] = 1
                self.one_hot[strand][chromosome][start-1:end, 0] = 0  
        
        if transition:
            def calculate_index(array, position, default, offset, condition):
                if condition:
                    return default
                else:
                    return np.argmax(array[position, :4]) + offset

            def update_one_hot(self, strand, chromosome, position, index):
                self.one_hot[strand][chromosome][position] = 0
                self.one_hot[strand][chromosome][position, index] = 1
                
            # states : Ir, I0, I1, I2, E0, E1, E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
            for chromosome, feature, strand, phase, start, end in self.gene_structures:                          
                if feature == 'CDS':
                    if strand == '+':
                        prev_condition = start - 2 < 0
                        prev = calculate_index(self.one_hot[strand][chromosome], start - 2, 7, 10, prev_condition)
                        prev = 7 if prev == 10 else prev
                        
                        end_condition = end >= len(self.one_hot[strand][chromosome])
                        after = calculate_index(self.one_hot[strand][chromosome], end, 14, 7, end_condition)
                        after = 14 if after == 7 else after

                        update_one_hot(self, strand, chromosome, start-1, prev)
                        update_one_hot(self, strand, chromosome, end-1, after)                        
                    else:
                        end_condition = end >= len(self.one_hot[strand][chromosome])
                        prev = calculate_index(self.one_hot[strand][chromosome], end, 7, 10, end_condition)
                        prev = 7 if prev == 10 else prev

                        start_condition = start - 2 < 0
                        after = calculate_index(self.one_hot[strand][chromosome], start - 2, 14, 7, start_condition)
                        after = 14 if after == 7 else after

                        update_one_hot(self, strand, chromosome, end-1, prev)
                        update_one_hot(self, strand, chromosome, start-1, after)                        

    def get_flat_chunks_hmm(self, seq_names, strand='+', coords=False):
        """Get one-hot encoded chunks, chunks smaller than chunksize are removed.

        Arguments:
            seq_names (list): Names of sequences to chunk.
            strand (str): Strand to process ('+' or '-').
            coords (bool): get coordinates of each chunk

        Returns:
            tuple: One hot encoded chunks of labels
        """
        self.chunks = []
        chunk_coords = []
        for seq_name in seq_names:
            num_chunks = (len(self.one_hot[strand][seq_name]) - self.overlap) \
                // (self.chunksize - self.overlap) + 1
            
            if num_chunks-1 == 0:
                continue
            if coords:
                chunk_coords += [[
                        seq_name, strand,
                        i * (self.chunksize - self.overlap)+1, 
                        i * (self.chunksize - self.overlap) + self.chunksize] \
                        for i in range(num_chunks)]
            self.chunks += [self.one_hot[strand][seq_name][i * (self.chunksize - self.overlap):\
                i * (self.chunksize - self.overlap) + self.chunksize, :] \
                for i in range(num_chunks-1)]
            
        self.chunks = np.array(self.chunks)
        if strand == '-':
            self.chunks = self.chunks[::-1,::-1,:]
            chunk_coords.reverse()
        if coords:
            return self.chunks, chunk_coords
        return self.chunks
    