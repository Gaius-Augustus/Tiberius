# ==============================================================
# Authors: Lars Gabriel
#
# Class handling GTF information to generate trainings examples
# ==============================================================

import numpy as np
import sys 
from collections import defaultdict

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
        self.tx_ranges = {}

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
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('track'):
                    continue
                parts = line.strip().split('\t')
                chromosome, source, feature, start, end, score, strand, phase, attributes = parts
                # Extract transcript_id from attributes
                transcript_id = None
                if 'transcript_id' in attributes:
                    transcript_id = attributes.split('transcript_id "')[1].split('"')[0]
                elif 'ID' in attributes:
                    transcript_id = attributes.split('ID=')[1].split(';')[0]
                else:
                    # Fallback to gene_id if transcript_id is missing
                    transcript_id = attributes.split('gene_id "')[1].split('"')[0]
                if transcript_id and transcript_id not in self.tx_ranges:
                    self.tx_ranges[transcript_id] = [int(start), int(end)]
                elif transcript_id:
                    self.tx_ranges[transcript_id][0] = min(int(start), self.tx_ranges[transcript_id][0])
                    self.tx_ranges[transcript_id][1] = min(int(end), self.tx_ranges[transcript_id][1])
                # Append as a 7-tuple
                self.gene_structures.append((chromosome, feature, strand, phase, int(start), int(end), transcript_id))

    
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
        """
        A faster, more vectorized version to translate gene structure information to one-hot encoding.

        If transition is True, then for multi-exon genes we use 15 labels:
            0: Intergenic (IR)
            1-3: Intron (I0, I1, I2) based on reading frame from preceding CDS
            4-6: Exon interior (E0, E1, E2)
            7: START (first base of a multi-exon transcript)
            8-10: Exon-to-intron transition (EI0, EI1, EI2)
            11-13: Intron-to-exon transition (IE0, IE1, IE2)
            14: STOP (last base of a multi-exon transcript)
        Single-exon genes (when transition=True) get additional 5 labels:
            15: START_single
            16-18: Exon_Single (depending on reading frame)
            19: STOP_single

        If transition is False, only 7 labels are used:
            0: IR, 1-3: Intron, 4-6: Exon.
        """

        # Determine total number of labels.
        total_labels = 20 if transition else 7

        # Pre-allocate one-hot arrays for each strand and sequence.
        self.one_hot = {strand: {} for strand in ['+', '-']}
        for strand in self.one_hot:
            for seq, seq_len in zip(sequence_names, sequence_lengths):
                arr = np.zeros((seq_len, total_labels), dtype=np.int8)
                # Default label is intergenic (IR, index 0)
                arr[:, 0] = 1
                self.one_hot[strand][seq] = arr

        # Group CDS features by transcript_id.
        # Each gene structure tuple is assumed to be: 
        # (chrom, feature, strand, phase, start, end, transcript_id)
        transcripts = defaultdict(lambda: {"CDS": []})
        for gs in self.gene_structures:
            if len(gs) < 7:
                continue  # Skip entries without transcript_id.
            chrom, feature, strand, phase, start, end, tx_id = gs
            if chrom not in sequence_names:
                continue
            if feature == "CDS":
                transcripts[tx_id]["CDS"].append((start, end))
                transcripts[tx_id]["chrom"] = chrom
                transcripts[tx_id]["strand"] = strand

        for tx_id, info in transcripts.items():
            if not info["CDS"]:
                continue
            chrom = info["chrom"]
            strand = info["strand"]

            # Convert CDS list to a NumPy array for vectorized operations.
            # Each row is (start, end, phase)
            cds_list = info["CDS"]
            if strand == '+':
                cds_list.sort(key=lambda x: x[0])
            else:
                cds_list.sort(key=lambda x: x[0], reverse=True)
            is_single_exon = (len(cds_list) == 1)

            # Process exons while computing phases from scratch.
            exon_info = []  # Will store tuples: (start, end, computed_phase, frames_array)
            running_total = 0  # Sum of exon lengths processed so far.
            for seg in cds_list:
                start, end = seg
                # Compute the phase for this exon.
                computed_phase = (3 - (running_total % 3)) % 3  # First exon gets 0.
                s_idx = start - 1  # Convert 1-indexed to 0-indexed.
                e_idx = end        # Python slice end is exclusive.
                # Calculate an offset to assign reading frame labels.
                # (The idea is that after "removing" computed_phase bases from the start,
                # the remaining bases are in frame 0.)
                offset = (3 - computed_phase) % 3
                length = e_idx - s_idx
                # Compute reading frame for each base in this exon.
                frames = (np.arange(length) + offset) % 3

                # Clear the exon region (remove default IR label) and assign the exon interior.
                self.one_hot[strand][chrom][s_idx:e_idx, :] = 0
                if transition:
                    if is_single_exon:
                        self.one_hot[strand][chrom][s_idx:e_idx, 16:19] = np.eye(3, dtype=np.int8)[frames]
                    else:
                        self.one_hot[strand][chrom][s_idx:e_idx, 4:7] = np.eye(3, dtype=np.int8)[frames]
                else:
                    self.one_hot[strand][chrom][s_idx:e_idx, 4:7] = np.eye(3, dtype=np.int8)[frames]
                if strand == '-':
                    self.one_hot[strand][chrom][s_idx:e_idx, :] = self.one_hot[strand][chrom][s_idx:e_idx][::-1]

                exon_info.append((start, end, computed_phase, frames))
                running_total += (e_idx - s_idx)

            exon_info.sort(key=lambda x: x[0])
            if transition:
                # position of START label
                start_pos = exon_info[0][0]-1 if strand == '+' else exon_info[-1][1]-1
                # position of STOP label
                stop_pos = exon_info[-1][1]-1 if strand == '+' else exon_info[0][0]-1
                self.one_hot[strand][chrom][start_pos] = np.zeros(total_labels, dtype=np.int8)                
                self.one_hot[strand][chrom][stop_pos] = np.zeros(total_labels, dtype=np.int8)
                if is_single_exon:
                    self.one_hot[strand][chrom][start_pos, 15] = 1  # START_single
                    self.one_hot[strand][chrom][stop_pos, 19] = 1  # STOP_single
                else:
                    self.one_hot[strand][chrom][start_pos, 7] = 1  # START.
                    self.one_hot[strand][chrom][stop_pos, 14] = 1  # STOP.

                    # Process intron regions and transitions between consecutive CDS segments.
                    for i in range(len(exon_info) - 1):
                        cur_start, cur_end, cur_phase, cur_frames = exon_info[i]
                        nxt_start, nxt_end, nxt_phase, nxt_frames = exon_info[i+1]
                        # Define the intron region between current CDS and next CDS.
                        intron_start = cur_end
                        intron_end = nxt_start - 1
                        if intron_start < intron_end:
                            # Compute reading frame at the end of current CDS.
                            last_frame = (cur_frames[-1]-1)%3 if strand == '+' else (nxt_frames[-1]-1)%3
                            intron_label = 1 + last_frame  # Intron labels are at indices 1-3.
                            self.one_hot[strand][chrom][intron_start:intron_end, :] = 0
                            self.one_hot[strand][chrom][intron_start:intron_end, intron_label] = 1

                            # Set the exon-to-intron (EI) transition at the last base of the current exon.
                            pos_EI = cur_end - 1 if strand == '+' else nxt_start - 1
                            self.one_hot[strand][chrom][pos_EI] = np.zeros(total_labels, dtype=np.int8)
                            self.one_hot[strand][chrom][pos_EI, 8 + last_frame] = 1

                            # Set the intron-to-exon (IE) transition at the first base of the next exon.
                            pos_IE = nxt_start - 1 if strand == '+' else cur_end - 1
                            self.one_hot[strand][chrom][pos_IE] = np.zeros(total_labels, dtype=np.int8)
                            first_frame = (nxt_frames[0]+1)%3  if strand == '+' else (cur_frames[0]+1)%3 
                            self.one_hot[strand][chrom][pos_IE, 11 + first_frame] = 1

        return self.one_hot
              

    def get_flat_chunks_hmm(self, seq_names, strand='+', coords=False, transcript_ids=False):
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
        transcript_list = []
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
            if transcript_ids:
                # compute list of transcript ids for each chunk and generate list of transcript ids and overlap of transcript and chunk
                for i in range(num_chunks-1):
                    chunk_transcripts = []
                    for transcript_id, (tx_start, tx_end) in self.tx_ranges.items():
                        if tx_start < (i * (self.chunksize - self.overlap) + self.chunksize) \
                            and tx_end > i * (self.chunksize - self.overlap):
                            chunk_transcripts = [
                                transcript_id, str(max(tx_start, i * (self.chunksize - self.overlap)+1)), 
                                    str(min(tx_end, i * (self.chunksize - self.overlap) + self.chunksize))]
                            chunk_transcripts = ['', '', ''] if not chunk_transcripts else chunk_transcripts
                            transcript_list.append(chunk_transcripts)
                

        self.chunks = np.array(self.chunks)
        if strand == '-':
            self.chunks = self.chunks[::-1,::-1,:]
            chunk_coords.reverse()
        if coords:
            return self.chunks, chunk_coords
        if transcript_ids:
            return self.chunks, np.array(transcript_list, dtype='<U50')
        return self.chunks
    