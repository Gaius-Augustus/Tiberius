# ==============================================================
# Authors: Lars Gabriel
#
# Class handling GTF information to generate trainings examples
# ==============================================================

import numpy as np
import sys
from collections import defaultdict
from typing import List, Dict, Optional
import numpy as np

class GeneFeature:
    """Represents a single feature (exon or intron) in a transcript.
    """
    type: str  # 'CDS' or 'intron'
    start: int
    end: int
    strand: str  # '+' or '-'
    phase: int  # Phase of the exon (0, 1, 2) or previous exon phase for introns

    def __init__(
        self,
        feature_type: str,
        start: int,
        end: int,
        strand: str,
        phase: int,
    ) -> None:
        self.type = feature_type
        self.start = start
        self.end = end
        self.strand = strand
        self.phase = phase

    def to_class_labels(self) -> np.ndarray:
        """Produce a 1D array of integer labels (0-14) for each base in [start, end].
        All CDS are processed as inner CDS
        Label mapping follows the HMM scheme.
        """
        length: int = self.end - self.start + 1
        offset: int = (3 - self.phase) % 3
        if self.type == 'CDS':
            # Compute reading frame for each base in this exon.
            labels: int = (np.arange(length, dtype=np.int32) + offset) % 3 + 4 # Exon label start at index 4
            labels[0] = 11 + (offset +1)%3
            labels[-1] = 8 + (length-int(self.phase)+1)%3
            if self.strand == '-':
                # Reverse the labels for negative strand
                labels = labels[::-1]
        elif self.type == 'intron':
            labels = np.zeros(length) + 1 + (offset +1)%3
        return labels

    def to_one_hot(self) -> np.ndarray:
        """Convert the integer labels into a one-hot matrix of shape (length, 15).
        """
        labels: np.ndarray = self.to_class_labels()
        one_hot: np.ndarray = np.eye(15, dtype=np.int32)[labels]
        return one_hot

class Transcript:
    """Represents a single transcript, composed of multiple GeneFeatures.
    """
    id: str
    sequence_name: str
    strand: str
    features: List[GeneFeature]
    start: Optional[int]
    end: Optional[int]

    def __init__(
        self,
        tx_id: str
    ) -> None:
        self.id = tx_id
        self.sequence_name = None
        self.strand = None
        self.features = []
        self.start = None
        self.end = None

    def read_input(self, lines: List[str]) -> None:
        """Parse GTF/GFF lines for this transcript and populate features.
        """
        for line in lines:
            parts = line.strip().split('\t')
            chromosome, source, feature, start, end, score, strand, phase, attributes = parts
            if self.sequence_name and self.sequence_name != chromosome:
                raise ValueError(
                    f"Transcript {self.id} spans multiple chromosomes: "
                    f"{self.sequence_name} and {chromosome}"
                )
            self.sequence_name = chromosome
            if self.strand and self.strand != strand:
                raise ValueError(
                    f"Transcript {self.id} has inconsistent strand: "
                    f"{self.strand} and {strand}"
                )
            self.strand = strand
            self.features.append(
                GeneFeature(
                    feature_type=feature,
                    start=int(start),
                    end=int(end),
                    strand=strand,
                    phase=phase
                )
            )
        # Sort features by start position
        self.features.sort(key=lambda x: x.start)

        # Set sequence name and strand from the first feature
        if self.features:
            # Set start and end based on features
            self.start = self.features[0].start
            self.end = self.features[-1].end

        # check if an intron feature is missing
        introns_added = []
        for i in range(len(self.features) - 1):
            if self.features[i].type == 'CDS' and self.features[i+1].type == 'CDS':
                # check if an intron feature is missing
                if self.features[i].end + 1 < self.features[i+1].start:
                    introns_added.append(GeneFeature(
                        feature_type='intron',
                        start=self.features[i].end + 1,
                        end=self.features[i+1].start - 1,
                        strand=self.strand,
                        phase=self.features[i].phase if self.strand == '+' \
                                else self.features[i+1].phase
                    ))
        self.features.extend(introns_added)
        self.features.sort(key=lambda x: x.start)

        # redo phases
        # interate through features and set phases reverse iteration if strand is '-'
        indices = range(len(self.features)) if self.strand == '+' \
                else range(len(self.features) - 1, -1, -1)
        prev_CDS_phase = 0
        for i in indices:
            feat = self.features[i]
            if feat.type == 'CDS':
                # calculate phase based on previous feature
                feat.phase = prev_CDS_phase
                prev_CDS_phase = (3 - (feat.end - feat.start + 1 - prev_CDS_phase)%3)%3
        for k, i in enumerate(indices):
            feat = self.features[i]
            if feat.type == 'intron':
                # calculate phase based on previous feature
                try:
                    feat.phase = self.features[list(indices)[k+1]].phase
                except IndexError:
                    print(f"Warning: Intron feature {feat.start}-{feat.end} in transcript {self.id} has no next CDS feature. Setting phase to 0.")


    def to_class_labels(self) -> np.ndarray:
        """Assemble a full integer-label sequence for the transcript.
        """
        length: int = self.end - self.start + 1
        labels: np.ndarray = np.zeros(length)
        for feat in self.features:
            rel_start: int = feat.start - self.start
            rel_end: int = feat.end - self.start
            labels[rel_start:rel_end+1] = feat.to_class_labels()
        labels[0] = 7 if self.strand == '+' else 14
        labels[-1] = 14 if self.strand == '+' else 7
        return labels

    def to_one_hot(self) -> np.ndarray:
        """Convert the transcript's class-label sequence into one-hot of shape (L, 15).
        """
        labels: np.ndarray = self.to_class_labels()
        return np.eye(15, dtype=np.int32)[labels]

class Annotation:
    """Represents a full GTF/GFF annotation, split into chunks.
    """
    file_path: str
    seqnames: List[str]
    seq_lens: List[int]
    chunk_len: int
    transcripts: Dict[str, Transcript]
    num_chunks: int
    chunk2transcripts: Dict[int, List[int]]

    def __init__(
        self,
        file_path: str,
        seqnames: List[str],
        seq_lens: List[int],
        chunk_len: int,
    ) -> None:
        self.file_path = file_path
        self.seqnames = seqnames
        self.seq_lens = seq_lens
        self.chunk_len = chunk_len
        self.transcripts = []
        self.seq2chunk_pos = {"-": {self.seqnames[i] : sum(s // self.chunk_len for s in self.seq_lens[:i]) \
                    for i in range(len(self.seqnames))}}
        self.seq2chunk_pos.update({
            "+":  {self.seqnames[i] : sum(s // self.chunk_len for s in self.seq_lens + self.seq_lens[:i]) \
                    for i in range(len(self.seqnames))}})

        self.chunk2transcripts = {}

    def transcript2chunknumb(self, seq_name: str, start: int, end: int, strand: str) -> List[int]:
        """Given a sequence name and start-end coordinates, return the chunk indices
        that this transcript overlaps.
        """
        start_chunk = self.seq2chunk_pos[strand][seq_name] + start // self.chunk_len
        end_chunk = self.seq2chunk_pos[strand][seq_name] + (end) // self.chunk_len

        return [start_chunk, end_chunk]


    def read_inputfile(self) -> None:
        """Read GTF or GFF file and build Transcript objects.
        """
        transcript_lines = {}
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('track'):
                    continue
                parts = line.strip().split('\t')
                chromosome, source, feature, start, end, score, strand, phase, attributes = parts
                if chromosome not in self.seq2chunk_pos["+"]:
                    continue
                # Extract transcript_id from attributes
                if chromosome not in self.seq2chunk_pos["+"]:
                    continue
                if feature not in ["intron", "CDS"]:
                    continue
                transcript_id = None
                if 'transcript_id' in attributes:
                    transcript_id = attributes.split('transcript_id "')[1].split('"')[0]
                elif 'ID' in attributes:
                    transcript_id = attributes.split('ID=')[1].split(';')[0]
                else:
                    # throw an error if no transcript_id is found
                    raise ValueError(
                        f"Transcript ID not found in line: {line.strip()}"
                    )
                if transcript_id not in transcript_lines:
                    transcript_lines[transcript_id] = []
                transcript_lines[transcript_id].append(line.strip())

        for k, (tx_id, tx) in enumerate(transcript_lines.items()):
            # Create a Transcript object for each transcript_id
            self.transcripts.append(Transcript(tx_id))
            self.transcripts[-1].read_input(tx)

            chunk_numb = self.transcript2chunknumb(
                self.transcripts[-1].sequence_name,
                self.transcripts[-1].start,
                self.transcripts[-1].end,
                self.transcripts[-1].strand
            )
            for c in range(chunk_numb[0], chunk_numb[1] + 1):
                if c not in self.chunk2transcripts:
                    self.chunk2transcripts[c] = []
                self.chunk2transcripts[c].append(k)

    def get_chunk_labels(self, chunk_idx: int, get_tx_ids: bool = False) -> np.ndarray:
        """Return a 1D integer-label array of size chunk_len for the specified chunk.
        """
        strand: str = "+"
        tx_out: [str, str, str] = [] # txID, start pos in chunk, end pos in chunk
        labels: np.ndarray = np.zeros(self.chunk_len, dtype=np.int32)
        if chunk_idx not in self.chunk2transcripts:
            if get_tx_ids:
                return labels, np.array(tx_out, dtype='<U50')
            return labels
        for tx_num in self.chunk2transcripts[chunk_idx]:
            tx = self.transcripts[tx_num]
            strand = tx.strand
            tx_label = tx.to_class_labels()
            chunk_start = self.chunk_len * \
                (chunk_idx - self.seq2chunk_pos[strand][tx.sequence_name])
            chunk_end = chunk_start + self.chunk_len

            # take the overlap of transcript and chunk
            overlap_start_chunk = max(0, tx.start - chunk_start - 1)
            overlap_end_chunk = min(self.chunk_len, tx.end - chunk_start)
            overlap_start_tx = max(0, chunk_start - tx.start  + 1)
            overlap_end_tx = min(len(tx_label), len(tx_label) - (tx.end - chunk_end))
            # fill the labels array with the overlap
            labels[overlap_start_chunk:overlap_end_chunk] = \
                tx_label[overlap_start_tx:overlap_end_tx]
            if get_tx_ids and strand == '+':
                tx_out.append([tx.id, str(overlap_start_chunk), str(overlap_end_chunk)])
            elif get_tx_ids and strand == '-':
                tx_out.append([tx.id, str(self.chunk_len - overlap_end_chunk),
                str(self.chunk_len - overlap_start_chunk)])
        if strand == '-':
            # reverse the labels for negative strand
            labels = labels[::-1]
        if get_tx_ids:
            return labels, np.array(tx_out, dtype='<U50')
        return labels


    def get_onehot(self, chunk_idx: int, get_tx_ids: bool = False) -> np.ndarray:
        """Fetch the integer-label chunk (from memory or disk) and convert to one-hot.
        """
        if get_tx_ids:
            labels, tx_ids = self.get_chunk_labels(chunk_idx, get_tx_ids=True)
            return np.eye(15, dtype=np.int32)[labels], tx_ids
        labels: np.ndarray = self.get_chunk_labels(chunk_idx, get_tx_ids=get_tx_ids)
        return np.eye(15, dtype=np.int32)[labels]


class GeneStructure: # deprecated for now
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
                elif 'gene_id' in attributes:
                    # Fallback to gene_id if transcript_id is missing
                    transcript_id = attributes.split('gene_id "')[1].split('"')[0]
                else:
                    transcript_id = attributes.strip()
                if transcript_id and transcript_id not in self.tx_ranges:
                    self.tx_ranges[transcript_id] = [int(start), int(end)]
                elif transcript_id:
                    self.tx_ranges[transcript_id][0] = min(int(start), self.tx_ranges[transcript_id][0])
                    self.tx_ranges[transcript_id][1] = max(int(end), self.tx_ranges[transcript_id][1])
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

        If transition is False, only 7 labels are used:
            0: IR, 1-3: Intron, 4-6: Exon.
        """

        # Determine total number of labels.
        total_labels = 15 if transition else 7

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
            self.chunks = self.chunks[:,::-1,:]
            chunk_coords.reverse()
        if coords:
            return self.chunks, chunk_coords
        if transcript_ids:
            return self.chunks, np.array(transcript_list, dtype='<U50')
        return self.chunks
