import os, sys
import numpy as np
import subprocess as sp

class Wig_util:    
    def __init__(self):
        self.wig_arrays = None
        pass
    
    def addWig2numpy(self, wig_file, seq_lens, strand='+'):
        """Args:
            wig_file: Path to WIG file
            seq_lens: List of pairs of (Sequence name, Sequence length)
            srand: 
        """
        wig_seq_dict = {}
        with open(wig_file, 'r') as f_wig:
            wig_content = f_wig.read().split('fixedStep')
            
        for k, section in enumerate(wig_content[1:]):
            section = section.strip().split('\n')
            # print(section[0])
            header = section[0].split()
            chrom_info = {part.split('=')[0]: part.split('=')[1] for part in header}
            current_seq = chrom_info.get('chrom')
            pos = int(chrom_info.get('start')) - 1
            step = int(chrom_info.get('step'))
            
            if not current_seq in wig_seq_dict:
                wig_seq_dict[current_seq] = {}
            # print(section[0])
            for t, line in enumerate(section[1:]):
                val = float(line)
                if not str(pos) in wig_seq_dict[current_seq]:
                    wig_seq_dict[current_seq][str(pos)] = []
                wig_seq_dict[current_seq][str(pos)].append(val)
                pos += step
        
        if not self.wig_arrays:
            self.wig_arrays = {}
            for s_n, s_l in seq_lens:
                self.wig_arrays[s_n] = np.zeros((int(s_l),4), float)
                self.wig_arrays[s_n][:,0] = np.log(1e-5)
                self.wig_arrays[s_n][:,1] = np.log(1e-5)
        if strand == '+':
            i = 0
        else:
            i = 1
        
        for seq in wig_seq_dict:
            for pos in wig_seq_dict[seq]:                
                if seq in self.wig_arrays and 0 <= int(pos) < self.wig_arrays[seq].shape[0]:    
                    if self.wig_arrays[seq][int(pos), 2+i] > 0:
                        print(seq, pos)
                    self.wig_arrays[seq][int(pos), i] = np.mean(wig_seq_dict[seq][pos])
                    self.wig_arrays[seq][int(pos), 2+i] = len(wig_seq_dict[seq][pos])
    
#         for k, section in enumerate(wig_content[1:]):
#             section = section.strip().split('\n')
#             # print(section[0])
#             header = section[0].split()
#             chrom_info = {part.split('=')[0]: part.split('=')[1] for part in header}
#             current_seq = chrom_info.get('chrom')
#             pos = int(chrom_info.get('start')) - 1
#             step = int(chrom_info.get('step'))
#             # print(section[0])
#             for t, line in enumerate(section[1:]):
#                 val = float(line)
#                 # print(val, current_seq, pos, i,  self.wig_arrays[current_seq].shape, self.wig_arrays[current_seq][pos])
#                 if current_seq in self.wig_arrays and 0 <= pos < self.wig_arrays[current_seq].shape[0]:
#                     # print(self.wig_arrays[current_seq][pos, i], val, max(val, self.wig_arrays[current_seq][pos, i]))
#                     self.wig_arrays[current_seq][pos, i] = max(val, self.wig_arrays[current_seq][pos, i])
#                     self.wig_arrays[current_seq][pos, 2] = 0.
#                 # print( self.wig_arrays[current_seq][pos])
#                 pos += step  # Increment the start position for the next value
                
#                 if t > 200:
#                     exit()
                    
    def addWig2numpy_old(self, wig_file, seq_lens, strand='+'):
        """Args:
            wig_file: Path to WIG file
            seq_lens: List of pairs of (Sequence name, Sequence length)
            srand: 
        """
        if not self.wig_arrays:
            self.wig_arrays = {}
            for s_n, s_l in seq_lens:
                self.wig_arrays[s_n] = np.ones((int(s_l),3), float)
                self.wig_arrays[s_n][:,0] = np.log(1e-3)
                self.wig_arrays[s_n][:,1] = np.log(1e-3)
        if strand == '+':
            i = 0
        else:
            i = 1
         
        with open(wig_file, 'r') as f_wig:
            wig_content = f_wig.read().split('fixedStep')
        
        for k, section in enumerate(wig_content[1:]):
            section = section.strip().split('\n')
            # print(section[0])
            header = section[0].split()
            chrom_info = {part.split('=')[0]: part.split('=')[1] for part in header}
            current_seq = chrom_info.get('chrom')
            pos = int(chrom_info.get('start')) - 1
            step = int(chrom_info.get('step'))
            # print(section[0])
            for t, line in enumerate(section[1:]):
                val = float(line)
                # print(val, current_seq, pos, i,  self.wig_arrays[current_seq].shape, self.wig_arrays[current_seq][pos])
                if current_seq in self.wig_arrays and 0 <= pos < self.wig_arrays[current_seq].shape[0]:
                    # print(self.wig_arrays[current_seq][pos, i], val, max(val, self.wig_arrays[current_seq][pos, i]))
                    self.wig_arrays[current_seq][pos, i] = max(val, self.wig_arrays[current_seq][pos, i])
                    self.wig_arrays[current_seq][pos, 2] = 0.
                # print( self.wig_arrays[current_seq][pos])
                pos += step  # Increment the start position for the next value
                        
    def write_wigs(self, out_dir, prefix='', bw=False):
        for i, strand in enumerate(['plus', 'minus']):
            for phase in [0, 1, 2]:
                out_str = ''
                for seq_name in self.wig_arrays.keys():
                    # print(phase, seq_name)
                    # pos = phase+1
                    for pos in range((phase)%3,self.wig_arrays[seq_name].shape[0],4):
                    # for v in self.wig_arrays[seq_name][phase:-3:3, i]:                        
                        if self.wig_arrays[seq_name][pos, i] > np.log(1e-5):
                            # print(self.wig_arrays[seq_name][pos-3, i],self.wig_arrays[seq_name][pos, i])
                            if pos == (phase+1)%3 or \
                                self.wig_arrays[seq_name][pos-3, i] == np.log(1e-5):
                                out_str += f'fixedStep chrom={seq_name} start={pos+1} step=3\n'
                            out_str += f'{self.wig_arrays[seq_name][pos, i]}\n'
                        # pos += 3
                with open(f'{out_dir}/{prefix}_{(phase+1)%3}-{strand}.wig', 'w+') as f_wig:
                    f_wig.write(out_str)
                    
    def get_chunks(self, chunk_len, sequence_names):
        # Use a list comprehension to construct all_chunks
        all_chunks = [
            self.wig_arrays[seq][:numb_chunks*chunk_len].reshape(numb_chunks, chunk_len, 4)
            for seq in sequence_names
            if (numb_chunks := self.wig_arrays[seq].shape[0] // chunk_len) > 0
        ]
        # Concatenate along the first axis only if all_chunks is not empty
        return np.concatenate(all_chunks, axis=0) if all_chunks else np.array([])