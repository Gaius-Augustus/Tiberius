import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DataLoader_h5(Dataset):
    def __init__(self, file_paths, output_size=5):
        # implemented outputsize only for 5 and 15
        self.files = file_paths
        self.lengths = []
        self.cumulative_lengths = []
        self.output_size = output_size
        cumulative_length = 0

        for file_path in self.files:
            with h5py.File(file_path, 'r') as f:
                length = len(f['input'])
                self.lengths.append(length)
                cumulative_length += length
                self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        file_index = next(i for i, total_length in enumerate(self.cumulative_lengths) if total_length > idx)
        if file_index != 0:
            idx -= self.cumulative_lengths[file_index - 1]

        with h5py.File(self.files[file_index], 'r') as file:
            input_data = file['input'][idx]
            output_data = file['output'][idx]
        if self.output_size == 5:
            out_new = np.zeros((output_data.shape[0], 5), 
                               np.float32)
            out_new[:,0] = output_data[:,0]
            out_new[:,1] = np.sum(output_data[:,1:4], axis=-1)            
            out_new[:,2] = np.sum(output_data[:,[4, 7, 10, 12]], axis=-1)   
            out_new[:,3] = np.sum(output_data[:,[5, 8, 13]], axis=-1)
            out_new[:,4] = np.sum(output_data[:,[6, 9, 11, 14]], axis=-1)
            output_data = out_new
        elif not self.output_size == 15:
            print("Outputsize only supported for 5 and 15, returning outputsize 15!")
            out_new = output_data
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(out_new, dtype=torch.float32)
