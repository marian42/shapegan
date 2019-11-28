import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class VoxelsSingleTensor(Dataset):
    def __init__(self, filename, clamp=0.1):
        self.data = torch.load(filename)
        if clamp is not None:
            self.data.clamp_(-clamp, clamp)

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.data.shape[0]

class VoxelsMultipleFiles(Dataset):
    def __init__(self, directory, extension='.npy', clamp=0.1):
        self.files = glob.glob(os.path.join(directory, '**' + extension), recursive=True)
        self.clamp = clamp

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = np.load(self.files[index])
        result = torch.from_numpy(array)
        if self.clamp is not None:
            result.clamp_(-self.clamp, self.clamp)
        return result
