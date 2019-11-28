import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class VoxelsSingleTensor(Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename)

    def __getitem__(self, index):
        return self.data[index, :, :, :]

    def __len__(self):
        return self.data.shape[0]

class VoxelsMultipleFiles(Dataset):
    def __init__(self, directory, extension='.npy'):
        self.files = glob.glob(os.path.join(directory, '**' + extension), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = np.load(self.files[index])
        return torch.from_numpy(array)
