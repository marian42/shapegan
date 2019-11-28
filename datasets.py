import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from util import device

class VoxelsSingleTensor(Dataset):
    def __init__(self, filename, clamp=0.1):
        self.data = torch.load(filename).to(device)
        if clamp is not None:
            self.data.clamp_(-clamp, clamp)

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
        return torch.from_numpy(array).to(device)
