import numpy as np
import os
from tqdm import tqdm
import torch

# https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py
from binvox_rw import read_as_3d_array
from scipy.ndimage import zoom


def load_voxels(filename, size):    
    voxels = read_as_3d_array(open(filename, 'rb'))
    voxels = voxels.data.astype(np.float32)
    voxels = (zoom(voxels, size / voxels.shape[0]) > 0.3).astype(np.float32)
    return voxels


def prepare_dataset(directory, size, out_file_name = "dataset.to"):
    filenames = []
    print("Scanning directory " + directory + "...")
    for subdirectory in os.listdir(directory):
        filename = os.path.join(directory, subdirectory, "models" , "model_normalized.solid.binvox")
        if os.path.isfile(filename):
            filenames.append(filename)
    models = []
    print("Loading models...")
    for filename in tqdm(filenames):
        models.append(load_voxels(filename, size))
    batch = np.stack(models).astype(np.float32)
    tensor = torch.tensor(batch)

    torch.save(tensor, out_file_name)