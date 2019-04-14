import numpy as np

# https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py
from binvox_rw import read_as_3d_array
from scipy.ndimage import zoom


def load_voxels(filename, size):    
    voxels = read_as_3d_array(open(filename, 'rb'))
    voxels = voxels.data.astype(np.float32)
    voxels = (zoom(voxels, size / voxels.shape[0]) > 0.3).astype(np.float32)
    return voxels
