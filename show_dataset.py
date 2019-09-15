import random
import torch
import time
import sys
from tqdm import tqdm

from voxel.viewer import VoxelViewer

PATH = 'data/shapenet/03001627/'

viewer = VoxelViewer()
from dataset import dataset as dataset


if "mesh" in sys.argv:
    import trimesh
    import os

    for directory in dataset.get_models():
        model_filename = os.path.join(directory, 'model_normalized.obj')        
        mesh = trimesh.load(model_filename)
        viewer.set_mesh(mesh, center_and_scale=True)
        time.sleep(0.5)
    
else:
    dataset.load_voxels('cpu')
    for i in tqdm(list(range(dataset.voxels.shape[0]))):
        try:
            viewer.set_voxels(dataset.voxels[i, :, :, :].squeeze().detach().cpu().numpy())
            time.sleep(0.5)
        except KeyboardInterrupt:
            viewer.stop()
            break
