import random
import torch
import time
import sys
from tqdm import tqdm

from voxel.viewer import VoxelViewer

PATH = 'data/shapenet/03001627/'

viewer = VoxelViewer()

if "mesh" in sys.argv:
    import trimesh
    import os

    model_filenames = []
    for directory, _, files in os.walk(PATH):
        model_filename = os.path.join(directory, 'model_normalized.obj')
        if os.path.isfile(model_filename):
            model_filenames.append(model_filename)

    while True:
        filename = random.choice(model_filenames)
        mesh = trimesh.load(filename)
        viewer.set_mesh(mesh, center_and_scale=True)
        time.sleep(2)
else:
    from dataset import dataset as dataset

    for i in tqdm(list(range(dataset.voxels.shape[0]))):
        try:
            viewer.set_voxels(dataset.voxels[i, :, :, :].squeeze().detach().cpu().numpy())
            time.sleep(2)
        except KeyboardInterrupt:
            viewer.stop()
            break
