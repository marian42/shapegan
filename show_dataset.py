import random
import torch
import time

from voxel.viewer import VoxelViewer
from dataset import dataset as dataset

viewer = VoxelViewer()

while True:
    try:
        viewer.set_voxels(dataset.voxels[random.randint(0, dataset.size - 1), :, :, :].squeeze().detach().cpu().numpy())
        time.sleep(0.4)
    except KeyboardInterrupt:
        viewer.stop()
        break
