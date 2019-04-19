import random
import torch
import time

from voxel.viewer import VoxelViewer

dataset = torch.load("data/airplanes-64.to")
dataset_size = dataset.shape[0]

viewer = VoxelViewer()

while True:
    try:
        viewer.set_voxels(dataset[random.randint(0, dataset_size - 1), :, :, :].squeeze().detach().numpy())
        time.sleep(0.4)
    except KeyboardInterrupt:
        viewer.stop()
        break
