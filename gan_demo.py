from itertools import count
import torch
import time
import numpy as np
import sys

from voxel.viewer import VoxelViewer
from model import Generator, standard_normal_distribution, LATENT_CODE_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

generator = Generator()
if "wgan" in sys.argv:
    generator.filename = "wgan-generator.to"
generator.load()
generator.eval()

viewer = VoxelViewer()

STEPS = 20

SHAPE = (2, LATENT_CODE_SIZE, 1, 1, 1)

TRANSITION_TIME = 0.4
WAIT_TIME = 0.8

def get_random():
    return standard_normal_distribution.sample(sample_shape=SHAPE).to(device)


previous_model = None
next_model = get_random()

for epoch in count():
    try:
        previous_model = next_model
        next_model = get_random()

        for step in range(STEPS + 1):
            progress = step / STEPS
            model = None
            if step < STEPS:
                model = previous_model * (1 - progress) + next_model * progress
            else:
                model = next_model

            viewer.set_voxels(generator.forward(model)[0, :, :, :, :].squeeze().detach().cpu())
            time.sleep(TRANSITION_TIME / STEPS)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break