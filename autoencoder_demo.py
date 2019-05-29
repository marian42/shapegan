from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import time

import random

import numpy as np

from voxel.viewer import VoxelViewer

from model import Autoencoder, LATENT_CODE_SIZE, standard_normal_distribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import dataset as dataset

autoencoder = Autoencoder()
autoencoder.load()
autoencoder.eval()

viewer = VoxelViewer()

STEPS = 40

SHAPE = (LATENT_CODE_SIZE, )

TRANSITION_TIME = 0.8
WAIT_TIME = 0.8

def get_random_():
    return standard_normal_distribution.sample(sample_shape=SHAPE).to(device)


def get_random():
    index = random.randint(0, dataset.size -1)
    return autoencoder.create_latent_code(*autoencoder.encode(dataset.voxels[index, :, :, :]), device)


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

            viewer.set_voxels(autoencoder.decode(model).detach().cpu())
            time.sleep(TRANSITION_TIME / STEPS)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break