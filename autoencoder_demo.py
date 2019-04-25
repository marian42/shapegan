from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import time

import random

import numpy as np

from voxel.viewer import VoxelViewer

from model import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

autoencoder = Autoencoder()
autoencoder.load()

viewer = VoxelViewer()

distribution = torch.distributions.uniform.Uniform(0, 100)

STEPS = 40

SHAPE = (200, )

TRANSITION_TIME = 1
WAIT_TIME = 0.5

def get_random_():
    return distribution.sample(sample_shape=SHAPE).to(device)

def get_random():
    index = random.randint(0, dataset_size -1)
    return autoencoder.encode(dataset[index, :, :, :])

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