from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import time
import random
import numpy as np
import sys

from voxel.viewer import VoxelViewer
from model import Autoencoder, LATENT_CODE_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataset import dataset as dataset

autoencoder = Autoencoder(is_variational='classic' not in sys.argv)
autoencoder.load()
autoencoder.eval()

viewer = VoxelViewer()

STEPS = 40

SHAPE = (LATENT_CODE_SIZE, )

TRANSITION_TIME = 0.8
WAIT_TIME = 0.8

SAMPLE_FROM_LATENT_DISTRIBUTION = 'sample' in sys.argv

def get_latent_distribution():
    print("Calculating latent distribution...")
    indices = random.sample(list(range(dataset.size)), 1000)
    voxels = dataset.voxels[indices, :, :, :]
    with torch.no_grad():
        codes = autoencoder.encode(voxels, device)
    latent_codes_flattened = codes.detach().cpu().numpy().reshape(-1)
    mean = np.mean(latent_codes_flattened)
    variance = np.var(latent_codes_flattened) ** 0.5
    print('Latent distribution: µ = {:.3f}, σ = {:.3f}'.format(mean, variance))
    return torch.distributions.normal.Normal(mean, variance)

latent_distribution = get_latent_distribution()

def get_random():
    if SAMPLE_FROM_LATENT_DISTRIBUTION:
        return latent_distribution.sample(sample_shape=SHAPE).to(device)
    else:
        index = random.randint(0, dataset.size -1)
        return autoencoder.encode(dataset.voxels[index, :, :, :], device)


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
            voxels = autoencoder.decode(model).detach().cpu()
            viewer.set_voxels(voxels)

            time.sleep(TRANSITION_TIME / STEPS)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break