from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import time
import random
import numpy as np
import sys

from rendering import MeshRenderer
from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
from util import device
from datasets import VoxelDataset

dataset = VoxelDataset.glob('data/chairs/voxels_32/**.npy')

autoencoder = Autoencoder(is_variational='classic' not in sys.argv)
autoencoder.load()
autoencoder.eval()

viewer = MeshRenderer()

STEPS = 40

SHAPE = (LATENT_CODE_SIZE, )

TRANSITION_TIME = 1.2
WAIT_TIME = 1.2

SAMPLE_FROM_LATENT_DISTRIBUTION = 'sample' in sys.argv

def get_latent_distribution():
    print("Calculating latent distribution...")
    indices = random.sample(list(range(len(dataset))), min(1000, len(dataset)))
    voxels = torch.stack([dataset[i] for i in indices]).to(device)
    with torch.no_grad():
        codes = autoencoder.encode(voxels)
    latent_codes_flattened = codes.detach().cpu().numpy().reshape(-1)
    mean = np.mean(latent_codes_flattened)
    variance = np.var(latent_codes_flattened) ** 0.5
    print('Latent distribution: µ = {:.3f}, σ = {:.3f}'.format(mean, variance))
    return torch.distributions.normal.Normal(mean, variance)

if SAMPLE_FROM_LATENT_DISTRIBUTION:
    latent_distribution = get_latent_distribution()

def get_random():
    if SAMPLE_FROM_LATENT_DISTRIBUTION:
        return latent_distribution.sample(sample_shape=SHAPE).to(device)
    else:
        index = random.randint(0, len(dataset) -1)
        return autoencoder.encode(dataset[index].to(device))


previous_model = None
next_model = get_random()

for epoch in count():
    try:
        previous_model = next_model
        next_model = get_random()

        start = time.perf_counter()
        end = start + TRANSITION_TIME
        while time.perf_counter() < end:
            progress = min((time.perf_counter() - start) / TRANSITION_TIME, 1.0)
            model = previous_model * (1 - progress) + next_model * progress
            voxels = autoencoder.decode(model).detach().cpu()
            viewer.set_voxels(voxels)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break