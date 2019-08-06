import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from itertools import count
import time
import random
from voxel.viewer import VoxelViewer
from tqdm import tqdm
import sys

from model import SDFAutoencoder, LATENT_CODE_SIZE
from util import device

if "nogui" not in sys.argv:
    viewer = VoxelViewer()

POINTCLOUD_SIZE = 100000
LIMIT_MODEL_COUNT = 20

data = torch.load("data/dataset-sdf-clouds.to")
points = data[:LIMIT_MODEL_COUNT * POINTCLOUD_SIZE, :3]
points = points.cuda()
sdf = data[:LIMIT_MODEL_COUNT * POINTCLOUD_SIZE, 3].to(device)
del data

MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE
BATCH_SIZE = 4000
SDF_CUTOFF = 0.1

SIGMA = 0.01

torch.clamp_(sdf, -SDF_CUTOFF, SDF_CUTOFF)

sdf_autoencoder = SDFAutoencoder()
sdf_autoencoder.to(device)

if "continue" in sys.argv:
    sdf_autoencoder.load()

optimizer = optim.Adam(sdf_autoencoder.parameters(), lr=1e-5)
criterion = nn.MSELoss()

def create_batches():
    batches = []
    batch_count = int(POINTCLOUD_SIZE / BATCH_SIZE)
    for i in range(MODEL_COUNT):
        indices = np.arange(POINTCLOUD_SIZE) + i * POINTCLOUD_SIZE
        np.random.shuffle(indices)
        for j in range(batch_count):
            batches.append(indices[j * BATCH_SIZE:(j+1)*BATCH_SIZE])
    random.shuffle(batches)
    return batches

def train():
    for epoch in count():
        loss_values = []
        batch_index = 0
        epoch_start_time = time.time()
        for batch in tqdm(create_batches()):
            indices = torch.tensor(batch, device = device)
            
            indices_encoder = indices[:BATCH_SIZE // 2]
            indices_decoder = indices[BATCH_SIZE // 2:]
            
            sdf_autoencoder.zero_grad()
            latent_code = sdf_autoencoder.encode(points[indices_encoder], sdf[indices_encoder])
            output = sdf_autoencoder.decode(latent_code, points[indices_decoder])
            output = output.clamp(-SDF_CUTOFF, SDF_CUTOFF)
            loss = torch.mean(torch.abs(output - sdf[indices_decoder])) + SIGMA * torch.mean(torch.pow(latent_code, 2))
            loss.backward()
            loss_values.append(loss.item())
            optimizer.step()

            if batch_index % 200 == 0 and "nogui" not in sys.argv:
                try:
                    viewer.set_mesh(sdf_autoencoder.decoder.get_mesh(latent_code))
                except ValueError:
                    pass

            batch_index += 1
        
        print("Epoch {:d}. Loss: {:.8f}".format(epoch, np.mean(loss_values)))
        sdf_autoencoder.save()

train()