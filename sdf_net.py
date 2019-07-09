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
from model import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

viewer = VoxelViewer()
data = torch.load("data/dataset-sdf-clouds.to")
points = data[:, :3]
points = points.cuda()
sdf = data[:, 3].to(device)
del data

POINTCLOUD_SIZE = 100000
MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE
BATCH_SIZE = 2048
SDF_CUTOFF = 0.1

torch.clamp_(sdf, -SDF_CUTOFF, SDF_CUTOFF)
sdf /= SDF_CUTOFF

sdf_net = SDFNet()
if "continue" in sys.argv:
    sdf_net.load()
    latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)
else:    
    standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
    latent_codes = standard_normal_distribution.sample((MODEL_COUNT, LATENT_CODE_SIZE)).to(device)
latent_codes.requires_grad = True

network_optimizer = optim.Adam(sdf_net.parameters(), lr=1e-5)
latent_code_optimizer = optim.Adam([latent_codes], lr=1e-5)
criterion = nn.MSELoss()

def create_batches():
    size = points.shape[0]
    batch_count = int(size / BATCH_SIZE)
    indices = np.arange(size)
    np.random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield indices[(batch_count - 1) * BATCH_SIZE:]

def train():
    loss_values = []
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in tqdm(list(create_batches())):
            indices = torch.tensor(batch, device = device)
            model_indices = indices / POINTCLOUD_SIZE
            
            batch_latent_codes = latent_codes[model_indices, :]
            batch_points = points[indices, :]
            batch_sdf = sdf[indices]

            sdf_net.zero_grad()
            output = sdf_net.forward(batch_points, batch_latent_codes)
            loss = criterion(output, batch_sdf)
            loss.backward()
            network_optimizer.step()
            latent_code_optimizer.step()
            loss_values.append(loss.item())

            if batch_index % 1000 == 0:
                try:
                    viewer.set_mesh(sdf_net.get_mesh(latent_codes[random.randrange(MODEL_COUNT), :], device))
                except ValueError:
                    pass

            batch_index += 1
        
        print("Epoch {:d}. Loss: {:.8f}".format(epoch, np.mean(loss_values)))
        sdf_net.save()
        torch.save(latent_codes, LATENT_CODES_FILENAME)

train()