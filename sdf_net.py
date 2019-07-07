import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import numpy as np

from itertools import count
import time
import random
import skimage
import trimesh
from voxel.viewer import VoxelViewer

from tqdm import tqdm
from model import SavableModule

import sys

LATENT_CODE_SIZE = 32
LATENT_CODES_FILENAME = "models/sdf_net_latent_codes.to"

class SDFNet(SavableModule):
    def __init__(self):
        super(SDFNet, self).__init__(filename="sdf_net.to")

        self.layers = nn.Sequential(
            nn.Linear(in_features = 3 + LATENT_CODE_SIZE, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 1),
            nn.Tanh()
        )

        self.cuda()

    def forward(self, points, latent_codes):
        x = torch.cat((points, latent_codes), dim=1)
        return self.layers.forward(x).squeeze()

    def get_mesh(self, latent_code, voxel_count = 64):
        sample_points = np.meshgrid(
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count)
        )
        sample_points = np.stack(sample_points)
        sample_points = sample_points.reshape(3, -1).transpose()
        sample_points = torch.tensor(sample_points, dtype=torch.float, device = device)
        with torch.no_grad():
            distances = self.forward(sample_points, latent_code.repeat(sample_points.shape[0], 1)).cpu().numpy()
        distances = distances.reshape(voxel_count, voxel_count, voxel_count)
        distances = np.swapaxes(distances, 0, 1)

        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(distances, level=0, spacing=(2.0 / voxel_count, 2.0 / voxel_count, 2.0 / voxel_count))
        vertices -= 1
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh




viewer = VoxelViewer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sdf_net = SDFNet()

data = torch.load("data/dataset-sdf-clouds.to")

POINTCLOUD_SIZE = 100000


points = data[:, :3]
points = points.cuda()
sdf = data[:, 3].to(device)
torch.clamp_(sdf, -0.1, 0.1)
del data

MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE


if "continue" in sys.argv:
    sdf_net.load()
    latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)
else:    
    standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
    latent_codes = standard_normal_distribution.sample((MODEL_COUNT, LATENT_CODE_SIZE)).to(device)

latent_codes.requires_grad = True


BATCH_SIZE = 2048

network_optimizer = optim.Adam(sdf_net.parameters(), lr=1e-4)
latent_code_optimizer = optim.Adam([latent_codes], lr=1e-6)
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

            batch_index += 1

            if batch_index % 1000 == 0:
                try:
                    #viewer.set_mesh(sdf_net.get_mesh(latent_codes[random.randrange(MODEL_COUNT), :]))
                    viewer.set_mesh(sdf_net.get_mesh(latent_codes[0, :]))
                except ValueError:
                    pass
        
        print("Epoch {:d}. Loss: {:.8f}".format(epoch, loss.item()))
        sdf_net.save()
        torch.save(latent_codes, LATENT_CODES_FILENAME)

train()