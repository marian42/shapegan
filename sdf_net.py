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

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()        

        self.layers = nn.Sequential(
            nn.Linear(in_features = 3, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 1),
            nn.Tanh()
        )

        self.cuda()

    def forward(self, x):
        return self.layers.forward(x).squeeze()

    def get_mesh(self, voxel_count = 64):
        sample_points = np.meshgrid(
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count)
        )
        sample_points = np.stack(sample_points)
        sample_points = sample_points.reshape(3, -1).transpose()
        sample_points = torch.tensor(sample_points, dtype=torch.float, device = device)
        with torch.no_grad():
            distances = self.forward(sample_points).cpu().numpy()
        distances = distances.reshape(voxel_count, voxel_count, voxel_count)
        distances = np.swapaxes(distances, 0, 1)

        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(distances, level=0, spacing=(2.0 / voxel_count, 2.0 / voxel_count, 2.0 / voxel_count))
        vertices -= 1
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh


viewer = VoxelViewer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load("sdf_test.npy")
points = torch.tensor(data[:, :3], device=device, dtype=torch.float)
sdf = torch.tensor(data[:, 3], device=device, dtype=torch.float)
sdf = torch.clamp(sdf, -0.1, 0.1)

SIZE = sdf.shape[0]
BATCH_SIZE = 128

sdf_net = SDFNet()

optimizer = optim.Adam(sdf_net.parameters(), lr=1e-4)
criterion = nn.MSELoss()

def create_batches():
    batch_count = int(SIZE / BATCH_SIZE)
    indices = list(range(SIZE))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield indices[(batch_count - 1) * BATCH_SIZE:]

def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches():
            indices = torch.tensor(batch, device = device)
            sample = points[indices, :]
            labels = sdf[indices]

            sdf_net.zero_grad()
            output = sdf_net.forward(sample)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            batch_index += 1
        
        test_output = sdf_net.forward(points[:1000, :])
        loss = criterion(test_output, sdf[:1000]).item()
        print("Epoch {:d}. Loss: {:.8f}".format(epoch, loss))

        mesh = sdf_net.get_mesh()
        viewer.set_mesh(mesh)
train()