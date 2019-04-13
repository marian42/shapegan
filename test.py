from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py
from binvox_rw import read_as_3d_array

from voxelviewer import VoxelViewer

FILENAME = '/home/marian/shapenet/ShapeNetCore.v2/02747177/2f00a785c9ac57c07f48cf22f91f1702/models/model_normalized.solid.binvox'

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

voxels = read_as_3d_array(open(FILENAME, 'rb'))
voxels = voxels.data[::2, ::2, ::2].astype(float) # 128^3 to 64^3, bool to float
voxels = torch.tensor(voxels, device = device, dtype = torch.float)
voxels = torch.unsqueeze(voxels, dim = 0) # add channel dimension, final layer has one channel


# Based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convT1 = nn.ConvTranspose3d(in_channels = 200, out_channels = 512, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm3d(512)
        self.convT2 = nn.ConvTranspose3d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(256)
        self.convT3 = nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(128)
        self.convT4 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm3d(64)
        self.convT5 = nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.convT1(x)))
        x = self.bn2(F.relu(self.convT2(x)))
        x = self.bn3(F.relu(self.convT3(x)))
        x = self.bn4(F.relu(self.convT4(x)))
        x = self.sigmoid(self.convT5(x))
        return x

    def generate(self, batch_size = 1):
        shape = [batch_size, 200, 1, 1, 1]
        distribution = torch.distributions.Normal(torch.zeros(shape), torch.ones(shape))
        x = distribution.sample().to(device)
        return self.forward(x)


generator = Generator()
generator.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)

viewer = VoxelViewer()

for epoch in count():
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = generator.generate(batch_size=20)
    loss = criterion(outputs, voxels)
    loss.backward()
    optimizer.step()

    sample = generator.generate().squeeze()
    viewer.set_voxels(sample.detach().cpu().numpy())

    print(str(epoch) + ": " + str(loss.item()))