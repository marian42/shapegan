from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

import numpy as np

from voxelviewer import VoxelViewer
from voxels import load_voxels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("airplanes.to").to(device)
dataset_size = dataset.shape[0]


# Based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convT1 = nn.ConvTranspose3d(in_channels = 200, out_channels = 256, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm3d(256)
        self.convT2 = nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm3d(128)
        self.convT3 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2)
        self.bn3 = nn.BatchNorm3d(64)
        self.convT4 = nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.convT1(x)))
        x = self.bn2(F.relu(self.convT2(x)))
        x = self.bn3(F.relu(self.convT3(x)))
        x = self.sigmoid(self.convT4(x))
        return x

    def generate(self, batch_size = 1):
        shape = [batch_size, 200, 1, 1, 1]
        distribution = torch.distributions.Normal(torch.zeros(shape), torch.ones(shape))
        x = distribution.sample().to(device)
        return self.forward(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
        x = self.bn1(F.leaky_relu(self.conv1(x), 0.2))
        x = self.bn2(F.leaky_relu(self.conv2(x), 0.2))
        x = self.bn3(F.leaky_relu(self.conv3(x), 0.2))
        x = self.sigmoid(self.conv4(x))
        x = x.squeeze()
        return x
    

generator = Generator()
generator.cuda()

discriminator = Discriminator()
discriminator.cuda()

generator_criterion = nn.MSELoss()
generator_optimizer = optim.SGD(generator.parameters(), lr=0.02, momentum=0.9)

discriminator_criterion = nn.MSELoss()
discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=0.02, momentum=0.9)

viewer = VoxelViewer()

BATCH_SIZE = 10

generator_quality = 0.5
valid_sample_prediction = 0.5

for epoch in count():
    try:
        generator_optimizer.zero_grad()

        # generate samples
        fake_sample = generator.generate(batch_size = BATCH_SIZE)
        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())

        if generator_quality < 0.99 or True:
            # train generator
            fake_discriminator_output = discriminator.forward(fake_sample)
            generator_quality = np.average(fake_discriminator_output.detach().cpu().numpy())      
            fake_loss = generator_criterion(fake_discriminator_output, torch.ones(BATCH_SIZE, device = device))
            fake_loss.backward()
            generator_optimizer.step()
        
        if generator_quality > 0.01 or True:
            # train discriminator on fake samples
            discriminator_optimizer.zero_grad()
            fake_discriminator_output = discriminator.forward(fake_sample.detach())
            loss = discriminator_criterion(fake_discriminator_output, torch.zeros(BATCH_SIZE, device = device))
            loss.backward()
            discriminator_optimizer.step()
            generator_quality = np.average(fake_discriminator_output.detach().cpu().numpy())
        
            # train discriminator on real samples
            discriminator_optimizer.zero_grad()
            indices = torch.tensor(random.sample(range(dataset_size), BATCH_SIZE), device = device)
            valid_sample = dataset[indices, :, :, :]
            valid_discriminator_output = discriminator.forward(valid_sample)
            loss = discriminator_criterion(valid_discriminator_output, torch.ones(BATCH_SIZE, device = device))
            loss.backward()
            discriminator_optimizer.step()
            valid_sample_prediction = np.average(valid_discriminator_output.detach().cpu().numpy())

        print("epoch " + str(epoch) + ": prediction on fake samples: " + '{0:.4f}'.format(generator_quality) + ", prediction on valid samples: " + '{0:.4f}'.format(valid_sample_prediction))
    except KeyboardInterrupt:
        viewer.stop()
        break