import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

GENERATOR_FILENAME = "generator.to"
DISCRIMINATOR_FILENAME = "discriminator.to"

import os

# Based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convT1 = nn.ConvTranspose3d(in_channels = 200, out_channels = 256, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm3d(256)
        self.convT2 = nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.convT3 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(64)
        self.convT4 = nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()

        self.cuda()

    def forward(self, x):
        x = self.bn1(F.relu(self.convT1(x)))
        x = self.bn2(F.relu(self.convT2(x)))
        x = self.bn3(F.relu(self.convT3(x)))
        x = self.sigmoid(self.convT4(x))
        return x

    def generate(self, device, batch_size = 1):
        shape = [batch_size, 200, 1, 1, 1]
        distribution = torch.distributions.Normal(torch.zeros(shape), torch.ones(shape))
        x = distribution.sample().to(device)
        return self.forward(x)

    def load(self):
        if os.path.isfile(GENERATOR_FILENAME):
            self.load_state_dict(torch.load(GENERATOR_FILENAME))
    
    def save(self):
        torch.save(self.state_dict(), GENERATOR_FILENAME)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1)
        self.sigmoid = nn.Sigmoid()

        self.cuda()


    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
        x = self.bn1(F.leaky_relu(self.conv1(x), 0.2))
        x = self.bn2(F.leaky_relu(self.conv2(x), 0.2))
        x = self.bn3(F.leaky_relu(self.conv3(x), 0.2))
        x = self.sigmoid(self.conv4(x))
        x = x.squeeze()
        return x
    
    def load(self):
        if os.path.isfile(DISCRIMINATOR_FILENAME):
            self.load_state_dict(torch.load(DISCRIMINATOR_FILENAME))
    
    def save(self):
        torch.save(self.state_dict(), DISCRIMINATOR_FILENAME)