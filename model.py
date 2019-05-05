import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

GENERATOR_FILENAME = "data/generator.to"
DISCRIMINATOR_FILENAME = "data/discriminator.to"
AUTOENCODER_FILENAME = "data/autoencoder.to"

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
        distribution = torch.distributions.uniform.Uniform(0, 1)
        x = distribution.sample(torch.Size(shape)).to(device)
        return self.forward(x)

    def load(self):
        if os.path.isfile(GENERATOR_FILENAME):
            self.load_state_dict(torch.load(GENERATOR_FILENAME))
    
    def save(self):
        torch.save(self.state_dict(), GENERATOR_FILENAME)

    def copy_autoencoder_weights(self, autoencoder):
        self.convT1.weight = autoencoder.convT5.weight
        self.convT2.weight = autoencoder.convT6.weight
        self.convT3.weight = autoencoder.convT7.weight
        self.convT4.weight = autoencoder.convT8.weight


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
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.sigmoid(self.conv4(x))
        x = x.squeeze()
        return x
    
    def load(self):
        if os.path.isfile(DISCRIMINATOR_FILENAME):
            self.load_state_dict(torch.load(DISCRIMINATOR_FILENAME))
    
    def save(self):
        torch.save(self.state_dict(), DISCRIMINATOR_FILENAME)

AUTOENCODER_BOTTLENECK = 200

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(in_channels = 256, out_channels = AUTOENCODER_BOTTLENECK, kernel_size = 4, stride = 1)

        self.convT5 = nn.ConvTranspose3d(in_channels = AUTOENCODER_BOTTLENECK, out_channels = 256, kernel_size = 4, stride = 1)
        self.bn5 = nn.BatchNorm3d(256)
        self.convT6 = nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn6 = nn.BatchNorm3d(128)
        self.convT7 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn7 = nn.BatchNorm3d(64)
        self.convT8 = nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()

        self.cuda()

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        x = self.bn1(F.leaky_relu(self.conv1(x), 0.2))
        x = self.bn2(F.leaky_relu(self.conv2(x), 0.2))
        x = self.bn3(F.leaky_relu(self.conv3(x), 0.2))
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.squeeze()
        return x

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim = 0)  # add dimension for channels
        while len(x.shape) < 5: 
            x = x.unsqueeze(dim = len(x.shape)) # add 3 voxel dimensions
        
        x = self.bn5(F.relu(self.convT5(x)))
        x = self.bn6(F.relu(self.convT6(x)))
        x = self.bn7(F.relu(self.convT7(x)))
        x = self.sigmoid(self.convT8(x))
        x = x.squeeze()
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def load(self):
        if os.path.isfile(AUTOENCODER_FILENAME):
            self.load_state_dict(torch.load(AUTOENCODER_FILENAME))
    
    def save(self):
        torch.save(self.state_dict(), AUTOENCODER_FILENAME)