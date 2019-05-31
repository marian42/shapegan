import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

MODEL_PATH = "models"

import os
from loss import inception_score

LATENT_CODE_SIZE = 32

standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

# Based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convT1 = nn.ConvTranspose3d(in_channels = LATENT_CODE_SIZE, out_channels = 256, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm3d(256)
        self.convT2 = nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.convT3 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(64)
        self.convT4 = nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        self.tanh = nn.Tanh()

        self.filename = "generator.to"

        self.cuda()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.convT1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.convT2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.convT3(x)), 0.2)
        x = self.tanh(self.convT4(x))
        return x

    def generate(self, device, sample_size = 1):
        shape = torch.Size([sample_size, LATENT_CODE_SIZE, 1, 1, 1])
        x = standard_normal_distribution.sample(shape).to(device)
        return self.forward(x)

    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

    def copy_autoencoder_weights(self, autoencoder):
        def copy(source, destination):
            destination.load_state_dict(source.state_dict(), strict=False)

        copy(autoencoder.convT5, self.convT1)
        copy(autoencoder.convT6, self.convT2)
        copy(autoencoder.convT7, self.convT3)
        copy(autoencoder.convT8, self.convT4)
        copy(autoencoder.bn5, self.bn1)
        copy(autoencoder.bn6, self.bn2)
        copy(autoencoder.bn7, self.bn3)

    def get_inception_score(self, device, sample_size = 1000):
        sample = self.generate(device, sample_size)
        return inception_score(sample)
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv3d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1)
        self.sigmoid = nn.Sigmoid()

        self.use_sigmoid = True
        self.filename = "discriminator.to"

        self.cuda()


    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        x = x.squeeze()
        return x
    
    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

    def clip_weights(self, value):
        for parameter in self.parameters():
            parameter.data.clamp_(-value, value)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4_mean = nn.Conv3d(in_channels = 64, out_channels = LATENT_CODE_SIZE, kernel_size = 4, stride = 1)
        self.conv4_log_variance = nn.Conv3d(in_channels = 64, out_channels = LATENT_CODE_SIZE, kernel_size = 4, stride = 1)
        
        self.convT5 = nn.ConvTranspose3d(in_channels = LATENT_CODE_SIZE, out_channels = 64, kernel_size = 4, stride = 1)
        self.bn5 = nn.BatchNorm3d(64)
        self.convT6 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.bn6 = nn.BatchNorm3d(32)
        self.convT7 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding = 1)
        self.bn7 = nn.BatchNorm3d(16)
        self.convT8 = nn.ConvTranspose3d(in_channels = 16, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)
        self.tanh = nn.Tanh()

        self.filename = "autoencoder.to"

        self.cuda()

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        mean = self.conv4_mean(x).squeeze()
        log_variance = self.conv4_log_variance(x).squeeze()
        return mean, log_variance

    def create_latent_code(self, mean, log_variance, device):
        standard_deviation = torch.exp(log_variance * 0.5)
        eps = standard_normal_distribution.sample(mean.shape).to(device)
        return mean + standard_deviation * eps

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim = 0)  # add dimension for channels
        while len(x.shape) < 5: 
            x = x.unsqueeze(dim = len(x.shape)) # add 3 voxel dimensions
        
        x = F.leaky_relu(self.bn5(self.convT5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.convT6(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.convT7(x)), 0.2)
        x = self.tanh(self.convT8(x))
        x = x.squeeze()
        return x

    def forward(self, x, device):
        mean, log_variance = self.encode(x)
        z = self.create_latent_code(mean, log_variance, device)
        x = self.decode(z)
        return x, mean, log_variance
    
    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        from dataset import dataset as dataset
        
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 4)
        self.mp1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(in_channels = 12, out_channels = 16, kernel_size = 4)
        self.mp2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 4)
        self.fc = nn.Linear(in_features = 32 * 2 * 2 * 2, out_features = dataset.label_count)
        
        self.softmax = torch.nn.Softmax(dim=1)

        self.filename = "classifier.to"

        self.cuda()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.softmax(self.fc(x))
        return x
    
    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

