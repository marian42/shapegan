import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

MODEL_PATH = "models"

import os
from loss import inception_score

LATENT_CODE_SIZE = 32

standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

# Based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels = LATENT_CODE_SIZE, out_channels = 256, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope = 0.2),
            
            nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope = 0.2),

            nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope = 0.2),

            nn.ConvTranspose3d(in_channels = 64, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

        self.filename = "generator.to"
        self.inception_score_latent_codes = dict()
        self.cuda()

    def forward(self, x):
        return self.layers.forward(x)

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

        raise Exception("Not implemented.")        

    def get_inception_score(self, device, sample_size = 1000):
        with torch.no_grad():
            if sample_size not in self.inception_score_latent_codes:
                shape = torch.Size([sample_size, LATENT_CODE_SIZE, 1, 1, 1])
                self.inception_score_latent_codes[sample_size] = standard_normal_distribution.sample(shape).to(device)

            sample = self.forward(self.inception_score_latent_codes[sample_size])
            return inception_score(sample)
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.use_sigmoid = True
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv3d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1),
            Lambda(lambda x: torch.sigmoid(x) if self.use_sigmoid else x)
        )

        self.filename = "discriminator.to"
        self.cuda()

    def forward(self, x):
        if (len(x.shape) < 5):
            x = x.unsqueeze(dim = 1) # add dimension for channels
            
        return self.layers.forward(x).squeeze()
    
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

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.encode_mean = nn.Conv3d(in_channels = 64, out_channels = LATENT_CODE_SIZE, kernel_size = 4, stride = 1)
        self.encode_log_variance = nn.Conv3d(in_channels = 64, out_channels = LATENT_CODE_SIZE, kernel_size = 4, stride = 1)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels = LATENT_CODE_SIZE, out_channels = 64, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 16, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            Lambda(lambda x: torch.clamp(x, -1, 1))
        )
        
        self.filename = "autoencoder-{:d}.to".format(LATENT_CODE_SIZE)
        self.inception_score_latent_codes = dict()
        self.cuda()

    def encode(self, x, device, return_mean_and_log_variance = False):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        
        x = self.encoder.forward(x)
        mean = self.encode_mean(x).squeeze()
        
        if self.training or return_mean_and_log_variance:
            log_variance = self.encode_log_variance(x).squeeze()
            standard_deviation = torch.exp(log_variance * 0.5)
            eps = standard_normal_distribution.sample(mean.shape).to(device)
        
        if self.training:
            x = mean + standard_deviation * eps
        else:
            x = mean

        if return_mean_and_log_variance:
            return x, mean, log_variance
        else:
            return x

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim = 0)  # add dimension for channels
        while len(x.shape) < 5: 
            x = x.unsqueeze(dim = len(x.shape)) # add 3 voxel dimensions
        
        x = self.decoder.forward(x)
        return x.squeeze()

    def forward(self, x, device):
        z, mean, log_variance = self.encode(x, device, return_mean_and_log_variance = True)
        x = self.decode(z)
        return x, mean, log_variance
    
    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

    def get_inception_score(self, device, sample_size = 1000):
        with torch.no_grad():
            if sample_size not in self.inception_score_latent_codes:
                shape = torch.Size([sample_size, LATENT_CODE_SIZE])
                self.inception_score_latent_codes[sample_size] = standard_normal_distribution.sample(shape).to(device)

            sample = self.decode(self.inception_score_latent_codes[sample_size])
            return inception_score(sample)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        from dataset import dataset as dataset
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 5),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(in_channels = 12, out_channels = 16, kernel_size = 5),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 5),
            nn.ReLU(inplace=True),

            Lambda(lambda x: x.view(x.shape[0], -1)),

            nn.Linear(in_features = 32, out_features = dataset.label_count),
            nn.Softmax(dim=1)
        )

        self.filename = "classifier.to"
        self.cuda()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim = 0)  # add dimension for batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim = 1)  # add dimension for channels
        
        return self.layers.forward(x)
    
    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        if os.path.isfile(self.get_filename()):
            self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

