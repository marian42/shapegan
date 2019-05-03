from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

from voxel.viewer import VoxelViewer
from model import Generator, Discriminator, Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

generator = Generator()
generator.load()

discriminator = Discriminator()
discriminator.load()

def load_from_autoencoder():
    autoencoder = Autoencoder()
    autoencoder.load()
    generator.copy_autoencoder_weights(autoencoder)


#load_from_autoencoder()

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0025, betas = (0.5, 0.5))

discriminator_criterion = torch.nn.functional.binary_cross_entropy
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas = (0.5, 0.5))

viewer = VoxelViewer()

BATCH_SIZE = 20

generator_quality = 0.5
valid_sample_prediction = 0.5

def create_valid_sample(size = BATCH_SIZE):
    indices = torch.tensor(random.sample(range(dataset_size), BATCH_SIZE), device = device)
    valid_sample = dataset[indices, :, :, :]
    return valid_sample
            

for epoch in count():
    try:
        generator_optimizer.zero_grad()

        # generate samples
        fake_sample = generator.generate(device, batch_size = 2)
        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())

        if generator_quality < 0.8:
            # train generator
            generator_optimizer.zero_grad()
             
            valid_sample = create_valid_sample()
            valid_discriminator_output = discriminator.forward(valid_sample)
            
            fake_sample = generator.generate(device, batch_size = BATCH_SIZE)
            fake_discriminator_output = discriminator.forward(fake_sample)
            fake_loss = torch.mean(torch.log(valid_discriminator_output) + torch.log(1 - fake_discriminator_output))
            fake_loss.backward()
            generator_optimizer.step()
            generator_quality = np.average(fake_discriminator_output.detach().cpu().numpy())
            
        
        if generator_quality > 0.2:
            # train discriminator
            
            discriminator_optimizer.zero_grad()
            fake_sample = generator.generate(device, batch_size = BATCH_SIZE).detach()
            valid_sample = create_valid_sample().unsqueeze(dim=1)
            combined_sample = torch.cat((fake_sample, valid_sample), dim = 0)
            discriminator_output = discriminator.forward(combined_sample)
            
            desired_output = torch.cat((torch.zeros(BATCH_SIZE), torch.ones(BATCH_SIZE))).to(device)
            loss = discriminator_criterion(discriminator_output, desired_output)
            loss.backward()
            discriminator_optimizer.step()

            generator_quality = np.average(discriminator_output[:BATCH_SIZE].detach().cpu().numpy())
            valid_sample_prediction = np.average(discriminator_output[BATCH_SIZE:].detach().cpu().numpy())
            
        if epoch % 100 == 0:
            generator.save()
            discriminator.save()
            print("Model parameters saved.")

        print("epoch " + str(epoch) + ": prediction on fake samples: " + '{0:.4f}'.format(generator_quality) + ", prediction on valid samples: " + '{0:.4f}'.format(valid_sample_prediction))
    except KeyboardInterrupt:
        viewer.stop()
        break
