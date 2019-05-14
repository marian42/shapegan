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
dataset = dataset
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

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0025)

discriminator_criterion = torch.nn.functional.mse_loss
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

viewer = VoxelViewer()

BATCH_SIZE = 64

def contains_nan(tensor):
    return torch.sum(torch.isnan(tensor)).item() > 0

valid_target_default = torch.ones(BATCH_SIZE, requires_grad=False).to(device)
fake_target_default = torch.zeros(BATCH_SIZE, requires_grad=False).to(device)

def create_batches(sample_count, batch_size):
    batch_count = int(sample_count / batch_size)
    indices = list(range(sample_count))
    random.shuffle(list(range(sample_count)))
    for i in range(batch_count - 1):
        yield indices[i * batch_size:(i+1)*batch_size]
    yield indices[(batch_count - 1) * batch_size:]

def train():
    fake_sample_prediction = 0.5
    valid_sample_prediction = 0.5

    for epoch in count():
        batch_index = 0
        for batch in create_batches(dataset_size, BATCH_SIZE):
            try:
                # train generator
                generator_optimizer.zero_grad()
                    
                fake_sample = generator.generate(device, batch_size = BATCH_SIZE)
                viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                if contains_nan(fake_sample):
                    print("fake_sample contains NaN values. Skipping...")
                    break
                
                fake_discriminator_output = discriminator.forward(fake_sample)
                fake_loss = torch.mean(-torch.log(fake_discriminator_output))
                fake_loss.backward()
                generator_optimizer.step()
                    
                
                # train discriminator
                indices = torch.tensor(batch, device = device)
                current_batch_size = indices.shape[0] # equals BATCH_SIZE for all batches except the last one
                fake_target = fake_target_default if current_batch_size == BATCH_SIZE else torch.zeros(current_batch_size, requires_grad=False).to(device)
                valid_target = valid_target_default if current_batch_size == BATCH_SIZE else torch.ones(current_batch_size, requires_grad=False).to(device)

                discriminator_optimizer.zero_grad()
                fake_sample = generator.generate(device, batch_size = current_batch_size).detach()
                discriminator_output_fake = discriminator.forward(fake_sample)
                fake_loss = discriminator_criterion(discriminator_output_fake, fake_target)
                fake_loss.backward()
                discriminator_optimizer.step()

                discriminator_optimizer.zero_grad()
                valid_sample = dataset[indices, :, :, :]
                discriminator_output_valid = discriminator.forward(valid_sample)
                valid_loss = discriminator_criterion(discriminator_output_valid, valid_target)
                valid_loss.backward()
                discriminator_optimizer.step()
                
                fake_sample_prediction = torch.mean(discriminator_output_fake).item()
                valid_sample_prediction = torch.mean(discriminator_output_valid).item()
                batch_index += 1
                print("epoch " + str(epoch) + ", batch " + str(batch_index) + ": prediction on fake samples: " + '{0:.4f}'.format(fake_sample_prediction) + ", prediction on valid samples: " + '{0:.4f}'.format(valid_sample_prediction))
            except KeyboardInterrupt:
                viewer.stop()
                return
        
        generator.save()
        discriminator.save()
        print("Model parameters saved.")


train()                
