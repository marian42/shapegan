from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import sys

from model import Generator, Discriminator, Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset = dataset
dataset_size = dataset.shape[0]

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer()

generator = Generator()
generator.filename = "wgan-generator.to"
generator.load()

critic = Discriminator()
critic.filename = "wgan-critic.to"
critic.use_sigmoid = False
critic.load()

def load_from_autoencoder():
    autoencoder = Autoencoder()
    autoencoder.load()
    generator.copy_autoencoder_weights(autoencoder)
#load_from_autoencoder()

LEARN_RATE = 0.00005
BATCH_SIZE = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
CRITIC_WEIGHT_LIMIT = 0.01

generator_optimizer = optim.RMSprop(generator.parameters(), lr=LEARN_RATE)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARN_RATE)


def create_batches(sample_count, batch_size):
    batch_count = int(sample_count / batch_size)
    indices = list(range(sample_count))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * batch_size:(i+1)*batch_size]
    yield indices[(batch_count - 1) * batch_size:]

def train():
    fake_sample_prediction = 0.5
    valid_sample_prediction = 0.5

    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches(dataset_size, BATCH_SIZE):
            try:
                # train critic
                indices = torch.tensor(batch, device = device)
                current_batch_size = indices.shape[0] # equals BATCH_SIZE for all batches except the last one
                
                generator.zero_grad()
                critic.zero_grad()

                valid_sample = dataset[indices, :, :, :]                
                fake_sample = generator.generate(device, batch_size = current_batch_size).detach()
                fake_critic_output = critic.forward(fake_sample)
                valid_critic_output = critic.forward(valid_sample)
                critic_loss = -(torch.mean(valid_critic_output) - torch.mean(fake_critic_output))
                critic_loss.backward()
                critic_optimizer.step()
                critic.clip_weights(CRITIC_WEIGHT_LIMIT)
               
                # train generator
                if batch_index % CRITIC_UPDATES_PER_GENERATOR_UPDATE == 0:
                    generator.zero_grad()
                    critic.zero_grad()
                       
                    fake_sample = generator.generate(device, batch_size = BATCH_SIZE)
                    if show_viewer:
                        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                    fake_critic_output = critic.forward(fake_sample)
                    generator_loss = -torch.mean(fake_critic_output)                
                    generator_loss.backward()
                    generator_optimizer.step()
                
                
                    fake_sample_prediction = torch.mean(fake_critic_output).item()
                    valid_sample_prediction = torch.mean(valid_critic_output).item()
                    print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                        + ": fake value: " + '{0:.1f}'.format(fake_sample_prediction) \
                        + ", valid value: " + '{0:.1f}'.format(valid_sample_prediction))
                batch_index += 1                
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        generator.save()
        critic.save()
        print("Model parameters saved. Epoch took " + '{0:.1f}'.format(time.time() - epoch_start_time) + "s.")


train()                
