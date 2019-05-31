from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import sys

from model import Generator, Discriminator, Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import dataset as dataset

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer()

generator = Generator()
generator.filename = "wgan-generator.to"

critic = Discriminator()
critic.filename = "wgan-critic.to"
critic.use_sigmoid = False

if "continue" in sys.argv:
    generator.load()
    critic.load()

if "copy_autoencoder_weights" in sys.argv:
    autoencoder = Autoencoder()
    autoencoder.load()
    generator.copy_autoencoder_weights(autoencoder)


LEARN_RATE = 0.00005
BATCH_SIZE = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
CRITIC_WEIGHT_LIMIT = 0.01

generator_optimizer = optim.RMSprop(generator.parameters(), lr=LEARN_RATE)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARN_RATE)

print('Inception score: {:.4f}'.format(generator.get_inception_score(device)))

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
        for batch in create_batches(dataset.size, BATCH_SIZE):
            try:
                # train critic
                indices = torch.tensor(batch, device = device)
                current_batch_size = indices.shape[0] # equals BATCH_SIZE for all batches except the last one
                
                generator.zero_grad()
                critic.zero_grad()

                valid_sample = dataset.voxels[indices, :, :, :]                
                fake_sample = generator.generate(device, sample_size = current_batch_size).detach()
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
                       
                    fake_sample = generator.generate(device, sample_size = BATCH_SIZE)
                    if show_viewer:
                        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                    fake_critic_output = critic.forward(fake_sample)
                    generator_loss = -torch.mean(fake_critic_output)                
                    generator_loss.backward()
                    generator_optimizer.step()
                
                
                    fake_sample_prediction = torch.mean(fake_critic_output).item()
                    valid_sample_prediction = torch.mean(valid_critic_output).item()
                    if "verbose" in sys.argv:
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
        print('Epoch {:d} ({:.1f}s), inception score: {:.4f}, critic values: {:.2f}, {:.2f}'.format(
            epoch,
            time.time() - epoch_start_time,
            generator.get_inception_score(device, sample_size=800),
            fake_sample_prediction,
            valid_sample_prediction))


train()                
