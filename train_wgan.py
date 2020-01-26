from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import sys
from collections import deque

from model.gan import Generator, Discriminator
from util import device

from dataset import dataset as dataset
from util import create_text_slice
dataset.load_voxels(device)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

generator = Generator()
generator.filename = "wgan-generator.to"

critic = Discriminator()
critic.filename = "wgan-critic.to"
critic.use_sigmoid = False

if "continue" in sys.argv:
    generator.load()
    critic.load()

LEARN_RATE = 0.00005
BATCH_SIZE = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
CRITIC_WEIGHT_LIMIT = 0.01

generator_optimizer = optim.RMSprop(generator.parameters(), lr=LEARN_RATE)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARN_RATE)

log_file = open("plots/wgan_training.csv", "a" if "continue" in sys.argv else "w")

def create_batches(sample_count, batch_size):
    batch_count = int(sample_count / batch_size)
    indices = list(range(sample_count))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * batch_size:(i+1)*batch_size]
    yield indices[(batch_count - 1) * batch_size:]

def train():
    history_fake = deque(maxlen=50)
    history_real = deque(maxlen=50)

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
                fake_sample = generator.generate(sample_size = current_batch_size).detach()
                fake_critic_output = critic(fake_sample)
                valid_critic_output = critic(valid_sample)
                critic_loss = torch.mean(fake_critic_output) - torch.mean(valid_critic_output)
                critic_loss.backward()
                critic_optimizer.step()
                critic.clip_weights(CRITIC_WEIGHT_LIMIT)
               
                # train generator
                if batch_index % CRITIC_UPDATES_PER_GENERATOR_UPDATE == 0:
                    generator.zero_grad()
                    critic.zero_grad()
                       
                    fake_sample = generator.generate(sample_size = BATCH_SIZE)
                    if show_viewer:
                        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                    fake_critic_output = critic(fake_sample)
                    generator_loss = -torch.mean(fake_critic_output)                
                    generator_loss.backward()
                    generator_optimizer.step()
                
                    history_fake.append(torch.mean(fake_critic_output).item())
                    history_real.append(torch.mean(valid_critic_output).item())
                    if "verbose" in sys.argv:
                        print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                            + ": fake value: " + '{0:.1f}'.format(history_fake[-1]) \
                            + ", valid value: " + '{0:.1f}'.format(history_real[-1]))
                batch_index += 1                
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        generator.save()
        critic.save()

        if epoch % 20 == 0:
            generator.save(epoch=epoch)
            critic.save(epoch=epoch)

        if "show_slice" in sys.argv:
            voxels = generator.generate().squeeze()
            print(create_text_slice(voxels))

        epoch_duration = time.time() - epoch_start_time
        fake_prediction = np.mean(history_fake)
        valid_prediction = np.mean(history_real)
        print('Epoch {:d} ({:.1f}s), critic values: {:.2f}, {:.2f}'.format(
            epoch, epoch_duration, fake_prediction, valid_prediction))
        log_file.write("{:d} {:.1f} {:.2f} {:.2f}\n".format(
            epoch, epoch_duration, fake_prediction, valid_prediction))
        log_file.flush()


train()
