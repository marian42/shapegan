from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import sys
from collections import deque

from model.sdf_net import SDFNet
from model.gan import Discriminator, LATENT_CODE_SIZE
from util import create_text_slice, device, standard_normal_distribution

from dataset import dataset as dataset, VOXEL_RESOLUTION, SDF_CLIPPING
from util import create_text_slice, get_voxel_coordinates

dataset.rescale_sdf = False
dataset.load_voxels(device)

LEARN_RATE = 0.00001
BATCH_SIZE = 8
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
CRITIC_WEIGHT_LIMIT = 0.01

generator = SDFNet()
generator.filename = 'hybrid_wgan_generator.to'

critic = Discriminator()
critic.filename = 'hybrid_wgan_critic.to'
critic.use_sigmoid = False

if "continue" in sys.argv:
    generator.load()
    critic.load()

LOG_FILE_NAME = "plots/hybrid_wgan_training.csv"
first_epoch = 0
if 'continue' in sys.argv:
    log_file_contents = open(LOG_FILE_NAME, 'r').readlines()
    first_epoch = len(log_file_contents)

log_file = open(LOG_FILE_NAME, "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.Adam(generator.parameters(), lr=LEARN_RATE)

critic_criterion = torch.nn.functional.binary_cross_entropy
critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARN_RATE)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

valid_target = torch.ones(BATCH_SIZE, requires_grad=False).to(device)
fake_target = torch.zeros(BATCH_SIZE, requires_grad=False).to(device)

def create_batches(sample_count):
    batch_count = int(sample_count / BATCH_SIZE)
    indices = list(range(sample_count))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]

def sample_latent_codes():
    latent_codes = standard_normal_distribution.sample(sample_shape=[BATCH_SIZE, LATENT_CODE_SIZE]).to(device)
    latent_codes = latent_codes.repeat((1, 1, VOXEL_RESOLUTION**3)).reshape(-1, LATENT_CODE_SIZE)
    return latent_codes


grid_points = get_voxel_coordinates(VOXEL_RESOLUTION, return_torch_tensor=True).repeat((BATCH_SIZE, 1))
history_fake = deque(maxlen=50)
history_real = deque(maxlen=50)

def train():
    for epoch in count(start=first_epoch):
        batch_index = 0
        epoch_start_time = time.time()
        for batch in list(create_batches(dataset.size)):
            try:
                indices = torch.tensor(batch, device = device)
                
                # train critic
                critic_optimizer.zero_grad()                
                latent_codes = sample_latent_codes()
                fake_sample = generator(grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                valid_sample = dataset.voxels[indices, :, :, :]

                critic_output_fake = critic(fake_sample)
                critic_output_valid = critic(valid_sample)

                critic_loss = torch.mean(critic_output_fake) - torch.mean(critic_output_valid)
                critic_loss.backward()
                critic_optimizer.step()
                critic.clip_weights(CRITIC_WEIGHT_LIMIT)

                # train generator
                if batch_index % CRITIC_UPDATES_PER_GENERATOR_UPDATE == 0:
                    generator_optimizer.zero_grad()
                    critic.zero_grad()
                    
                    latent_codes = sample_latent_codes()
                    fake_sample = generator(grid_points, latent_codes)
                    fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                    if batch_index % 20 == 0 and show_viewer:
                        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                    if batch_index % 20 == 0 and "show_slice" in sys.argv:
                        print(create_text_slice(fake_sample[0, :, :, :] / SDF_CLIPPING))
                    
                    critic_output_fake = critic(fake_sample)
                    fake_loss = torch.mean(-torch.log(critic_output_fake))
                    fake_loss.backward()
                    generator_optimizer.step()
                    
                    history_fake.append(torch.mean(critic_output_fake).item())
                    history_real.append(torch.mean(critic_output_valid).item())

                if "verbose" in sys.argv and batch_index % 20 == 0:
                    print("Epoch " + str(epoch) + ", batch " + str(batch_index) +
                        ": prediction on fake samples: " + '{0:.4f}'.format(history_fake[-1]) +
                        ", prediction on valid samples: " + '{0:.4f}'.format(history_real[-1]))
                
                batch_index += 1
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        generator.save()
        critic.save()

        generator.save(epoch=epoch)
        critic.save(epoch=epoch)

        prediction_fake = np.mean(history_fake)
        prediction_real = np.mean(history_real)
        print('Epoch {:d} ({:.1f}s), prediction on fake: {:.4f}, prediction on real: {:.4f}'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        log_file.write('{:d} {:.1f} {:.4f} {:.4f}\n'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        log_file.flush()


train()
log_file.close()
