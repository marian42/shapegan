from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import sys
from collections import deque
from tqdm import tqdm

from model.sdf_net import SDFNet
from model.gan import Discriminator, LATENT_CODE_SIZE
from util import create_text_slice, device, standard_normal_distribution, get_voxel_coordinates

from dataset import dataset as dataset, VOXEL_RESOLUTION, SDF_CLIPPING
from util import create_text_slice
dataset.rescale_sdf = False
dataset.load_voxels(device)

generator = SDFNet()
generator.filename = 'hybrid_gan_generator.to'

discriminator = Discriminator()
discriminator.filename = 'hybrid_gan_discriminator.to'

if "continue" in sys.argv:
    generator.load()
    discriminator.load()

LOG_FILE_NAME = "plots/hybrid_gan_training.csv"
first_epoch = 0
if 'continue' in sys.argv:
    log_file_contents = open(LOG_FILE_NAME, 'r').readlines()
    first_epoch = len(log_file_contents)

log_file = open(LOG_FILE_NAME, "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)

discriminator_criterion = torch.nn.functional.binary_cross_entropy
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

BATCH_SIZE = 8

valid_target_default = torch.ones(BATCH_SIZE, requires_grad=False).to(device)
fake_target_default = torch.zeros(BATCH_SIZE, requires_grad=False).to(device)

def create_batches(sample_count, batch_size):
    batch_count = int(sample_count / batch_size)
    indices = list(range(sample_count))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * batch_size:(i+1)*batch_size]
    yield indices[(batch_count - 1) * batch_size:]

def sample_latent_codes(current_batch_size):
    latent_codes = standard_normal_distribution.sample(sample_shape=[current_batch_size, LATENT_CODE_SIZE]).to(device)
    latent_codes = latent_codes.repeat((1, 1, grid_points.shape[0])).reshape(-1, LATENT_CODE_SIZE)
    return latent_codes

grid_points = get_voxel_coordinates(VOXEL_RESOLUTION, return_torch_tensor=True)
history_fake = deque(maxlen=50)
history_real = deque(maxlen=50)

def train():
    for epoch in count(start=first_epoch):
        batch_index = 0
        epoch_start_time = time.time()
        for batch in tqdm(list(create_batches(dataset.size, BATCH_SIZE)), desc='Epoch {:d}'.format(epoch)):
            try:
                indices = torch.tensor(batch, device = device)
                current_batch_size = indices.shape[0] # equals BATCH_SIZE for all batches except the last one
                batch_grid_points = grid_points.repeat((current_batch_size, 1))

                # train generator
                generator_optimizer.zero_grad()
                
                latent_codes = sample_latent_codes(current_batch_size)
                fake_sample = generator(batch_grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                if batch_index % 20 == 0 and show_viewer:
                    viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                if batch_index % 20 == 0 and "show_slice" in sys.argv:
                    print(create_text_slice(fake_sample[0, :, :, :] / SDF_CLIPPING))
                
                fake_discriminator_output = discriminator(fake_sample)
                fake_loss = torch.mean(-torch.log(fake_discriminator_output))
                fake_loss.backward()
                generator_optimizer.step()
                    
                
                # train discriminator on fake samples                
                fake_target = fake_target_default if current_batch_size == BATCH_SIZE else torch.zeros(current_batch_size, requires_grad=False).to(device)
                valid_target = valid_target_default if current_batch_size == BATCH_SIZE else torch.ones(current_batch_size, requires_grad=False).to(device)

                discriminator_optimizer.zero_grad()                
                latent_codes = sample_latent_codes(current_batch_size)
                fake_sample = generator(batch_grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                discriminator_output_fake = discriminator(fake_sample)
                fake_loss = discriminator_criterion(discriminator_output_fake, fake_target)
                fake_loss.backward()
                discriminator_optimizer.step()

                # train discriminator on real samples
                discriminator_optimizer.zero_grad()
                valid_sample = dataset.voxels[indices, :, :, :]
                discriminator_output_valid = discriminator(valid_sample)
                valid_loss = discriminator_criterion(discriminator_output_valid, valid_target)
                valid_loss.backward()
                discriminator_optimizer.step()
                
                history_fake.append(torch.mean(discriminator_output_fake).item())
                history_real.append(torch.mean(discriminator_output_valid).item())
                batch_index += 1

                if "verbose" in sys.argv:
                    print("Epoch " + str(epoch) + ", batch " + str(batch_index) +
                        ": prediction on fake samples: " + '{0:.4f}'.format(history_fake[-1]) +
                        ", prediction on valid samples: " + '{0:.4f}'.format(history_real[-1]))
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        prediction_fake = np.mean(history_fake)
        prediction_real = np.mean(history_real)

        print('Epoch {:d} ({:.1f}s), prediction on fake: {:.4f}, prediction on real: {:.4f}'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        
        if abs(prediction_fake - prediction_real) > 0.1:
            print("Network diverged.")
            exit()

        generator.save()
        discriminator.save()

        generator.save(epoch=epoch)
        discriminator.save(epoch=epoch)

        if "show_slice" in sys.argv:
            latent_code = sample_latent_codes(1)
            voxels = generator(grid_points, latent_code)
            voxels = voxels.reshape(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
            print(create_text_slice(voxels / SDF_CLIPPING))
        
        log_file.write('{:d} {:.1f} {:.4f} {:.4f}\n'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        log_file.flush()


train()
log_file.close()
