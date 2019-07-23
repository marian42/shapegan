from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import time
import sys

from model import SDFNet, Discriminator, standard_normal_distribution, LATENT_CODE_SIZE
from util import create_text_slice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import dataset as dataset, VOXEL_SIZE, SDF_CLIPPING
from loss import inception_score
from util import create_text_slice

generator = SDFNet()
generator.filename = 'hybrid_gan_generator.to'

discriminator = Discriminator()
discriminator.filename = 'hybrid_gan_discriminator.to'

dataset.voxels *= SDF_CLIPPING # Undo scaling of SDF values that is done by the dataset loader

if "continue" in sys.argv:
    generator.load()
    discriminator.load()

log_file = open("plots/hybrid_gan_training.csv", "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

discriminator_criterion = torch.nn.functional.binary_cross_entropy
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer()

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

def create_grid_points():
    sample_points = np.meshgrid(
        np.linspace(-1, 1, VOXEL_SIZE),
        np.linspace(-1, 1, VOXEL_SIZE),
        np.linspace(-1, 1, VOXEL_SIZE)
    )
    sample_points = np.stack(sample_points).astype(np.float32)
    sample_points = np.swapaxes(sample_points, 1, 2)
    sample_points = sample_points.reshape(3, -1).transpose()
    sample_points = torch.tensor(sample_points, device=device)
    return sample_points

def sample_latent_codes(current_batch_size):
    latent_codes = standard_normal_distribution.sample(sample_shape=[current_batch_size, LATENT_CODE_SIZE]).to(device)
    latent_codes = latent_codes.repeat((1, 1, grid_points.shape[0])).reshape(-1, LATENT_CODE_SIZE)
    return latent_codes
                

grid_points = create_grid_points()

def train():
    fake_sample_prediction = 0.5
    valid_sample_prediction = 0.5

    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches(dataset.size, BATCH_SIZE):
            try:
                indices = torch.tensor(batch, device = device)
                current_batch_size = indices.shape[0] # equals BATCH_SIZE for all batches except the last one
                batch_grid_points = grid_points.repeat((current_batch_size, 1))

                # train generator
                generator_optimizer.zero_grad()
                
                latent_codes = sample_latent_codes(current_batch_size)
                fake_sample = generator.forward(batch_grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
                if batch_index % 20 == 0 and show_viewer:
                    viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                if batch_index % 20 == 0 and "show_slice" in sys.argv:
                    print(create_text_slice(fake_sample[0, :, :, :] / SDF_CLIPPING))
                
                fake_discriminator_output = discriminator.forward(fake_sample)
                fake_loss = torch.mean(-torch.log(fake_discriminator_output))
                fake_loss.backward()
                generator_optimizer.step()
                    
                
                # train discriminator on fake samples
                fake_target = fake_target_default if current_batch_size == BATCH_SIZE else torch.zeros(current_batch_size, requires_grad=False).to(device)
                valid_target = valid_target_default if current_batch_size == BATCH_SIZE else torch.ones(current_batch_size, requires_grad=False).to(device)

                discriminator_optimizer.zero_grad()                
                latent_codes = sample_latent_codes(current_batch_size)
                fake_sample = generator.forward(batch_grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
                discriminator_output_fake = discriminator.forward(fake_sample)
                fake_loss = discriminator_criterion(discriminator_output_fake, fake_target)
                fake_loss.backward()
                discriminator_optimizer.step()

                # train discriminator on real samples
                discriminator_optimizer.zero_grad()
                valid_sample = dataset.voxels[indices, :, :, :]
                discriminator_output_valid = discriminator.forward(valid_sample)
                valid_loss = discriminator_criterion(discriminator_output_valid, valid_target)
                valid_loss.backward()
                discriminator_optimizer.step()
                
                fake_sample_prediction = torch.mean(discriminator_output_fake).item()
                valid_sample_prediction = torch.mean(discriminator_output_valid).item()
                batch_index += 1

                if "verbose" in sys.argv:
                    print("Epoch " + str(epoch) + ", batch " + str(batch_index) +
                        ": prediction on fake samples: " + '{0:.4f}'.format(fake_sample_prediction) +
                        ", prediction on valid samples: " + '{0:.4f}'.format(valid_sample_prediction))
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        generator.save()
        discriminator.save()

        if "show_slice" in sys.argv:
            latent_code = sample_latent_codes(1)
            voxels = generator.forward(grid_points, latent_code)
            voxels = voxels.reshape(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
            print(create_text_slice(voxels / SDF_CLIPPING))

        score = 0
        print('Epoch {:d} ({:.1f}s), inception score: {:.4f}'.format(epoch, time.time() - epoch_start_time, score))
        log_file.write('{:d} {:.1f} {:.4f}\n'.format(epoch, time.time() - epoch_start_time, score))
        log_file.flush()


train()
log_file.close()
