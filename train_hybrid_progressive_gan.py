from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.autograd as autograd

import random
import time
import sys
from collections import deque
from tqdm import tqdm

from model.sdf_net import SDFNet
from model.progressive_gan import Discriminator, LATENT_CODE_SIZE, RESOLUTIONS
from util import create_text_slice, device, standard_normal_distribution, get_voxel_coordinates

SDF_CLIPPING = 0.1
from util import create_text_slice
from datasets import VoxelDataset
from torch.utils.data import DataLoader

def get_parameter(name, default):
    for arg in sys.argv:
        if arg.startswith(name + '='):
            return arg[len(name) + 1:]
    return default


ITERATION = int(get_parameter('iteration', 0))
# Continue with model parameters that were previously trained at the SAME iteration
# Otherwise, it will use the model parameters of the previous iteration or initialize randomly at iteration 0
CONTINUE = "continue" in sys.argv

FADE_IN_EPOCHS = 10
BATCH_SIZE = 16
GRADIENT_PENALTY_WEIGHT = 10
NUMBER_OF_EPOCHS = int(get_parameter('epochs', 250))

VOXEL_RESOLUTION = RESOLUTIONS[ITERATION]

dataset = VoxelDataset.from_split('data/chairs/voxels_{:d}/{{:s}}.npy'.format(VOXEL_RESOLUTION), 'data/chairs/train.txt')
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def get_generator_filename(iteration):
    return 'hybrid_progressive_gan_generator_{:d}.to'.format(iteration)

generator = SDFNet(device='cpu')
discriminator = Discriminator()
if not CONTINUE and ITERATION > 0:
    generator.filename = get_generator_filename(ITERATION - 1)
    generator.load()
    discriminator.set_iteration(ITERATION - 1)
    discriminator.load()
discriminator.set_iteration(ITERATION)
generator.filename = get_generator_filename(ITERATION)
if CONTINUE:
    generator.load()
    discriminator.load()

if torch.cuda.device_count() > 1:
    print("Using dataparallel with {:d} GPUs.".format(torch.cuda.device_count()))
    generator_parallel = nn.DataParallel(generator)
    discriminator_parallel = nn.DataParallel(discriminator)
else:
    generator_parallel = generator
    discriminator_parallel = discriminator

generator_parallel.to(device)
discriminator_parallel.to(device)

LOG_FILE_NAME = "plots/hybrid_gan_training_{:d}.csv".format(ITERATION)
first_epoch = 0
if 'continue' in sys.argv:
    log_file_contents = open(LOG_FILE_NAME, 'r').readlines()
    first_epoch = len(log_file_contents)

log_file = open(LOG_FILE_NAME, "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.RMSprop(generator_parallel.parameters(), lr=0.0001)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

def sample_latent_codes(current_batch_size):
    latent_codes = standard_normal_distribution.sample(sample_shape=[current_batch_size, LATENT_CODE_SIZE]).to(device)
    latent_codes = latent_codes.repeat((1, 1, grid_points.shape[0])).reshape(-1, LATENT_CODE_SIZE)
    return latent_codes

grid_points = get_voxel_coordinates(VOXEL_RESOLUTION, return_torch_tensor=True)
grid_points_default_batch = grid_points.repeat((BATCH_SIZE, 1))

history_fake = deque(maxlen=50)
history_real = deque(maxlen=50)
history_gradient_penalty = deque(maxlen=50)

def get_gradient_penalty(real_sample, fake_sample):
    alpha = torch.rand((real_sample.shape[0], 1, 1, 1), device=device).expand(real_sample.shape)

    interpolated_sample = alpha * real_sample + ((1 - alpha) * fake_sample)
    interpolated_sample.requires_grad = True
    
    discriminator_output = discriminator_parallel(interpolated_sample)

    gradients = autograd.grad(outputs=discriminator_output, inputs=interpolated_sample, grad_outputs=torch.ones(discriminator_output.shape).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((gradients.norm(2, dim=(1,2,3)) - 1) ** 2).mean() * GRADIENT_PENALTY_WEIGHT

def train():
    progress = tqdm(total=NUMBER_OF_EPOCHS * (len(dataset) // BATCH_SIZE + 1), initial=first_epoch * (len(dataset) // BATCH_SIZE + 1))

    for epoch in range(first_epoch, NUMBER_OF_EPOCHS):
        progress.desc = 'Epoch {:d}/{:d} ({:d}³)'.format(epoch, NUMBER_OF_EPOCHS, VOXEL_RESOLUTION)
        batch_index = 0
        epoch_start_time = time.time()
        for valid_sample in data_loader:
            try:
                if valid_sample.shape[0] == 1: # Skip final batch if it contains only one object
                    continue
                valid_sample = valid_sample.to(device)
                current_batch_size = valid_sample.shape[0]
                if current_batch_size == BATCH_SIZE:
                    batch_grid_points = grid_points_default_batch
                else:
                    batch_grid_points = grid_points.repeat((current_batch_size, 1))

                if not CONTINUE and ITERATION > 0:
                    discriminator.fade_in_progress = (epoch + batch_index / (len(dataset) / BATCH_SIZE)) / FADE_IN_EPOCHS

                # train generator
                if batch_index % 5 == 0:
                    generator_optimizer.zero_grad()
                    
                    latent_codes = sample_latent_codes(current_batch_size)
                    fake_sample = generator_parallel(batch_grid_points, latent_codes)
                    fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                    if batch_index % 50 == 0 and show_viewer:
                        viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                    if batch_index % 50 == 0 and "show_slice" in sys.argv:
                        tqdm.write(create_text_slice(fake_sample[0, :, :, :] / SDF_CLIPPING))
                    
                    fake_discriminator_output = discriminator_parallel(fake_sample)
                    fake_loss = -fake_discriminator_output.mean()
                    fake_loss.backward()
                    generator_optimizer.step()
                    
                
                # train discriminator on fake samples                
                discriminator_optimizer.zero_grad()
                latent_codes = sample_latent_codes(current_batch_size)
                fake_sample = generator_parallel(batch_grid_points, latent_codes)
                fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                discriminator_output_fake = discriminator_parallel(fake_sample)

                # train discriminator on real samples
                discriminator_output_valid = discriminator_parallel(valid_sample)
                
                gradient_penalty = get_gradient_penalty(valid_sample.detach(), fake_sample.detach())
                loss = discriminator_output_fake.mean() - discriminator_output_valid.mean() + gradient_penalty
                loss.backward()

                discriminator_optimizer.step()
                
                history_fake.append(discriminator_output_fake.mean().item())
                history_real.append(discriminator_output_valid.mean().item())
                history_gradient_penalty.append(gradient_penalty.item())
                batch_index += 1

                if "verbose" in sys.argv and batch_index % 50 == 0:
                    tqdm.write("Epoch " + str(epoch) + ", batch " + str(batch_index) +
                        ": D(x'): " + '{0:.4f}'.format(history_fake[-1]) +
                        ", D(x): " + '{0:.4f}'.format(history_real[-1]) +
                        ", loss: " + '{0:.4f}'.format(history_real[-1] - history_fake[-1]) +
                        ", gradient penalty: " + '{0:.4f}'.format(gradient_penalty.item()))
                progress.update()
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        
        prediction_fake = np.mean(history_fake)
        prediction_real = np.mean(history_real)
        recent_gradient_penalty = np.mean(history_gradient_penalty)

        tqdm.write('Epoch {:d} ({:.1f}s), D(x\'): {:.4f}, D(x): {:.4f}, loss: {:4f}, gradient penalty: {:.4f}'.format(
            epoch,
            time.time() - epoch_start_time,
            prediction_fake,
            prediction_real,
            prediction_real - prediction_fake,
            recent_gradient_penalty))
        
        generator.save()
        discriminator.save()
        
        if epoch % 10 == 0:
            generator.save(epoch=epoch)
            discriminator.save(epoch=epoch)

        if "show_slice" in sys.argv:
            latent_code = sample_latent_codes(1)
            slice_voxels = generator_parallel(grid_points, latent_code)
            slice_voxels = slice_voxels.reshape(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
            tqdm.write(create_text_slice(slice_voxels / SDF_CLIPPING))
        
        log_file.write('{:d} {:.1f} {:.4f} {:.4f} {:.4f}\n'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real, recent_gradient_penalty))
        log_file.flush()


train()
log_file.close()
