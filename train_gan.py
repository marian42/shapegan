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

from util import create_text_slice, device
from datasets import VoxelDataset
from torch.utils.data import DataLoader

generator = Generator()
discriminator = Discriminator()

if "continue" in sys.argv:
    generator.load()
    discriminator.load()

log_file = open("plots/gan_training.csv", "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)

discriminator_criterion = torch.nn.functional.binary_cross_entropy
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

BATCH_SIZE = 64

dataset = VoxelDataset.glob('data/chairs/voxels_32/**.npy')
data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

valid_target_default = torch.ones(BATCH_SIZE, requires_grad=False).to(device)
fake_target_default = torch.zeros(BATCH_SIZE, requires_grad=False).to(device)

def train():
    history_fake = deque(maxlen=50)
    history_real = deque(maxlen=50)

    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in data_loader:
            try:

                # train generator
                generator_optimizer.zero_grad()
                    
                fake_sample = generator.generate(sample_size = BATCH_SIZE)
                if show_viewer:
                    viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
                
                fake_discriminator_output = discriminator(fake_sample)
                fake_loss = -torch.mean(torch.log(fake_discriminator_output))
                fake_loss.backward()
                generator_optimizer.step()
                    
                
                # train discriminator
                current_batch_size = batch.shape[0] # equals BATCH_SIZE for all batches except the last one
                fake_target = fake_target_default if current_batch_size == BATCH_SIZE else torch.zeros(current_batch_size, requires_grad=False).to(device)
                valid_target = valid_target_default if current_batch_size == BATCH_SIZE else torch.ones(current_batch_size, requires_grad=False).to(device)

                discriminator_optimizer.zero_grad()
                fake_sample = generator.generate(sample_size = current_batch_size).detach()
                discriminator_output_fake = discriminator(fake_sample)
                fake_loss = discriminator_criterion(discriminator_output_fake, fake_target)
                fake_loss.backward()
                discriminator_optimizer.step()

                discriminator_optimizer.zero_grad()
                discriminator_output_valid = discriminator(batch.to(device))
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
        
        generator.save()
        discriminator.save()

        if epoch % 20 == 0:
            generator.save(epoch=epoch)
            discriminator.save(epoch=epoch)

        if "show_slice" in sys.argv:
            voxels = generator.generate().squeeze()
            print(create_text_slice(voxels))

        prediction_fake = np.mean(history_fake)
        prediction_real = np.mean(history_real)
        print('Epoch {:d} ({:.1f}s), prediction on fake: {:.4f}, prediction on real: {:.4f}'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        log_file.write('{:d} {:.1f} {:.4f} {:.4f}\n'.format(epoch, time.time() - epoch_start_time, prediction_fake, prediction_real))
        log_file.flush()


train()
log_file.close()
