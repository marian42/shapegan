from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random

import numpy as np

import sys
import time

from model import Autoencoder

from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

BATCH_SIZE = 64
KLD_LOSS_WEIGHT = 0.03
TEST_SPLIT = 0.1

all_indices = list(range(dataset_size))
random.shuffle(all_indices)
test_indices = all_indices[:int(dataset_size * TEST_SPLIT)]
training_indices = list(all_indices[int(dataset_size * TEST_SPLIT):])
test_data = dataset[test_indices]

autoencoder = Autoencoder()
autoencoder.load()

cross_entropy = torch.nn.functional.binary_cross_entropy
optimizer = optim.Adam(autoencoder.parameters(), lr=0.00005)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer()

error_history = deque(maxlen = BATCH_SIZE)

def create_batches():
    batch_count = int(len(training_indices) / BATCH_SIZE)
    random.shuffle(training_indices)
    for i in range(batch_count - 1):
        yield training_indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield training_indices[(batch_count - 1) * BATCH_SIZE:]

def test():
    with torch.no_grad():
        output, _, _ = autoencoder.forward(test_data, device)
        reconstruction_loss = cross_entropy(output / 2 + 0.5, test_data / 2 + 0.5)
        print("Reconstruction loss on test data: " + str(reconstruction_loss.item()))


def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches():
            try:
                indices = torch.tensor(batch, device = device)
                sample = dataset[indices, :, :, :]

                autoencoder.zero_grad()
                output, mean, log_variance = autoencoder.forward(sample, device)
                reconstruction_loss = cross_entropy(output / 2 + 0.5, sample / 2 + 0.5)
                error_history.append(reconstruction_loss.item())
                kld_loss = torch.mean(((mean.pow(2) + torch.exp(log_variance)) * -1 + 1 + log_variance) * -0.5)
                loss = reconstruction_loss + kld_loss * KLD_LOSS_WEIGHT
                
                loss.backward()
                optimizer.step()        

                if show_viewer:
                    viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())
                error = loss.item()

                print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                    + ', reconstruction loss: {0:.8f}'.format(reconstruction_loss.item()) \
                    + ' (average: {0:.8f}), '.format(sum(error_history) / len(error_history)) \
                    + 'KLD loss: {0:.8f}'.format(kld_loss.item()))     
                batch_index += 1
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        autoencoder.save()
        print("Model parameters saved. Epoch took " + '{0:.1f}'.format(time.time() - epoch_start_time) + "s.")
        test()

train()
