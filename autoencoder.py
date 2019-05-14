from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random

import numpy as np

from voxel.viewer import VoxelViewer

from model import Autoencoder

from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

BATCH_SIZE = 100


autoencoder = Autoencoder()
autoencoder.load()

cross_entropy = torch.nn.functional.binary_cross_entropy
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas = (0.5, 0.5))

viewer = VoxelViewer()

error_history = deque(maxlen = 50)

def create_batches(sample_count, batch_size):
    batch_count = int(sample_count / batch_size)
    indices = list(range(sample_count))
    random.shuffle(list(range(sample_count)))
    for i in range(batch_count - 1):
        yield indices[i * batch_size:(i+1)*batch_size]
    yield indices[(batch_count - 1) * batch_size:]


def train():
    for epoch in count():
        batch_index = 0
        for batch in create_batches(dataset_size, BATCH_SIZE):
            try:
                indices = torch.tensor(batch, device = device)
                sample = dataset[indices, :, :, :]

                autoencoder.zero_grad()
                output, mean, log_variance = autoencoder.forward(sample, device)
                reconstruction_loss = cross_entropy(output / 2 + 0.5, sample / 2 + 0.5)
                error_history.append(reconstruction_loss.item())
                kld_loss = torch.mean(((mean.pow(2) + torch.exp(log_variance)) * -1 + 1 + log_variance) * -0.5)
                loss = reconstruction_loss + kld_loss
                
                loss.backward()
                optimizer.step()        

                viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())
                error = loss.item()

                print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                    + ', reconstruction error: {0:.8f}'.format(reconstruction_loss.item()) \
                    + ' (average: {0:.8f}), '.format(sum(error_history) / len(error_history)) \
                    + 'KLD loss: {0:.8f}'.format(kld_loss.item()))     
                batch_index += 1
            except KeyboardInterrupt:
                viewer.stop()
                return
        autoencoder.save()
        print("Model parameters saved.")

train()
