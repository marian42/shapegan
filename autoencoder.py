from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random

import numpy as np

from voxel.viewer import VoxelViewer

from model import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("data/chairs-32.to").to(device)
dataset_size = dataset.shape[0]

BATCH_SIZE = 100


autoencoder = Autoencoder()
autoencoder.load()

cross_entropy = torch.nn.functional.binary_cross_entropy
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas = (0.5, 0.5))

viewer = VoxelViewer()

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
                output = autoencoder.forward(sample)
                loss = cross_entropy(output / 2 + 0.5, sample / 2 + 0.5)
                loss.backward()
                optimizer.step()        

                viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())
                error = loss.item()

                print("epoch " + str(epoch) + ", batch " + str(batch_index) + " error: " + '{0:.8f}'.format(error))                
                batch_index += 1
            except KeyboardInterrupt:
                viewer.stop()
                return
        autoencoder.save()
        print("Model parameters saved.")

train()
