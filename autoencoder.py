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

BATCH_SIZE = 200


autoencoder = Autoencoder()
autoencoder.load()

criterion = torch.nn.functional.binary_cross_entropy
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas = (0.5, 0.5))

viewer = VoxelViewer()

for epoch in count():
    try:
        indices = torch.tensor(random.sample(range(dataset_size), BATCH_SIZE), device = device)
        sample = dataset[indices, :, :, :]

        autoencoder.zero_grad()
        output = autoencoder.forward(sample)
        loss = criterion(output, sample)
        loss.backward()
        optimizer.step()        

        viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())
        error = loss.item()

        if epoch % 50 == 0:
            autoencoder.save()
            print("Model parameters saved.")
        
        print("epoch " + str(epoch) + ": error: " + '{0:.8f}'.format(error))
    except KeyboardInterrupt:
        viewer.stop()
        break