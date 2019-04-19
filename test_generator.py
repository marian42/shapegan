from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import random

import numpy as np

from voxel.viewer import VoxelViewer

from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator()
generator.load()

generator_criterion = nn.MSELoss()
generator_optimizer = optim.SGD(generator.parameters(), lr=0.05, momentum=0.9)

viewer = VoxelViewer()

sample_plane = torch.load("data/airplane-32.to").to(device)
zeros = torch.zeros([1, 200, 1, 1, 1], device = device)
view_plane = sample_plane.squeeze().cpu().numpy()

for epoch in count():
    try:
        generator_optimizer.zero_grad()
        output = generator.forward(zeros)
        loss = generator_criterion(output, sample_plane)
        loss.backward()
        generator_optimizer.step()        

        viewer.set_voxels(output.squeeze().detach().cpu().numpy())
        error = loss.item()

        if epoch % 50 == 0:
            generator.save()
            print("Model parameters saved.")
        
        print("epoch " + str(epoch) + ": error: " + '{0:.8f}'.format(error))
    except KeyboardInterrupt:
        viewer.stop()
        break