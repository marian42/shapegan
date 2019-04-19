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

dataset = torch.load("data/airplanes-32.to").to(device)
sample_plane = torch.unsqueeze(dataset[0, :, :, :], dim = 1)
zeros = torch.zeros([1, 200, 1, 1, 1], device = device)
view_plane = sample_plane.squeeze().cpu().numpy()

for epoch in count():
    try:
        generator_optimizer.zero_grad()
        output = generator.forward(zeros)
        loss = generator_criterion(output, sample_plane)
        loss.backward()
        generator_optimizer.step()        

        if epoch % 2 == 0 or True:
            viewer.set_voxels(output.squeeze().detach().cpu().numpy())
        else:
            viewer.set_voxels(view_plane)
        error = loss.item()

        if epoch % 50 == 0:
            generator.save()
            print("Model parameters saved.")
        
        print("epoch " + str(epoch) + ": error: " + '{0:.8f}'.format(error))
    except KeyboardInterrupt:
        viewer.stop()
        break