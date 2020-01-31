import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from itertools import count
import time
import random
from tqdm import tqdm
import sys

from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device

if "nogui" not in sys.argv:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

POINTCLOUD_SIZE = 200000

points = torch.load('data/sdf_points.to').to(device)
sdf = torch.load('data/sdf_values.to').to(device)

MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE
BATCH_SIZE = 20000
SDF_CUTOFF = 0.1
sdf.clamp_(-SDF_CUTOFF, SDF_CUTOFF)
signs = sdf.cpu().numpy() > 0

SIGMA = 0.01

LOG_FILE_NAME = "plots/sdf_net_training.csv"

sdf_net = SDFNet()
if "continue" in sys.argv:
    sdf_net.load()
    latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)
else:    
    normal_distribution = torch.distributions.normal.Normal(0, 0.0001)
    latent_codes = normal_distribution.sample((MODEL_COUNT, LATENT_CODE_SIZE)).to(device)
latent_codes.requires_grad = True

network_optimizer = optim.Adam(sdf_net.parameters(), lr=1e-5)
latent_code_optimizer = optim.Adam([latent_codes], lr=1e-5)
criterion = nn.MSELoss()

first_epoch = 0
if 'continue' in sys.argv:
    log_file_contents = open(LOG_FILE_NAME, 'r').readlines()
    first_epoch = len(log_file_contents)

log_file = open(LOG_FILE_NAME, "a" if "continue" in sys.argv else "w")

def create_batches():
    indices_positive = np.nonzero(signs)[0]
    indices_negative = np.nonzero(~signs)[0]
    if indices_negative.shape[0] > indices_positive.shape[0]:
        np.random.shuffle(indices_negative)
        indices_negative = indices_negative[:indices_positive.shape[0]]
    else:
        np.random.shuffle(indices_positive)
        indices_positive = indices_positive[:indices_negative.shape[0]]
    indices = np.concatenate((indices_negative, indices_positive))
    np.random.shuffle(indices)
    batch_count = int(indices.shape[0] / BATCH_SIZE)
    for i in range(batch_count - 1):
        yield indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield indices[(batch_count - 1) * BATCH_SIZE:]

def train():
    for epoch in count(start=first_epoch):
        epoch_start_time = time.time()
        loss_values = []
        batch_index = 0
        for batch in tqdm(list(create_batches())):
            indices = torch.tensor(batch, device = device)
            model_indices = indices / POINTCLOUD_SIZE

            batch_latent_codes = latent_codes[model_indices, :]
            batch_points = points[indices, :]
            batch_sdf = sdf[indices]

            sdf_net.zero_grad()
            if latent_codes.grad is not None:
                latent_codes.grad.data.zero_()
            output = sdf_net.forward(batch_points, batch_latent_codes)
            loss = torch.mean(torch.abs(output - batch_sdf)) + SIGMA * torch.mean(torch.pow(batch_latent_codes, 2))
            loss.backward()
            network_optimizer.step()
            latent_code_optimizer.step()
            loss_values.append(loss.item())

            if batch_index % 400 == 0 and "nogui" not in sys.argv:
                try:
                    viewer.set_mesh(sdf_net.get_mesh(latent_codes[random.randrange(MODEL_COUNT), :]))
                except ValueError:
                    pass

            batch_index += 1

        variance = np.var(latent_codes.detach().reshape(-1).cpu().numpy()) ** 0.5
        epoch_duration = time.time() - epoch_start_time
        
        print("Epoch {:d}, {:.1f}s. Loss: {:.8f}".format(epoch, epoch_duration, np.mean(loss_values)))

        sdf_net.save()
        torch.save(latent_codes, LATENT_CODES_FILENAME)
        
        sdf_net.save(epoch=epoch)
        torch.save(latent_codes, sdf_net.get_filename(epoch=epoch, filename='sdf_net_latent_codes.to'))

        log_file.write('{:d} {:.1f} {:.6f} {:.6f}\n'.format(epoch, epoch_duration, np.mean(loss_values), variance))
        log_file.flush()

train()