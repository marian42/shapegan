from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import VoxelDataset
from torch.utils.data import DataLoader

import random
random.seed(0)
torch.manual_seed(0)

import numpy as np
import sys
import time
from tqdm import tqdm

from model.autoencoder import Autoencoder
from collections import deque
from util import create_text_slice, device

BATCH_SIZE = 32

dataset = VoxelDataset.glob('data/chairs/voxels_32/**.npy')
data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

VIEWER_UPDATE_STEP = 20

IS_VARIATIONAL = 'classic' not in sys.argv

autoencoder = Autoencoder(is_variational=IS_VARIATIONAL)
if "continue" in sys.argv:
    autoencoder.load()

optimizer = optim.Adam(autoencoder.parameters(), lr=0.00005)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

reconstruction_error_history = deque(maxlen = BATCH_SIZE)
kld_error_history = deque(maxlen = BATCH_SIZE)

criterion = nn.functional.mse_loss

log_file = open("plots/{:s}autoencoder_training.csv".format('variational_' if autoencoder.is_variational else ''), "a" if "continue" in sys.argv else "w")

def voxel_difference(input, target):
    wrong_signs = (input * target) < 0
    return torch.sum(wrong_signs).item() / wrong_signs.nelement()

def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()

def get_reconstruction_loss(input, target):
    difference = input - target
    wrong_signs = target < 0
    difference[wrong_signs] *= 32

    return torch.mean(torch.abs(difference))

def test(epoch_index, epoch_time, test_set):
    with torch.no_grad():
        autoencoder.eval()

        if IS_VARIATIONAL:
            output, mean, log_variance = autoencoder(test_set)
            kld = kld_loss(mean, log_variance)
        else:
            output = autoencoder(test_set)
            kld = 0

        reconstruction_loss = criterion(output, test_set)
        
        voxel_diff = voxel_difference(output, test_set)

        if "show_slice" in sys.argv:
            print(create_text_slice(output[0, :, :, :]))

        print("Epoch {:d} ({:.1f}s): ".format(epoch_index, epoch_time) +
            "Reconstruction loss: {:.4f}, ".format(reconstruction_loss) +
            "Voxel diff: {:.4f}, ".format(voxel_diff) + 
            "KLD loss: {:4f}, ".format(kld) + 
            "training loss: {:4f}, ".format(np.mean(reconstruction_error_history))
        )

        log_file.write('{:d} {:.1f} {:.6f} {:.6f} {:.6f}\n'.format(epoch_index, epoch_time, reconstruction_loss, kld, voxel_diff))
        log_file.flush()

def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in tqdm(data_loader, desc='Epoch {:d}'.format(epoch)):
            try:
                batch = batch.to(device)

                autoencoder.zero_grad()
                autoencoder.train()
                if IS_VARIATIONAL:
                    output, mean, log_variance = autoencoder(batch)
                    kld = kld_loss(mean, log_variance)
                else:
                    output = autoencoder(batch)
                    kld = 0

                reconstruction_loss = get_reconstruction_loss(output, batch)

                loss = reconstruction_loss + kld

                reconstruction_error_history.append(reconstruction_loss.item())
                kld_error_history.append(kld.item() if IS_VARIATIONAL else 0)
                
                loss.backward()
                optimizer.step()

                if show_viewer and batch_index == 0:
                    viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())

                if show_viewer and (batch_index + 1) % VIEWER_UPDATE_STEP == 0 and 'verbose' in sys.argv:
                    viewer.set_voxels(output[0, :, :, :].squeeze().detach().cpu().numpy())
                    print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                        + ', reconstruction loss: {0:.4f}'.format(reconstruction_loss.item()) \
                        + ' (average: {0:.4f}), '.format(np.mean(reconstruction_error_history)) \
                        + 'KLD loss: {0:.4f}'.format(np.mean(kld_error_history)))
                batch_index += 1
            except KeyboardInterrupt:
                if show_viewer:
                    viewer.stop()
                return
        autoencoder.save()
        if epoch % 20 == 0:
            autoencoder.save(epoch=epoch)
        #test(epoch, time.time() - epoch_start_time, test_set)
        print("Epoch {:d} ({:.1f}s): reconstruction loss: {:.4f}, KLD loss: {:.4f}".format(
            epoch,
            time.time() - epoch_start_time,
            np.mean(reconstruction_error_history),
            np.mean(kld_error_history)))

train()