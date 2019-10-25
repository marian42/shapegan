import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from skimage.measure import block_reduce

sample_rate, samples = wavfile.read('wire-and-flashing-lights.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nfft=128, nperseg=128)

#spectrogram = np.log(spectrogram)
spectrogram = block_reduce(spectrogram, (1, 5), np.max)

spectrogram = spectrogram[:-1, :]
mean = np.mean(spectrogram, axis=1)

spectrogram -= mean[:, np.newaxis]

variance = np.std(spectrogram, axis=1)
spectrogram /= variance[:, np.newaxis]
spectrogram = np.nan_to_num(spectrogram)
spectrogram *= 0.4

duration = samples.shape[0] / sample_rate
spectrogram_sample_rate = spectrogram.shape[1] / duration



from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device, standard_normal_distribution
from dataset import dataset
import scipy
import numpy as np
from rendering import MeshRenderer
import time
import torch
from tqdm import tqdm
import cv2
import random
import sys
import datetime

SURFACE_LEVEL = 0.048 

sdf_net = SDFNet()
sdf_net.filename = 'hybrid_gan_generator.to'
sdf_net.load()
sdf_net.eval()


TRANSITION_TIME = 2
viewer = MeshRenderer()
import sounddevice

time.sleep(1)

sounddevice.play(samples, sample_rate)
sounddevice
start = datetime.datetime.now()
last_code_raw = torch.zeros(LATENT_CODE_SIZE, dtype=torch.float32, device=device)
last_code = torch.zeros(LATENT_CODE_SIZE, dtype=torch.float32, device=device)

old_new = torch.zeros((2, LATENT_CODE_SIZE), dtype=torch.float32, device=device)
r = torch.arange(LATENT_CODE_SIZE, dtype=torch.int64, device=device)

while True:
    try:
        time_elapsed = (datetime.datetime.now() - start).total_seconds()
        index = int(time_elapsed * spectrogram_sample_rate)
        code = spectrogram[:, index]
        code = torch.tensor(code, dtype=torch.float32, device=device)
        old_new[0, :] = last_code_raw * 0.95
        old_new[1, :] = code
        old_new_abs = torch.abs(old_new)
        _, indices = torch.max(old_new_abs, dim=0)
        code = old_new[indices, r]
        last_code_raw = code

        code = last_code * 0.8 + code * 0.2
        last_code = code
        
        viewer.set_mesh(sdf_net.get_mesh(code, voxel_resolution=32, sphere_only=False, level=SURFACE_LEVEL))
        
    except KeyboardInterrupt:
        viewer.stop()
        break