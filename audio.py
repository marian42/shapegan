from model.sdf_net import SDFNet, LATENT_CODE_SIZE
from rendering import MeshRenderer
import time
import torch
import datetime
from scipy import signal
from scipy.io import wavfile
import numpy as np
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from util import device

sample_rate, samples = wavfile.read('wire-and-flashing-lights.wav')
_, _, spectrogram = signal.spectrogram(samples, sample_rate)

spectrogram = block_reduce(spectrogram, (1, 5), np.max)
spectrogram = PCA(n_components=64).fit_transform(spectrogram.T).T

smooth_spectrogram = gaussian_filter(spectrogram.reshape(-1), sigma=5, order=0).reshape(spectrogram.shape)
spectrogram -= smooth_spectrogram

mean = np.mean(spectrogram, axis=1)
spectrogram -= mean[:, np.newaxis]

variance = np.std(spectrogram, axis=1)
spectrogram /= variance[:, np.newaxis]
spectrogram = np.nan_to_num(spectrogram)
spectrogram *= 0.9

duration = samples.shape[0] / sample_rate
spectrogram_sample_rate = spectrogram.shape[1] / duration

sdf_net = SDFNet()
sdf_net.filename = 'hybrid_gan_generator.to'
sdf_net.load()
sdf_net.eval()
SURFACE_LEVEL = 0.048

viewer = MeshRenderer()

import sounddevice
time.sleep(1)
sounddevice.play(samples, sample_rate)

start = datetime.datetime.now()
last_code_raw = torch.zeros(LATENT_CODE_SIZE, dtype=torch.float32, device=device)
last_code = torch.zeros(LATENT_CODE_SIZE, dtype=torch.float32, device=device)

old_new = torch.zeros((2, LATENT_CODE_SIZE), dtype=torch.float32, device=device)
latent_code_indices = torch.arange(LATENT_CODE_SIZE, dtype=torch.int64, device=device)

while True:
    try:
        time_elapsed = (datetime.datetime.now() - start).total_seconds()
        index = int(time_elapsed * spectrogram_sample_rate)
        print(index)
        code = torch.tensor(spectrogram[:, index], dtype=torch.float32, device=device)
        old_new[0, :] = last_code_raw * 0.9
        old_new[1, :] = code
        old_new_abs = torch.abs(old_new)
        _, indices = torch.max(old_new_abs, dim=0)
        code = old_new[indices, latent_code_indices]
        last_code_raw = code

        code = last_code * 0.7 + code * 0.3
        last_code = code
        
        viewer.set_mesh(sdf_net.get_mesh(code, voxel_resolution=32, sphere_only=False, level=SURFACE_LEVEL))
        
    except KeyboardInterrupt:
        viewer.stop()
        break