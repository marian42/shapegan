import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE
from dataset import dataset as dataset
from model import Autoencoder
import random

tsne = TSNE(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices = random.sample(list(range(dataset.size)), 5000)
voxels = dataset.voxels[indices, :, :, :]
autoencoder = Autoencoder()
autoencoder.load()
print("Generating codes...")
with torch.no_grad():
    codes = autoencoder.create_latent_code(*autoencoder.encode(voxels), device).cpu().numpy()
labels = dataset.label_indices[indices].cpu().numpy()

print("Calculating t-sne embedding...")
embedded = tsne.fit_transform(codes)

print("Plotting...")
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s = 4)
plt.show()