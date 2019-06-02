import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import sys
from tqdm import tqdm

from sklearn.manifold import TSNE
from model import Autoencoder, Generator, LATENT_CODE_SIZE
import random

from dataset import dataset as dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_tsne_plot(codes, voxels = None, labels = None, filename = "plot.pdf"):
    print("Calculating t-sne embedding...")
    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(codes)
    
    print("Plotting...")
    fig, ax = plt.subplots()
    ax.scatter(embedded[:, 0], embedded[:, 1], c=labels, s = 40, cmap='Set1')
    fig.set_size_inches(40, 40)

    if voxels is not None:
        print("Creating images...")
        from voxel.viewer import VoxelViewer
        viewer = VoxelViewer(start_thread=False, background_color = (1.0, 1.0, 1.0, 1.0))
        for i in tqdm(range(voxels.shape[0])):
            viewer.set_voxels(voxels[i, :, :, :].cpu().numpy())
            image = viewer.get_image()
            box = AnnotationBbox(OffsetImage(image, zoom = 0.5, cmap='gray'), embedded[i, :], frameon=True)
            ax.add_artist(box)
        
    print("Saving PDF...")
    plt.savefig(filename, dpi=200, bbox_inches='tight')

if "autoencoder" in sys.argv:
    indices = random.sample(list(range(dataset.size)), 1000)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = Autoencoder()
    autoencoder.load()
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.create_latent_code(*autoencoder.encode(voxels), device).cpu().numpy()
    labels = dataset.label_indices[indices].cpu().numpy()
    create_tsne_plot(codes, voxels, labels, "plots/autoencoder-images.pdf")
    #create_tsne_plot(codes, None, labels, "plots/autoencoder-dots.pdf")

if "gan" in sys.argv:
    generator = Generator()
    generator.load()
    standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
    shape = torch.Size([500, LATENT_CODE_SIZE, 1, 1, 1])
    x = standard_normal_distribution.sample(shape).to(device)
    with torch.no_grad():
        voxels = generator.forward(x).squeeze()
    codes = x.squeeze().cpu().numpy()
    print(codes.shape)
    print(voxels.shape)
    create_tsne_plot(codes, voxels, labels = None, filename = "plots/gan-images.pdf")

        

    
