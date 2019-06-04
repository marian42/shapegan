import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import sys
from tqdm import tqdm
from numpy import genfromtxt

from sklearn.manifold import TSNE
from model import Autoencoder, Generator, LATENT_CODE_SIZE
import random


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
    from dataset import dataset as dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if "autoencoder_hist" in sys.argv:
    from dataset import dataset as dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices = random.sample(list(range(dataset.size)), 5000)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = Autoencoder()
    autoencoder.load()
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.create_latent_code(*autoencoder.encode(voxels), device).cpu().numpy()
    
    print("Plotting...")
    plt.hist(codes, bins=50, range=(-3, 3), histtype='step')
    plt.savefig("plots/autoencoder-histogram.pdf")
    codes = codes.flatten()
    plt.clf()
    plt.hist(codes, bins=100, range=(-3, 3))
    plt.savefig("plots/autoencoder-histogram-combined.pdf")

if "autoencoder_examples" in sys.argv:
    from dataset import dataset as dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer(start_thread=False, background_color = (1.0, 1.0, 1.0, 1.0))
    
    indices = random.sample(list(range(dataset.size)), 20)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = Autoencoder()
    autoencoder.load()
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.create_latent_code(*autoencoder.encode(voxels), device)
        reconstructed = autoencoder.decode(codes).cpu().numpy()
        codes = codes.cpu().numpy()

    print("Plotting")
    fig, axs = plt.subplots(len(indices), 3, figsize=(10, 32))
    for i in range(len(indices)):
        viewer.set_voxels(voxels[i, :, :, :].cpu().numpy())
        image = viewer.get_image(output_size=512)
        axs[i, 0].imshow(image, cmap='gray')
        axs[i, 0].axis('off')

        axs[i, 1].bar(range(codes.shape[1]), codes[i, :])
        axs[i, 1].set_ylim((-3, 3))

        viewer.set_voxels(reconstructed[i, :, :, :])
        image = viewer.get_image(output_size=512)
        axs[i, 2].imshow(image, cmap='gray')
        axs[i, 2].axis('off')
    plt.savefig("plots/autoencoder-examples.pdf", bbox_inches='tight', dpi=400)
    

if "gan" in sys.argv:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

if "wgan" in sys.argv:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = Generator()
    generator.filename = "wgan-generator.to"
    generator.load()
    standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
    shape = torch.Size([500, LATENT_CODE_SIZE, 1, 1, 1])
    x = standard_normal_distribution.sample(shape).to(device)
    with torch.no_grad():
        voxels = generator.forward(x).squeeze()
    codes = x.squeeze().cpu().numpy()
    print(codes.shape)
    print(voxels.shape)
    create_tsne_plot(codes, voxels, labels = None, filename = "plots/wgan-images.pdf")

def get_moving_average(data, window_size):
    moving_average = []
    for i in range(data.shape[0] - window_size):
        moving_average.append(np.mean(data[i:i+window_size]))
    
    return range(window_size, data.shape[0]), moving_average

if "gan_training" in sys.argv:
    data = genfromtxt('plots/gan_training.csv', delimiter=' ')
        
    plt.plot(data[:, 2])
    plt.plot(*get_moving_average(data[:, 2], 10))

    plt.ylabel('Inception Score')
    plt.xlabel('Epoch')
    plt.title('GAN Training')
    plt.savefig("plots/gan-training.pdf")

if "wgan_training" in sys.argv:
    data = genfromtxt('plots/wgan_training.csv', delimiter=' ')
        
    plt.plot(data[:, 2])
    plt.plot(*get_moving_average(data[:, 2], 10))

    plt.ylabel('Inception Score')
    plt.xlabel('Epoch')
    plt.title('WGAN Training')
    plt.savefig("plots/wgan-training.pdf")
