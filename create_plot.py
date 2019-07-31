import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import sys
from tqdm import tqdm
from numpy import genfromtxt
import scipy

from sklearn.manifold import TSNE
from model import Autoencoder, Generator, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
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
        codes = autoencoder.encode(voxels, device).cpu().numpy()
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
        codes = autoencoder.encode(voxels, device).cpu().numpy()
    
    print("Plotting...")
    plt.hist(codes, bins=50, range=(-3, 3), histtype='step')
    plt.savefig("plots/autoencoder-histogram.pdf")
    codes = codes.flatten()
    plt.clf()
    plt.hist(codes, bins=100, range=(-3, 3))
    plt.savefig("plots/autoencoder-histogram-combined.pdf")

if "autodecoder_hist" in sys.argv:
    latent_codes = torch.load(LATENT_CODES_FILENAME).cpu().detach().flatten().numpy()
    latent_codes = latent_codes.reshape(-1)
    mean, variance = np.mean(latent_codes), np.var(latent_codes) ** 0.5
    print("mean: ", mean)
    print("variance: ", variance)

    x_range = 0.42

    x = np.linspace(-x_range, x_range, 500)
    y = scipy.stats.norm.pdf(x, mean, variance)
    plt.plot(x, y, 'r')
    plt.hist(latent_codes, bins=50, range=(-x_range, x_range), density=1)

    plt.savefig("plots/autodecoder-histogram.pdf")

if "autoencoder_examples" in sys.argv:
    from dataset import dataset as dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from voxel.viewer import VoxelViewer
    viewer = VoxelViewer(start_thread=False)
    
    indices = random.sample(list(range(dataset.size)), 20)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = Autoencoder()
    autoencoder.load()
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.encode(voxels, device)
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
    
    return np.arange(window_size / 2, data.shape[0] - window_size / 2, dtype=int), moving_average

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

if "sdf_slice" in sys.argv:
    from sdf.mesh_to_sdf import MeshSDF, scale_to_unit_sphere
    import trimesh
    import cv2

    model_filename = '/home/marian/shapenet/ShapeNetCore.v2/03001627/4c6c364af4b52751ca6910e4922d61aa/models/model_normalized.obj'
    
    print("Loading mesh...")
    mesh = trimesh.load(model_filename)
    mesh = scale_to_unit_sphere(mesh)
    mesh_sdf = MeshSDF(mesh)

    resolution = 1280
    slice_position = 0.4
    clip = 0.1
    points = np.meshgrid(
        np.linspace(slice_position, slice_position, 1),
        np.linspace(1, -1, resolution),
        np.linspace(-1, 1, resolution)
    )

    points = np.stack(points)
    points = points.reshape(3, -1).transpose()

    print("Calculating SDF values...")
    sdf = mesh_sdf.get_sdf_in_batches(points)
    sdf = sdf.reshape(1, resolution, resolution)
    sdf = sdf[0, :, :]
    sdf = np.clip(sdf, -clip, clip) / clip

    print("Creating image...")
    image = np.ones((resolution, resolution, 3))
    image[:,:,:2][sdf > 0] = (1.0 - sdf[sdf > 0])[:, np.newaxis]
    image[:,:,1:][sdf < 0] = (1.0 + sdf[sdf < 0])[:, np.newaxis]
    mask = np.abs(sdf) < 0.03
    image[mask, :] = 0
    image *= 255
    cv2.imwrite("plots/sdf_example.png", image)

if "voxel_occupancy" in sys.argv:
    from dataset import dataset as dataset
    voxels = dataset.voxels.cpu()
    mask = voxels < 0
    occupied = torch.sum(mask, dim=[1, 2, 3]).numpy()

    plt.hist(occupied, bins=100, range=(0, 10000))
    plt.savefig("plots/voxel-occupancy-histogram.pdf")