'''
This file contains code to create various plots and figures.
It is currently not maintained and not all of the figures work.
But it can serve as inspiration for further development.
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from tqdm import tqdm

from model import LATENT_CODE_SIZE, LATENT_CODES_FILENAME
import random
from util import device

class ImageGrid():
    def __init__(self, width, height=1, cell_width = 3, cell_height = None, margin=0.2, create_viewer=True, crop=True):
        print("Plotting...")
        self.width = width
        self.height = height
        cell_height = cell_height if cell_height is not None else cell_width

        self.figure, self.axes = plt.subplots(height, width,
            figsize=(width * cell_width, height * cell_height),
            gridspec_kw={'left': 0, 'right': 1, 'top': 1, 'bottom': 0, 'wspace': margin, 'hspace': margin})
        self.figure.patch.set_visible(False)

        self.crop = crop
        if create_viewer:
            from rendering import MeshRenderer
            self.viewer = MeshRenderer(start_thread=False)
        else:
            self.viewer = None

    def set_image(self, image, x = 0, y = 0):
        cell = self.axes[y, x] if self.height > 1 and self.width > 1 else self.axes[x + y]
        cell.imshow(image)
        cell.axis('off')
        cell.patch.set_visible(False)

    def set_voxels(self, voxels, x = 0, y = 0, color=None):
        if color is not None:
            self.viewer.model_color = color
        self.viewer.set_voxels(voxels)
        image = self.viewer.get_image(crop=self.crop)
        self.set_image(image, x, y)

    def save(self, filename):
        plt.axis('off')
        extent = self.figure.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        plt.savefig(filename, bbox_inches=extent, dpi=400)
        if self.viewer is not None:
            self.viewer.delete_buffers()

def load_autoencoder(is_variational=False):
    from model.autoencoder import Autoencoder
    autoencoder = Autoencoder(is_variational=is_variational)
    autoencoder.load()
    autoencoder.eval()
    return autoencoder

def load_generator(is_wgan=False):
    from model.gan import Generator
    generator = Generator()
    if is_wgan:
        generator.filename = "wgan-generator.to"
    generator.load()
    generator.eval()
    return generator

def load_sdf_net(filename=None, return_latent_codes = False):
    from model.sdf_net import SDFNet, LATENT_CODES_FILENAME
    sdf_net = SDFNet()
    if filename is not None:
        sdf_net.filename = filename
    sdf_net.load()
    sdf_net.eval()

    if return_latent_codes:
        latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)
        latent_codes.requires_grad = False
        return sdf_net, latent_codes
    else:
        return sdf_net

def create_tsne_plot(codes, voxels = None, labels = None, filename = "plot.pdf", indices=None):
    from sklearn.manifold import TSNE
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    width, height = 40, 52

    print("Calculating t-sne embedding...")
    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(codes)
    
    print("Plotting...")
    fig, ax = plt.subplots()
    plt.axis('off')
    margin = 0.0128
    plt.margins(margin * height / width, margin)

    x = embedded[:, 0]
    y = embedded[:, 1]
    x = np.interp(x, (x.min(), x.max()), (0, 1))
    y = np.interp(y, (y.min(), y.max()), (0, 1))

    ax.scatter(x, y, c=labels, s = 40, cmap='Set1')
    fig.set_size_inches(width, height)

    if voxels is not None:
        print("Creating images...")
        from rendering import MeshRenderer
        viewer = MeshRenderer(start_thread=False)
        for i in tqdm(range(voxels.shape[0])):
            viewer.set_voxels(voxels[i, :, :, :].cpu().numpy())
            viewer.model_color = dataset.get_color(labels[i])
            image = viewer.get_image(crop=True, output_size=128)
            box = AnnotationBbox(OffsetImage(image, zoom = 0.5, cmap='gray'), (x[i], y[i]), frameon=True)
            ax.add_artist(box)

    if indices is not None:
        print("Creating images...")
        dataset_directories = open('data/models.txt', 'r').readlines()
        from rendering import MeshRenderer
        viewer = MeshRenderer(start_thread=False)
        import trimesh
        import logging
        logging.getLogger('trimesh').setLevel(1000000)
        for i in tqdm(range(len(indices))):
            mesh = trimesh.load(os.path.join(dataset_directories[index].strip(), 'model_normalized.obj'))
            viewer.set_mesh(mesh, center_and_scale=True)
            viewer.model_color = dataset.get_color(labels[i])
            image = viewer.get_image(crop=True, output_size=128)
            box = AnnotationBbox(OffsetImage(image, zoom = 0.5, cmap='gray'), (x[i], y[i]), frameon=True)
            ax.add_artist(box)
        
    print("Saving PDF...")
    
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(filename, bbox_inches=extent, dpi=200)

if "color-test" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels('cpu')
    dataset.load_labels()
    voxels = dataset.voxels

    COUNT = dataset.label_count

    plot = ImageGrid(COUNT)
    
    for label in tqdm(range(COUNT)):
        objects = (dataset.labels == label).nonzero()
        index = objects[random.randint(0, objects.shape[0] - 1)].item()
        tensor = voxels[index, :, :, :].clone()
        plot.set_voxels(tensor, label, color=dataset.get_color(label))

    plot.save("plots/test.pdf")

if "autoencoder-classes" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    dataset.load_labels()

    COUNT = dataset.label_count

    vae = load_autoencoder(is_variational=True)
    indices = []
    for label in tqdm(range(COUNT)):
        objects = (dataset.labels == label).nonzero()
        indices.append(objects[random.randint(0, objects.shape[0] - 1)].item())    
    voxels = dataset.voxels[indices, :, :, :]

    print("Generating codes...")
    with torch.no_grad():
        codes_vae = vae.encode(voxels)
        reconstructed_vae = vae.decode(codes_vae).cpu().numpy()
    
    plot = ImageGrid(COUNT, 2)

    for i in range(COUNT):
        plot.set_voxels(voxels[i, :, :, :], i, 0, color=dataset.get_color(i))
        plot.set_voxels(reconstructed_vae[i, :, :, :], i, 1)

    plot.save("plots/vae-reconstruction-classes.pdf")

if "autodecoder-classes" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_labels(device='cpu')
    from rendering.raymarching import render_image
    from rendering import MeshRenderer
    import logging
    logging.getLogger('trimesh').setLevel(1000000)

    viewer = MeshRenderer(start_thread=False)

    COUNT = dataset.label_count

    sdf_net, latent_codes = load_sdf_net(return_latent_codes=True)
    indices = []
    for label in range(COUNT):
        objects = (dataset.labels == label).nonzero()
        indices.append(objects[random.randint(0, objects.shape[0] - 1)].item())
    
    latent_codes = latent_codes[indices, :]
    
    plot = ImageGrid(COUNT, 2, create_viewer=False)
    dataset_directories = directories = open('data/models.txt', 'r').readlines()
        
    for i in range(COUNT):
        mesh = trimesh.load(os.path.join(dataset_directories[index].strip(), 'model_normalized.obj'))
        viewer.set_mesh(mesh, center_and_scale=True)
        viewer.model_color = dataset.get_color(i)
        image = viewer.get_image(crop=True)
        plot.set_image(image, i, 0)

        image = render_image(sdf_net, latent_codes[i, :], color=dataset.get_color(i), crop=True)
        plot.set_image(image, i, 1)
    viewer.delete_buffers()
    plot.save("plots/deepsdf-reconstruction-classes.pdf")

if "autoencoder" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    dataset.load_labels(device)
    
    indices = random.sample(list(range(dataset.size)), 1000)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = load_autoencoder(is_variational='clasic' not in sys.argv)
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.encode(voxels).cpu().numpy()
    create_tsne_plot(codes, voxels, dataset.labels[indices].cpu().numpy(), "plots/{:s}autoencoder-tsne.pdf".format('' if 'classic' in sys.argv else 'variational-'))

if "autodecoder_tsne" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_labels('cpu')
    dataset.load_labels()
    from model.sdf_net import LATENT_CODES_FILENAME
    latent_codes = torch.load(LATENT_CODES_FILENAME).detach().cpu().numpy()
    
    indices = random.sample(range(latent_codes.shape[0]), 1000)
    latent_codes = latent_codes[indices, :]
    labels = dataset.labels[indices]
    
    create_tsne_plot(latent_codes, labels=labels, filename="plots/deepsdf-tsne.pdf", indices=indices)


if "autoencoder_hist" in sys.argv:
    import scipy.stats
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    is_variational = 'classic' not in sys.argv

    x_range = 4 if is_variational else 1

    indices = random.sample(list(range(dataset.size)), min(5000, dataset.size))
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = load_autoencoder(is_variational=is_variational)
    print("Generating codes...")
    with torch.no_grad():
        autoencoder.train()
        codes = autoencoder.encode(voxels).cpu().numpy()
    
    print("Plotting...")
    plt.hist(codes[:, ::4], bins=100, range=(-x_range, x_range), histtype='step', density=1, color=['#1f77b4' for _ in range(0, codes.shape[1], 4)])
    plt.xlabel("$\mathbf{z}^{(i)}$")
    plt.ylabel("relative abundance")
    plt.savefig("plots/{:s}autoencoder-histogram.pdf".format('variational-' if is_variational else ''), bbox_inches='tight')
    codes = codes.flatten()
    plt.clf()
    x = np.linspace(-x_range, x_range, 500)
    y = scipy.stats.norm.pdf(x, 0, 1)
    if is_variational:
        plt.plot(x, y, color='green')
    plt.hist(codes, bins=100, range=(-x_range, x_range), density=1)
    plt.xlabel("$\mathbf{z}$")
    plt.ylabel("relative abundance")
    plt.savefig("plots/{:s}autoencoder-histogram-combined.pdf".format('variational-' if is_variational else ''), bbox_inches='tight')

if "autodecoder_hist" in sys.argv:
    import scipy.stats
    codes = torch.load(LATENT_CODES_FILENAME).cpu().detach().numpy()
    
    x_range = 0.42

    print("Plotting...")
    plt.hist(codes[:, ::4], bins=100, range=(-x_range, x_range), histtype='step', density=1, color=['#1f77b4' for _ in range(0, codes.shape[1], 4)])
    plt.xlabel("$\mathbf{z}^{(i)}$")
    plt.ylabel("relative abundance")
    plt.savefig("plots/autodecoder-histogram.pdf", bbox_inches='tight')
    codes = codes.flatten()
    plt.clf()
    x = np.linspace(-x_range, x_range, 500)
    y = scipy.stats.norm.pdf(x, 0, 1)
    plt.hist(codes, bins=100, range=(-x_range, x_range), density=1)
    plt.xlabel("$\mathbf{z}$")
    plt.ylabel("relative abundance")
    plt.savefig("plots/autodecoder-histogram-combined.pdf", bbox_inches='tight')

if "autoencoder_examples" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)

    from rendering import MeshRenderer
    viewer = MeshRenderer(start_thread=False)
    
    indices = random.sample(list(range(dataset.size)), 20)
    voxels = dataset.voxels[indices, :, :, :]
    autoencoder = load_autoencoder()
    print("Generating codes...")
    with torch.no_grad():
        codes = autoencoder.encode(voxels)
        reconstructed = autoencoder.decode(codes).cpu().numpy()
        codes = codes.cpu().numpy()

    print("Plotting...")
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

if "autoencoder_examples_2" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)

    indices = random.sample(list(range(dataset.size)), 5)
    voxels = dataset.voxels[indices, :, :, :]
    ae = load_autoencoder(is_variational=False)
    vae = load_autoencoder(is_variational=True)

    print("Generating codes...")
    with torch.no_grad():
        codes_ae = ae.encode(voxels)
        reconstructed_ae = ae.decode(codes_ae).cpu().numpy()
        codes_vae = vae.encode(voxels)
        reconstructed_vae = vae.decode(codes_vae).cpu().numpy()
    
    plot = ImageGrid(len(indices), 3)

    for i in range(len(indices)):
        plot.set_voxels(voxels[i, :, :, :], i, 0)
        plot.set_voxels(reconstructed_ae[i, :, :, :], i, 1)
        plot.set_voxels(reconstructed_vae[i, :, :, :], i, 2)

    plot.save("plots/ae-vae-examples.pdf")

if "autoencoder_generate" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    from sklearn.metrics import pairwise_distances

    SAMPLES = 5

    voxels = dataset.voxels
    ae = load_autoencoder(is_variational=False)
    vae = load_autoencoder(is_variational=True)
    print("Generating codes...")
    with torch.no_grad():
        codes_ae = ae.encode(voxels).cpu().numpy()
        codes_vae = vae.encode(voxels).cpu().numpy()
    codes_ae_flattented = codes_ae.reshape(-1)
    codes_vae_flattented = codes_vae.reshape(-1)
    
    ae_distribution = torch.distributions.normal.Normal(
        np.mean(codes_ae_flattented),
        np.var(codes_ae_flattented) ** 0.5
    )
    vae_distribution = torch.distributions.normal.Normal(
        np.mean(codes_vae_flattented),
        np.var(codes_vae_flattented) ** 0.5
    )

    samples_ae = ae_distribution.sample([SAMPLES, LATENT_CODE_SIZE]).to(device)
    samples_vae = vae_distribution.sample([SAMPLES, LATENT_CODE_SIZE]).to(device)
    with torch.no_grad():
        reconstructed_ae = ae.decode(samples_ae).cpu().numpy()
        reconstructed_vae = vae.decode(samples_vae).cpu().numpy()

    distances_ae = pairwise_distances(codes_ae, samples_ae.cpu().numpy(), metric='cosine')
    indices_ae = np.argmin(distances_ae, axis=0)
    reference_codes_ae = torch.tensor(codes_ae[indices_ae, :], device=device)
    with torch.no_grad():
        reconstructed_references_ae = ae.decode(reference_codes_ae).cpu().numpy()

    distances_vae = pairwise_distances(codes_vae, samples_vae.cpu().numpy(), metric='cosine')
    indices_vae = np.argmin(distances_vae, axis=0)
    reference_codes_vae = torch.tensor(codes_vae[indices_vae, :], device=device)
    with torch.no_grad():
        reconstructed_references_vae = vae.decode(reference_codes_vae).cpu().numpy()

    plot = ImageGrid(SAMPLES, 4)
    
    for i in range(SAMPLES):
        plot.set_voxels(reconstructed_ae[i, :, :, :], i, 0)
        plot.set_voxels(reconstructed_references_ae[i, :, :, :], i, 1)
        plot.set_voxels(reconstructed_vae[i, :, :, :], i, 2)
        plot.set_voxels(reconstructed_references_vae[i, :, :, :], i, 3)

    plot.save("plots/ae-vae-samples.pdf")

if "autoencoder_interpolation" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    voxels = dataset.voxels

    STEPS = 6
    
    indices = random.sample(list(range(dataset.size)), 2)
    print(indices)
    
    ae = load_autoencoder(is_variational=False)
    vae = load_autoencoder(is_variational=True)

    print("Generating codes...")
    with torch.no_grad():
        codes_ae = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        codes_start_end = ae.encode(voxels[indices, :, :, :])
        code_start = codes_start_end[0, :]
        code_end = codes_start_end[1, :]
        for i in range(STEPS):
            codes_ae[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
        reconstructed_ae = ae.decode(codes_ae)
        
        codes_vae = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        codes_start_end = vae.encode(voxels[indices, :, :, :])
        code_start = codes_start_end[0, :]
        code_end = codes_start_end[1, :]
        for i in range(STEPS):
            codes_vae[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
        reconstructed_vae = vae.decode(codes_vae)

    plot = ImageGrid(STEPS, 2)
    
    for i in range(STEPS):
        plot.set_voxels(reconstructed_ae[i, :, :, :], i, 0)
        plot.set_voxels(reconstructed_vae[i, :, :, :], i, 1)

    plot.save("plots/ae-vae-interpolation.pdf")

if "autoencoder_interpolation_2" in sys.argv:
    from dataset import dataset as dataset
    dataset.load_voxels(device)
    voxels = dataset.voxels

    STEPS = 6
    
    indices = random.sample(list(range(dataset.size)), 2)
    print(indices)
    
    vae = load_autoencoder(is_variational=True)

    print("Generating codes...")
    with torch.no_grad():
        codes_vae = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        codes_start_end = vae.encode(voxels[indices, :, :, :])
        code_start = codes_start_end[0, :]
        code_end = codes_start_end[1, :]
        for i in range(STEPS):
            codes_vae[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
        reconstructed_vae = vae.decode(codes_vae)

    plot = ImageGrid(STEPS)
    
    for i in range(STEPS):
        plot.set_voxels(reconstructed_vae[i, :, :, :], i)

    plot.save("plots/vae-interpolation.pdf")

if "gan_tsne" in sys.argv:
    generator = load_generator(is_wgan='wgan' in sys.argv)
    from util import standard_normal_distribution

    shape = torch.Size([500, LATENT_CODE_SIZE])
    x = standard_normal_distribution.sample(shape).to(device)
    with torch.no_grad():
        voxels = generator(x).squeeze()
    codes = x.squeeze().cpu().numpy()
    filename = "plots/gan-images.pdf" if 'wgan' in sys.argv else "plots/wgan-images.pdf"
    create_tsne_plot(codes, voxels, labels = None, filename = filename)

if "gan_examples" in sys.argv:    
    generator = load_generator(is_wgan='wgan' in sys.argv)

    COUNT = 5
    with torch.no_grad():
        voxels = generator.generate(sample_size=COUNT)

    plot = ImageGrid(COUNT)
    for i in range(COUNT):
        plot.set_voxels(voxels[i, :, :, :], i)
    
    filename = "plots/wgan-examples.pdf" if 'wgan' in sys.argv else "plots/gan-examples.pdf"
    plot.save(filename)

if "gan_interpolation" in sys.argv:
    from util import standard_normal_distribution

    STEPS = 6

    generator = load_generator(is_wgan='wgan' in sys.argv)

    print("Generating codes...")
    with torch.no_grad():
        codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        codes_start_end = standard_normal_distribution.sample((2, LATENT_CODE_SIZE))
        code_start = codes_start_end[0, :]
        code_end = codes_start_end[1, :]
        for i in range(STEPS):
            codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
        voxels = generator(codes)

    plot = ImageGrid(STEPS)
    for i in range(STEPS):
        plot.set_voxels(voxels[i, :, :, :], i)
    
    filename = "plots/wgan-interpolation.pdf" if 'wgan' in sys.argv else "plots/gan-interpolation.pdf"
    plot.save(filename)

def get_moving_average(data, window_size):
    moving_average = []
    for i in range(data.shape[0] - window_size):
        moving_average.append(np.mean(data[i:i+window_size]))
    
    return np.arange(window_size / 2, data.shape[0] - window_size / 2, dtype=int), moving_average

if "wgan_training" in sys.argv:
    data = np.genfromtxt('plots/wgan_training.csv', delimiter=' ')
        
    plt.ylim((-400, 1000))
    plt.plot(data[:, 4], label="Assessment of real objects")
    plt.plot(data[:, 3], label="Assessment of fake objects")

    plt.xlabel('Epoch')
    plt.ylabel('Critic output')
    plt.legend()
    plt.savefig("plots/wgan-training-critic.pdf", bbox_inches='tight')

if "sdf_training" in sys.argv:
    data = np.genfromtxt('plots/sdf_net_training.csv', delimiter=' ')
    
    plt.clf()
    plt.plot(np.arange(1, data.shape[0] + 1), data[:, 2], linestyle='-', linewidth=0.5, color='grey')
    plt.plot(np.arange(1, data.shape[0] + 1), data[:, 2], 'x')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("plots/deepsdf-training-loss.pdf", bbox_inches='tight')


def create_autoencoder_training_plot(data_file, title, plot_file):
    if not os.path.isfile(data_file):
        return

    data = np.genfromtxt(data_file, delimiter=' ')
    
    #plt.yscale('log')
    max_reconstruction_loss = np.max(data[:, 2])
    reconstruction_loss = data[:, 2] / max_reconstruction_loss
    kld_loss = data[:, 3] / max_reconstruction_loss
    voxel_error = data[:, 4] / np.max(data[:, 4])
    #plt.axhline(y=data[-1, 2], color='black', linewidth=1)
    plt.plot(reconstruction_loss, label='Reconstruction loss ({:.3f})'.format(data[-1, 2]))
    #plt.plot(kld_loss, label='KLD loss ({:.3f})'.format(data[-1, 3]))
    plt.plot(voxel_error, label='Voxel error ({:.3f})'.format(data[-1, 4]))
    
    plt.xlabel('Epoch')
    plt.yticks([])
    plt.title(title)
    plt.legend(loc='center right')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()

def create_autoencoder_training_plot_latex():
    data = np.genfromtxt('plots/variational_autoencoder_training.csv', delimiter=' ')
    
    plt.plot(data[:, 2], label='Reconstruction loss')
    plt.plot(data[:, 3], label='KLD loss')    
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylabel('Loss')
    plt.savefig('plots/vae-training-loss.pdf', bbox_inches='tight')

    plt.clf()
    plt.plot(data[:, 4]) 
    plt.xlabel('Epoch')
    plt.ylabel('Voxel error')
    plt.savefig('plots/vae-training-error.pdf', bbox_inches='tight')
    
    plt.clf()

if "autoencoder_training" in sys.argv:
    if 'latex' in sys.argv:
        create_autoencoder_training_plot_latex()
    else:
        create_autoencoder_training_plot('plots/autoencoder_training.csv', 'Autoencoder Training', 'plots/autoencoder-training.pdf')
        create_autoencoder_training_plot('plots/variational_autoencoder_training.csv', 'Variational Autoencoder Training', 'plots/variational-autoencoder-training.pdf')

if "sdf_slice" in sys.argv:
    from mesh_to_sdf import mesh_to_sdf, scale_to_unit_sphere
    import trimesh
    import cv2

    model_filename = 'data/shapenet/03001627/6ae8076b0f9c74199c2009e4fd70d135/models/model_normalized.obj'
    
    print("Loading mesh...")
    mesh = trimesh.load(model_filename)
    mesh = scale_to_unit_sphere(mesh)

    resolution = 1280
    slice_position = 0.0
    clip = 0.1
    points = np.meshgrid(
        np.linspace(slice_position, slice_position, 1),
        np.linspace(1, -1, resolution),
        np.linspace(-1, 1, resolution)
    )

    points = np.stack(points)
    points = points.reshape(3, -1).transpose()

    print("Calculating SDF values...")
    sdf = mesh_to_sdf(mesh, points)
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
    dataset.load_voxels(device)
    voxels = dataset.voxels.cpu()
    mask = voxels < 0
    occupied = torch.sum(mask, dim=[1, 2, 3]).numpy()

    plt.hist(occupied, bins=100, range=(0, 10000))
    plt.savefig("plots/voxel-occupancy-histogram.pdf")

if "model_images" in sys.argv:
    from rendering import MeshRenderer
    viewer = MeshRenderer(start_thread=False)
    import trimesh
    import cv2
    import logging

    logging.getLogger('trimesh').setLevel(1000000)

    filenames = open('data/sdf-clouds.txt', 'r').read().split("\n")
    index = 0

    for filename in tqdm(filenames):
        model_filename = filename.replace('sdf-pointcloud.npy', 'model_normalized.obj')
        image_filename = 'screenshots/sdf_meshes/{:d}.png'.format(index)
        index += 1
        if os.path.isfile(image_filename):
            continue

        mesh = trimesh.load(model_filename)
        viewer.set_mesh(mesh, center_and_scale=True)
        image = viewer.get_image(crop=False, output_size=viewer.size, greyscale=False)
        cv2.imwrite(image_filename, image)

if "wgan-results" in sys.argv:
    from util import crop_image

    COUNT = 5
    
    plot = ImageGrid(COUNT, create_viewer=False)
    
    for i in range(COUNT):
        image = plt.imread('screenshots/wgan/{:d}.png'.format(i))
        plot.set_image(crop_image(image, background=1), i)
        
    plot.save('plots/wgan-results.pdf')

if 'sdf_net_reconstruction' in sys.argv:
    from rendering.raymarching import render_image_for_index
    from PIL import Image
    from util import crop_image
    sdf_net, latent_codes = load_sdf_net(return_latent_codes=True)

    COUNT = 5
    MESH_FILENAME = 'screenshots/sdf_meshes/{:d}.png'

    indices = random.sample(range(latent_codes.shape[0]), COUNT)
    print(indices)

    plot = ImageGrid(COUNT, 2, create_viewer=False)

    for i in range(COUNT):
        mesh = Image.open(MESH_FILENAME.format(indices[i]))
        mesh = np.array(mesh)
        mesh = crop_image(mesh)
        plot.set_image(mesh, i, 0)

        image = render_image_for_index(sdf_net, latent_codes, indices[i], crop=True)
        plot.set_image(image, i, 1)

    plot.save('plots/deepsdf-reconstruction.pdf')

if "sdf_net_interpolation" in sys.argv:
    from rendering.raymarching import render_image_for_index, render_image
    sdf_net, latent_codes = load_sdf_net(return_latent_codes=True)
    
    STEPS = 6

    indices = random.sample(list(range(latent_codes.shape[0])), 2)
    print(indices)
    code_start = latent_codes[indices[0], :]
    code_end = latent_codes[indices[1], :]

    print("Generating codes...")
    with torch.no_grad():
        codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        for i in range(STEPS):
            codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)

    plot = ImageGrid(STEPS, create_viewer=False)

    for i in range(STEPS):
        plot.set_image(render_image(sdf_net, codes[i, :], crop=True), i)

    plot.save("plots/deepsdf-interpolation.pdf")

if "sdf_net_sample" in sys.argv:
    from rendering.raymarching import render_image    
    sdf_net, latent_codes = load_sdf_net(return_latent_codes=True)
    latent_codes_flattened = latent_codes.detach().reshape(-1).cpu().numpy()

    COUNT = 5
    
    mean, variance = np.mean(latent_codes_flattened), np.var(latent_codes_flattened) ** 0.5
    print("mean: ", mean)
    print("variance: ", variance)
    distribution = torch.distributions.normal.Normal(mean, variance)
    codes = distribution.sample([COUNT, LATENT_CODE_SIZE]).to(device)
    
    plot = ImageGrid(COUNT, create_viewer=False)

    for i in range(COUNT):
        plot.set_image(render_image(sdf_net, codes[i, :], crop=True), i)

    plot.save("plots/deepsdf-samples.pdf")
    
if "hybrid_gan" in sys.argv:
    from rendering.raymarching import render_image
    from util import standard_normal_distribution
    generator = load_sdf_net(filename='hybrid_gan_generator.to')

    COUNT = 5
    
    codes = standard_normal_distribution.sample([COUNT, LATENT_CODE_SIZE]).to(device)
    
    plot = ImageGrid(COUNT, create_viewer=False)

    for i in range(COUNT):
        plot.set_image(render_image(generator, codes[i, :], radius=1.6, crop=True, sdf_offset=-0.045, vertical_cutoff=1), i)

    plot.save("plots/hybrid-gan-samples.pdf")


if "hybrid_gan_interpolation" in sys.argv:
    from rendering.raymarching import render_image_for_index, render_image
    from util import standard_normal_distribution
    import cv2
    sdf_net = load_sdf_net(filename='hybrid_gan_generator.to')

    OPTIONS = 10
    
    codes = standard_normal_distribution.sample([OPTIONS, LATENT_CODE_SIZE]).to(device)
    for i in range(OPTIONS):
        image = render_image(sdf_net, codes[i, :], resolution=200, radius=1.6, sdf_offset=-0.045, vertical_cutoff=1, crop=True)
        image.save('plots/option-{:d}.png'.format(i))
    
    STEPS = 6
        
    code_start = codes[int(input('Enter index for starting shape: ')), :]
    code_end = codes[int(input('Enter index for ending shape: ')), :]

    with torch.no_grad():
        codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        for i in range(STEPS):
            codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)

    plot = ImageGrid(STEPS, create_viewer=False)
    
    for i in range(STEPS):
        plot.set_image(render_image(sdf_net, codes[i, :], crop=True, radius=1.6, sdf_offset=-0.045, vertical_cutoff=1), i)

    plot.save("plots/hybrid-gan-interpolation.pdf")

if "hybrid_gan_upscaling" in sys.argv:
    from rendering.raymarching import render_image_for_index, render_image
    from util import standard_normal_distribution
    sdf_net = load_sdf_net(filename='hybrid_gan_generator.to')
        
    code = standard_normal_distribution.sample([LATENT_CODE_SIZE]).to(device)

    plot = ImageGrid(4)
    
    voxels_32 = sdf_net.get_voxels(code, 32, sphere_only=False)
    plot.set_voxels(voxels_32, 0)
    voxels_32 = voxels_32[1:-2, 1:-2, 1:-2]

    import scipy.ndimage
    voxels_upscaled = scipy.ndimage.zoom(voxels_32, 4)
    voxels_upscaled = np.pad(voxels_upscaled, 1, mode='constant', constant_values=1)
    plot.set_voxels(voxels_upscaled, 1)

    voxels_128 = sdf_net.get_voxels(code, 128, sphere_only=False)
    plot.set_voxels(voxels_128, 2)

    plot.set_image(render_image(sdf_net, code, radius=1.6, crop=True, vertical_cutoff=1, sdf_offset=-0.045), 3)

    plot.save("plots/hybrid-gan-upscaling.pdf")

if "shapenet-errors" in sys.argv:
    from PIL import Image
    from util import crop_image
    plot = ImageGrid(6, create_viewer=False)

    for i in range(6):
        image = Image.open('screenshots/errors/error-{:d}.png'.format(i+1))
        image = np.array(image)
        image = crop_image(image)
        plot.set_image(image, i)

    plot.save("plots/errors.pdf")

from model import CHECKPOINT_PATH

if 'vae_checkpoints' in sys.argv:
    COUNT = 5
    checkpoints = os.listdir(CHECKPOINT_PATH)
    checkpoints = [i for i in checkpoints if i.startswith('variational-autoencoder-64-')]
    checkpoints = sorted(checkpoints)
    checkpoints = checkpoints[:6]
    checkpoints = [checkpoints[i * (len(checkpoints) - 1) // (COUNT - 1)] for i in range(COUNT)]
    print('\n'.join(checkpoints))
    
    from dataset import dataset
    dataset.load_voxels(device=device)

    MODEL_INDEX = random.randint(0, dataset.voxels.shape[0]-1)
    print(MODEL_INDEX)
    model = dataset.voxels[(MODEL_INDEX, MODEL_INDEX), :, :, :]

    from model.autoencoder import Autoencoder
    vae = Autoencoder()
    vae.eval()

    plot = ImageGrid(COUNT)
    with torch.no_grad():
        for i in range(COUNT):
            vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, checkpoints[i])))
            reconstructed, _, _ = vae(model)
            plot.set_voxels(reconstructed[0, :, :, :], i)

    plot.save('plots/vae-checkpoints.pdf')

if 'sdf_checkpoints' in sys.argv:
    from rendering.raymarching import render_image
    COUNT = 5
    checkpoints = os.listdir(CHECKPOINT_PATH)
    checkpoints_network = [i for i in checkpoints if i.startswith('sdf_net-epoch-')]
    checkpoints_latent_codes = [i for i in checkpoints if i.startswith('sdf_net_latent_codes-epoch-')]
    checkpoints_network = sorted(checkpoints_network)
    checkpoints_latent_codes = sorted(checkpoints_latent_codes)
    indices = [i * (len(checkpoints_network) - 1) // (COUNT - 1) for i in range(COUNT)]

    checkpoints_network = [checkpoints_network[i] for i in indices]
    checkpoints_latent_codes = [checkpoints_latent_codes[i] for i in indices]
    print('\n'.join(checkpoints_network))

    MODEL_INDEX = 1000
    print(MODEL_INDEX)

    from model.sdf_net import SDFNet
    sdf_net = SDFNet()
    sdf_net.eval()

    plot = ImageGrid(COUNT, create_viewer=False)
    for i in range(COUNT):
        sdf_net.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, checkpoints_network[i])))
        latent_codes = torch.load(os.path.join(CHECKPOINT_PATH, checkpoints_latent_codes[i])).detach()
        latent_code = latent_codes[MODEL_INDEX, :]
        plot.set_image(render_image(sdf_net, latent_code, crop=True), i)

    plot.save('plots/deepsdf-checkpoints.pdf')



if "deepsdf-interpolation-stl" in sys.argv:
    from rendering.raymarching import render_image_for_index, render_image
    sdf_net, latent_codes = load_sdf_net(return_latent_codes=True)
    
    STEPS = 5

    indices = random.sample(list(range(latent_codes.shape[0])), 2)
    print(indices)
    code_start = latent_codes[indices[0], :]
    code_end = latent_codes[indices[1], :]

    print("Generating codes...")
    with torch.no_grad():
        codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        for i in range(STEPS):
            codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
    
    for i in range(STEPS):
        print(i)
        mesh = sdf_net.get_mesh(codes[i, :], voxel_resolution=256, sphere_only=False)
        mesh.export('plots/mesh-{:d}.stl'.format(i))