from util import device, ensure_directory, standard_normal_distribution
from dataset import dataset
import scipy
import numpy as np
from rendering import MeshRenderer
import torch
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import Bbox
from sklearn.cluster import KMeans
from model.classifier import Classifier

SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

SURFACE_LEVEL = 0.011

FRAMES = SAMPLE_COUNT * TRANSITION_FRAMES
progress = np.arange(FRAMES, dtype=float) / TRANSITION_FRAMES

from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
vae = Autoencoder()
vae.load()
vae.eval()

classifier = Classifier()
classifier.load()
classifier.eval()

stop_latent_codes = standard_normal_distribution.sample((SAMPLE_COUNT + 1, LATENT_CODE_SIZE)).numpy()
stop_latent_codes[-1, :] = stop_latent_codes[0, :]

spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), stop_latent_codes, axis=0, bc_type='periodic')
frame_latent_codes = spline(progress)

width, height = 40, 40

PLOT_FILE_NAME = 'plot.png'
ensure_directory('images')

colors = np.array([dataset.get_color(i) for i in range(dataset.label_count)])
names = [i.name.split(',')[0] for i in dataset.categories]
category_indices = np.arange(dataset.label_count)

def create_plot(predicted_classes, resolution=1080, filename=PLOT_FILE_NAME, dpi=200):
    size_inches = resolution / dpi

    
    fig, ax = plt.subplots(1, figsize=(size_inches, size_inches), dpi=dpi)
    
    ax.bar(category_indices, predicted_classes, color=colors)
    ax.set_ylim((0, 1))
    ax.set_xticks(category_indices)
    ax.set_xticklabels(names)
    ax.get_yaxis().set_visible(False)

    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

frame_latent_codes = torch.tensor(frame_latent_codes, dtype=torch.float32, device=device)

print("Rendering...")
viewer = MeshRenderer(size=1080, start_thread=False)

def render_frame(frame_index):
    with torch.no_grad():
        voxels = vae.decode(frame_latent_codes[frame_index, :])
        predicted_classes = classifier.forward(voxels).detach().cpu().squeeze().numpy()

    weighted_colors = colors * predicted_classes[:, np.newaxis]
    viewer.model_color = tuple(np.sum(weighted_colors, axis=0))
    viewer.set_voxels(voxels)
    
    image_mesh = viewer.get_image(flip_red_blue=True)

    create_plot(predicted_classes)
    image_tsne = plt.imread(PLOT_FILE_NAME)[:, :, [2, 1, 0]] * 255

    image = np.concatenate((image_mesh, image_tsne), axis=1)

    cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)


for frame_index in tqdm(range(SAMPLE_COUNT * TRANSITION_FRAMES)):
    render_frame(frame_index)
    frame_index += 1

print("\n\nUse this command to create a video:\n")
print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')