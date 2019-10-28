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
import matplotlib.pyplot as plt
from dataset import dataset

SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

ROTATE_MODEL = False

SURFACE_LEVEL = 0.011

FRAMES = SAMPLE_COUNT * TRANSITION_FRAMES
progress = np.arange(FRAMES, dtype=float) / TRANSITION_FRAMES

latent_codes = torch.load(LATENT_CODES_FILENAME).detach().cpu().numpy()
labels = dataset.load_labels().detach().cpu().numpy()

SIZE = latent_codes.shape[0]

indices = random.sample(list(range(SIZE)), SAMPLE_COUNT + 1)
indices[0] = indices[-1] # Make animation periodic
stop_latent_codes = latent_codes[indices, :]

colors = np.zeros((labels.shape[0], 3))
for i in range(labels.shape[0]):
    colors[i, :] = dataset.get_color(labels[i])

spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), stop_latent_codes, axis=0, bc_type='periodic')
frame_latent_codes = spline(progress)

color_spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), colors[indices, :], axis=0, bc_type='periodic')
frame_colors = color_spline(progress)
frame_colors = np.clip(frame_colors, 0, 1)

from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, Bbox

width, height = 40, 40

print("Calculating t-sne embedding...")
tsne = TSNE(n_components=2)
x = np.concatenate((latent_codes[:SIZE, :], frame_latent_codes), axis=0)
y = tsne.fit_transform(x)
latent_codes_embedded = y[:SIZE, :]
print("Done calculating t-sne")

frame_latent_codes_embedded = y[SIZE:, :]
frame_latent_codes_embedded[0, :] = frame_latent_codes_embedded[-1, :]

PLOT_FILE_NAME = 'tsne.png'

margin = 2
range_x = (y[:, 0].min() - margin, y[:, 0].max() + margin)
range_y = (y[:, 1].min() - margin, y[:, 1].max() + margin)

del x, y, tsne
plt.ioff()

def create_plot(index, resolution=1080, filename=PLOT_FILE_NAME, dpi=100):
    frame_color = frame_colors[index, :]
    frame_color = (frame_color[0], frame_color[1], frame_color[2], 1.0)

    size_inches = resolution / dpi

    fig, ax = plt.subplots(1, figsize=(size_inches, size_inches), dpi=dpi)
    ax.set_position([0, 0, 1, 1])
    plt.axis('off')
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)

    ax.plot(frame_latent_codes_embedded[:, 0], frame_latent_codes_embedded[:, 1], c=(0.2, 0.2, 0.2, 1.0), zorder=1)
    ax.scatter(latent_codes_embedded[:, 0], latent_codes_embedded[:, 1], c=colors[:SIZE], s = 10, zorder=0)
    ax.scatter(frame_latent_codes_embedded[index, 0], frame_latent_codes_embedded[index, 1], facecolors=frame_color, s = 200, linewidths=2, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=2)
    ax.scatter(latent_codes_embedded[indices, 0], latent_codes_embedded[indices, 1], facecolors=colors[indices, :], s = 140, linewidths=1, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=3)
    
    fig.savefig(filename, bbox_inches=Bbox([[0, 0], [size_inches, size_inches]]), dpi=dpi)
    plt.close(fig)

sdf_net = SDFNet()
sdf_net.load()
sdf_net.eval()

frame_latent_codes = torch.tensor(frame_latent_codes, dtype=torch.float32, device=device)

viewer = MeshRenderer(size=1080, start_thread=False)

def render_frame(frame_index):
    viewer.model_color = frame_colors[frame_index, :]
    viewer.set_mesh(sdf_net.get_mesh(frame_latent_codes[frame_index, :], voxel_resolution=128, sphere_only=True, level=SURFACE_LEVEL))
    image_mesh = viewer.get_image(flip_red_blue=True)

    create_plot(frame_index)
    image_tsne = plt.imread(PLOT_FILE_NAME)[:, :, [2, 1, 0]] * 255

    image = np.concatenate((image_mesh, image_tsne), axis=1)

    cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)


for frame_index in tqdm(range(SAMPLE_COUNT * TRANSITION_FRAMES)):
    render_frame(frame_index)
    frame_index += 1

print("\n\nUse this command to create a video:\n")
print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')