from util import device, ensure_directory
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

SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

USE_VAE = False

SURFACE_LEVEL = 0.011

FRAMES = SAMPLE_COUNT * TRANSITION_FRAMES
progress = np.arange(FRAMES, dtype=float) / TRANSITION_FRAMES


if USE_VAE:
    print("Loading voxels...")
    dataset.load_voxels(device)

    from model.autoencoder import Autoencoder, LATENT_CODE_SIZE
    vae = Autoencoder()
    vae.load()
    vae.eval()
    print("Calculating latent codes...")

    latent_codes = torch.zeros((dataset.size, LATENT_CODE_SIZE))
    batch_size = 1000
    with torch.no_grad():
        for i in tqdm(range(dataset.size // batch_size + 1)):
            start = i * batch_size
            end = min((i + 1) * batch_size, dataset.size)
            latent_codes[start:end, :] = vae.encode(dataset.voxels[start:end, :, :, :]).detach().cpu()
    del dataset.voxels
    latent_codes = latent_codes.numpy()
else:
    from model.sdf_net import SDFNet, LATENT_CODES_FILENAME
    latent_codes = torch.load(LATENT_CODES_FILENAME).detach().cpu().numpy()

    sdf_net = SDFNet()
    sdf_net.load()
    sdf_net.eval()

print("Calculating embedding...")
tsne = TSNE(n_components=2)
latent_codes_embedded = tsne.fit_transform(latent_codes)
print("Calculating clusters...")
kmeans = KMeans(n_clusters=SAMPLE_COUNT)

indices = np.zeros(SAMPLE_COUNT, dtype=int)
kmeans_clusters = kmeans.fit_predict(latent_codes_embedded)
for i in range(SAMPLE_COUNT):
    center = kmeans.cluster_centers_[i, :]
    dist = np.linalg.norm(latent_codes_embedded - center[np.newaxis, :], axis=1)
    indices[i] = np.argmin(dist)

def try_find_shortest_roundtrip(indices):
    best_order = indices
    best_distance = None
    for _ in range(5000):
        candiate = best_order.copy()
        a = random.randint(0, SAMPLE_COUNT-1)
        b = random.randint(0, SAMPLE_COUNT-1)
        candiate[a] = best_order[b]
        candiate[b] = best_order[a]
        dist = np.sum(np.linalg.norm(latent_codes_embedded[candiate, :] - latent_codes_embedded[np.roll(candiate, 1), :], axis=1)).item()
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_order = candiate

    return best_order, best_distance

def find_shortest_roundtrip(indices):
    best_order, best_distance = try_find_shortest_roundtrip(indices)

    for _ in tqdm(range(100)):
        np.random.shuffle(indices)
        order, distance = try_find_shortest_roundtrip(indices)
        if distance < best_distance:
            best_order = order
    return best_order

print("Calculating trip...")
indices = find_shortest_roundtrip(indices)
indices = np.concatenate((indices, indices[0][np.newaxis]))

SIZE = latent_codes.shape[0]

stop_latent_codes = latent_codes[indices, :]

spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), stop_latent_codes, axis=0, bc_type='periodic')
frame_latent_codes = spline(progress)

embedded_spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), latent_codes_embedded[indices, :], axis=0, bc_type='periodic')
frame_latent_codes_embedded = embedded_spline(progress)
frame_latent_codes_embedded[0, :] = frame_latent_codes_embedded[-1, :]

width, height = 40, 40

PLOT_FILE_NAME = 'tsne.png'
ensure_directory('images')

margin = 2
range_x = (latent_codes_embedded[:, 0].min() - margin, latent_codes_embedded[:, 0].max() + margin)
range_y = (latent_codes_embedded[:, 1].min() - margin, latent_codes_embedded[:, 1].max() + margin)

plt.ioff()

def create_plot(index, resolution=1080, filename=PLOT_FILE_NAME, dpi=100):
    size_inches = resolution / dpi

    fig, ax = plt.subplots(1, figsize=(size_inches, size_inches), dpi=dpi)
    ax.set_position([0, 0, 1, 1])
    plt.axis('off')
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)

    ax.plot(frame_latent_codes_embedded[:, 0], frame_latent_codes_embedded[:, 1], c=(0.2, 0.2, 0.2, 1.0), zorder=1, linewidth=2)
    ax.scatter(latent_codes_embedded[:, 0], latent_codes_embedded[:, 1], c='red', s = 10, zorder=0)
    ax.scatter(frame_latent_codes_embedded[index, 0], frame_latent_codes_embedded[index, 1], facecolors='red', s = 200, linewidths=2, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=2)
    ax.scatter(latent_codes_embedded[indices, 0], latent_codes_embedded[indices, 1], facecolors='red', s = 140, linewidths=1, edgecolors=(0.1, 0.1, 0.1, 1.0), zorder=3)
    
    fig.savefig(filename, bbox_inches=Bbox([[0, 0], [size_inches, size_inches]]), dpi=dpi)
    plt.close(fig)

frame_latent_codes = torch.tensor(frame_latent_codes, dtype=torch.float32, device=device)

print("Rendering...")
viewer = MeshRenderer(size=1080, start_thread=False)

def render_frame(frame_index):
    with torch.no_grad():
        if USE_VAE:
            viewer.set_voxels(vae.decode(frame_latent_codes[frame_index, :]))
        else:
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