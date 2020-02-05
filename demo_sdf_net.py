from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device, standard_normal_distribution, ensure_directory
import scipy.interpolate
import numpy as np
from rendering import MeshRenderer
import time
import torch
from tqdm import tqdm
import cv2
import random
import sys

SAMPLE_COUNT = 30 # Number of distinct objects to generate and interpolate between
TRANSITION_FRAMES = 60

ROTATE_MODEL = False
USE_HYBRID_GAN = True

SURFACE_LEVEL = 0.04 if USE_HYBRID_GAN else 0.011

sdf_net = SDFNet()
if USE_HYBRID_GAN:
    sdf_net.filename = 'hybrid_progressive_gan_generator_3.to'
sdf_net.load()
sdf_net.eval()

if USE_HYBRID_GAN:
    codes = standard_normal_distribution.sample((SAMPLE_COUNT + 1, LATENT_CODE_SIZE)).numpy()
else:
    latent_codes = torch.load(LATENT_CODES_FILENAME).detach().cpu().numpy()
    indices = random.sample(list(range(latent_codes.shape[0])), SAMPLE_COUNT + 1)
    codes = latent_codes[indices, :]

codes[0, :] = codes[-1, :] # Make animation periodic
spline = scipy.interpolate.CubicSpline(np.arange(SAMPLE_COUNT + 1), codes, axis=0, bc_type='periodic')

def create_image_sequence():
    ensure_directory('images')
    frame_index = 0
    viewer = MeshRenderer(size=1080, start_thread=False)
    progress_bar = tqdm(total=SAMPLE_COUNT * TRANSITION_FRAMES)

    for sample_index in range(SAMPLE_COUNT):
        for step in range(TRANSITION_FRAMES):
            code = torch.tensor(spline(float(sample_index) + step / TRANSITION_FRAMES), dtype=torch.float32, device=device)
            if ROTATE_MODEL:
                viewer.rotation = (147 + frame_index / (SAMPLE_COUNT * TRANSITION_FRAMES) * 360 * 6, 40)
            viewer.set_mesh(sdf_net.get_mesh(code, voxel_resolution=128, sphere_only=False, level=SURFACE_LEVEL))
            image = viewer.get_image(flip_red_blue=True)
            cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)
            frame_index += 1
            progress_bar.update()
    
    print("\n\nUse this command to create a video:\n")
    print('ffmpeg -framerate 30 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')

def show_models():
    TRANSITION_TIME = 2
    viewer = MeshRenderer()

    while True:
        for sample_index in range(SAMPLE_COUNT):
            try:
                start = time.perf_counter()
                end = start + TRANSITION_TIME
                while time.perf_counter() < end:
                    progress = min((time.perf_counter() - start) / TRANSITION_TIME, 1.0)
                    if ROTATE_MODEL:
                        viewer.rotation = (147 + (sample_index + progress) / SAMPLE_COUNT * 360 * 6, 40)
                    code = torch.tensor(spline(float(sample_index) + progress), dtype=torch.float32, device=device)
                    viewer.set_mesh(sdf_net.get_mesh(code, voxel_resolution=64, sphere_only=False, level=SURFACE_LEVEL))
                
            except KeyboardInterrupt:
                viewer.stop()
                return

def create_objects():
    from util import ensure_directory
    from rendering.raymarching import render_image
    from rendering.math import get_rotation_matrix
    import os
    ensure_directory('generated_objects/')
    image_filename = 'generated_objects/chair-{:03d}.png'
    mesh_filename = 'generated_objects/chair-{:03d}.stl'
    index = 0
    while True:
        if os.path.exists(image_filename.format(index)) or os.path.exists(mesh_filename.format(index)):
            index += 1
            continue
        latent_code = standard_normal_distribution.sample((LATENT_CODE_SIZE,)).to(device)
        image = render_image(sdf_net, latent_code, resolution=128, sdf_offset=-SURFACE_LEVEL, ssaa=2, radius=1.4, color=(0.7, 0.7, 0.7))
        image.save(image_filename.format(index))
        mesh = sdf_net.get_mesh(latent_code, voxel_resolution=256, sphere_only=False, level=SURFACE_LEVEL)
        mesh.apply_transform(get_rotation_matrix(90, 'x'))
        mesh.apply_translation((0, 0, -np.min(mesh.vertices[:, 2])))
        mesh.export(mesh_filename.format(index))
        print("Created mesh for index {:d}".format(index))
        index += 1


if 'save' in sys.argv:
    create_image_sequence()
elif 'create_objects' in sys.argv:
    create_objects()
else:
    show_models()