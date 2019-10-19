from itertools import count
import torch
import time
import numpy as np
import sys
import random
from tqdm import tqdm
import cv2
import pyrender

from voxel.viewer import VoxelViewer
from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device

USE_HYBRID_GAN = 'hybrid_gan' in sys.argv or 'hybrid_wgan' in sys.argv

sdf_net = SDFNet()
if USE_HYBRID_GAN:
    sdf_net.filename = 'hybrid_gan_generator.to' if 'hybrid_gan' in sys.argv else 'hybrid_wgan_generator.to'
sdf_net.load()
sdf_net.eval()

SAMPLE_FROM_LATENT_DISTRIBUTION = 'sample_latent' in sys.argv
SAMPLE_FROM_NORMAL_DISTRIBUTION = 'sample_normal' in sys.argv

if not SAMPLE_FROM_NORMAL_DISTRIBUTION:
    latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)

def get_sdf_latent_distribution():
    print("Calculating latent distribution...")
    latent_codes_flattened = latent_codes.detach().cpu().numpy().reshape(-1)
    mean = np.mean(latent_codes_flattened)
    variance = np.var(latent_codes_flattened) ** 0.5
    print('Latent distribution: µ = {:.3f}, σ = {:.3f}'.format(mean, variance))
    return torch.distributions.normal.Normal(mean, variance)

if SAMPLE_FROM_LATENT_DISTRIBUTION:
    latent_code_distribution = get_sdf_latent_distribution()
if SAMPLE_FROM_NORMAL_DISTRIBUTION:
    latent_code_distribution = torch.distributions.normal.Normal(0, 1)
    
TRANSITION_FRAMES = 40
WAIT_FRAMES = 10
SAMPLE_COUNT = 40

TRANSITION_TIME = 2
WAIT_TIME = 1.2

def get_random_latent_code():
    if SAMPLE_FROM_LATENT_DISTRIBUTION or SAMPLE_FROM_NORMAL_DISTRIBUTION:
        return latent_code_distribution.sample((LATENT_CODE_SIZE,)).to(device)
    else:
        return latent_codes[random.randrange(latent_codes.shape[0]), :]

start_model = get_random_latent_code()

def save_image(image, frame_index):
    cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)

def create_image_sequence():
    previous_model = None
    frame_index = 0
    viewer = VoxelViewer(size=1280, start_thread=False)
    progress_bar = tqdm(total=SAMPLE_COUNT * (TRANSITION_FRAMES + 1))
    next_model = start_model

    for sample_index in range(SAMPLE_COUNT):
        previous_model = next_model
        if sample_index == SAMPLE_COUNT - 1:
            next_model = start_model
        else:
            next_model = get_random_latent_code()

        for step in range(TRANSITION_FRAMES):
            progress = step / TRANSITION_FRAMES
            latent_code = previous_model * (1 - progress) + next_model * progress
            
            viewer.set_mesh(sdf_net.get_mesh(latent_code, voxel_count=140, sphere_only=not USE_HYBRID_GAN))
            
            image = viewer.get_image(flip_red_blue=True)
            save_image(image, frame_index)
            frame_index += 1
            progress_bar.update()
        
        image = viewer.get_image(flip_red_blue=True)
        for _ in range(WAIT_FRAMES):
            save_image(image, frame_index)
            frame_index += 1

    print("\n\nUse this command to create a video:\n")
    print('ffmpeg -framerate 24 -i images/frame-%05d.png -c:v libx264 -profile:v high -crf 19 -pix_fmt yuv420p video.mp4')

def show_random_pointclouds():
    while True:
        latent_code = get_random_latent_code()
        points, normals = sdf_net.get_surface_points(latent_code, return_normals=True)

        scene = pyrender.Scene()
        pyrender_pointcloud = pyrender.Mesh.from_points(points.detach().cpu().numpy(), normals=normals.detach().cpu().numpy())
        scene.add(pyrender_pointcloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

def show_models():
    previous_model = None
    viewer = VoxelViewer()
    next_model = start_model

    for _ in count():
        try:
            previous_model = next_model
            next_model = get_random_latent_code()

            start = time.perf_counter()
            end = start + TRANSITION_TIME
            while time.perf_counter() < end:
                progress = min((time.perf_counter() - start) / TRANSITION_TIME, 1.0)
                model = previous_model * (1 - progress) + next_model * progress
                
                viewer.set_mesh(sdf_net.get_mesh(model, voxel_count=48, sphere_only=not USE_HYBRID_GAN))

            time.sleep(WAIT_TIME)
            
        except KeyboardInterrupt:
            viewer.stop()
            return

if "save" in sys.argv:
    create_image_sequence()
elif "pointcloud" in sys.argv:
    show_random_pointclouds()
else:
    show_models()