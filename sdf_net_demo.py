from itertools import count
import torch
import time
import numpy as np
import sys
import random

from voxel.viewer import VoxelViewer
from model import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sdf_net = SDFNet()
sdf_net.load()
sdf_net.eval()
latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)

MODEL_COUNT = latent_codes.shape[0]
TRANSITION_FRAMES = 40
WAIT_FRAMES = 10
MODEL_COUNT = 30

WAIT_TIME = 0.8
TRANSITION_TIME = 0.4

def get_random_latent_code():
    return latent_codes[random.randrange(MODEL_COUNT), :]

frame_index = 0

previous_model = None
start_model = get_random_latent_code()


def save_image(image, frame_index):
    cv2.imwrite("images/frame-{:05d}.png".format(frame_index), image)

def create_image_sequence():
    viewer = VoxelViewer(start_thread=False)
    progress_bar = tqdm(total=MODEL_COUNT * (TRANSITION_FRAMES + WAIT_FRAMES))
    next_model = start_model

    for model_index in range(MODEL_COUNT):
        previous_model = next_model
        if model_index == MODEL_COUNT - 1:
            next_model = start_model
        else:
            next_model = get_random_latent_code()

        for step in tqdm(list(range(TRANSITION_FRAMES))):
            progress = step / TRANSITION_FRAMES
            latent_code = previous_model * (1 - progress) + next_model * progress
            
            try:
                viewer.set_mesh(sdf_net.get_mesh(latent_code, device, voxel_count=140))
            except ValueError:
                pass
            image = viewer.get_image(crop = False, output_size = 800)
            save_image(image, frame_index)
            frame_index += 1
            progress_bar.update()
        
        for _ in range(WAIT_FRAMES):
            image = viewer.get_image(crop = False, output_size = 800)
            save_image(image, frame_index)
            frame_index += 1
            progress_bar.update()

def show_models():
    viewer = VoxelViewer()
    next_model = start_model

    for epoch in count():
        try:
            previous_model = next_model
            next_model = get_random_latent_code()

            for step in range(TRANSITION_FRAMES + 1):
                progress = step / TRANSITION_FRAMES
                model = previous_model * (1 - progress) + next_model * progress
                
                try:
                    viewer.set_mesh(sdf_net.get_mesh(model, device, voxel_count=64))
                except ValueError:
                    pass
                time.sleep(TRANSITION_TIME / TRANSITION_FRAMES)

            time.sleep(WAIT_TIME)
            
        except KeyboardInterrupt:
            viewer.stop()
            return

if "save" in sys.argv:
    create_image_sequence()
else:
    show_models()