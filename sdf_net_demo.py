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

viewer = VoxelViewer()

STEPS = 20


TRANSITION_TIME = 0.4
WAIT_TIME = 0.8

def get_random():
    return latent_codes[random.randrange(MODEL_COUNT), :]


previous_model = None
next_model = get_random()

for epoch in count():
    try:
        previous_model = next_model
        next_model = get_random()

        for step in range(STEPS + 1):
            progress = step / STEPS
            model = None
            if step < STEPS:
                model = previous_model * (1 - progress) + next_model * progress
            else:
                model = next_model

            
            viewer.set_mesh(sdf_net.get_mesh(model, device))
            #time.sleep(TRANSITION_TIME / STEPS)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break