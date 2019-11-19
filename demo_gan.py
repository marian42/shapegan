from itertools import count
import torch
import time
import numpy as np
import sys

from rendering import MeshRenderer
from model.gan import Generator, LATENT_CODE_SIZE
from util import device, standard_normal_distribution

generator = Generator()
if "wgan" in sys.argv:
    generator.filename = "wgan-generator.to"
generator.load()
generator.eval()

viewer = MeshRenderer()

STEPS = 20

TRANSITION_TIME = 0.4
WAIT_TIME = 0.8

def get_random():
    return standard_normal_distribution.sample(sample_shape=(LATENT_CODE_SIZE,)).to(device)

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

            viewer.set_voxels(generator(model).squeeze().detach().cpu())
            time.sleep(TRANSITION_TIME / STEPS)

        time.sleep(WAIT_TIME)
        
    except KeyboardInterrupt:
        viewer.stop()
        break