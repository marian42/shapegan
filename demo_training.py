from mesh_to_sdf import sample_sdf_near_surface
import numpy as np
import trimesh
import torch
from util import device, ensure_directory

from model.sdf_net import SDFNet
from rendering import MeshRenderer
import sys
import cv2
LATENT_CODE_SIZE = 0

MODEL_PATH = 'examples/chair.obj'

mesh = trimesh.load(MODEL_PATH)
points, sdf = sample_sdf_near_surface(mesh)

save_images = 'save' in sys.argv

if save_images:
    viewer = MeshRenderer(start_thread=False, size=1080)
    ensure_directory('images')
else:
    viewer = MeshRenderer()

points = torch.tensor(points, dtype=torch.float32, device=device)
sdf = torch.tensor(sdf, dtype=torch.float32, device=device)
sdf.clamp_(-0.1, 0.1)

sdf_net = SDFNet(latent_code_size=LATENT_CODE_SIZE).to(device)
optimizer = torch.optim.Adam(sdf_net.parameters(), lr=1e-5)

BATCH_SIZE = 20000
latent_code = torch.zeros((BATCH_SIZE, LATENT_CODE_SIZE), device=device)
indices = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=device)

positive_indices = (sdf > 0).nonzero().squeeze().cpu().numpy()
negative_indices = (sdf < 0).nonzero().squeeze().cpu().numpy()

step = 0
error_targets = np.logspace(np.log10(0.02), np.log10(0.0005), num=500)
image_index = 0

while True:
    try:
        indices[:BATCH_SIZE//2] = torch.tensor(np.random.choice(positive_indices, BATCH_SIZE//2), device=device)
        indices[BATCH_SIZE//2:] = torch.tensor(np.random.choice(negative_indices, BATCH_SIZE//2), device=device)

        sdf_net.zero_grad()
        predicted_sdf = sdf_net(points[indices, :], latent_code)
        batch_sdf = sdf[indices]
        loss = torch.mean(torch.abs(predicted_sdf - batch_sdf))
        loss.backward()
        optimizer.step()

        if loss.item() < error_targets[image_index]:
            try:
                viewer.set_mesh(sdf_net.get_mesh(latent_code[0, :], voxel_resolution=64, raise_on_empty=True))
                if save_images:
                    image = viewer.get_image(flip_red_blue=True)
                    cv2.imwrite("images/frame-{:05d}.png".format(image_index), image)
                image_index += 1
            except ValueError:
                pass
        step += 1
        print('Step {:04d}, Image {:04d}, loss: {:.6f}, target: {:.6f}'.format(step, image_index, loss.item(), error_targets[image_index]))
    except KeyboardInterrupt:
        viewer.stop()
        break
