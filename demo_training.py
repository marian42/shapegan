from sdf.mesh_to_sdf import MeshSDF, scale_to_unit_sphere
import numpy as np
import trimesh
import torch
from util import device

from model.sdf_net import SDFNet
from rendering import MeshRenderer
latent_code_size = 0

MODEL_PATH = 'examples/chair.obj'

mesh = trimesh.load(MODEL_PATH)
mesh = scale_to_unit_sphere(mesh)
mesh_sdf = MeshSDF(mesh)
points, sdf = mesh_sdf.get_sample_points()

points = torch.tensor(points, dtype=torch.float32, device=device)
sdf = torch.tensor(sdf, dtype=torch.float32, device=device)
sdf.clamp_(-0.1, 0.1)

viewer = MeshRenderer()

sdf_net = SDFNet(latent_code_size=latent_code_size).to(device)
optimizer = torch.optim.Adam(sdf_net.parameters(), lr=1e-4)

BATCH_SIZE = 20000
latent_code = torch.zeros((BATCH_SIZE, latent_code_size), device=device)
indices = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=device)

positive_indices = (sdf > 0).nonzero().squeeze().cpu().numpy()
negative_indices = (sdf < 0).nonzero().squeeze().cpu().numpy()

step = 0
while True:
    try:
        #indices.random_(points.shape[0] - 1)
        indices[:BATCH_SIZE//2] = torch.tensor(np.random.choice(positive_indices, BATCH_SIZE//2), device=device)
        indices[BATCH_SIZE//2:] = torch.tensor(np.random.choice(negative_indices, BATCH_SIZE//2), device=device)

        sdf_net.zero_grad()
        predicted_sdf = sdf_net.forward(points[indices, :], latent_code)
        batch_sdf = sdf[indices]
        loss = torch.mean(torch.abs(predicted_sdf - batch_sdf))
        print(step, loss.item())
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            viewer.set_mesh(sdf_net.get_mesh(latent_code[0, :], voxel_resolution=192))
        step += 1
    except KeyboardInterrupt:
        viewer.stop()
        break
