import os.path as osp
import sys

import numpy as np
import argparse
import torch
import trimesh
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from generator import SDFGenerator

# sys.path.insert(0, '../..')
# sys.path.insert(0, '../../latent_3d_points/external/structural_losses')
# from latent_3d_points.src.evaluation_metrics import (
#     minimum_mathing_distance, jsd_between_point_cloud_sets, coverage)  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--G', type=str, required=True)
args = parser.parse_args()

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
G = G.to(device)
G.load_state_dict(torch.load(args.G, map_location=device))
G.eval()

torch.manual_seed(1234)
generated = []
for i in range(100):
    print(i)
    z = torch.randn((LATENT_SIZE, ), device=device)
    mesh = G.get_mesh(z, 64, level=0.05)
    generated += [torch.from_numpy(mesh.sample(2048))]
generated = torch.stack(generated, dim=0)
print(generated.size())
torch.save(generated, '/data/SDF_GAN/chairs/results/uniform_05.pt')

# root = osp.join('/data/SDF_GAN', args.category)
# with open(osp.join(root, 'val.txt'), 'r') as f:
#     names = f.read().split('\n')
#     names = names[:-1] if names[-1] == '' else names

# def scale_to_unit_sphere(mesh):
#     origin = mesh.bounding_box.centroid
#     vertices = mesh.vertices - origin
#     distances = np.linalg.norm(vertices, axis=1)
#     size = np.max(distances)
#     vertices /= size
#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

# pos = []
# for name in names:
#     path = osp.join(root, 'raw', name, 'model.obj')
#     mesh = trimesh.load_mesh(path)
#     mesh = mesh.dump().sum() if isinstance(mesh, trimesh.Scene) else mesh
#     mesh = scale_to_unit_sphere(mesh)
#     pos += [torch.from_numpy(mesh.sample(2048))]
# pos = torch.stack(pos, dim=0)
# print(pos.size())
# print(pos.min(), pos.max())
