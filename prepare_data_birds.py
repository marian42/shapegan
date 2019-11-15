import os
from rendering import MeshRenderer
import time
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np

DIRECTORY = 'data/birds/'

def get_obj_files():
    for directory, _, files in os.walk(DIRECTORY):
        for filename in files:
            if filename.endswith('.obj'):
                yield os.path.join(directory, filename)

files = list(get_obj_files())

viewer = MeshRenderer()
rotation = np.matmul(
    Rotation.from_euler('y', 90, degrees=True).as_dcm(),
    Rotation.from_euler('x', -90, degrees=True).as_dcm())

for filename in tqdm(files):
    mesh = trimesh.load(filename)
    vertices = np.matmul(rotation, mesh.vertices.transpose()).transpose()
    normals = np.matmul(rotation, mesh.vertex_normals.transpose()).transpose()
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, vertex_normals=normals)
    viewer.set_mesh(mesh, center_and_scale=True)
    #time.sleep(1)