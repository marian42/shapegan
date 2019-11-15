import os
from rendering import MeshRenderer
import time
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np
from sdf.mesh_to_sdf import MeshSDF
from util import ensure_directory

DIRECTORY = 'data/birds/'
DIRECTORY_SDF = 'data/birds_sdf'

def get_obj_files():
    for directory, _, files in os.walk(DIRECTORY):
        for filename in files:
            if filename.endswith('.obj'):
                yield os.path.join(directory, filename)

files = list(get_obj_files())
rotation = np.matmul(
    Rotation.from_euler('y', 90, degrees=True).as_dcm(),
    Rotation.from_euler('x', -90, degrees=True).as_dcm())

def process_obj_file(in_file, out_file, transform_file):
    mesh = trimesh.load(in_file)
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    vertices = np.matmul(rotation, mesh.vertices.transpose()).transpose()
    normals = np.matmul(rotation, mesh.vertex_normals.transpose()).transpose()
    centroid = np.matmul(rotation, mesh.bounding_box.centroid)
    vertices -= centroid[np.newaxis, :]
    scale = np.max(np.linalg.norm(vertices, axis=1)) * 1.05
    vertices /= scale
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, vertex_normals=normals)
    
    transform_file.write('{:s},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(in_file, centroid[0], centroid[1], centroid[2], scale))
    transform_file.flush()

    mesh_sdf = MeshSDF(mesh, use_scans=False)
    points, sdf = mesh_sdf.get_sample_points()
    combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
    np.save(out_file, combined)


def process_obj_files():
    transform_file = open('data/birds_transforms.csv', 'a')
    ensure_directory(DIRECTORY_SDF)

    for filename in tqdm(files):
        out_file = os.path.join(DIRECTORY_SDF, filename.split('/')[-1].replace('.obj', '.npy'))
        if os.path.exists(out_file):
            continue
        process_obj_file(filename, out_file, transform_file)
        
    transform_file.close()

process_obj_files()