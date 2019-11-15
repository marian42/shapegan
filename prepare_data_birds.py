import os
import time
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np
from sdf.mesh_to_sdf import MeshSDF
from util import ensure_directory
from multiprocessing import Pool

DIRECTORY = 'data/birds/'
DIRECTORY_SDF = 'data/birds_sdf'

def get_obj_files():
    for directory, _, files in os.walk(DIRECTORY):
        for filename in files:
            if filename.endswith('.obj'):
                yield os.path.join(directory, filename)

rotation = np.matmul(
    Rotation.from_euler('y', 90, degrees=True).as_dcm(),
    Rotation.from_euler('x', -90, degrees=True).as_dcm())

def process_obj_file(filename):
    out_file = os.path.join(DIRECTORY_SDF, filename.split('/')[-1].replace('.obj', '.npy'))
    if os.path.isfile(out_file):
        return
    mesh = trimesh.load(filename)
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    vertices = np.matmul(rotation, mesh.vertices.transpose()).transpose()
    normals = np.matmul(rotation, mesh.vertex_normals.transpose()).transpose()
    centroid = np.matmul(rotation, mesh.bounding_box.centroid)
    vertices -= centroid[np.newaxis, :]
    scale = np.max(np.linalg.norm(vertices, axis=1)) * 1.05
    vertices /= scale
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, vertex_normals=normals)

    mesh_sdf = MeshSDF(mesh, use_scans=False)
    points, sdf = mesh_sdf.get_sample_points()
    combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
    np.save(out_file, combined)


def process_obj_files():
    ensure_directory(DIRECTORY_SDF)
    files = list(get_obj_files())
    
    worker_count = os.cpu_count()
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(files))
    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_obj_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()

process_obj_files()