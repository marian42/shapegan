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

def get_npy_filename(obj_filename):
    return os.path.join(DIRECTORY_SDF, obj_filename.split('/')[-1].replace('.obj', '.npy'))

def process_obj_file(filename):
    out_file = get_npy_filename(filename)
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

def combine_files():
    import torch
    npy_files = sorted([get_npy_filename(f) for f in get_obj_files()])
    
    N = len(npy_files)
    POINTCLOUD_SIZE = 200000
    points = torch.zeros((N * POINTCLOUD_SIZE, 3))
    sdf = torch.zeros((N * POINTCLOUD_SIZE))
    position = 0

    for npy_filename in tqdm(npy_files):
        numpy_array = np.load(npy_filename)
        points[position * POINTCLOUD_SIZE:(position + 1) * POINTCLOUD_SIZE, :] = torch.tensor(numpy_array[:, :3])
        sdf[position * POINTCLOUD_SIZE:(position + 1) * POINTCLOUD_SIZE] = torch.tensor(numpy_array[:, 3])
        del numpy_array
        position += 1
    
    torch.save(points, os.path.join('data', 'sdf_points.to'))
    torch.save(sdf, os.path.join('data', 'sdf_values.to'))

process_obj_files()
combine_files()