import trimesh
import numpy as np
from tqdm import tqdm
import skimage
import os
from queue import Queue
from threading import Thread
import logging
import random


PATH = 'data/shapenet/03001627/' # chairs

def get_voxel_coordinates(centroid, size, voxel_count):
    points = np.meshgrid(
        np.linspace(centroid[0] - size, centroid[0] + size, voxel_count),
        np.linspace(centroid[1] - size, centroid[1] + size, voxel_count),
        np.linspace(centroid[2] - size, centroid[2] + size, voxel_count)
    )
    points = np.stack(points)
    return points.reshape(3, -1).transpose()

def get_sdf(mesh, points, chunk_size = 2000):
    result = []
    chunks = points.shape[0] // chunk_size + 1
    for i in range(chunks):
        sdf = trimesh.proximity.signed_distance(mesh, points[i * chunk_size : min((i + 1) * chunk_size, points.shape[0]), :])
        result.append(sdf)
    return np.concatenate(result)


VOXEL_COUNT = 32
THREAD_COUNT = 3

def process_model(directory):
    filename_in = os.path.join(directory, "model_normalized.obj")
    filename_voxels = os.path.join(directory, "sdf-{0}.npy".format(VOXEL_COUNT))
    filename_reconstructed = os.path.join(directory, "sdf-{0}-reconstructed.obj".format(VOXEL_COUNT))

    mesh = trimesh.load(filename_in, )

    bb = mesh.bounding_box
    size = np.max(bb.extents) / 2

    points = get_voxel_coordinates(bb.centroid, size, VOXEL_COUNT)
    sdf = get_sdf(mesh, points)

    voxels = sdf.reshape(VOXEL_COUNT, VOXEL_COUNT, VOXEL_COUNT)
    voxels = np.transpose(voxels, (1, 0, 2))
    np.save(filename_voxels, voxels)

    vertices, faces, normals, vals = skimage.measure.marching_cubes_lewiner(voxels * -1, level=0)
    reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    reconstructed.export(filename_reconstructed)

trimesh_logger = logging.getLogger('trimesh')
trimesh_logger.setLevel(9000)

directories = []
for directory, _, files in os.walk(PATH):
    for file in files:
        if not file.endswith("model_normalized.obj"):
            continue
        
        filename_voxels = os.path.join(directory, "sdf-{0}.npy".format(VOXEL_COUNT))
        filename_reconstructed = os.path.join(directory, "sdf-{0}-reconstructed.obj".format(VOXEL_COUNT))

        if os.path.isfile(filename_voxels) and os.path.isfile(filename_reconstructed):
            continue
    
        directories.append(directory)

random.shuffle(directories)
queue = Queue()
for i in directories:
    queue.put(i)

def worker():
    while queue.not_empty:
        directory = queue.get()
        try:
            process_model(directory)
        except e:
            print("Failed to process item.")
        print(queue.qsize(), " items left.")

for _ in range(3):
    thread = Thread(target = worker)
    thread.start()

queue.join()