import os
from mesh_to_sdf import *
from tqdm import tqdm
from queue import Queue
from threading import Thread
import time

PATH = 'data/shapenet/03001627/'

BAD_MODEL_FILENAME = "bad_model"
VOXEL_RESOLUTION = 32
MODEL_FILENAME = "model_normalized.obj"
VOXEL_FILENAME = "sdf-{:d}.npy".format(VOXEL_RESOLUTION)
SDF_CLOUD_FILENAME = "sdf-pointcloud.npy"
SURFACE_POINTCLOUD_FILENAME = "surface-pointcloud.npy"

def mark_bad_model(directory):
    open(os.path.join(directory, BAD_MODEL_FILENAME), 'w')

def process_directory(directory):
    if os.path.isfile(os.path.join(directory, BAD_MODEL_FILENAME)):
        return False
    
    model_filename = os.path.join(directory, MODEL_FILENAME)
    voxels_filename = os.path.join(directory, VOXEL_FILENAME)
    cloud_filename = os.path.join(directory, SDF_CLOUD_FILENAME)
    surface_cloud_filename = os.path.join(directory, SURFACE_POINTCLOUD_FILENAME)

    if os.path.isfile(voxels_filename) and os.path.isfile(cloud_filename) and os.path.isfile(surface_cloud_filename):
        return True

    mesh = trimesh.load(model_filename)
    mesh = scale_to_unit_sphere(mesh)

    mesh_sdf = MeshSDF(mesh)

    if not os.path.isfile(surface_cloud_filename):
        pointcloud = mesh_sdf.get_surface_points_and_normals()
        np.save(surface_cloud_filename, pointcloud)

    if not os.path.isfile(voxels_filename):
        try:
            voxels = mesh_sdf.get_voxel_sdf(voxel_count=VOXEL_RESOLUTION)
            np.save(voxels_filename, voxels)
        except BadMeshException:
            mark_bad_model(directory)
            return False

    if not os.path.isfile(cloud_filename):
        try:
            points, sdf = mesh_sdf.get_sample_points()
            combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
            np.save(cloud_filename, combined)
        except BadMeshException as exception:
            mark_bad_model(directory)
            return False
    return True
    
def get_directorries():
    for directory, _, files in os.walk(PATH):
        if MODEL_FILENAME in files:
            yield directory

def delete_existing_data(directories):
    for directory in directories:
        files = [
            os.path.join(directory, BAD_MODEL_FILENAME),
            os.path.join(directory, VOXEL_FILENAME),
            os.path.join(directory, SDF_CLOUD_FILENAME)
        ]
        
        for file in files:
            if os.path.isfile(file):
                os.remove(file)

print("Scanning for directories.")
directories = Queue()
for directory in get_directorries():
    directories.put(directory)
print("Found {:d} models.".format(directories.qsize()))

progress = tqdm(total=directories.qsize())

def worker():
    while not directories.empty():
        directory = directories.get()
        process_directory(directory)
        progress.update()
        time.sleep(0.001)

for _ in range(4):
    thread = Thread(target=worker)
    thread.start()
directories.join()
