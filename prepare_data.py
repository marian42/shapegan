import os
import trimesh
from tqdm import tqdm
import numpy as np
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere, BadMeshException
from util import ensure_directory
from multiprocessing import Pool
from rendering.math import get_rotation_matrix

DIRECTORY_MODELS = 'data/meshes/'
MODEL_EXTENSION = '.stl'
DIRECTORY_SDF = 'data/sdf/'

CREATE_VOXELS = True
VOXEL_RESOLUTION = 32

CREATE_SDF_CLOUDS = True
SDF_CLOUD_SAMPLE_SIZE = 200000

ROTATION = None # get_rotation_matrix(90, axis='x')

def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_npy_filename(model_filename, qualifier=''):
    return DIRECTORY_SDF + model_filename[len(DIRECTORY_MODELS):-len(MODEL_EXTENSION)] + qualifier + '.npy'

def get_voxel_filename(model_filename):
    return get_npy_filename(model_filename, '-voxels-{:d}'.format(VOXEL_RESOLUTION))

def get_sdf_cloud_filename(model_filename):
    return get_npy_filename(model_filename, '-sdf')

def get_bad_mesh_filename(model_filename):
    return DIRECTORY_SDF + model_filename[len(DIRECTORY_MODELS):-len(MODEL_EXTENSION)] + '.badmesh'

def mark_bad_mesh(model_filename):
    filename = get_bad_mesh_filename(model_filename)
    ensure_directory(os.path.dirname(filename))            
    open(filename, 'w').close()

def is_bad_mesh(model_filename):
    return os.path.exists(get_bad_mesh_filename(model_filename))

def process_model_file(filename):
    voxels_filename = get_voxel_filename(filename)
    sdf_cloud_filename = get_sdf_cloud_filename(filename)

    if is_bad_mesh(filename):
        return
    if not (CREATE_VOXELS and not os.path.isfile(voxels_filename) or CREATE_SDF_CLOUDS and not os.path.isfile(sdf_cloud_filename)):
        return
    
    mesh = trimesh.load(filename)
    if ROTATION is not None:
        mesh.apply_transform(ROTATION)
    mesh = scale_to_unit_sphere(mesh)

    surface_point_cloud = get_surface_point_cloud(mesh)
    if CREATE_SDF_CLOUDS:
        try:
            points, sdf = surface_point_cloud.sample_sdf_near_surface(number_of_points=SDF_CLOUD_SAMPLE_SIZE, sign_method='depth', min_size=0.015)
            combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
            ensure_directory(os.path.dirname(sdf_cloud_filename))
            np.save(sdf_cloud_filename, combined)
        except BadMeshException:
            tqdm.write("Skipping bad mesh. ({:s})".format(filename))
            mark_bad_mesh(filename)
            return

    if CREATE_VOXELS:
        try:
            voxels = surface_point_cloud.get_voxels(voxel_resolution=VOXEL_RESOLUTION, use_depth_buffer=True)
            ensure_directory(os.path.dirname(voxels_filename))
            np.save(voxels_filename, voxels)
        except BadMeshException:
            tqdm.write("Skipping bad mesh. ({:s})".format(filename))
            mark_bad_mesh(filename)
            return


def process_model_files():
    ensure_directory(DIRECTORY_SDF)
    files = list(get_model_files())
    
    worker_count = os.cpu_count() // 2
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(files))
    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_model_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()

def combine_pointcloud_files():
    import torch
    print("Combining SDF point clouds...")
    npy_files = sorted([get_sdf_cloud_filename(f) for f in get_model_files()])
    npy_files = [f for f in npy_files if os.path.exists(f)]
    
    N = len(npy_files)
    points = torch.zeros((N * SDF_CLOUD_SAMPLE_SIZE, 3))
    sdf = torch.zeros((N * SDF_CLOUD_SAMPLE_SIZE))
    position = 0

    for npy_filename in tqdm(npy_files):
        numpy_array = np.load(npy_filename)
        points[position * SDF_CLOUD_SAMPLE_SIZE:(position + 1) * SDF_CLOUD_SAMPLE_SIZE, :] = torch.tensor(numpy_array[:, :3])
        sdf[position * SDF_CLOUD_SAMPLE_SIZE:(position + 1) * SDF_CLOUD_SAMPLE_SIZE] = torch.tensor(numpy_array[:, 3])
        del numpy_array
        position += 1
    
    print("Saving combined SDF clouds...")
    torch.save(points, os.path.join('data', 'sdf_points.to'))
    torch.save(sdf, os.path.join('data', 'sdf_values.to'))

if __name__ == '__main__':
    process_model_files()
    if CREATE_SDF_CLOUDS:
        combine_pointcloud_files()