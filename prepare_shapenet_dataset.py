import os
# Enable this when running on a computer without a screen:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
from tqdm import tqdm
import numpy as np
from util import ensure_directory
from multiprocessing import Pool
import traceback
from mesh_to_sdf import BadMeshException, get_surface_point_cloud
from mesh_to_sdf.utils import scale_to_unit_cube, scale_to_unit_sphere

DATASET_NAME = 'chairs'
DIRECTORY_MODELS = 'data/shapenet/03001627'
MODEL_EXTENSION = '.obj'
DIRECTORY_VOXELS = 'data/{:s}/voxels_{{:d}}/'.format(DATASET_NAME)
DIRECTORY_UNIFORM = 'data/{:s}/uniform/'.format(DATASET_NAME)
DIRECTORY_SURFACE = 'data/{:s}/surface/'.format(DATASET_NAME)
DIRECTORY_BAD_MESHES = 'data/{:s}/bad_meshes/'.format(DATASET_NAME)

VOXEL_RESOLUTIONS = [8, 16, 32, 64]
POINT_CLOUD_SAMPLE_SIZE = 64**3

USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 1024

def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_hash(filename):
    return filename.split('/')[-3]

def get_voxel_filename(model_filename, resolution):
    return os.path.join(DIRECTORY_VOXELS.format(resolution), get_hash(model_filename) + '.npy')

def get_sdf_cloud_filename(model_filename):
    return os.path.join(DIRECTORY_UNIFORM, get_hash(model_filename) + '.npy')

def get_surface_filename(model_filename):
    return os.path.join(DIRECTORY_SURFACE, get_hash(model_filename) + '.npy')

def get_bad_mesh_filename(model_filename):
    return os.path.join(DIRECTORY_BAD_MESHES, get_hash(model_filename))

def mark_bad_mesh(model_filename):
    filename = get_bad_mesh_filename(model_filename)
    ensure_directory(os.path.dirname(filename))            
    open(filename, 'w').close()

def is_bad_mesh(model_filename):
    return os.path.exists(get_bad_mesh_filename(model_filename))

def get_uniform_and_surface_points(surface_point_cloud, number_of_points = 200000):
        unit_sphere_points = np.random.uniform(-1, 1, size=(number_of_points * 2, 3)).astype(np.float32)
        unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
        uniform_points = unit_sphere_points[:number_of_points, :]

        distances, indices = surface_point_cloud.kd_tree.query(uniform_points)
        uniform_sdf = distances.astype(np.float32).reshape(-1) * -1
        uniform_sdf[surface_point_cloud.is_outside(uniform_points)] *= -1

        surface_points = surface_point_cloud.points[indices[:, 0], :]
        near_surface_points = surface_points + np.random.normal(scale=0.0025, size=surface_points.shape).astype(np.float32)
        near_surface_sdf = surface_point_cloud.get_sdf(near_surface_points, use_depth_buffer=USE_DEPTH_BUFFER)
        
        model_size = np.count_nonzero(uniform_sdf < 0) / number_of_points
        if model_size < 0.01:
            raise BadMeshException()

        return uniform_points, uniform_sdf, near_surface_points, near_surface_sdf

def process_model_file(filename):
    try:
        if is_bad_mesh(filename):
            return
        
        mesh = trimesh.load(filename)

        voxel_filenames = [get_voxel_filename(filename, resolution) for resolution in VOXEL_RESOLUTIONS]
        if not all(os.path.exists(f) for f in voxel_filenames):
            mesh_unit_cube = scale_to_unit_cube(mesh)
            surface_point_cloud = get_surface_point_cloud(mesh_unit_cube, bounding_radius=3**0.5, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            try:
                for resolution in VOXEL_RESOLUTIONS:
                    voxels = surface_point_cloud.get_voxels(resolution, use_depth_buffer=USE_DEPTH_BUFFER, check_result=True)
                    np.save(get_voxel_filename(filename, resolution), voxels)
                    del voxels
                
            except BadMeshException:
                tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                mark_bad_mesh(filename)
                return
            del mesh_unit_cube, surface_point_cloud
        
        if not os.path.exists(get_sdf_cloud_filename(filename)) or not os.path.exists(get_surface_filename(filename)):
            mesh_unit_sphere = scale_to_unit_sphere(mesh)
            surface_point_cloud = get_surface_point_cloud(mesh_unit_sphere, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            try:
                uniform_points, uniform_sdf, near_surface_points, near_surface_sdf = get_uniform_and_surface_points(surface_point_cloud, number_of_points=POINT_CLOUD_SAMPLE_SIZE)
                
                combined_uniform = np.concatenate((uniform_points, uniform_sdf[:, np.newaxis]), axis=1)
                np.save(get_sdf_cloud_filename(filename), combined_uniform)

                combined_surface = np.concatenate((near_surface_points, near_surface_sdf[:, np.newaxis]), axis=1)
                np.save(get_surface_filename(filename), combined_surface)
            except BadMeshException:
                tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                mark_bad_mesh(filename)
                return
            del mesh_unit_sphere, surface_point_cloud
            
    except:
        traceback.print_exc()


def process_model_files():
    for res in VOXEL_RESOLUTIONS:
        ensure_directory(DIRECTORY_VOXELS.format(res))
    ensure_directory(DIRECTORY_UNIFORM)
    ensure_directory(DIRECTORY_SURFACE)
    ensure_directory(DIRECTORY_BAD_MESHES)

    files = list(get_model_files())

    worker_count = 4 #os.cpu_count()
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(files))
    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_model_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()

if __name__ == '__main__':
    process_model_files()