import os
# Enable this when running on a computer without a screen:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
from tqdm import tqdm
import numpy as np
from util import ensure_directory
from multiprocessing import Pool
import traceback
from mesh_to_sdf import get_surface_point_cloud,scale_to_unit_cube, scale_to_unit_sphere, BadMeshException

DATASET_NAME = 'chairs'
DIRECTORY_MODELS = 'data/shapenet/03001627'
MODEL_EXTENSION = '.obj'
DIRECTORY_VOXELS = 'data/{:s}/voxels_{{:d}}/'.format(DATASET_NAME)
DIRECTORY_UNIFORM = 'data/{:s}/uniform/'.format(DATASET_NAME)
DIRECTORY_SURFACE = 'data/{:s}/surface/'.format(DATASET_NAME)
DIRECTORY_SDF_CLOUD = 'data/{:s}/cloud/'.format(DATASET_NAME)
DIRECTORY_BAD_MESHES = 'data/{:s}/bad_meshes/'.format(DATASET_NAME)

# Voxel resolutions to create.
# Set to [] if no voxels are needed.
# Set to [32] for for all models except for the progressively growing DeepSDF/Voxel GAN
VOXEL_RESOLUTIONS = [8, 16, 32, 64]

CREATE_SDF_CLOUDS = False # For DeepSDF autodecoder, contains uniformly and non-uniformly sampled points as proposed in the DeepSDF paper
CREATE_UNIFORM_AND_SURFACE = True # Uniformly sampled points for the Pointnet-based GAN and surface point clouds for the pointnet-based GAN with refinement

SDF_POINT_CLOUD_SIZE = 200000 # For DeepSDF point clouds (CREATE_SDF_CLOUDS)
POINT_CLOUD_SAMPLE_SIZE = 64**3 # For uniform and surface points (CREATE_UNIFORM_AND_SURFACE)

# Options for virtual scans used to generate SDFs
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

def get_uniform_filename(model_filename):
    return os.path.join(DIRECTORY_UNIFORM, get_hash(model_filename) + '.npy')

def get_surface_filename(model_filename):
    return os.path.join(DIRECTORY_SURFACE, get_hash(model_filename) + '.npy')

def get_sdf_cloud_filename(model_filename):
    return os.path.join(DIRECTORY_SDF_CLOUD, get_hash(model_filename) + '.npy')

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
        

        create_uniform_and_surface = CREATE_UNIFORM_AND_SURFACE and (not os.path.exists(get_uniform_filename(filename)) or not os.path.exists(get_surface_filename(filename)))
        create_sdf_clouds = CREATE_SDF_CLOUDS and not os.path.exists(get_sdf_cloud_filename(filename))

        if create_uniform_and_surface or create_sdf_clouds:
            mesh_unit_sphere = scale_to_unit_sphere(mesh)
            surface_point_cloud = get_surface_point_cloud(mesh_unit_sphere, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
            try:
                if create_uniform_and_surface:
                    uniform_points, uniform_sdf, near_surface_points, near_surface_sdf = get_uniform_and_surface_points(surface_point_cloud, number_of_points=POINT_CLOUD_SAMPLE_SIZE)
                    
                    combined_uniform = np.concatenate((uniform_points, uniform_sdf[:, np.newaxis]), axis=1)
                    np.save(get_uniform_filename(filename), combined_uniform)

                    combined_surface = np.concatenate((near_surface_points, near_surface_sdf[:, np.newaxis]), axis=1)
                    np.save(get_surface_filename(filename), combined_surface)

                if create_sdf_clouds:
                    sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth' if USE_DEPTH_BUFFER else 'normal', number_of_points=SDF_POINT_CLOUD_SIZE, min_size=0.015)
                    combined = np.concatenate((sdf_points, sdf_values[:, np.newaxis]), axis=1)
                    np.save(get_sdf_cloud_filename(filename), combined)
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
    if CREATE_UNIFORM_AND_SURFACE:
        ensure_directory(DIRECTORY_UNIFORM)
        ensure_directory(DIRECTORY_SURFACE)
    if CREATE_SDF_CLOUDS:
        ensure_directory(DIRECTORY_SDF_CLOUD)
    ensure_directory(DIRECTORY_BAD_MESHES)

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

def combine_sdf_clouds():
    import torch
    print("Combining SDF point clouds...")

    files = list(sorted(get_model_files()))
    files = [f for f in files if os.path.exists(get_sdf_cloud_filename(f))]
    
    N = len(files)
    points = torch.zeros((N * SDF_POINT_CLOUD_SIZE, 3))
    sdf = torch.zeros((N * SDF_POINT_CLOUD_SIZE))
    position = 0

    for file_name in tqdm(files):
        numpy_array = np.load(get_sdf_cloud_filename(file_name))
        points[position * SDF_POINT_CLOUD_SIZE:(position + 1) * SDF_POINT_CLOUD_SIZE, :] = torch.tensor(numpy_array[:, :3])
        sdf[position * SDF_POINT_CLOUD_SIZE:(position + 1) * SDF_POINT_CLOUD_SIZE] = torch.tensor(numpy_array[:, 3])
        del numpy_array
        position += 1
    
    print("Saving combined SDF clouds...")
    torch.save(points, os.path.join('data', 'sdf_points.to'))
    torch.save(sdf, os.path.join('data', 'sdf_values.to'))

if __name__ == '__main__':
    process_model_files()
    if CREATE_SDF_CLOUDS:
        combine_sdf_clouds()