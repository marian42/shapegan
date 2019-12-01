import os
# Enable this when running on a computer without a screen
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
from tqdm import tqdm
import numpy as np
from sdf.mesh_to_sdf import MeshSDF, scale_to_unit_sphere, scale_to_unit_cube, BadMeshException
from util import ensure_directory
from multiprocessing import Pool
import traceback

DIRECTORY_MODELS = 'data/shapenet/03001627'
MODEL_EXTENSION = '.obj'
DIRECTORY_VOXELS = 'data/chairs/voxels_{:d}/'
DIRECTORY_CLOUDS = 'data/chairs/uniform/'
DIRECTORY_SURFACE = 'data/chairs/surface/'
DIRECTORY_BAD_MESHES = 'data/chairs/bad_meshes/'

VOXEL_RESOLUTIONS = [8, 16, 32, 64]
SDF_CLOUD_SAMPLE_SIZE = 64**3

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
    return os.path.join(DIRECTORY_CLOUDS, get_hash(model_filename) + '.npy')

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

def process_model_file(filename):
    try:
        if is_bad_mesh(filename):
            return
        
        mesh = trimesh.load(filename)
        mesh_unit_cube = scale_to_unit_cube(mesh)

        voxel_filenames = [get_voxel_filename(filename, resolution) for resolution in VOXEL_RESOLUTIONS]
        if not all(os.path.exists(f) for f in voxel_filenames):
            mesh_sdf = MeshSDF(mesh_unit_cube, object_size=3**0.5)
            try:
                for resolution in VOXEL_RESOLUTIONS:
                    voxels = mesh_sdf.get_voxel_sdf(voxel_resolution=resolution)
                    np.save(get_voxel_filename(filename, resolution), voxels)
            except BadMeshException:
                tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                mark_bad_mesh(filename)
                return
        
        if not os.path.exists(get_sdf_cloud_filename(filename)) or not os.path.exists(get_surface_filename(filename)):
            mesh_unit_sphere = scale_to_unit_sphere(mesh)
            mesh_sdf = MeshSDF(mesh_unit_sphere)
            try:
                points, sdf, near_surface_points, near_surface_sdf = mesh_sdf.get_sample_points(number_of_points=SDF_CLOUD_SAMPLE_SIZE)
                
                combined_uniform = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
                np.save(get_sdf_cloud_filename(filename), combined_uniform)

                combined_surface = np.concatenate((near_surface_points, near_surface_sdf[:, np.newaxis]), axis=1)
                np.save(get_surface_filename(filename), combined_surface)
            except BadMeshException:
                tqdm.write("Skipping bad mesh. ({:s})".format(get_hash(filename)))
                mark_bad_mesh(filename)
                return
            
    except:
        traceback.print_exc()


def process_model_files():
    for res in VOXEL_RESOLUTIONS:
        ensure_directory(DIRECTORY_VOXELS.format(res))
    ensure_directory(DIRECTORY_CLOUDS)
    ensure_directory(DIRECTORY_SURFACE)
    ensure_directory(DIRECTORY_BAD_MESHES)

    files = list(get_model_files())

    worker_count = os.cpu_count()
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