import os
from mesh_to_sdf import *
from tqdm import tqdm

PATH = '/home/marian/shapenet/ShapeNetCore.v2/03001627/'

BAD_MODEL_FILENAME = "bad_model"
VOXEL_RESOLUTION = 32
MODEL_FILENAME = "model_normalized.obj"
VOXEL_FILENAME = "sdf-{:d}.npy".format(VOXEL_RESOLUTION)
CLOUD_FILENAME = "sdf-pointcloud.npy"

def mark_bad_model(directory):
    open(os.path.join(directory, BAD_MODEL_FILENAME), 'w')

def process_directory(directory):
    if os.path.isfile(os.path.join(directory, BAD_MODEL_FILENAME)):
        return False
    
    model_filename = os.path.join(directory, MODEL_FILENAME)
    voxels_filename = os.path.join(directory, VOXEL_FILENAME)
    cloud_filename = os.path.join(directory, CLOUD_FILENAME)

    if os.path.isfile(voxels_filename) and os.path.isfile(cloud_filename):
        return True

    mesh = trimesh.load(model_filename)
    mesh = scale_to_unit_sphere(mesh)

    mesh_sdf = MeshSDF(mesh)

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
            os.path.join(directory, CLOUD_FILENAME)
        ]
        
        for file in files:
            if os.path.isfile(file):
                os.remove(file)

print("Scanning for directories.")
directories = list(get_directorries())
print("Found {:d} models.".format(len(directories)))

good_meshes = 0
bad_meshes = 0

for directory in tqdm(directories):
    success = process_directory(directory)
    if success:
        good_meshes += 1
    else:
        bad_meshes += 1
    if good_meshes % 10 == 0:
        print("Success rate: {:d} / {:d}".format(good_meshes, good_meshes + bad_meshes))

print("Done. Success rate: {:d} / {:d}".format(good_meshes, len(directories)))