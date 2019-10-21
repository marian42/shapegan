from sdf.mesh_to_sdf import MeshSDF, scale_to_unit_sphere
from sdf.pyrender_wrapper import render_normal_and_depth_buffers
from sdf.scan import get_camera_transform
import pyrender
import trimesh
import numpy as np
import math
from matplotlib import pyplot as plt

MODEL_PATH = 'examples/chair.obj'

def show_image(image, grayscale=False):
    from matplotlib import pyplot as plt
    plt.axis('off')
    if grayscale:
        plt.gray()
    plt.tight_layout()
    plt.imshow(image, interpolation='nearest')
    plt.show()

mesh = trimesh.load(MODEL_PATH)
mesh = scale_to_unit_sphere(mesh)

scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
print("Now showing the input model as a triangle mesh.\nClose the window to continue.")
pyrender.Viewer(scene, use_raymond_lighting=True)

camera_pose = get_camera_transform(-140, -20)
camera = pyrender.PerspectiveCamera(yfov=2 * math.asin(1.0 / 2) * 0.97, aspectRatio=1.0, znear = 2 - 1.0, zfar = 2 + 1.0)
normal_buffer, depth_buffer = render_normal_and_depth_buffers(mesh, camera, camera_pose, 1080)
print("Now showing the normal map of a render of the mesh.\nClose the window to continue.")
show_image(normal_buffer)

print("Now showing the depth map of a render of the mesh.\nClose the window to continue.")
show_image(depth_buffer, grayscale=True)

mesh_sdf = MeshSDF(mesh)

print("Now showing the surface point cloud with normals.\nClose the window to continue.")
mesh_sdf.show_pointcloud()

print('Calculating...')
resolution = 800
slice_position = 0.35
points = np.meshgrid(
    np.linspace(slice_position, slice_position, 1),
    np.linspace(1, -1, resolution),
    np.linspace(-1, 1, resolution)
)

points = np.stack(points).reshape(3, -1).transpose()
sdf = mesh_sdf.get_sdf_in_batches(points).reshape(1, resolution, resolution)[0, :, :]
clip = 0.2
sdf = np.clip(sdf, -clip, clip) / clip

image = np.ones((resolution, resolution, 3))
image[:,:,:2][sdf > 0] = (1.0 - sdf[sdf > 0])[:, np.newaxis]
image[:,:,1:][sdf < 0] = (1.0 + sdf[sdf < 0])[:, np.newaxis]
image[np.abs(sdf) < 0.02, :] = 0
print("Now showing a slice through the SDF of the model.\nClose the window to continue.")
show_image(image)

print("Now showing a voxel volume reconstructed with Marching Cubes.\nClose the window to continue.")
mesh_sdf.show_reconstructed_mesh(voxel_size=64)

points, sdf = mesh_sdf.get_sample_points()
scene = pyrender.Scene()
colors = np.zeros((points.shape[0], 3))
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene.add(cloud)
print("Now showing a point cloud of non-uniformly sampled SDF data. Negative distances are red, positive distances are blue.")
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
