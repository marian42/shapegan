import trimesh
import pyrender
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import skimage
from tqdm import tqdm
from PIL import Image
import time

PATH = '/home/marian/shapenet/ShapeNetCore.v2/02942699/1cc93f96ad5e16a85d3f270c1c35f1c7/models/model_normalized.obj'

CAMERA_DISTANCE = 1.3
VOXEL_COUNT = 128
VIEWPORT_SIZE = VOXEL_COUNT * 4

def get_rotation_matrix(angle):
    rotation = Rotation.from_euler('y', angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

class DepthMap():
    def __init__(self, mesh, camera_angle):
        camera_pose = np.identity(4)
        camera_pose[2, 3] = CAMERA_DISTANCE
        camera_pose = np.matmul(get_rotation_matrix(camera_angle), camera_pose)

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear=0.001, zfar = 2)
        scene.add(camera, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
        _, depth = renderer.render(scene)

        #img = Image.fromarray(depth / np.max(depth) * 255, 'F')
        #img.show()

        depth[depth == 0] = float('inf')
        self.depth = depth
        
        half_viewport_size = 0.5 * VIEWPORT_SIZE
        clipping_to_viewport = np.array([
            [half_viewport_size, 0.0, 0.0, half_viewport_size],
            [0.0, half_viewport_size, 0.0, half_viewport_size],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0.0, 1.0]
        ])

        world_to_clipping = np.matmul(camera.get_projection_matrix(), np.linalg.inv(camera_pose))
        world_to_viewport = np.matmul(clipping_to_viewport , world_to_clipping)
        self.world_to_clipping_transpose = world_to_viewport.transpose()

    def check_inside(self, points):
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        points = np.matmul(points, self.world_to_clipping_transpose)
        sample_depth = np.array(points[:, 2])
        points = points[:, :2] / points[:, 3][:, np.newaxis]
        points = np.clip(points.astype(int), 0, VIEWPORT_SIZE - 1)
        map_depth = self.depth[points[:, 1], points[:, 0]]
        return map_depth > sample_depth

def get_voxel_coordinates(bounding_box, voxel_count):
    centroid = bounding_box.centroid
    size = np.max(bounding_box.extents) / 2
    points = np.meshgrid(
        np.linspace(centroid[0] - size, centroid[0] + size, voxel_count),
        np.linspace(centroid[1] - size, centroid[1] + size, voxel_count),
        np.linspace(centroid[2] - size, centroid[2] + size, voxel_count)
    )
    points = np.stack(points)
    return points.reshape(3, -1).transpose()

def voxelize_mesh(mesh, voxel_count):
    angles = [0, 90, 180, 270]
    depth_maps = [DepthMap(mesh, angle) for angle in angles]

    points = get_voxel_coordinates(mesh.bounding_box, voxel_count)

    voxels = np.ones(points.shape[0])
    for depth_map in depth_maps:
        voxels_inside = depth_map.check_inside(points)
        voxels[voxels_inside] = -1

    voxels = voxels.reshape(voxel_count, voxel_count, voxel_count)
    voxels = np.transpose(voxels, (1, 0, 2))
    return voxels


mesh = trimesh.load(PATH)
voxels = voxelize_mesh(mesh, VOXEL_COUNT)
voxel_size = np.max(mesh.bounding_box.extents) / VOXEL_COUNT

vertices, faces, normals, vals = skimage.measure.marching_cubes_lewiner(voxels * -1, level=0, spacing=(voxel_size, voxel_size, voxel_size))
reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

scene = pyrender.Scene()
mesh_in = pyrender.Mesh.from_trimesh(mesh)
mesh_out = pyrender.Mesh.from_trimesh(reconstructed)

scene.add(mesh_in)
scene.add(mesh_out)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)