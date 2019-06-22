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
VIEWPORT_SIZE = 800
VOXEL_COUNT = 128

def get_rotation_matrix(angle):
    rot = Rotation.from_euler('y', angle, degrees=True)
    rotmatrix = np.identity(4)
    rotmatrix[:3, :3] = rot.as_dcm()
    return rotmatrix

class DepthMap():
    def __init__(self, mesh, camera_angle):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh))

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear=0.001, zfar = 2)
        
        camera_pose = np.identity(4)
        camera_pose[2, 3] = CAMERA_DISTANCE
        camera_pose = np.matmul(get_rotation_matrix(camera_angle), camera_pose)

        scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
        color, depth = r.render(scene)

        #img = Image.fromarray(depth / np.max(depth) * 255, 'F')
        #img.show()

        self.depth = depth
        self.depth[depth == 0] = float('inf')
        self.world_to_clipping = np.matmul(camera.get_projection_matrix(), np.linalg.inv(camera_pose))

    def is_inside(self, point):
        sample = np.ones(4)
        sample[:3] = point
        sample = np.matmul(self.world_to_clipping, sample)
        point_depth = sample[2]
        uv = (sample / sample[3] * 0.5 + 0.5) * VIEWPORT_SIZE
        uv = uv.astype(int)
        uv = np.clip(uv, 0, VIEWPORT_SIZE - 1)
        map_depth = self.depth[uv[1], uv[0]]
        return map_depth < point_depth

    def check_inside(self, points):
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).transpose()
        points = np.matmul(self.world_to_clipping, points).transpose()
        sample_depth = np.array(points[:, 2])
        points /= points[:, 3][:, np.newaxis]
        half_viewport_size = 0.5 * VIEWPORT_SIZE
        uv = points * half_viewport_size + half_viewport_size
        uv = uv.astype(int)
        uv = np.clip(uv, 0, VIEWPORT_SIZE - 1)
        map_depth = self.depth[uv[:, 1], uv[:, 0]]
        
        return map_depth > sample_depth

def get_voxel_coordinates(centroid, size, voxel_count):
    points = np.meshgrid(
        np.linspace(centroid[0] - size, centroid[0] + size, voxel_count),
        np.linspace(centroid[1] - size, centroid[1] + size, voxel_count),
        np.linspace(centroid[2] - size, centroid[2] + size, voxel_count)
    )
    points = np.stack(points)
    return points.reshape(3, -1).transpose()




scene = pyrender.Scene()
mesh = trimesh.load(PATH)
mesh_in = pyrender.Mesh.from_trimesh(mesh)

scene.add(mesh_in)



depth_maps = [DepthMap(mesh, angle) for angle in [0 + 45, 90 + 45, 180 + 45, 270 + 45]]

bb = mesh.bounding_box
size = np.max(bb.extents) / 2
points = get_voxel_coordinates(bb.centroid, size, VOXEL_COUNT)


voxels = np.ones(points.shape[0])

for depth_map in depth_maps:
    voxels_inside = depth_map.check_inside(points)
    voxels[voxels_inside] = -1

voxels = voxels.reshape(VOXEL_COUNT, VOXEL_COUNT, VOXEL_COUNT)
#voxels = np.transpose(voxels, (1, 0, 2))

voxel_size = size / VOXEL_COUNT * 2

vertices, faces, normals, vals = skimage.measure.marching_cubes_lewiner(voxels * -1, level=0, spacing=(voxel_size, voxel_size, voxel_size))
reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

mesh_out = pyrender.Mesh.from_trimesh(reconstructed)

scene.add(mesh_out, pose = get_rotation_matrix(0))

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)