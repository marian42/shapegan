import trimesh
import pyrender
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
from scipy.spatial.transform import Rotation
import time
from sklearn.neighbors import KDTree
import skimage
import logging
from threading import Lock
from tqdm import tqdm
import math
import random

CAMERA_DISTANCE = 2
VIEWPORT_SIZE = 512

logging.getLogger("trimesh").setLevel(9000)

render_lock = Lock()

class BadMeshException(Exception):
    pass

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("sdf/shaders/mesh.vert", "sdf/shaders/mesh.frag", defines=defines)
        return self.program

class Scan():
    def __init__(self, mesh, camera_pose):
        self.camera_pose = camera_pose
        self.camera_direction = np.matmul(camera_pose, np.array([0, 0, 1, 0]))[:3]
        self.camera_position = np.matmul(camera_pose, np.array([0, 0, 0, 1]))[:3]
        
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
        camera = pyrender.PerspectiveCamera(yfov=2 * math.asin(1.0 / CAMERA_DISTANCE), aspectRatio=1.0, znear = CAMERA_DISTANCE - 1.0, zfar = CAMERA_DISTANCE + 1.0)
        scene.add(camera, pose=camera_pose)
        self.projection_matrix = camera.get_projection_matrix()

        renderer = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
        renderer._renderer._program_cache = CustomShaderCache()

        color, depth = renderer.render(scene)
        self.depth = depth * 2 - 1
        indices = np.argwhere(self.depth != 1)

        points = np.ones((indices.shape[0], 4))
        points[:, [1, 0]] = indices.astype(float) / VIEWPORT_SIZE * 2 - 1
        points[:, 1] *= -1
        points[:, 2] = self.depth[indices[:, 0], indices[:, 1]]
        
        clipping_to_world = np.matmul(camera_pose, np.linalg.inv(camera.get_projection_matrix()))

        points = np.matmul(points, clipping_to_world.transpose())
        points /= points[:, 3][:, np.newaxis]
        self.points = points[:, :3]

        normals = color[indices[:, 0], indices[:, 1]] / 255 * 2 - 1
        camera_to_points = self.camera_position - self.points
        normal_orientation = np.einsum('ij,ij->i', camera_to_points, normals)
        normals[normal_orientation < 0] *= -1
        self.normals = normals

    def convert_world_space_to_viewport(self, points):
        half_viewport_size = 0.5 * VIEWPORT_SIZE
        clipping_to_viewport = np.array([
            [half_viewport_size, 0.0, 0.0, half_viewport_size],
            [0.0, -half_viewport_size, 0.0, half_viewport_size],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0.0, 1.0]
        ])

        world_to_clipping = np.matmul(self.projection_matrix, np.linalg.inv(self.camera_pose))
        world_to_viewport = np.matmul(clipping_to_viewport, world_to_clipping)
        
        world_space_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        viewport_points = np.matmul(world_space_points, world_to_viewport.transpose())
        viewport_points /= viewport_points[:, 3][:, np.newaxis]
        return viewport_points

    def is_visible(self, points):
        viewport_points = self.convert_world_space_to_viewport(points)
        pixels = viewport_points[:, :2].astype(int)
        pixels = np.clip(pixels, 0, VIEWPORT_SIZE - 1)
        return viewport_points[:, 2] < self.depth[pixels[:, 1], pixels[:, 0]]


def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

def get_camera_transform(rotation_y, rotation_x = 0):
    camera_pose = np.identity(4)
    camera_pose[2, 3] = CAMERA_DISTANCE
    camera_pose = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_pose)
    camera_pose = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_pose)
    return camera_pose

def create_scans(mesh, camera_count = 20):
    scans = []

    render_lock.acquire()
    scans.append(Scan(mesh, get_camera_transform(0, 90)))
    scans.append(Scan(mesh, get_camera_transform(0, -90)))

    for i in range(camera_count):
        camera_pose = get_camera_transform(360.0 * i / camera_count, random.uniform(-60, 60))
        scans.append(Scan(mesh, camera_pose))

    render_lock.release()
    return scans

def scale_to_unit_sphere(mesh):
    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    distances = np.linalg.norm(vertices, axis=1)
    size = np.max(distances)
    vertices /= size
    return trimesh.base.Trimesh(vertices=vertices, faces=mesh.faces)

class MeshSDF:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bounding_box = mesh.bounding_box
        self.scans = create_scans(mesh)

        self.points = np.concatenate([scan.points for scan in self.scans], axis=0)
        self.normals = np.concatenate([scan.normals for scan in self.scans], axis=0)

        self.kd_tree = KDTree(self.points)

    def get_sdf(self, query_points):
        distances, _ = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1) * -1
        end = time.time()
        distances[self.is_outside(query_points)] *= -1
        return distances

    def get_sdf_in_batches(self, points, batch_size=100000):
        result = np.zeros(points.shape[0])
        for i in tqdm(range(int(math.ceil(points.shape[0] / batch_size)))):
            start = i * batch_size
            end = min(result.shape[0], (i + 1) * batch_size)
            result[start:end] = self.get_sdf(points[start:end, :])
        return result

    def get_pyrender_pointcloud(self):
        return pyrender.Mesh.from_points(self.points, normals=self.normals)

    def get_voxel_sdf(self, voxel_count = 32):
        voxels = self.get_sdf(self.get_voxel_coordinates(voxel_count=voxel_count))
        voxels = voxels.reshape(voxel_count, voxel_count, voxel_count)
        return voxels

    def get_voxel_coordinates(self, voxel_count = 32):
        centroid = self.bounding_box.centroid
        size = np.max(self.bounding_box.extents) / 2
        points = np.meshgrid(
            np.linspace(centroid[0] - size, centroid[0] + size, voxel_count),
            np.linspace(centroid[1] - size, centroid[1] + size, voxel_count),
            np.linspace(centroid[2] - size, centroid[2] + size, voxel_count)
        )
        points = np.stack(points)
        return points.reshape(3, -1).transpose()

    def get_sample_points(self, number_of_points = 100000):
        ''' Use sample points as described in the DeepSDF paper '''
        points = []

        surface_sample_count = int(number_of_points * 0.4)
        surface_indices = np.random.choice(self.points.shape[0], surface_sample_count)
        surface_points = self.points[surface_indices, :]
        points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

        unit_sphere_sample_count = int(number_of_points * 0.2)
        unit_sphere_points = np.random.uniform(-1, 1, size=(unit_sphere_sample_count * 2, 3))
        unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
        points.append(unit_sphere_points[:unit_sphere_sample_count, :])
        points = np.concatenate(points).astype(np.float32)

        return points, self.get_sdf(points)

    def get_surface_points_and_normals(self, number_of_points = 50000):
        count = self.points.shape[0]
        if count < number_of_points:
            print("Warning: Less than {:d} points sampled.".format(number_of_points))
        indices = np.arange(count)
        np.random.shuffle(indices)
        indices = indices[:number_of_points]
        return np.concatenate([self.points[indices, :], self.normals[indices, :]], axis=1)
    
    def show_pointcloud(self):
        scene = pyrender.Scene()
        scene.add(self.get_pyrender_pointcloud())
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

    def show_reconstructed_mesh(self, voxel_size=64):
        scene = pyrender.Scene()
        voxels = self.get_voxel_sdf(voxel_count=voxel_size)
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(voxel_size, voxel_size, voxel_size))
        reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        reconstructed_pyrender = pyrender.Mesh.from_trimesh(reconstructed, smooth=False)
        scene.add(reconstructed_pyrender)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)
        
    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result

def show_mesh(mesh):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    PATH = '/home/marian/shapenet/ShapeNetCore.v2/03001627/64871dc28a21843ad504e40666187f4e/models/model_normalized.obj'
    mesh = trimesh.load(PATH)
    mesh = scale_to_unit_sphere(mesh)

    show_mesh(mesh)
    mesh_sdf = MeshSDF(mesh)

    #mesh_sdf.show_pointcloud()
    mesh_sdf.show_reconstructed_mesh()

    #points, sdf = mesh_sdf.get_sample_points()
    #combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
    #np.save("sdf_test.npy", combined)