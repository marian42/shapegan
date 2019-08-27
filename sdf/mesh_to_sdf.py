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
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear = 0.5, zfar = 4)
        scene.add(camera, pose=camera_pose)
        self.projection_matrix = camera.get_projection_matrix()

        renderer = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
        renderer._renderer._program_cache = CustomShaderCache()

        self.color, self.depth = renderer.render(scene)
        self.depth = self.depth * 2 - 1
        indices = np.argwhere(self.depth != 1)

        points = np.ones((indices.shape[0], 4))
        points[:, [1, 0]] = indices.astype(float) / VIEWPORT_SIZE * 2 - 1
        points[:, 1] *= -1
        points[:, 2] = self.depth[indices[:, 0], indices[:, 1]]
        
        clipping_to_world = np.matmul(camera_pose, np.linalg.inv(camera.get_projection_matrix()))

        points = np.matmul(points, clipping_to_world.transpose())
        points /= points[:, 3][:, np.newaxis]
        self.points = points[:, :3]

        normals = self.color[indices[:, 0], indices[:, 1]] / 255 * 2 - 1
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

    def remove_thin_geometry(self, other_scans):
        thin_points = np.zeros(self.points.shape[0], dtype=np.uint8)
        for scan in other_scans:
            if np.dot(self.camera_direction, scan.camera_direction) > 0.2:
                continue

            viewport_points = self.convert_world_space_to_viewport(self.points)            

            pixels = viewport_points[:, :2].astype(int)
            current_depth = viewport_points[:, 2]
            scan_depth = scan.depth[pixels[:, 1], pixels[:, 0]]
            
            # Points that are seen by two cameras pointing in opposing directions
            candidates = current_depth < scan_depth + 0.007
            
            camera_to_points = scan.camera_position - self.points 
            candidates = np.logical_and(candidates, np.einsum('ij,ij->i', camera_to_points, self.normals) < -0.5)            

            thin_points = np.logical_or(thin_points, candidates)
        self.points = self.points[~thin_points]
        self.normals = self.normals[~thin_points]


def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

def get_camera_transform(rotation_y, rotation_x = 0):
    camera_pose = np.identity(4)
    camera_pose[2, 3] = CAMERA_DISTANCE
    camera_pose = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_pose)
    camera_pose = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_pose)
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


def get_thin_triangles(mesh):
    EPSILON = 0.005

    scans = create_scans(mesh, camera_count=20)

    triangle_positions = mesh.triangles_center
    triangle_normals = mesh.triangles_cross
    
    triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, np.newaxis]
    
    points_a = triangle_positions + triangle_normals * EPSILON
    points_b = triangle_positions - triangle_normals * EPSILON

    outside_a = np.zeros(points_a.shape[0], dtype=np.uint8)
    outside_b = np.zeros(points_a.shape[0], dtype=np.uint8)

    for scan in scans:
        outside_a = np.logical_or(outside_a, scan.is_visible(points_a))
        outside_b = np.logical_or(outside_b, scan.is_visible(points_b))

    return outside_a & outside_b


def remove_thin_triangles(mesh, max_iter = 4):
    mesh.remove_degenerate_faces()
    
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        thin_triangles = get_thin_triangles(mesh)
        if np.count_nonzero(thin_triangles) == 0:
            return
        mesh.update_faces(~thin_triangles)


def scale_to_unit_sphere(mesh):
    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    distances = np.linalg.norm(vertices, axis=1)
    size = np.max(distances)
    vertices /= size
    return trimesh.base.Trimesh(vertices=vertices, faces=mesh.faces)


def count_thin_triangles(mesh):
    thin_triangles = get_thin_triangles(mesh)
    return np.count_nonzero(thin_triangles) / thin_triangles.shape[0]


class MeshSDF:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bounding_box = mesh.bounding_box
        self.scans = create_scans(mesh)

        self.points = np.concatenate([scan.points for scan in self.scans], axis=0)
        self.normals = np.concatenate([scan.normals for scan in self.scans], axis=0)

        self.kd_tree = KDTree(self.points)

    def get_sdf(self, query_points, sample_count = 30, perform_sanity_check=True):
        start = time.time()
        distances, indices = self.kd_tree.query(query_points, k=sample_count)
        distances = distances.astype(np.float32)
        end = time.time()
        #print('Time for KD-Tree query: {:.1f}s'.format(end - start))

        closest_points = self.points[indices]
        direction_to_surface = query_points[:, np.newaxis, :] - closest_points
        inside = np.einsum('ijk,ijk->ij', direction_to_surface, self.normals[indices]) < 0
        inside = np.sum(inside, axis=1) > sample_count * 0.5
        distances = distances[:, 0]
        distances[inside] *= -1
        if perform_sanity_check:
            self.sanity_check(query_points, distances)
        return distances

    def get_sdf_in_batches(self, points, sample_count=30, batch_size=100000):
        result = np.zeros(points.shape[0])
        for i in tqdm(range(int(math.ceil(points.shape[0] / batch_size)))):
            start = i * batch_size
            end = min(result.shape[0], (i + 1) * batch_size)
            result[start:end] = self.get_sdf(points[start:end, :], sample_count=sample_count)
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

    def get_points_and_normals(self, number_of_points = 50000):
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

    def sanity_check(self, points, sdf, bad_points_threshold=0.005):
        high_distance = np.abs(sdf) > 0.02
        points = points[high_distance]
        sdf = sdf[high_distance]

        outside_visibility = self.is_outside(points)
        outside_sdf = sdf > 0
        bad_points = np.count_nonzero(outside_visibility != outside_sdf) / sdf.shape[0]
        
        #print('SDF sanity check with {:d} samples: {:.1f}% inconsistent signs.'.format(sdf.shape[0], bad_points * 100))
        if bad_points > bad_points_threshold:
            raise BadMeshException("{:.1f}% of the SDF values have the wrong sign. Make sure the supplied mesh is watertight.".format(bad_points * 100))


def show_mesh(mesh):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    PATH = '/home/marian/shapenet/ShapeNetCore.v2/03001627/64871dc28a21843ad504e40666187f4e/models/model_normalized.obj'
    mesh = trimesh.load(PATH)
    mesh = scale_to_unit_sphere(mesh)

    print("{:.2f}% thin triangles".format(count_thin_triangles(mesh) * 100))

    remove_thin_triangles(mesh)
    show_mesh(mesh)
    mesh_sdf = MeshSDF(mesh)

    #mesh_sdf.show_pointcloud()
    mesh_sdf.show_reconstructed_mesh()

    #points, sdf = mesh_sdf.get_sample_points()
    #combined = np.concatenate((points, sdf[:, np.newaxis]), axis=1)
    #np.save("sdf_test.npy", combined)