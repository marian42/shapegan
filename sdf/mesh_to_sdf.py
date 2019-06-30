import trimesh
import pyrender
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
from scipy.spatial.transform import Rotation
import time
from scipy.spatial import KDTree
import skimage

PATH = '/home/marian/shapenet/ShapeNetCore.v2/02942699/1cc93f96ad5e16a85d3f270c1c35f1c7/models/model_normalized.obj'

CAMERA_DISTANCE = 1.2
VIEWPORT_SIZE = 256

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
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear = 0.5, zfar = 2)
        scene.add(camera, pose=camera_pose)
        self.projection_matrix = camera.get_projection_matrix()

        renderer = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
        renderer._renderer._program_cache = CustomShaderCache()

        self.color, self.depth = renderer.render(scene)
        indices = np.argwhere(self.depth != 1)

        points = np.ones((indices.shape[0], 4))
        points[:, [1, 0]] = indices.astype(float) / VIEWPORT_SIZE * 2 - 1
        points[:, 1] *= -1
        points[:, 2] = self.depth[indices[:, 0], indices[:, 1]] * 2 - 1
        
        clipping_to_world = np.matmul(camera_pose, np.linalg.inv(camera.get_projection_matrix()))

        points = np.matmul(points, clipping_to_world.transpose())
        points /= points[:, 3][:, np.newaxis]
        self.points = points[:, :3]

        normals = self.color[indices[:, 0], indices[:, 1]] / 255 * 2 - 1
        camera_to_points = self.camera_position - self.points
        normal_orientation = np.einsum('ij,ij->i', camera_to_points, normals)
        normals[normal_orientation < 0] *= -1
        self.normals = normals

    def remove_thin_geometry(self, other_scans):
        thin_points = np.zeros(self.points.shape[0], dtype=np.uint8)
        for scan in other_scans:
            if np.dot(self.camera_direction, scan.camera_direction) > 0.2:
                continue

            world_space_points = np.concatenate([self.points, np.ones((self.points.shape[0], 1))], axis=1)

            half_viewport_size = 0.5 * VIEWPORT_SIZE
            clipping_to_viewport = np.array([
                [half_viewport_size, 0.0, 0.0, half_viewport_size],
                [0.0, -half_viewport_size, 0.0, half_viewport_size],
                [0.0, 0.0, 1.0, 0.0],
                [0, 0, 0.0, 1.0]
            ])

            world_to_clipping = np.matmul(scan.projection_matrix, np.linalg.inv(scan.camera_pose))
            world_to_viewport = np.matmul(clipping_to_viewport, world_to_clipping)
            
            viewport_points = np.matmul(world_space_points, world_to_viewport.transpose())
            viewport_points /= viewport_points[:, 3][:, np.newaxis]

            pixels = viewport_points[:, :2].astype(int)
            current_depth = viewport_points[:, 2]
            scan_depth = scan.depth[pixels[:, 1], pixels[:, 0]] * 2 - 1
            
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

def create_scans(mesh, camera_count = 30):
    scans = []

    for i in range(camera_count):
        camera_pose = np.identity(4)
        camera_pose[2, 3] = CAMERA_DISTANCE
        camera_pose = np.matmul(get_rotation_matrix(360.0 * i / camera_count), camera_pose)
        camera_pose = np.matmul(get_rotation_matrix(45 if i % 2 == 0 else -45, axis='x'), camera_pose)

        scans.append(Scan(mesh, camera_pose))

    return scans

class MeshSDF:
    def __init__(self, mesh):
        self.bounding_box = mesh.bounding_box
        self.scans = create_scans(mesh)

        for scan in self.scans:
            scan.remove_thin_geometry(self.scans)

        self.points = np.concatenate([scan.points for scan in self.scans], axis=0)
        self.normals = np.concatenate([scan.normals for scan in self.scans], axis=0)

        self.points += self.normals * 0.0005
        self.kd_tree = KDTree(self.points, leafsize=100)

    def get_sdf(self, query_points):
        start = time.time()
        distances, indices = self.kd_tree.query(query_points, eps=0.001)
        end = time.time()
        print(end - start)
        closest_points = self.points[indices]
        direction_to_surface = query_points - closest_points
        inside = np.einsum('ij,ij->i', direction_to_surface, self.normals[indices]) < 0
        distances[inside] *= -1
        return distances

    def get_pyrender_pointcloud(self):
        return pyrender.Mesh.from_points(self.points, normals=self.normals)

    def get_voxel_sdf(self, voxel_count = 32):
        return self.get_sdf(self.get_voxel_coordinates(voxel_count=voxel_count))

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


mesh = trimesh.load(PATH)
mesh_sdf = MeshSDF(mesh)

scene = pyrender.Scene()

voxel_size = 32
voxels = mesh_sdf.get_voxel_sdf(voxel_count=voxel_size)
voxels = voxels.reshape(voxel_size, voxel_size, voxel_size)

vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(voxel_size, voxel_size, voxel_size))
reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

reconstructed_pyrender = pyrender.Mesh.from_trimesh(reconstructed, smooth=False)
scene.add(reconstructed_pyrender)

#scene.add(mesh_sdf.get_pyrender_pointcloud())
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)