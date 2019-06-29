import trimesh
import pyrender
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
from scipy.spatial.transform import Rotation
import time

PATH = '/home/marian/shapenet/ShapeNetCore.v2/02942699/1cc93f96ad5e16a85d3f270c1c35f1c7/models/model_normalized.obj'

CAMERA_DISTANCE = 1.2
VIEWPORT_SIZE = 800

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("sdf/shaders/mesh.vert", "sdf/shaders/mesh.frag", defines=defines)
        return self.program

def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

def render_to_pointcloud(mesh, camera_pose):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear = 0.5, zfar = 2)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(VIEWPORT_SIZE, VIEWPORT_SIZE)
    renderer._renderer._program_cache = CustomShaderCache()

    color, depth = renderer.render(scene)
    indices = np.argwhere(depth != 1)

    normals = color[indices[:, 0], indices[:, 1]] / 255 * 2 - 1
    camera_direction = np.matmul(camera_pose, np.array([0, 0, 1, 0]))[:3]
    normal_orientation = np.dot(normals, camera_direction)
    normals[normal_orientation < 0] *= -1

    points = np.ones((indices.shape[0], 4))
    points[:, [1, 0]] = indices.astype(float) / VIEWPORT_SIZE * 2 - 1
    points[:, 1] *= -1
    points[:, 2] = depth[indices[:, 0], indices[:, 1]] * 2 - 1
    
    clipping_to_world = np.matmul(camera_pose, np.linalg.inv(camera.get_projection_matrix()))

    points = np.matmul(points, clipping_to_world.transpose())
    points /= points[:, 3][:, np.newaxis]

    return points[:, :3], normals

def mesh_to_pointcloud(mesh, camera_count = 10):
    points_list = []
    normals_list = []

    for i in range(camera_count):
        camera_pose = np.identity(4)
        camera_pose[2, 3] = CAMERA_DISTANCE
        camera_pose = np.matmul(get_rotation_matrix(360.0 * i / camera_count), camera_pose)
        camera_pose = np.matmul(get_rotation_matrix(45 if i % 2 == 0 else -45, axis='x'), camera_pose)

        points, normals = render_to_pointcloud(mesh, camera_pose)
        points_list.append(points)
        normals_list.append(normals)

    points = np.concatenate(points_list, axis=0)
    normals = np.concatenate(normals_list, axis=0)

    return points, normals

mesh = trimesh.load(PATH)
points, normals = mesh_to_pointcloud(mesh)
pyrender_mesh = pyrender.Mesh.from_points(points, normals=normals)
scene = pyrender.Scene()
#scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
scene.add(pyrender_mesh)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True)