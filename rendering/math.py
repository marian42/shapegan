
import numpy as np
from scipy.spatial.transform import Rotation

PROJECTION_MATRIX = np.array(
    [[ 1.73205081, 0,           0,           0,         ],
     [ 0,          1.73205081,  0,           0,         ],
     [ 0,          0,          -1.02020202, -0.2020202, ],
     [ 0,          0,          -1,           0,         ]], dtype=float)

def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

def get_camera_transform(camera_distance, rotation_y, rotation_x=0, project=False):
    camera_pose = np.identity(4)
    camera_pose[2, 3] = -camera_distance
    camera_pose = np.matmul(camera_pose, get_rotation_matrix(rotation_x, axis='x'))
    camera_pose = np.matmul(camera_pose, get_rotation_matrix(rotation_y, axis='y'))

    if project:
        camera_pose = np.matmul(PROJECTION_MATRIX, camera_pose)
    return camera_pose