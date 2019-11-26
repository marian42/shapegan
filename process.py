import time
import os.path as osp
import sys
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from tqdm import tqdm
from queue import Queue
from threading import Thread
import glob

from sdf.mesh_to_sdf import scale_to_unit_sphere, MeshSDF

NUMBER_OF_THREADS = 2


def visualize(pts, dist=None, perm=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if perm is not None:
        pts = pts[perm]
        dist = None if dist is None else dist[perm]

    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]

    ax.scatter(xs, ys, zs, s=2, c='blue' if dist is None else dist)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def process(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    mesh = scale_to_unit_sphere(mesh)
    mesh_sdf = MeshSDF(mesh)

    uniform_pos = 2 * torch.rand(250000, 3) - 1
    uniform_pos = uniform_pos.numpy()

    def get_sdf(self, query_points):
        distances, indices = self.kd_tree.query(query_points)
        distances = distances.astype(np.float32).reshape(-1) * -1
        distances[self.is_outside(query_points)] *= -1
        closest_points = self.points[indices].squeeze(1)
        return distances, closest_points

    uniform_dist, closest_point = get_sdf(mesh_sdf, uniform_pos)
    surface_pos = closest_point + np.random.normal(
        scale=0.025, size=(closest_point.shape[0], 3))
    surface_dist, _ = get_sdf(mesh_sdf, surface_pos)

    uniform_pos = torch.from_numpy(uniform_pos).to(torch.float)
    uniform_dist = torch.from_numpy(uniform_dist).to(torch.float)

    surface_pos = torch.from_numpy(surface_pos).to(torch.float)
    surface_dist = torch.from_numpy(surface_dist).to(torch.float)

    uniform = torch.cat([uniform_pos, uniform_dist.view(-1, 1)], dim=-1)
    surface = torch.cat([surface_pos, surface_dist.view(-1, 1)], dim=-1)

    return uniform, surface


# if __name__ == '__main__':
#     path = '/Users/rusty1s/Downloads/model.obj'
#     data = process(path)
#     torch.save(data, '/Users/rusty1s/Downloads/model.pt')
#     print("TEST SUCCEEDED")


def get_directories(root):
    return list(glob.glob(osp.join(root, '*')))


# def delete_existing_data(directories):
#     for directory in directories:
#         files = [
#             os.path.join(directory, BAD_MODEL_FILENAME),
#             os.path.join(directory, VOXEL_FILENAME),
#             os.path.join(directory, SDF_CLOUD_FILENAME)
#         ]

#         for file in files:
#             if os.path.isfile(file):
#                 print("Deleting: ", file)
#                 os.remove(file)

# if 'delete' in sys.argv:
#     directories = get_directorries()
#     delete_existing_data(directories)
#     exit()

target_dir = '/Users/rusty1s/Downloads/ShapeNetSDFChairs'

# print("Scanning for directories.")
# directories = Queue()
# for directory in get_directories('/Users/rusty1s/Downloads/03001627'):
#     directories.put(directory)
# print("Found {:d} models.".format(directories.qsize()))

for directory in tqdm(get_directories('/Users/rusty1s/Downloads/03001627')):
    name = directory.split('/')[-1]
    target = osp.join(target_dir, '{}.pt'.format(name))
    if not osp.exists(target):
        data = process(osp.join(directory, 'model.obj'))
        torch.save(
            data, osp.join(target_dir,
                           '{}.pt'.format(directory.split('/')[-1])))

# progress = tqdm(total=directories.qsize())

# def worker():
#     while not directories.empty():
#         directory = directories.get()
#         name = directory.split('/')[-1]
#         target = osp.join(target_dir, '{}.pt'.format(name))
#         if not osp.exists(target):
#             data = process(osp.join(directory, 'model.obj'))
#             torch.save(
#                 data,
#                 osp.join(target_dir, '{}.pt'.format(directory.split('/')[-1])))
#         progress.update()
#         directories.task_done()
#         time.sleep(0.001)

# for _ in range(NUMBER_OF_THREADS):
#     thread = Thread(target=worker)
#     thread.start()
# directories.join()
