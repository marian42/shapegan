import os.path as osp

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def visualize(pos, dist=None, perm=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if perm is not None:
        pos = pos[perm]
        dist = None if dist is None else dist[perm]

    xs = pos[:, 0]
    ys = pos[:, 1]
    zs = pos[:, 2]

    ax.scatter(xs, ys, zs, s=2, c='blue' if dist is None else dist)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()


class ShapeNetPointSDF(torch.utils.data.Dataset):
    def __init__(self, root, category, num_points, split='train',
                 transform=None):
        self.root = osp.expanduser(osp.join(osp.normpath(root), category))
        self.category = category
        self.num_points = num_points
        assert 0 < self.num_points <= 64**3  # TODO
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.transform = transform

        with open(osp.join(self.root, '{}.txt'.format(split)), 'r') as f:
            self.names = f.read().split('\n')
            if self.names[-1] == '':
                self.names = self.names[:-1]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        uniform = osp.join(self.root, 'uniform', '{}.npy'.format(name))
        uniform = torch.from_numpy(np.load(uniform))

        surface = osp.join(self.root, 'surface', '{}.npy'.format(name))
        surface = torch.from_numpy(np.load(surface))

        # Sample a subset of points.
        sample = np.random.choice(uniform.size(0), self.num_points)
        uniform, surface = uniform[sample], surface[sample]

        data = (uniform, surface)

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    root = 'SDF_GAN'
    dataset = ShapeNetPointSDF(root, category='chairs', num_points=16 * 1024)
    print(len(dataset))
    uniform, surface = dataset[0]
    print(uniform.size(), surface.size())

    pos = uniform[:, :3]
    dist = uniform[:, 3]
    perm = dist.abs() < 0.05
    visualize(pos, dist, perm)

    pos = surface[:, :3]
    dist = surface[:, 3]
    visualize(pos, dist)

    visualize(uniform[:3, :3])
    visualize(surface[:3, :3])
