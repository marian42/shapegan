import os.path as osp
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


class ShapeNetSDF(torch.utils.data.Dataset):
    def __init__(self, root, num_points, transform=None):
        self.root = osp.expanduser(osp.normpath(root))
        self.num_points = num_points
        assert 0 < self.num_points <= 250000
        self.transform = transform

        self.files = glob.glob(osp.join(self.root, '*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        uniform, surface = torch.load(self.files[idx])

        # Sample a subset of points.
        sample = np.random.choice(uniform.size(0), self.num_points)
        sample = torch.from_numpy(sample)
        uniform, surface = uniform[sample], surface[sample]

        data = (uniform, surface)

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    root = '/data/sdf_chairs/chairs'
    dataset = ShapeNetSDF(root, num_points=4096)
    print(len(dataset))
    uniform, surface = dataset[0]
    print(uniform.size(), surface.size())

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

        plt.show()

    pos = uniform[:, :3]
    dist = uniform[:, 3]
    visualize(pos, dist)
