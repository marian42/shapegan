import torch
from torch.utils.data import Dataset
import os
import numpy as np


class VoxelDataset(Dataset):
    def __init__(self, files, clamp=0.1, rescale_sdf=True):
        self.files = files
        self.clamp = clamp
        self.rescale_sdf = rescale_sdf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = np.load(self.files[index])
        result = torch.from_numpy(array)
        if self.clamp is not None:
            result.clamp_(-self.clamp, self.clamp)
            if self.rescale_sdf:
                result /= self.clamp
        return result

    @staticmethod
    def glob(pattern):
        import glob
        files = glob.glob(pattern, recursive=True)
        if len(files) == 0:
            raise Exception(
                'No files found for glob pattern {:s}.'.format(pattern))
        return VoxelDataset(sorted(files))

    @staticmethod
    def from_split(pattern, split_file_name):
        split_file = open(split_file_name, 'r')
        ids = split_file.readlines()
        files = [pattern.format(id.strip()) for id in ids]
        files = [file for file in files if os.path.exists(file)]
        return VoxelDataset(files)

    def show(self):
        from rendering import MeshRenderer
        import time
        from tqdm import tqdm

        viewer = MeshRenderer()
        for item in tqdm(self):
            viewer.set_voxels(item.numpy())
            time.sleep(0.5)


class PointDataset(Dataset):
    def __init__(self, root, filenames, num_points=1024, transform=None):
        self.root = os.path.expanduser(os.path.join(os.path.normpath(root)))
        self.filenames = filenames
        self.num_points = num_points
        assert 0 < self.num_points <= 64**3
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        uniform = os.path.join(self.root, 'uniform', f'{name}.npy')
        uniform = torch.from_numpy(np.load(uniform))

        surface = os.path.join(self.root, 'surface', f'{name}.npy')
        surface = torch.from_numpy(np.load(surface))

        # Sample a subset of points.
        sample = np.random.choice(uniform.size(0), self.num_points)
        uniform, surface = uniform[sample], surface[sample]

        data = (uniform, surface)

        if self.transform is not None:
            data = self.transform(data)

        return data

    @staticmethod
    def from_split(root, split, num_points=1024, transform=None):
        with open(os.path.join(root, f'{split}.txt'), 'r') as f:
            filenames = f.read().split('\n')
            if filenames[-1] == '':
                filenames = filenames[:-1]
        return PointDataset(root, filenames, num_points, transform)


if __name__ == '__main__':
    # dataset = VoxelDataset.glob('data/chairs/voxels_64/')
    dataset = VoxelDataset.from_split(
        'data/chairs/voxels_{:d}/{{:s}}.npy'.format(64),
        'data/chairs/train.txt')
    dataset.show()

    dataset = PointDataset.from_split('data/chairs', 'train')
