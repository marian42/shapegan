import sys

import trimesh
from skimage.measure import marching_cubes_lewiner
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

sys.path.insert(0, '..')
from model.sdf_net import SDFNet, SDFVoxelizationHelperData  # noqa

sdf_voxelization_helper = dict()


class SDFGenerator(SDFNet):
    def __init__(self, latent_channels, hidden_channels, num_layers, norm=True,
                 dropout=0.0):
        super(SDFGenerator, self).__init__(device='cpu')

        self.layers1 = None
        self.layers2 = None

        assert num_layers % 2 == 0

        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = dropout

        in_channels = 3
        out_channels = hidden_channels

        self.lins = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(Linear(in_channels, out_channels))
            self.norms.append(LayerNorm(out_channels))

            if i == (num_layers // 2) - 1:
                in_channels = hidden_channels + 3
            else:
                in_channels = hidden_channels

            if i == num_layers - 2:
                out_channels = 1

        self.z_lin1 = Linear(latent_channels, hidden_channels)
        self.z_lin2 = Linear(latent_channels, hidden_channels)

    def forward(self, pos, z):
        # pos: [batch_size, num_points, 3]
        # z: [batch_size, latent_channels]

        pos = pos.unsqueeze(0) if pos.dim() == 2 else pos

        assert pos.dim() == 3
        assert pos.size(-1) == 3

        z = z.unsqueeze(0) if z.dim() == 1 else z
        assert z.dim() == 2
        assert z.size(-1) == self.latent_channels

        assert pos.size(0) == z.size(0)

        x = pos
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            if i == self.num_layers // 2:
                x = torch.cat([x, pos], dim=-1)

            x = lin(x)

            if i == 0:
                x = self.z_lin1(z).unsqueeze(1) + x

            if i == self.num_layers // 2:
                x = self.z_lin2(z).unsqueeze(1) + x

            if i < self.num_layers - 1:
                x = norm(x) if self.norm else x
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def get_voxels(self, z, resolution):
        if not resolution in sdf_voxelization_helper:
            helper_data = SDFVoxelizationHelperData(z.device, resolution, True)
            sdf_voxelization_helper[resolution] = helper_data
        else:
            helper_data = sdf_voxelization_helper[resolution]

        with torch.no_grad():
            distances = self(helper_data.sample_points, z).view(-1)

        voxels = torch.ones((resolution, resolution, resolution))
        voxels[helper_data.unit_sphere_mask] = distances.to('cpu')
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
        voxels = torch.from_numpy(voxels)

        return voxels

    def get_mesh(self, z, resolution=64):
        voxels = self.get_voxels(z, resolution)

        try:
            vertices, faces, normals, _ = marching_cubes_lewiner(
                voxels.numpy(), level=0,
                spacing=(2 / resolution, 2 / resolution, 2 / resolution))
            vertices -= 1
        except ValueError as e:
            print(e)
            return None

        return trimesh.Trimesh(vertices=vertices, faces=faces,
                               vertex_normals=normals)


if __name__ == '__main__':
    model = SDFGenerator(latent_channels=16, hidden_channels=32, num_layers=4)
    out = model(torch.randn(128, 3), torch.randn(16, ))
    assert out.size() == (1, 128, 1)

    out = model(torch.randn(8, 128, 3), torch.randn(8, 16))
    assert out.size() == (8, 128, 1)
