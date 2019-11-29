import sys
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

sys.path.insert(0, '..')
from model.sdf_net import SDFNet  # noqa


class SDFGenerator(SDFNet):
    def __init__(self, latent_channels, hidden_channels, num_layers, norm=True,
                 dropout=0.0):
        super(SDFGenerator, self).__init__()

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

        self.reset_parameters()

    def reset_parameters(self):
        for lin, norm in zip(self.lins, self.norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.z_lin1.reset_parameters()
        self.z_lin2.reset_parameters()

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


if __name__ == '__main__':
    model = SDFGenerator(latent_channels=16, hidden_channels=32, num_layers=4)
    out = model(torch.randn(128, 3), torch.randn(16, ))
    print(out.size())
