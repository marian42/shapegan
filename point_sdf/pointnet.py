import torch
from torch.nn import Linear, Sequential, ReLU
from torch_scatter import scatter_max


class PointNet(torch.nn.Module):
    def __init__(self, out_channels):
        super(PointNet, self).__init__()

        self.nn1 = Sequential(
            Linear(4, 64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Linear(256, 512),
        )

        self.nn2 = Sequential(
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, out_channels),
        )

    def forward(self, pos, dist, batch=None):
        dist = dist.unsqueeze(-1) if dist.size(-1) != 1 else dist

        x = torch.cat([pos, dist], dim=-1)

        x = self.nn1(x)

        if batch is None:
            x = x.max(dim=-2)[0]
        else:
            x = scatter_max(x, batch, dim=-2)[0]

        x = self.nn2(x)

        return x


if __name__ == '__main__':
    model = PointNet(out_channels=1)

    out = model(torch.randn(128, 3), torch.randn(128, ))
    assert out.size() == (1, )

    out = model(torch.randn(16, 128, 3), torch.randn(16, 128))
    assert out.size() == (16, 1)

    out = model(torch.randn(16, 128, 3), torch.randn(16, 128, 1))
    assert out.size() == (16, 1)
