import torch
from torch.nn import Linear, Sequential, ReLU
from torch_scatter import scatter_max


class Encoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()

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
        x = torch.cat([pos, dist], dim=-1)

        x = self.nn1(x)

        if batch is None:
            x = x.max(dim=1)[0]
        else:
            x = scatter_max(x, batch, dim=0)[0]
        x = self.nn2(x)
        return x
