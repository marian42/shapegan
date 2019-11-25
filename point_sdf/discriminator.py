import torch
from torch.nn import Linear, Sequential, ReLU


class Encoder(torch.nn.Module):
    def __init__(self, out_channels, use_dist=False):
        super(Encoder, self).__init__()

        self.nn1 = Sequential(
            Linear(4 if use_dist else 3, 64),
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

    def forward(self, pos, dist=None):
        if dist is not None:
            dist = dist.unsqueeze(-1) if dist.dim() == 2 else dist
            x = torch.cat([pos, dist], dim=-1)
        else:
            x = pos
        x = self.nn1(x)
        x = x.max(dim=1)[0]
        x = self.nn2(x)
        return x
