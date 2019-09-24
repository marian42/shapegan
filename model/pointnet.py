from model import *

import torch
from torch_geometric.nn import PointConv, fps as sample_farthest_points, radius as find_neighbors_in_range, global_max_pool

class SetAbstractionModule(torch.nn.Module):
    def __init__(self, ratio, radius, local_nn):
        super(SetAbstractionModule, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = PointConv(local_nn)

    def forward(self, x, pos, batch):
        indices = sample_farthest_points(pos, batch, ratio=self.ratio)
        row, col = find_neighbors_in_range(pos, pos[indices], r=self.radius, batch_x=batch, batch_y=batch[indices], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[indices]), edge_index)
        pos, batch = pos[indices], batch[indices]
        return x, pos, batch


class GlobalSetAbstractionModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSetAbstractionModule, self).__init__()
        self.nn = nn

    def forward(self, features, points, batch):
        x = self.nn(torch.cat([features, points], dim=1))
        x = global_max_pool(x, batch)
        return x


def create_MLP(layer_sizes):
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            continue
        layers.append(Linear(layer_sizes[i - 1], layer_size))
        layers.append(ReLU())

    return Sequential(*layers)