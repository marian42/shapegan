from model import *

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
        points = points.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, points, batch


def create_MLP(layer_sizes):
    layers = []
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            continue
        layers.append(Linear(layer_sizes[i - 1], layer_size))
        layers.append(ReLU())
        layers.append(BatchNorm1d(layer_size))

    return Sequential(*layers)


class SDFDiscriminator(SavableModule):
    def __init__(self, output_size=1):
        super(SDFDiscriminator, self).__init__(filename='sdf_discriminator.to')

        self.set_abstraction_1 = SetAbstractionModule(ratio=0.5, radius=0.03, local_nn=create_MLP([1 + 3, 64, 64, 128]))
        self.set_abstraction_2 = SetAbstractionModule(ratio=0.25, radius=0.2, local_nn=create_MLP([128 + 3, 128, 128, 256]))
        self.set_abstraction_3 = GlobalSetAbstractionModule(create_MLP([256 + 3, 256, 512, 1024]))

        self.fully_connected_1 = Linear(1024, 512)
        self.fully_connected_2 = Linear(512, 256)
        self.fully_connected_3 = Linear(256, output_size)

    def forward(self, points, features):
        batch = torch.zeros(points.shape[0], device=points.device, dtype=torch.int64)
        
        x = features.unsqueeze(dim=1)
        x, points, batch = self.set_abstraction_1(x, points, batch)
        x, points, batch = self.set_abstraction_2(x, points, batch)
        x, _, _ = self.set_abstraction_3(x, points, batch)
        
        x = F.relu(self.fully_connected_1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fully_connected_2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fully_connected_3(x)
        x = torch.sigmoid(x)
        return x.squeeze()
