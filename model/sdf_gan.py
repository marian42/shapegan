from model import *
import torch.nn.functional as F

from model.pointnet import *

class SDFDiscriminator(SavableModule):
    def __init__(self, output_size=1):
        super(SDFDiscriminator, self).__init__(filename='sdf_discriminator.to')

        self.set_abstraction_1 = SetAbstractionModule(ratio=0.5, radius=0.22, local_nn=create_MLP([1 + 3, 64, 64, 128]))
        self.set_abstraction_2 = SetAbstractionModule(ratio=0.25, radius=0.3, local_nn=create_MLP([128 + 3, 128, 128, 256]))
        self.set_abstraction_3 = GlobalSetAbstractionModule(create_MLP([256 + 3, 256, 512, 1024]))

        self.fully_connected_1 = Linear(1024, 512)
        self.fully_connected_2 = Linear(512, 256)
        self.fully_connected_3 = Linear(256, output_size)

    def forward(self, points, features):
        batch = torch.zeros(points.shape[0], device=points.device, dtype=torch.int64)
        
        x = features.unsqueeze(dim=1)
        x, points, batch = self.set_abstraction_1(x, points, batch)
        x, points, batch = self.set_abstraction_2(x, points, batch)
        x = self.set_abstraction_3(x, points, batch)
        
        x = F.relu(self.fully_connected_1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fully_connected_2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fully_connected_3(x)
        x = torch.sigmoid(x)
        return x.squeeze()
