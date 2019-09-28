from model import *
import torch.nn.functional as F

from model.pointnet import *

from dataset import dataset as dataset

class PointcloudClassifier(SavableModule):
    def __init__(self, output_size=1):
        super(PointcloudClassifier, self).__init__(filename='pointcloud_classifier.to')

        self.set_abstraction_1 = SetAbstractionModule(ratio=0.5, radius=0.1, local_nn=create_MLP([3, 64, 64, 128]))
        self.set_abstraction_2 = SetAbstractionModule(ratio=0.25, radius=0.16, local_nn=create_MLP([128 + 3, 128, 128, 256]))
        self.set_abstraction_3 = GlobalSetAbstractionModule(create_MLP([256 + 3, 256, 512, 1024]))

        self.fully_connected_1 = Linear(1024, 512)
        self.fully_connected_2 = Linear(512, 256)
        self.fully_connected_3 = Linear(256, dataset.label_count)

        self.cuda()

    def forward(self, points):
        batch = torch.zeros(points.shape[0], device=points.device, dtype=torch.int64)
        
        x, points, batch = self.set_abstraction_1(None, points, batch)
        x, points, batch = self.set_abstraction_2(x, points, batch)
        x = self.set_abstraction_3(x, points, batch)
        
        x = F.relu(self.fully_connected_1(x))
        x = F.relu(self.fully_connected_2(x))
        x = self.fully_connected_3(x)
        x = torch.softmax(x.squeeze(), dim=0)
        return x.squeeze()