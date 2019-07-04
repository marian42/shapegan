import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import numpy as np

from itertools import count
import time
import random

class SDFNet(nn.Module):
    def __init__(self):
        super(SDFNet, self).__init__()        

        self.layers = nn.Sequential(
            nn.Linear(in_features = 3, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = 256, out_features = 1)
        )

        self.cuda()

    def forward(self, x):
        return self.layers.forward(x).squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load("sdf_test.npy")
points = torch.tensor(data[:, :3], device=device, dtype=torch.float)
sdf = torch.tensor(data[:, 3], device=device, dtype=torch.float)
sdf = torch.clamp(sdf, -0.1, 0.1)

SIZE = sdf.shape[0]
BATCH_SIZE = 128

sdf_net = SDFNet()

optimizer = optim.Adam(sdf_net.parameters(), lr=0.00001)
criterion = nn.MSELoss()

def create_batches():
    batch_count = int(SIZE / BATCH_SIZE)
    indices = list(range(SIZE))
    random.shuffle(indices)
    for i in range(batch_count - 1):
        yield indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield indices[(batch_count - 1) * BATCH_SIZE:]

def train():
    with torch.no_grad():
        test_output = sdf_net.forward(points[:10000])
    loss = criterion(test_output, sdf[:10000]).item()
    print("Loss: {:.5f}".format(loss))

    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches():
            indices = torch.tensor(batch, device = device)
            sample = points[indices, :]
            labels = sdf[indices]

            sdf_net.zero_grad()
            output = sdf_net.forward(sample)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            batch_index += 1
        
        test_output = sdf_net.forward(points[:10000])
        loss = criterion(test_output, sdf[:10000]).item()
        print("Epoch {:d}. Loss: {:.8f}".format(epoch, loss))

train()