from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm as tqdm

import sys
import time

import random
random.seed(0)
torch.manual_seed(0)

from model.pointcloud_classifier import PointcloudClassifier
from dataset import dataset as dataset, SURFACE_POINTCLOUD_SIZE
from util import device

dataset.load_labels(device=device)
points = dataset.load_surface_clouds('cpu')
points = points.to(device)
del dataset.surface_pointclouds

TEST_SPLIT = 0.75
POINTCLOUD_SIZE = points.shape[0] // dataset.size
SAMPLE_SIZE = 1000

all_indices = list(range(dataset.size))
test_indices = all_indices[:int(dataset.size * TEST_SPLIT)]
training_indices = list(all_indices[int(dataset.size * TEST_SPLIT):])
labels_onehot = dataset.get_labels_onehot(device)

classifier = PointcloudClassifier()
if "continue" in sys.argv:
    classifier.load()

optimizer = optim.Adam(classifier.parameters(), lr=0.0008)
criterion = nn.BCELoss()

error_history = deque(maxlen = dataset.size)
accuracy_history = deque(maxlen = dataset.size)

def test(epoch_index, epoch_time):
    print("Epoch {:d} ({:.1f}s): ".format(epoch_index, epoch_time) +
        "training loss: {:4f}, ".format(np.mean(error_history)) +
        "training accuracy: {:4f}, ".format(np.mean(accuracy_history)))

def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for i in tqdm(range(dataset.size)):
            indices = torch.LongTensor(SAMPLE_SIZE).random_(0, POINTCLOUD_SIZE - 1)
            indices += POINTCLOUD_SIZE * i
            sample = points[indices, :]

            classifier.zero_grad()
            output = classifier(sample)
            loss = criterion(output, labels_onehot[i, :])
            loss.backward()
            optimizer.step()
            error = loss.item()
            error_history.append(error)

            _, predicted_label = torch.max(output, 0)
            accuracy = 1 if (predicted_label.item() == dataset.labels[i].item()) else 0
            accuracy_history.append(accuracy)

            batch_index += 1
        classifier.save()
        test(epoch, time.time() - epoch_start_time)

train()