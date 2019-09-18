from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys
import time

import random
random.seed(0)
torch.manual_seed(0)

from model.classifier import Classifier
from dataset import dataset as dataset
from util import device
dataset.load_voxels(device)

BATCH_SIZE = 32
TEST_SPLIT = 0.1

all_indices = list(range(dataset.size))
test_indices = all_indices[:int(dataset.size * TEST_SPLIT)]
training_indices = list(all_indices[int(dataset.size * TEST_SPLIT):])
test_data = dataset.voxels[test_indices]
labels_onehot = dataset.get_labels_onehot(device)
test_labels_onehot = labels_onehot[test_indices]
test_labels = dataset.labels[test_indices]

classifier = Classifier()
if "continue" in sys.argv:
    classifier.load()

optimizer = optim.Adam(classifier.parameters(), lr=0.0005)
criterion = nn.BCELoss()

error_history = deque(maxlen = BATCH_SIZE)
accuracy_history = deque(maxlen = BATCH_SIZE)

def create_batches():
    batch_count = int(len(training_indices) / BATCH_SIZE)
    random.shuffle(training_indices)
    for i in range(batch_count - 1):
        yield training_indices[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
    yield training_indices[(batch_count - 1) * BATCH_SIZE:]

def test(epoch_index, epoch_time):
    with torch.no_grad():
        output = classifier.forward(test_data)
        loss = criterion(output, test_labels_onehot).item()
        _, output_classes = torch.max(output, 1)
        accuracy = float(torch.sum(output_classes == test_labels)) / test_data.shape[0]
        print("Epoch {:d} ({:.1f}s): ".format(epoch_index, epoch_time) +
            "training loss: {:4f}, ".format(np.mean(error_history)) +
            "training accuracy: {:4f}, ".format(np.mean(accuracy_history)) +
            "test loss: {:.4f}, ".format(loss) +
            "test accuracy: {:.4f}".format(accuracy))


def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in create_batches():
            indices = torch.tensor(batch, device = device)
            sample = dataset.voxels[indices, :, :, :]

            classifier.zero_grad()
            output = classifier.forward(sample)
            loss = criterion(output, labels_onehot[indices])
            loss.backward()
            optimizer.step()
            error = loss.item()
            error_history.append(error)

            _, output_classes = torch.max(output, 1)
            accuracy = float(torch.sum(output_classes == dataset.labels[indices])) / len(indices)
            accuracy_history.append(accuracy)

            batch_index += 1
        classifier.save()
        test(epoch, time.time() - epoch_start_time)

train()