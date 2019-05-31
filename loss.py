import torch
from operator import mul
from functools import reduce

bce_loss = lambda a, b: torch.nn.functional.binary_cross_entropy(a * 0.5 + 0.5, b * 0.5 + 0.5, reduction="sum")


class VoxelReconstructionLoss():
    def __init__(self):
        self.value = 0.5

    def __call__(self, input, target):
        target_signs = target > 0        
        loss_a = torch.mean(input[target_signs])
        loss_b = torch.mean(input[~target_signs])

        return loss_b - loss_a


voxel_reconstruction_loss = VoxelReconstructionLoss()


def voxel_difference(input, target):
    input_signs = input > 0
    target_signs = target > 0
    difference = input_signs != target_signs
    return torch.sum(difference).item() / reduce(mul, input.shape, 1)


def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

class InceptionScore():
    def __init__(self):
        self.classifier = None

    def __call__(self, input):
        if self.classifier is None:
            from model import Classifier
            self.classifier = Classifier()
            self.classifier.load()

        with torch.no_grad():
            label_distribution = self.classifier.forward(input)
            marginal_distribution = torch.mean(label_distribution, dim = 0)
            
            kld = -torch.sum(label_distribution * torch.log(marginal_distribution / label_distribution), dim = 1)
            
            score = torch.exp(torch.mean(kld))
        return score.item()

inception_score = InceptionScore()