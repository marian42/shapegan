import torch
from operator import mul
from functools import reduce


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
    wrong_signs = (input * target) < 0
    return torch.sum(wrong_signs).item() / wrong_signs.nelement()


def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()

# Inception score of a sample from the dataset
REFERENCE_INCEPTION_SCORE = 14.33

class InceptionScore():
    def __init__(self):
        self.classifier = None

    def __call__(self, input):
        return 0
        
        if self.classifier is None:
            from model.classifier import Classifier
            self.classifier = Classifier()
            self.classifier.load()

        with torch.no_grad():
            label_distribution = self.classifier.forward(input)
            marginal_distribution = torch.mean(label_distribution, dim = 0)
            
            kld = -torch.sum(label_distribution * torch.log(marginal_distribution / label_distribution), dim = 1)
            
            score = torch.exp(torch.mean(kld[torch.isfinite(kld)]))
        return score.item() / REFERENCE_INCEPTION_SCORE

inception_score = InceptionScore()