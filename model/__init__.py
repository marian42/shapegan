import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

import os

MODEL_PATH = "models"
LATENT_CODES_FILENAME = os.path.join(MODEL_PATH, "sdf_net_latent_codes.to")
LATENT_CODE_SIZE = 64

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class SavableModule(nn.Module):
    def __init__(self, filename):
        super(SavableModule, self).__init__()
        self.filename = filename

    def get_filename(self):
        return os.path.join(MODEL_PATH, self.filename)

    def load(self):
        self.load_state_dict(torch.load(self.get_filename()), strict=False)
    
    def save(self):
        torch.save(self.state_dict(), self.get_filename())

    @property
    def device(self):
        return next(self.parameters()).device