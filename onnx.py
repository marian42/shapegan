from model.sdf_net import SDFNet, LATENT_CODE_SIZE, LATENT_CODES_FILENAME
from util import device, standard_normal_distribution, ensure_directory
import numpy as np
import torch


sdf_net = SDFNet()
sdf_net.filename = 'hybrid_progressive_gan_generator_3.to'
sdf_net.load()
sdf_net.eval()

import torch.onnx

POINT_COUNT = 4096
dummy_points = torch.randn((POINT_COUNT, 3), device=device)
dummy_codes = torch.randn((POINT_COUNT, LATENT_CODE_SIZE), device=device)
torch.onnx.export(sdf_net, (dummy_points, dummy_codes), "chairs.onnx", verbose=True)
