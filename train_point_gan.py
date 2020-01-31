import os.path as osp

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from datasets import PointDataset
from model.point_sdf_net import PointNet, SDFGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)
G, D = G.to(device), D.to(device)
G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

root = osp.join(f'data/{args.category}')
dataset = PointDataset.from_split(root, split='train')

configuration = [  # num_points, batch_size, epochs
    (1024, 32, 300),
    (2048, 32, 300),
    (4096, 32, 300),
    (8192, 24, 300),
    (16384, 12, 300),
    (32768, 6, 900),
]

num_steps = 0
for num_points, batch_size, epochs in configuration:
    dataset.num_points = num_points
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for uniform, _ in loader:
            num_steps += 1

            uniform = uniform.to(device)
            u_pos, u_dist = uniform[..., :3], uniform[..., 3:]

            D_optimizer.zero_grad()

            z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
            fake = G(u_pos, z)
            out_real = D(u_pos, u_dist)
            out_fake = D(u_pos, fake)
            D_loss = out_fake.mean() - out_real.mean()

            alpha = torch.rand((uniform.size(0), 1, 1), device=device)
            interpolated = alpha * u_dist + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            out = D(u_pos, interpolated)

            grad = torch.autograd.grad(out, interpolated,
                                       grad_outputs=torch.ones_like(out),
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(dim=-1, p=2)
            gp = GRADIENT_PENALITY * ((grad_norm - 1).pow(2).mean())

            loss = D_loss + gp
            loss.backward()
            D_optimizer.step()

            if num_steps % 5 == 0:
                G_optimizer.zero_grad()
                z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
                fake = G(u_pos, z)
                out_fake = D(u_pos, fake)
                loss = -out_fake.mean()
                loss.backward()
                G_optimizer.step()

            total_loss += D_loss.abs().item()

        print('Num points: {}, Epoch: {:03d}, Loss: {:.6f}'.format(
            num_points, epoch, total_loss / len(loader)))
