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
THRESHOLD = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)
G, D = G.to(device), D.to(device)

root = osp.join(f'data/{args.category}')
dataset = PointDataset.from_split(root, split='train')


def generate_batch(u_pos, u_dist, s_pos, s_dist):
    u_batch = torch.arange(u_pos.size(0), device=u_pos.device)
    u_batch = u_batch.view(-1, 1).repeat(1, u_pos.size(1))

    mask = u_dist.abs().squeeze(-1) < THRESHOLD

    s_pos = s_pos[mask].view(-1, 3)
    s_dist = s_dist[mask].view(-1, 1)
    s_batch = u_batch[mask].view(-1)

    mask = mask | (torch.rand(mask.size(), device=mask.device) < 0.15)

    u_pos = u_pos[mask].view(-1, 3)
    u_dist = u_dist[mask].view(-1, 1)
    u_batch = u_batch[mask].view(-1)

    return (
        torch.cat([u_pos, s_pos], dim=0),
        torch.cat([u_dist, s_dist], dim=0),
        torch.cat([u_batch, s_batch], dim=0),
    )


class RefinementGenerator(torch.nn.Module):
    def __init__(self, generator):
        super(RefinementGenerator, self).__init__()
        self.generator = generator

    def forward(self, u_pos, z):
        u_pos.requires_grad_(True)
        u_dist = self.generator(u_pos, z)

        grad = torch.autograd.grad(u_dist, u_pos,
                                   grad_outputs=torch.ones_like(u_dist),
                                   retain_graph=True, only_inputs=True)[0]
        s_pos = u_pos - u_dist * grad
        s_pos = s_pos + 0.0025 * torch.randn_like(s_pos)
        s_dist = self.generator(s_pos, z)

        return u_pos, u_dist, s_pos, s_dist


# TODO: Load G and D from `train_point_gan.py`.
# G.load_state_dict(torch.load(..., map_location=device))
# D.load_state_dict(torch.load(..., map_location=device))
ref_G = RefinementGenerator(G).to(device)
G_optimizer = RMSprop(ref_G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

configuration = [  # num_points, batch_size, epochs
    (8192, 16, 60),
    (16384, 8, 60),
]

num_steps = 0
for num_points, batch_size, epochs in configuration:
    dataset.num_points = num_points
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for uniform, surface in loader:
            num_steps += 1

            uniform, surface = uniform.to(device), surface.to(device)
            u_pos, u_dist = uniform[..., :3], uniform[..., 3:]
            s_pos, s_dist = surface[..., :3], surface[..., 3:]

            D_optimizer.zero_grad()

            z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
            fake_u_pos, fake_u_dist, fake_s_pos, fake_s_dist = ref_G(u_pos, z)
            fake_pos, fake_dist, fake_batch = generate_batch(
                fake_u_pos, fake_u_dist, fake_s_pos, fake_s_dist)

            real_pos, real_dist, real_batch = generate_batch(
                u_pos, u_dist, s_pos, s_dist)

            out_real = D(real_pos, real_dist, real_batch)
            out_fake = D(fake_pos, fake_dist, fake_batch)
            D_loss = out_fake.mean() - out_real.mean()

            alpha = torch.rand((uniform.size(0), 1, 1), device=device)
            interpolated = alpha * u_dist + (1 - alpha) * fake_u_dist
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
                fake = ref_G(u_pos, z)
                fake_u_pos, fake_u_dist, fake_s_pos, fake_s_dist = fake
                fake_pos, fake_dist, fake_batch = generate_batch(
                    fake_u_pos, fake_u_dist, fake_s_pos, fake_s_dist)
                out_fake = D(fake_pos, fake_dist, fake_batch)
                loss = -out_fake.mean()
                loss.backward()
                G_optimizer.step()

            total_loss += D_loss.abs().item()

        print('Num points: {}, Epoch: {:03d}, Loss: {:.6f}'.format(
            num_points, epoch, total_loss / len(loader)))
