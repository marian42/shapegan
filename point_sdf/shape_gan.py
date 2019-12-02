import sys

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from shapenet import ShapeNetPointSDF, visualize
from generator import SDFGenerator
from pointnet import PointNet

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true')
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

if args.eval:
    G.load_state_dict(torch.load('G.pt', map_location=device))
    G.eval()
    torch.manual_seed(12345)
    for _ in range(5):
        pos = 2 * torch.rand((64 * 1024, 3), device=device) - 1
        pos.requires_grad_(True)
        z = torch.randn((LATENT_SIZE, ), device=device)
        dist = G(pos, z).squeeze()
        visualize(pos.detach(), dist.detach(), dist.detach().abs() < 0.05)

        mesh = G.get_mesh(z, 64)
        visualize(mesh.sample(2048))

        # grad = torch.autograd.grad(dist, pos,
        #                            grad_outputs=torch.ones_like(dist),
        #                            create_graph=True, retain_graph=True,
        #                            only_inputs=True)[0]
        # perm = dist.abs() < 0.1
        # pos = pos[perm] - grad[perm] * dist[perm].unsqueeze(-1)
        # visualize(pos.detach())

    sys.exit()

root = '/data/SDF_GAN'
dataset = ShapeNetPointSDF(root, category='chairs', split='train')

num_steps = 0
for num_points, batch_size in zip([1024, 2048, 4096, 8192, 16384, 32768],
                                  [32, 32, 32, 24, 12, 6]):

    dataset.num_points = num_points
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)

    for epoch in range(1, 301):
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

        torch.save(G.state_dict(), 'G.pt')
        torch.save(D.state_dict(), 'D.pt')

        if epoch % 100 == 0:
            torch.save(G.state_dict(), 'G_{}_{}.pt'.format(num_points, epoch))
            torch.save(D.state_dict(), 'D_{}_{}.pt'.format(num_points, epoch))
