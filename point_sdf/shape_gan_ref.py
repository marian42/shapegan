import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from shapenet import ShapeNetSDF
from generator import SDFGenerator
from discriminator import Encoder

LATENT_DIM = 128
GRADIENT_PENALITY = 10
THRESHOLD = 0.1


def generate_batch(u_pos, u_dist, s_pos, s_dist):
    u_batch = torch.stack([
        torch.full((u_pos.size(1), ), i, device=device, dtype=torch.long)
        for i in range(u_pos.size(0))
    ], dim=0)

    mask = u_dist.abs().squeeze(-1) < THRESHOLD

    s_pos = s_pos[mask]
    s_dist = s_dist[mask]
    s_batch = u_batch[mask]

    u_pos, u_dist = u_pos.view(-1, 3), u_dist.view(-1, 1)
    u_batch = u_batch.view(-1)

    s_pos, s_dist = s_pos.view(-1, 3), s_dist.view(-1, 1)
    s_batch = s_batch.view(-1)

    return (
        torch.cat([u_pos, s_pos], dim=0),
        torch.cat([u_dist, s_dist], dim=0),
        torch.cat([u_batch, s_batch], dim=0),
    )


class RefinementGenerator(torch.nn.Module):
    def __init__(self, generator):
        super(RefinementGenerator, self).__init__()
        self.generator = generator

    def forward(self, pos, z):
        pos.requires_grad_(True)
        dist = self.generator(pos, z)

        grad = torch.autograd.grad(dist, pos,
                                   grad_outputs=torch.ones_like(dist),
                                   retain_graph=True, only_inputs=True)[0]
        ref_pos = pos - dist * grad
        ref_dist = self.generator(pos, z)

        return pos, dist, ref_pos, ref_dist


root = '/data/sdf_chairs/chairs'
dataset = ShapeNetSDF(root, num_points=1024 * 8)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_DIM, 256, num_layers=8, dropout=0.0)
G = RefinementGenerator(G)
D = Encoder(1)
G, D = G.to(device), D.to(device)

G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

num_steps = 0
for epoch in range(1, 1001):
    total_loss = 0
    for uniform, surface in loader:
        num_steps += 1

        uniform, surface = uniform.to(device), surface.to(device)
        u_pos, u_dist = uniform[..., :3], uniform[..., 3:]
        s_pos, s_dist = surface[..., :3], surface[..., 3:]

        real_pos, real_dist, real_batch = generate_batch(
            u_pos, u_dist, s_pos, s_dist)

        D_optimizer.zero_grad()

        z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
        u_pos2, u_dist2, s_pos2, s_dist2 = G(u_pos, z)
        fake_pos, fake_dist, fake_batch = generate_batch(
            u_pos2, u_dist2, s_pos2, s_dist2)

        out_real = D(real_pos, real_dist, real_batch)
        out_fake = D(fake_pos, fake_dist, fake_batch)
        D_loss = out_fake.mean() - out_real.mean()

        alpha = torch.rand((uniform.size(0), 1, 1), device=device)
        interpolated = alpha * u_dist + (1 - alpha) * u_dist2
        interpolated.requires_grad_(True)
        batch = torch.cat([
            torch.full((u_pos.size(1), ), i, device=device, dtype=torch.long)
            for i in range(u_pos.size(0))
        ], dim=0)
        out = D(u_pos.view(-1, 3), interpolated.view(-1, 1), batch)

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
            z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
            pos = 2 * torch.rand_like(u_pos) - 1
            z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
            u_pos, u_dist, s_pos, s_dist = G(pos, z)
            fake_pos, fake_dist, fake_batch = generate_batch(
                u_pos, u_dist, s_pos, s_dist)
            out_fake = D(fake_pos, fake_dist, fake_batch)
            loss = -out_fake.mean()
            loss.backward()
            G_optimizer.step()

            print('D: {:.4f}, GP: {:.4f}, F: {:.4f} - {:.4f}'.format(
                -D_loss.item(), gp.item(),
                fake_dist.min().item(),
                fake_dist.max().item()))
        total_loss += D_loss.abs().item()

    print('Epoch {} done!'.format(epoch))
    print('Loss: {:.4f}'.format(total_loss / len(loader)))
    torch.save(G.state_dict(), 'G_ref.pt')
    torch.save(D.state_dict(), 'D_ref.pt')
