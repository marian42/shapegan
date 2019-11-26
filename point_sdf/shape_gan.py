import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from shapenet import ShapeNetSDF
from generator import SDFGenerator
from discriminator import Encoder

LATENT_DIM = 128
GRADIENT_PENALITY = 10

root = '/data/sdf_chairs/chairs'
dataset = ShapeNetSDF(root, num_points=1024)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_DIM, 256, num_layers=8, dropout=0.0)
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

        D_optimizer.zero_grad()

        z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
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
            z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
            pos = 2 * torch.rand_like(u_pos) - 1
            z = torch.randn(uniform.size(0), LATENT_DIM, device=device)
            fake = G(pos, z)
            out_fake = D(pos, fake)
            loss = -out_fake.mean()
            loss.backward()
            G_optimizer.step()

            print(
                'D: {:.4f}, GP: {:.4f}, R: {:.4f} - {:.4f}, F: {:.4f} - {:.4f}'
                .format(-D_loss.item(), gp.item(),
                        u_dist.min().item(),
                        u_dist.max().item(),
                        fake.min().item(),
                        fake.max().item()))
        total_loss += D_loss.abs().item()

    print('Epoch {} done!'.format(epoch))
    print('Loss: {:.4f}'.format(total_loss / len(loader)))
    torch.save(G.state_dict(), 'G.pt')
    torch.save(D.state_dict(), 'D.pt')
