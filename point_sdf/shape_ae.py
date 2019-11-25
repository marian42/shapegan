import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from shapenet import ShapeNetSDF
from generator import SDFGenerator
from discriminator import Encoder

LATENT_DIM = 64

root = '/data/sdf_chairs/chairs'
dataset = ShapeNetSDF(root, num_points=1024 * 16)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = Encoder(LATENT_DIM, use_dist=True)
decoder = SDFGenerator(LATENT_DIM, 256, num_layers=8, dropout=0.0)
encoder, decoder = encoder.to(device), decoder.to(device)
optimizer = Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

for epoch in range(1, 1001):
    for uniform, surface in loader:
        optimizer.zero_grad()
        uniform = uniform.to(device)
        surface = surface.to(device)
        u_pos, u_dist = uniform[..., :3], uniform[..., 3:]
        s_pos, s_dist = surface[..., :3], surface[..., 3:]

        u_dist = (10 * u_dist).tanh()
        s_dist = (10 * s_dist).tanh()

        z = encoder(u_pos, u_dist)
        out = decoder(u_pos, z)
        out_clamped = out.tanh()

        loss = (u_dist - out_clamped).abs().mean()
        loss.backward()
        optimizer.step()

        print('Loss: {:.4f}, Range: {:.2f} - {:.2f}'.format(
            loss, out.min(), out.max()))

    print('Epoch done!')
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
