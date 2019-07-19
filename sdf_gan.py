from model import SDFNet, SDFDiscriminator, LATENT_CODE_SIZE, standard_normal_distribution
from itertools import count
import torch
import random
from collections import deque
import numpy as np
from voxel.viewer import VoxelViewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("data/dataset-sdf-clouds.to")
points = data[:, :3]
points = points.cuda()
sdf = data[:, 3].to(device)
del data

POINTCLOUD_SIZE = 100000
MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE
POINT_SAMPLE_COUNT = 4096
SDF_CUTOFF = 0.1
torch.clamp_(sdf, -SDF_CUTOFF, SDF_CUTOFF)


generator = SDFNet()
generator.to(device)

discriminator = SDFDiscriminator()
discriminator.to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-6)

discriminator_criterion = torch.nn.functional.mse_loss

def create_latent_code(repeat=POINT_SAMPLE_COUNT):
    shape = torch.Size([1, LATENT_CODE_SIZE])
    x = standard_normal_distribution.sample(shape).to(device)
    x = x.repeat(repeat, 1)
    return x

def get_points_in_unit_sphere(n = POINT_SAMPLE_COUNT):
    x = torch.rand(n * 3, 3, device=device) * 2 - 1
    mask = (torch.norm(x, dim=1) < 1).nonzero().squeeze()
    mask = mask[:n]
    x = x[mask, :]
    return x

debug_points = get_points_in_unit_sphere(n = 10)
debug_latent_codes = create_latent_code(repeat=debug_points.shape[0])

history_fake = deque(maxlen=50)
history_real = deque(maxlen=50)
viewer = VoxelViewer()

for step in count():
    # train generator
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    code = create_latent_code()
    test_points = get_points_in_unit_sphere()
    fake_sdf = generator.forward(test_points, code)
    fake_discriminator_assessment = discriminator.forward(test_points, fake_sdf)
    loss = -torch.log(fake_discriminator_assessment)
    loss.backward()
    generator_optimizer.step()

    # train discriminator on real sample
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    indices = torch.LongTensor(POINT_SAMPLE_COUNT).random_(int(POINTCLOUD_SIZE * 0.8), POINTCLOUD_SIZE - 1)
    indices += POINTCLOUD_SIZE * random.randint(0, MODEL_COUNT - 1)
    
    output = discriminator.forward(points[indices, :], sdf[indices])
    history_real.append(output.item())    
    loss = discriminator_criterion(output, torch.tensor(1, device=device, dtype=torch.float32))
    loss.backward()
    discriminator_optimizer.step()

    # train discriminator on fake sample
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    code = create_latent_code()
    test_points = get_points_in_unit_sphere()
    with torch.no_grad():
        fake_sdf = generator.forward(test_points, code)
    output = discriminator.forward(test_points, fake_sdf)
    history_fake.append(output.item())

    loss = discriminator_criterion(output, torch.tensor(0, device=device, dtype=torch.float32))
    loss.backward()
    discriminator_optimizer.step()


    if step % 50 == 0 and step != 0:
        print("Prediction on fake samples: {:.3f}, on real samples: {:.3f}".format(np.mean(history_fake), np.mean(history_real)))
        try:
            viewer.set_mesh(generator.get_mesh(create_latent_code(repeat=1), device))
        except ValueError:
            print("(Voxel volume contains no sign changes)")
        print(generator.forward(debug_points, debug_latent_codes).detach())