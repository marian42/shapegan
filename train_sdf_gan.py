from model.sdf_net import SDFNet
from model.sdf_autoencoder import SDFDiscriminator, LATENT_CODE_SIZE
from itertools import count
import torch
import random
from collections import deque
import numpy as np
from rendering import MeshRenderer
from util import device, standard_normal_distribution, get_points_in_unit_sphere

points = torch.load("data/chairs-points.to").to(device)
sdf = torch.load("data/chairs-sdf.to").to(device)

POINTCLOUD_SIZE = 100000
MODEL_COUNT = points.shape[0] // POINTCLOUD_SIZE
POINT_SAMPLE_COUNT = 1000
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

debug_points = get_points_in_unit_sphere(n = 10, device=device)
debug_latent_codes = create_latent_code(repeat=debug_points.shape[0])

history_fake = deque(maxlen=50)
history_real = deque(maxlen=50)
viewer = MeshRenderer()

for step in count():
    # train generator
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    code = create_latent_code()
    test_points = get_points_in_unit_sphere(n = POINT_SAMPLE_COUNT, device=device)
    fake_sdf = generator(test_points, code)
    fake_discriminator_assessment = discriminator(test_points, fake_sdf)
    loss = -torch.log(fake_discriminator_assessment)
    loss.backward()
    generator_optimizer.step()

    # train discriminator on real sample
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    indices = torch.LongTensor(POINT_SAMPLE_COUNT).random_(int(POINTCLOUD_SIZE * 0.8), POINTCLOUD_SIZE - 1)
    indices += POINTCLOUD_SIZE * random.randint(0, MODEL_COUNT - 1)
    
    output = discriminator(points[indices, :], sdf[indices])
    history_real.append(output.item())    
    loss = discriminator_criterion(output, torch.tensor(1, device=device, dtype=torch.float32))
    loss.backward()
    discriminator_optimizer.step()

    # train discriminator on fake sample
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    code = create_latent_code()
    test_points = get_points_in_unit_sphere(n = POINT_SAMPLE_COUNT, device=device)
    with torch.no_grad():
        fake_sdf = generator(test_points, code)
    output = discriminator(test_points, fake_sdf)
    history_fake.append(output.item())

    loss = discriminator_criterion(output, torch.tensor(0, device=device, dtype=torch.float32))
    loss.backward()
    discriminator_optimizer.step()


    if step % 50 == 0 and step != 0:
        print("Step {:d}: Prediction on fake samples: {:.3f}, on real samples: {:.3f}".format(step, np.mean(history_fake), np.mean(history_real)))
        try:
            viewer.set_mesh(generator.get_mesh(create_latent_code(repeat=1)))
        except ValueError:
            print("(Voxel volume contains no sign changes)")
        print(generator(debug_points, debug_latent_codes).detach())