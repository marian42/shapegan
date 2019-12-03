import functools
import os.path as osp
import glob

import argparse
import torch

from generator import SDFGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--level', type=float, default=0.04)
args = parser.parse_args()

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
G = G.to(device)
G.eval()


def compare(item1, item2):
    p1, e1 = int(item1.split('_')[1]), int(item1.split('_')[2][:-3])
    p2, e2 = int(item2.split('_')[1]), int(item2.split('_')[2][:-3])
    return -1 if p1 < p2 or (p1 == p2 and e1 < e2) else 1


paths = sorted(glob.glob(osp.join(args.category, 'G_*00.pt')),
               key=functools.cmp_to_key(compare))

for i, path in enumerate(paths):
    print(path)
    G.load_state_dict(torch.load(path, map_location=device))
    torch.manual_seed(1234)
    generated = []
    while len(generated) < 200:
        z = torch.randn((LATENT_SIZE, ), device=device)
        mesh = G.get_mesh(z, 64, level=args.level)
        if mesh is not None:
            generated += [torch.from_numpy(mesh.sample(2048))]
    generated = torch.stack(generated, dim=0)
    path = '/data/SDF_GAN/{}/results/uniform_{:02d}.pt'.format(
        args.category, i + 1)
    torch.save(generated, path)
