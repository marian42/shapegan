import torch

CHARACTERS = '      `.-:/+osyhdmm###############'

def create_text_slice(voxels):
    voxel_size = voxels.shape[-1]
    center = voxels.shape[-1] // 4
    data = voxels[center, :, :]
    data = (data * -0.5 + 0.5) * (len(CHARACTERS) - 1)
    data = data.type(torch.int).cpu()
    lines = ['|' + ''.join([CHARACTERS[i] for i in line]) + '|' for line in data]
    result = []
    for i in range(voxel_size):
        if len(result) < i / 2.2:
            result.append(lines[i])
    frame = '+' + '—' * voxel_size + '+\n'
    return frame + '\n'.join(reversed(result)) + '\n' + frame


def get_points_in_unit_sphere(n, device):
    x = torch.rand(int(n * 2.2), 3, device=device) * 2 - 1
    mask = (torch.norm(x, dim=1) < 1).nonzero().squeeze()
    mask = mask[:n]
    x = x[mask, :]
    if x.shape[0] < n:
        print("Warning: Did not find enough points.")
    return x