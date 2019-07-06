import torch

CHARACTERS = '      `.-:/+osyhdmm###############'

def show_slice(voxels):
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