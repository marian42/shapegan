import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
import numpy as np
import os

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory('plots')
ensure_directory('models')
ensure_directory('data')

CHARACTERS = '      `.-:/+osyhdmm###############'

def create_text_slice(voxels):
    voxel_resolution = voxels.shape[-1]
    center = voxels.shape[-1] // 4
    data = voxels[center, :, :]
    data = torch.clamp(data * -0.5 + 0.5, 0, 1) * (len(CHARACTERS) - 1)
    data = data.type(torch.int).cpu()
    lines = ['|' + ''.join([CHARACTERS[i] for i in line]) + '|' for line in data]
    result = []
    for i in range(voxel_resolution):
        if len(result) < i / 2.2:
            result.append(lines[i])
    frame = '+' + '—' * voxel_resolution + '+\n'
    return frame + '\n'.join(reversed(result)) + '\n' + frame


def get_points_in_unit_sphere(n, device):
    x = torch.rand(int(n * 2.5), 3, device=device) * 2 - 1
    mask = (torch.norm(x, dim=1) < 1).nonzero().squeeze()
    mask = mask[:n]
    x = x[mask, :]
    if x.shape[0] < n:
        print("Warning: Did not find enough points.")
    return x

def crop_image(image, background=255):
    mask = image[:, :] != background
    coords = np.array(np.nonzero(mask))
    
    if coords.size != 0:
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
    else:
        top_left = np.array((0, 0))
        bottom_right = np.array(image.shape)
        print("Warning: Image contains only background pixels.")
        
    half_size = int(max(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]) / 2)
    center = ((top_left + bottom_right) / 2).astype(int)
    center = (min(max(half_size, center[0]), image.shape[0] - half_size), min(max(half_size, center[1]), image.shape[1] - half_size))
    if half_size > 100:
        image = image[center[0] - half_size : center[0] + half_size, center[1] - half_size : center[1] + half_size]
    return image

def get_voxel_coordinates(resolution = 32, size=1, center=0, return_torch_tensor=False):
    if type(center) == int:
        center = (center, center, center)
    points = np.meshgrid(
        np.linspace(center[0] - size, center[0] + size, resolution),
        np.linspace(center[1] - size, center[1] + size, resolution),
        np.linspace(center[2] - size, center[2] + size, resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose()
    if return_torch_tensor:
        return torch.tensor(points, dtype=torch.float32, device=device)
    else:
        return points.astype(np.float32)

def show_sdf_point_cloud(points, sdf):
    import pyrender
    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1
    colors[sdf > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)