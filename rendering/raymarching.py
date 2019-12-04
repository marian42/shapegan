import torch
import numpy as np
import random
import math
from tqdm import tqdm
from PIL import Image
import os

from model.sdf_net import SDFNet, LATENT_CODES_FILENAME
from util import device, ensure_directory
from rendering.math import get_camera_transform
from scipy.spatial.transform import Rotation

BATCH_SIZE = 100000

def get_default_coordinates():
    camera_transform = get_camera_transform(2.2, 147, 20)
    camera_position = np.matmul(np.linalg.inv(camera_transform), np.array([0, 0, 0, 1]))[:3]
    light_matrix = get_camera_transform(6, 164, 50)
    light_position = np.matmul(np.linalg.inv(light_matrix), np.array([0, 0, 0, 1]))[:3]
    return camera_position, light_position

camera_position, light_position = get_default_coordinates()

def get_normals(sdf_net, points, latent_code):
    batch_count = points.shape[0] // BATCH_SIZE
    result = torch.zeros((points.shape[0], 3), device=points.device)
    for i in range(batch_count):
        result[BATCH_SIZE * i:BATCH_SIZE * (i+1), :] = sdf_net.get_normals(latent_code, points[BATCH_SIZE * i:BATCH_SIZE * (i+1), :])
    
    if points.shape[0] > BATCH_SIZE * batch_count:
        result[BATCH_SIZE * batch_count:, :] = sdf_net.get_normals(latent_code, points[BATCH_SIZE * batch_count:, :])
    return result


def get_shadows(sdf_net, points, light_position, latent_code, threshold = 0.001, sdf_offset=0, radius=1.0):
    ray_directions = light_position[np.newaxis, :] - points
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
    ray_directions_t = torch.tensor(ray_directions, device=device, dtype=torch.float32)
    points = torch.tensor(points, device=device, dtype=torch.float32)
    
    points += ray_directions_t * 0.1

    indices = torch.arange(points.shape[0])
    shadows = torch.zeros(points.shape[0])

    for i in tqdm(range(200)):
        test_points = points[indices, :]
        sdf = sdf_net.evaluate_in_batches(test_points, latent_code, return_cpu_tensor=False) + sdf_offset
        sdf = torch.clamp_(sdf, -0.1, 0.1)
        points[indices, :] += ray_directions_t[indices, :] * sdf.unsqueeze(1)
        
        hits = (sdf > 0) & (sdf < threshold)
        shadows[indices[hits]] = 1
        indices = indices[~hits]
        
        misses = points[indices, 1] > radius
        indices = indices[~misses]
        
        if indices.shape[0] < 2:
            break

    shadows[indices] = 1
    return shadows.cpu().numpy()
    

def render_image(sdf_net, latent_code, resolution=800, threshold=0.0005, sdf_offset=0, iterations=1000, ssaa=2, radius=1.0, crop=False, color=(0.8, 0.1, 0.1), vertical_cutoff=None):
    camera_forward = camera_position / np.linalg.norm(camera_position) * -1
    camera_distance = np.linalg.norm(camera_position).item()
    up = np.array([0, 1, 0])
    camera_right = np.cross(camera_forward, up)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_forward, camera_right)
    camera_up /= np.linalg.norm(camera_up)
    
    screenspace_points = np.meshgrid(
        np.linspace(-1, 1, resolution * ssaa),
        np.linspace(-1, 1, resolution * ssaa),
    )
    screenspace_points = np.stack(screenspace_points)
    screenspace_points = screenspace_points.reshape(2, -1).transpose()
    
    points = np.tile(camera_position, (screenspace_points.shape[0], 1))
    points = points.astype(np.float32)
    
    focal_distance = 1.0 / math.tan(math.asin(radius / camera_distance))
    ray_directions = screenspace_points[:, 0] * camera_right[:, np.newaxis] \
        + screenspace_points[:, 1] * camera_up[:, np.newaxis] \
        + focal_distance * camera_forward[:, np.newaxis]
    ray_directions = ray_directions.transpose().astype(np.float32)
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]

    b = np.einsum('ij,ij->i', points, ray_directions) * 2
    c = np.dot(camera_position, camera_position) - radius * radius
    distance_to_sphere = (-b - np.sqrt(np.power(b, 2) - 4 * c)) / 2
    indices = np.argwhere(np.isfinite(distance_to_sphere)).reshape(-1)

    points[indices] += ray_directions[indices] * distance_to_sphere[indices, np.newaxis]

    points = torch.tensor(points, device=device, dtype=torch.float32)
    ray_directions_t = torch.tensor(ray_directions, device=device, dtype=torch.float32)

    indices = torch.tensor(indices, device=device, dtype=torch.int64)
    model_mask = torch.zeros(points.shape[0], dtype=torch.uint8)

    for i in tqdm(range(iterations)):
        test_points = points[indices, :]
        sdf = sdf_net.evaluate_in_batches(test_points, latent_code, return_cpu_tensor=False) + sdf_offset
        torch.clamp_(sdf, -0.02, 0.02)
        points[indices, :] += ray_directions_t[indices, :] * sdf.unsqueeze(1)
        
        hits = (sdf > 0) & (sdf < threshold)
        model_mask[indices[hits]] = 1
        indices = indices[~hits]
        
        misses = torch.norm(points[indices, :], dim=1) > radius
        indices = indices[~misses]
        
        if indices.shape[0] < 2:
            break
        
    model_mask[indices] = 1

    if vertical_cutoff is not None:
        model_mask[points[:, 1] > vertical_cutoff] = 0
        model_mask[points[:, 1] < -vertical_cutoff] = 0

    normal = get_normals(sdf_net, points[model_mask], latent_code).cpu().numpy()

    model_mask = model_mask.cpu().numpy().astype(bool)
    points = points.cpu().numpy()
    model_points = points[model_mask]
    
    seen_by_light = 1.0 - get_shadows(sdf_net, model_points, light_position, latent_code, radius=radius, sdf_offset=sdf_offset)
    
    light_direction = light_position[np.newaxis, :] - model_points
    light_direction /= np.linalg.norm(light_direction, axis=1)[:, np.newaxis]
    
    diffuse = np.einsum('ij,ij->i', light_direction, normal)
    diffuse = np.clip(diffuse, 0, 1) * seen_by_light

    reflect = light_direction - np.einsum('ij,ij->i', light_direction, normal)[:, np.newaxis] * normal * 2
    reflect /= np.linalg.norm(reflect, axis=1)[:, np.newaxis]
    specular = np.einsum('ij,ij->i', reflect, ray_directions[model_mask, :])
    specular = np.clip(specular, 0.0, 1.0)
    specular = np.power(specular, 20) * seen_by_light
    rim_light = -np.einsum('ij,ij->i', normal, ray_directions[model_mask, :])
    rim_light = 1.0 - np.clip(rim_light, 0, 1)
    rim_light = np.power(rim_light, 4) * 0.3

    color = np.array(color)[np.newaxis, :] * (diffuse * 0.5 + 0.5)[:, np.newaxis]
    color += (specular * 0.3 + rim_light)[:, np.newaxis]

    color = np.clip(color, 0, 1)

    ground_points = ray_directions[:, 1] < 0
    ground_points[model_mask] = 0
    ground_points = np.argwhere(ground_points).reshape(-1)
    ground_plane = np.min(model_points[:, 1]).item()
    points[ground_points, :] -= ray_directions[ground_points, :] * ((points[ground_points, 1] - ground_plane) / ray_directions[ground_points, 1])[:, np.newaxis]
    ground_points = ground_points[np.linalg.norm(points[ground_points, ::2], axis=1) < 3]
    
    ground_shadows = get_shadows(sdf_net, points[ground_points, :], light_position, latent_code, sdf_offset=sdf_offset)

    pixels = np.ones((points.shape[0], 3))
    pixels[model_mask] = color
    pixels[ground_points] -= ((1.0 - 0.65) * ground_shadows)[:, np.newaxis]
    pixels = pixels.reshape((resolution * ssaa, resolution * ssaa, 3))

    if crop:
        from util import crop_image
        pixels = crop_image(pixels, background=1)

    image = Image.fromarray(np.uint8(pixels * 255) , 'RGB')

    if ssaa != 1:
        image = image.resize((resolution, resolution), Image.ANTIALIAS)

    return image


def render_image_for_index(sdf_net, latent_codes, index, crop=False, resolution=800):
    ensure_directory('screenshots')
    FILENAME = 'screenshots/raymarching-examples/image-{:d}-{:d}.png'
    filename = FILENAME.format(index, resolution)

    if os.path.isfile(filename):
        return Image.open(filename)
    
    img = render_image(sdf_net, latent_codes[index], resolution=resolution, crop=crop)
    img.save(filename)
    return img