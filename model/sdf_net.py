from model import *
import trimesh
import skimage
from util import get_points_in_unit_sphere, get_voxel_coordinates
import numpy as np

class SDFVoxelizationHelperData():
    def __init__(self, device, voxel_resolution, sphere_only=True):
        sample_points = get_voxel_coordinates(voxel_resolution)

        if sphere_only:
            unit_sphere_mask = np.linalg.norm(sample_points, axis=1) < 1.1
            sample_points = sample_points[unit_sphere_mask, :]
            self.unit_sphere_mask = unit_sphere_mask.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
        
        self.sample_points = torch.tensor(sample_points, device=device)
        self.point_count = self.sample_points.shape[0]

sdf_voxelization_helper = dict()

SDF_NET_BREADTH = 256

class SDFNet(SavableModule):
    def __init__(self, latent_code_size=LATENT_CODE_SIZE, device='cuda'):
        super(SDFNet, self).__init__(filename="sdf_net.to")
        self.layers1 = nn.Sequential(
            nn.Linear(in_features = 3 + latent_code_size, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True)
        )

        self.layers2 = nn.Sequential(
            nn.Linear(in_features = SDF_NET_BREADTH + latent_code_size + 3, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, points, latent_codes):
        input = torch.cat((points, latent_codes), dim=1)
        x = self.layers1(input)
        x = torch.cat((x, input), dim=1)
        x = self.layers2(x)
        return x.squeeze()

    def evaluate_in_batches(self, points, latent_code, batch_size=100000, return_cpu_tensor=True):
        latent_codes = latent_code.repeat(batch_size, 1)
        with torch.no_grad():
            batch_count = points.shape[0] // batch_size
            if return_cpu_tensor:
                result = torch.zeros((points.shape[0]))
            else:
                result = torch.zeros((points.shape[0]), device=points.device)
            for i in range(batch_count):
                result[batch_size * i:batch_size * (i+1)] = self(points[batch_size * i:batch_size * (i+1), :], latent_codes)
            remainder = points.shape[0] - batch_size * batch_count
            result[batch_size * batch_count:] = self(points[batch_size * batch_count:, :], latent_codes[:remainder, :])
        return result

    def get_voxels(self, latent_code, voxel_resolution, sphere_only=True, pad=True):
        if not (voxel_resolution, sphere_only) in sdf_voxelization_helper:
            helper_data = SDFVoxelizationHelperData(self.device, voxel_resolution, sphere_only)
            sdf_voxelization_helper[(voxel_resolution, sphere_only)] = helper_data
        else:
            helper_data = sdf_voxelization_helper[(voxel_resolution, sphere_only)]

        with torch.no_grad():
            distances = self.evaluate_in_batches(helper_data.sample_points, latent_code).numpy()
        
        if sphere_only:
            voxels = np.ones((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
            voxels[helper_data.unit_sphere_mask] = distances
        else:
            voxels = distances.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
            if pad:
                voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

        return voxels

    def get_mesh(self, latent_code, voxel_resolution = 64, sphere_only = True, raise_on_empty=False, level=0):
        size = 2
        
        voxels = self.get_voxels(latent_code, voxel_resolution=voxel_resolution, sphere_only=sphere_only)
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
        try:
            vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=level, spacing=(size / voxel_resolution, size / voxel_resolution, size / voxel_resolution))
        except ValueError as value_error:
            if raise_on_empty:
                raise value_error
            else:
                return None
        
        vertices -= size / 2
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh

    def get_uniform_surface_points(self, latent_code, point_count=1000, voxel_resolution=64, sphere_only=True, level=0):
        mesh = self.get_mesh(latent_code, voxel_resolution=voxel_resolution, sphere_only=sphere_only, level=level)
        return mesh.sample(point_count)

    def get_normals(self, latent_code, points):
        if latent_code.requires_grad or points.requires_grad:
            raise Exception('get_normals may only be called with tensors that don\'t require grad.')
        
        points.requires_grad = True
        latent_codes = latent_code.repeat(points.shape[0], 1)
        sdf = self(points, latent_codes)
        sdf.backward(torch.ones(sdf.shape[0], device=self.device))
        normals = points.grad
        normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
        return normals

    def get_surface_points(self, latent_code, sample_size=100000, sdf_cutoff=0.1, return_normals=False, use_unit_sphere=True):
        if use_unit_sphere:
            points = get_points_in_unit_sphere(n=sample_size, device=self.device) * 1.1
        else:
            points = torch.rand((sample_size, 3), device=self.device) * 2.2 - 1
        points.requires_grad = True
        latent_codes = latent_code.repeat(points.shape[0], 1)
    
        sdf = self(points, latent_codes)

        sdf.backward(torch.ones((sdf.shape[0]), device=self.device))
        normals = points.grad
        normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
        points.requires_grad = False

        # Move points towards surface by the amount given by the signed distance
        points -= normals * sdf.unsqueeze(dim=1)

        # Discard points with truncated SDF values
        mask = (torch.abs(sdf) < sdf_cutoff) & torch.all(torch.isfinite(points), dim=1)
        points = points[mask, :]
        normals = normals[mask, :]
        
        if return_normals:
            return points, normals
        else:
            return points

    def get_surface_points_in_batches(self, latent_code, amount = 1000):
        result = torch.zeros((amount, 3), device=self.device)
        position = 0
        iteration_limit = 20
        while position < amount and iteration_limit > 0:
            points = self.get_surface_points(latent_code, sample_size=amount * 6)
            amount_used = min(amount - position, points.shape[0])
            result[position:position+amount_used, :] = points[:amount_used, :]
            position += amount_used
            iteration_limit -= 1
        return result
