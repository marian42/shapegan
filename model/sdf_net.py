from model import *
import trimesh
import skimage
from util import get_points_in_unit_sphere
import numpy as np

class SDFVoxelizationHelperData():
    def __init__(self, device, voxel_count):
        sample_points = np.meshgrid(
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count),
            np.linspace(-1, 1, voxel_count)
        )
        sample_points = np.stack(sample_points).astype(np.float32)
        sample_points = np.swapaxes(sample_points, 1, 2)
        sample_points = sample_points.reshape(3, -1).transpose()
        unit_sphere_mask = np.linalg.norm(sample_points, axis=1) < 1
        sample_points = sample_points[unit_sphere_mask, :]

        self.unit_sphere_mask = unit_sphere_mask.reshape(voxel_count, voxel_count, voxel_count)
        self.sphere_sample_points = torch.tensor(sample_points, device=device)
        self.point_count = self.sphere_sample_points.shape[0]

sdf_voxelization_helper = dict()

SDF_NET_BREADTH = 512

class SDFNet(SavableModule):
    def __init__(self):
        super(SDFNet, self).__init__(filename="sdf_net.to")

        # DeepSDF paper has one additional FC layer in the first and second part each
        
        self.layers1 = nn.Sequential(
            nn.Linear(in_features = 3 + LATENT_CODE_SIZE, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True)
        )

        self.layers2 = nn.Sequential(
            nn.Linear(in_features = SDF_NET_BREADTH + LATENT_CODE_SIZE + 3, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = 1),
            nn.Tanh()
        )

        self.cuda()

    def forward(self, points, latent_codes):
        input = torch.cat((points, latent_codes), dim=1)
        x = self.layers1.forward(input)
        x = torch.cat((x, input), dim=1)
        x = self.layers2.forward(x)
        return x.squeeze()

    def get_mesh(self, latent_code, voxel_count = 64):
        if not voxel_count in sdf_voxelization_helper:
            sdf_voxelization_helper[voxel_count] = SDFVoxelizationHelperData(self.device, voxel_count)
       
        helper_data = sdf_voxelization_helper[voxel_count]

        with torch.no_grad():
            latent_codes = latent_code.repeat(helper_data.point_count, 1)
            distances = self.forward(helper_data.sphere_sample_points, latent_codes).cpu().numpy()
        
        voxels = np.ones((voxel_count, voxel_count, voxel_count))
        voxels[helper_data.unit_sphere_mask] = distances
        
        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(2.0 / voxel_count, 2.0 / voxel_count, 2.0 / voxel_count))
        vertices -= 1
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        return mesh

    def get_normals(self, latent_code, points):
        points.requires_grad = True
        latent_codes = latent_code.repeat(points.shape[0], 1)
        sdf = self.forward(points, latent_codes)
        sdf.backward(torch.ones((sdf.shape[0]), device=self.device))
        normals = points.grad
        normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
        return normals

    def get_surface_points(self, latent_code, sample_size=100000, sdf_cutoff=0.1, return_normals=False):
        points = get_points_in_unit_sphere(n=sample_size, device=self.device)
        points.requires_grad = True
        latent_codes = latent_code.repeat(points.shape[0], 1)
    
        sdf = self.forward(points, latent_codes)

        sdf.backward(torch.ones((sdf.shape[0]), device=self.device))
        normals = points.grad
        normals /= torch.norm(normals, dim=1).unsqueeze(dim=1)
        points.requires_grad = False

        # Move points towards surface by the amount given by the signed distance
        points -= normals * sdf.unsqueeze(dim=1)

        # Discard points with truncated SDF values
        mask = torch.abs(sdf) < sdf_cutoff
        points = points[mask, :]
        normals = normals[mask, :]
        
        if return_normals:
            return points, normals
        else:
            return points
       
