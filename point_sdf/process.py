import torch
import numpy as np
import trimesh
from sdf.mesh_to_sdf import scale_to_unit_sphere, MeshSDF


def process(path):
    mesh = trimesh.load(path)
    mesh = mesh.dump().sum()

    mesh = scale_to_unit_sphere(mesh)
    mesh_sdf = MeshSDF(mesh)

    uniform_pos = 2 * torch.rand(250000, 3) - 1
    uniform_pos = uniform_pos.numpy()

    def get_sdf(self, query_points, sample_count=11):
        distances, indices = self.kd_tree.query(query_points, k=sample_count)
        distances = distances.astype(np.float32)

        closest_points = self.points[indices]
        direction_to_surface = query_points[:, np.newaxis, :] - closest_points
        inside = np.einsum('ijk,ijk->ij', direction_to_surface,
                           self.normals[indices]) < 0
        inside = np.sum(inside, axis=1) > sample_count * 0.5
        distances = distances[:, 0]
        distances[inside] *= -1
        return distances, closest_points[:, 0]

    uniform_dist, closest_point = get_sdf(mesh_sdf, uniform_pos)
    surface_pos = closest_point + np.random.normal(
        scale=0.0025, size=(closest_point.shape[0], 3))
    surface_dist, _ = get_sdf(mesh_sdf, surface_pos)

    uniform_pos = torch.from_numpy(uniform_pos).to(torch.float)
    uniform_dist = torch.from_numpy(uniform_dist).to(torch.float)

    surface_pos = torch.from_numpy(surface_pos).to(torch.float)
    surface_dist = torch.from_numpy(surface_dist).to(torch.float)

    uniform = torch.cat([uniform_pos, uniform_dist.view(-1, 1)], dim=-1)
    surface = torch.cat([surface_pos, surface_dist.view(-1, 1)], dim=-1)

    return uniform, surface


if __name__ == '__main__':
    path = 'model.obj'
    data = process(path)
    torch.save(data, 'model.pt')
