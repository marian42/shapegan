import numpy as np

# Creates a cube for every occupied (negative) voxel
def create_binary_voxel_mesh(voxels_array, threshold = 0.0):
    voxels = np.pad(voxels_array, 1, mode = 'constant')
    mask = voxels < threshold

    # X
    x, y, z = np.where(mask[:-1,:,:] & ~mask[1:,:,:])
    vertices = [
        x + 1, y, z,
        x + 1, y + 1, z,
        x + 1, y, z + 1,
        
        x + 1, y + 1, z,
        x + 1, y + 1, z + 1,
        x + 1, y, z + 1
    ]
    vertex_arrays = [np.array(vertices).transpose().flatten()]
    normals = [np.tile(np.array([1, 0, 0]), 6 * x.shape[0])]

    
    x, y, z = np.where(~mask[:-1,:,:] & mask[1:,:,:])
    vertices = [
        x + 1, y + 1, z,
        x + 1, y, z,
        x + 1, y, z + 1,
        
        x + 1, y, z + 1,
        x + 1, y + 1, z + 1,
        x + 1, y + 1, z
    ]

    vertex_arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([-1, 0, 0]), 6 * x.shape[0]))

    # Y
    x, y, z = np.where(mask[:,:-1,:] & ~mask[:,1:,:])
    vertices = [
        x + 1, y + 1, z,
        x, y + 1, z,
        x, y + 1, z + 1,

        x + 1, y + 1, z + 1,
        x + 1, y + 1, z,
        x, y + 1, z + 1
    ]
    vertex_arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([0, 1, 0]), 6 * x.shape[0]))

    x, y, z = np.where(~mask[:,:-1,:] & mask[:,1:,:])
    vertices = [
        x, y + 1, z,
        x + 1, y + 1, z,
        x, y + 1, z + 1,

        x + 1, y + 1, z,
        x + 1, y + 1, z + 1,
        x, y + 1, z + 1
    ]
    vertex_arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([0, -1, 0]), 6 * x.shape[0]))

    # Z
    x, y, z = np.where(mask[:,:,:-1] & ~mask[:,:,1:])
    vertices = [
        x, y, z + 1,
        x + 1, y, z + 1,
        x, y + 1, z + 1,

        x + 1, y, z + 1,
        x + 1, y + 1, z + 1,
        x, y + 1, z + 1
    ]
    vertex_arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([0, 0, 1]), 6 * x.shape[0]))

    x, y, z = np.where(~mask[:,:,:-1] & mask[:,:,1:])
    vertices = [
        x + 1, y, z + 1,
        x, y, z + 1,
        x, y + 1, z + 1,

        x + 1, y + 1, z + 1,
        x + 1, y, z + 1,
        x, y + 1, z + 1
    ]
    vertex_arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([0, 0, -1]), 6 * x.shape[0]))

    return np.concatenate(vertex_arrays).astype(np.float32), np.concatenate(normals).astype(np.float32)