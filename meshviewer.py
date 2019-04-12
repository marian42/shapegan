import pygame
from pygame.locals import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders


from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
    


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_CULL_FACE)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    glEnable( GL_BLEND )
    
    
    vertexBuffer = vbo.VBO(vertices)

    mouse = None
    rotation = (0, 0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        _, _, pressed = pygame.mouse.get_pressed()
        current_mouse = pygame.mouse.get_pos()
        if mouse is not None:
            movement = (current_mouse[0] - mouse[0], current_mouse[1] - mouse[1])
            if pressed:
                rotation = (rotation[0] + movement[0], max(-90, min(90, rotation[1] + movement[1])))
        
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, voxel_size * 4)
        glTranslatef(0.0,0.0, - voxel_size * 2)
        glRotatef(rotation[1], 1.0, 0, 0)
        glRotatef(rotation[0], 0, 1.0, 0)
        glTranslatef(-(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2)
        
        mouse = current_mouse

        glClearColor(0.01, 0.01, 0.01, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor4d(1.0, 1.0, 0.0, 1.0)        

        glBegin(GL_TRIANGLES)
        v = vertices.tolist()
        for i in range(int(len(v) / 3)):
            glVertex3fv((v[i * 3], v[i * 3 + 1], v[i * 3 + 2]))
        glEnd()
        
        #vertexBuffer.bind()
        #glEnableClientState(GL_VERTEX_ARRAY)
        #glVertexPointer(3, GL_FLOAT, 0, vertexBuffer)
        #glDrawElements(GL_TRIANGLES, len(v), GL_FLOAT, None)
        pygame.display.flip()
        pygame.time.wait(10)


def create_vertices(voxels_array):
    THRESHOLD = 0.5

    voxels = np.zeros([voxels_array.shape[0] + 2, voxels_array.shape[2] + 2, voxels_array.shape[2] + 2])
    voxels[1:-1, 1:-1, 1:-1] = voxels_array

    mask = voxels > THRESHOLD

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

    arrays = [np.array(vertices).transpose().flatten()]

    x, y, z = np.where(~mask[:-1,:,:] & mask[1:,:,:])
    vertices = [
        x + 1, y + 1, z,
        x + 1, y, z,
        x + 1, y, z + 1,
        
        x + 1, y, z + 1,
        x + 1, y + 1, z + 1,
        x + 1, y + 1, z
    ]

    # Y
    arrays.append(np.array(vertices).transpose().flatten())

    x, y, z = np.where(mask[:,:-1,:] & ~mask[:,1:,:])
    vertices = [
        x + 1, y + 1, z,
        x, y + 1, z,
        x, y + 1, z + 1,

        x + 1, y + 1, z + 1,
        x + 1, y + 1, z,
        x, y + 1, z + 1
    ]
    arrays.append(np.array(vertices).transpose().flatten())

    x, y, z = np.where(~mask[:,:-1,:] & mask[:,1:,:])
    vertices = [
        x, y + 1, z,
        x + 1, y + 1, z,
        x, y + 1, z + 1,

        x + 1, y + 1, z,
        x + 1, y + 1, z + 1,
        x, y + 1, z + 1
    ]
    arrays.append(np.array(vertices).transpose().flatten())

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
    arrays.append(np.array(vertices).transpose().flatten())

    x, y, z = np.where(~mask[:,:,:-1] & mask[:,:,1:])
    vertices = [
        x + 1, y, z + 1,
        x, y, z + 1,
        x, y + 1, z + 1,

        x + 1, y + 1, z + 1,
        x + 1, y, z + 1,
        x, y + 1, z + 1
    ]
    arrays.append(np.array(vertices).transpose().flatten())

    return np.concatenate(arrays)




FILENAME = '/home/marian/shapenet/ShapeNetCore.v2/02747177/2f00a785c9ac57c07f48cf22f91f1702/models/model_normalized.solid.binvox'


from binvox_rw import read_as_3d_array

voxels = read_as_3d_array(open(FILENAME, 'rb'))
voxels = voxels.data[::2, ::2, ::2].astype(float) # 128^3 to 64^3, bool to float

'''voxels = np.array([
    [[0, 1, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
])'''

voxel_shape = [voxels.shape[0] + 2, voxels.shape[2] + 2, voxels.shape[2] + 2]
voxel_size = max(voxel_shape)

vertices = create_vertices(voxels)

main()