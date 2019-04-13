import pygame
from pygame.locals import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders


from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
    
def printOpenGLError():
    err = glGetError()
    if (err != GL_NO_ERROR):
        print('GLERROR: ', gluErrorString(err))
        sys.exit()

class Shader(object):
    def initShader(self, vertex_shader_source, fragment_shader_source):
        self.program = glCreateProgram()
        printOpenGLError()

        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, [vertex_shader_source])
        glCompileShader(self.vs)
        glAttachShader(self.program, self.vs)
        printOpenGLError()

        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, [fragment_shader_source])
        glCompileShader(self.fs)
        glAttachShader(self.program, self.fs)
        printOpenGLError()

        glLinkProgram(self.program)
        printOpenGLError()

    def use(self):
        if glUseProgram(self.program):
            printOpenGLError()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_CULL_FACE)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    vertexBuffer = vbo.VBO(vertices)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(display[0]) / float(display[1]), 0.1, voxel_size * 4)

    mouse = None
    rotation = (0, 0)

    shader = Shader()
    shader.initShader('''
void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
    ''',
    '''
void main()
{
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
    ''')

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
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -voxel_size * 2)
        glRotatef(rotation[1], 1.0, 0, 0)
        glRotatef(rotation[0], 0, 1.0, 0)
        glTranslatef(-(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2)
        
        mouse = current_mouse

        glClearColor(0.01, 0.01, 0.01, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor4d(1.0, 1.0, 0.0, 1.0)


        shader.use()    
        glEnableClientState(GL_VERTEX_ARRAY)
        vertexBuffer.bind()
        glVertexPointer(3, GL_FLOAT, 0, vertexBuffer)
        glDrawArrays(GL_TRIANGLES, 0, vertices.shape[0])        
        glUseProgram(0)

        glFlush()

        pygame.display.flip()
        pygame.time.wait(10)


def create_vertices(voxels_array):
    THRESHOLD = 0.5

    voxels = np.zeros([voxels_array.shape[0] + 2, voxels_array.shape[2] + 2, voxels_array.shape[2] + 2])
    voxels[1:-1, 1:-1, 1:-1] = voxels_array

    mask = voxels > THRESHOLD
    print(mask.shape)
    print(str(np.count_nonzero(mask)) + " voxels")

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

    result = np.concatenate(arrays).astype(np.float32)
    return result

FILENAME = '/home/marian/shapenet/ShapeNetCore.v2/02747177/2f00a785c9ac57c07f48cf22f91f1702/models/model_normalized.solid.binvox'


from binvox_rw import read_as_3d_array

voxels = read_as_3d_array(open(FILENAME, 'rb'))
voxels = voxels.data[::2, ::2, ::2].astype(np.float32) # 128^3 to 64^3, bool to float

voxel_shape = [voxels.shape[0] + 2, voxels.shape[2] + 2, voxels.shape[2] + 2]
voxel_size = max(voxel_shape)

vertices = create_vertices(voxels)
vertices = vertices

main()