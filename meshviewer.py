import pygame
from pygame.locals import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import sys

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

class Shader(object):
    def initShader(self, vertex_shader_source, fragment_shader_source):
        self.program = glCreateProgram()

        self.vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vs, [vertex_shader_source])
        glAttachShader(self.program, self.vs)

        self.fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fs, [fragment_shader_source])
        glCompileShader(self.fs)
        glAttachShader(self.program, self.fs)

        glLinkProgram(self.program)

        try:
            glUseProgram(self.program)
        except GLError:
            err = glGetProgramInfoLog(self.program)
            print(err.decode("utf-8"))
            sys.exit()

    def use(self):
        glUseProgram(self.program)

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
    normalBuffer = vbo.VBO(normals)
    
    mouse = None
    rotation = (0, 0)

    shader = Shader()
    shader.initShader('''
        varying out vec3 normal;

        void main()
        {
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            normal = (gl_ModelViewProjectionMatrix * vec4(gl_Normal, 0.0)).xyz;
        }
    ''',
    '''
        in vec3 normal;

        const vec3 lightDirection = vec3(0.7, 0.7, -0.14);
        const float ambient = 0.2;
        const float diffuse = 0.8;
        const float specular = 0.3;
        const vec3 viewDirection = vec3(0.0, 0.0, 1.0);
        
        const vec3 albedo = vec3(0.5);
        void main() {
            vec3 color = albedo * (ambient
                + diffuse * (0.5 + 0.5 * dot(lightDirection, normal))
                + specular * pow(max(0.0, dot(reflect(-lightDirection, normal), viewDirection)), 2.0));
            gl_FragColor = vec4(color.r, color.g, color.b, 1.0);
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
        
        mouse = current_mouse
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(display[0]) / float(display[1]), 0.1, voxel_size * 4)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -voxel_size * 2)
        glRotatef(rotation[1], 1.0, 0, 0)
        glRotatef(rotation[0], 0, 1.0, 0)
        glTranslatef(-(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2, -(voxel_shape[0] - 1) / 2)        

        glClearColor(0.01, 0.01, 0.01, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor4d(1.0, 1.0, 0.0, 1.0)

        shader.use()

        glEnableClientState(GL_VERTEX_ARRAY)
        vertexBuffer.bind()
        glVertexPointer(3, GL_FLOAT, 0, vertexBuffer)

        glEnableClientState(GL_NORMAL_ARRAY)
        normalBuffer.bind()
        glNormalPointer(GL_FLOAT, 0, normalBuffer)

        glDrawArrays(GL_TRIANGLES, 0, vertices.shape[0])

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

    arrays.append(np.array(vertices).transpose().flatten())
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
    arrays.append(np.array(vertices).transpose().flatten())
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
    arrays.append(np.array(vertices).transpose().flatten())
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
    arrays.append(np.array(vertices).transpose().flatten())
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
    arrays.append(np.array(vertices).transpose().flatten())
    normals.append(np.tile(np.array([0, 0, -1]), 6 * x.shape[0]))

    return np.concatenate(arrays).astype(np.float32), np.concatenate(normals).astype(np.float32)

FILENAME = '/home/marian/shapenet/ShapeNetCore.v2/02747177/2f00a785c9ac57c07f48cf22f91f1702/models/model_normalized.solid.binvox'


from binvox_rw import read_as_3d_array

voxels = read_as_3d_array(open(FILENAME, 'rb'))
voxels = voxels.data[::2, ::2, ::2].astype(np.float32) # 128^3 to 64^3, bool to float

voxel_shape = [voxels.shape[0] + 2, voxels.shape[2] + 2, voxels.shape[2] + 2]
voxel_size = max(voxel_shape)

vertices, normals = create_vertices(voxels)

main()