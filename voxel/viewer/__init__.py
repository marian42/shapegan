import pygame
from pygame.locals import *
from OpenGL.arrays import vbo

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from voxel.viewer.voxels_to_mesh import create_vertices
from voxel.viewer.shader import Shader

from threading import Thread

class VoxelViewer():
    def __init__(self, size = (800, 600)):
        self.size = size
        
        self.mouse = None
        self.rotation = (45, 45)

        self.vertex_buffer = None
        self.normal_buffer = None

        self.voxel_shape = [1, 1, 1]
        self.voxel_size = 1

        self.request_render = False

        self.running = True

        thread = Thread(target = self._run)
        thread.start()

    def set_voxels(self, voxels):
        vertices, normals = create_vertices(voxels)
        
        if self.vertex_buffer is None:
            self.vertex_buffer = vbo.VBO(vertices)
        else:
            self.vertex_buffer.set_array(vertices)

        if self.normal_buffer is None:
            self.normal_buffer = vbo.VBO(normals)
        else:
            self.normal_buffer.set_array(normals)

        self.voxel_shape = [voxels.shape[0] + 2, voxels.shape[1] + 2, voxels.shape[2] + 2]
        self.voxel_size = max(self.voxel_shape)

        self.vertex_buffer_size = vertices.shape[0]
        self.request_render = True

    def _poll_mouse(self):
        _, _, pressed = pygame.mouse.get_pressed()
        current_mouse = pygame.mouse.get_pos()
        if self.mouse is not None:
            movement = (current_mouse[0] - self.mouse[0], current_mouse[1] - self.mouse[1])
            if pressed == 1:
                self.rotation = (self.rotation[0] + movement[0], max(-90, min(90, self.rotation[1] + movement[1])))
        self.mouse = current_mouse
        return pressed == 1

    def _render(self):
        self.request_render = False
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.size[0]) / float(self.size[1]), 0.1, self.voxel_size * 4)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.voxel_size * 2)
        glRotatef(self.rotation[1], 1.0, 0, 0)
        glRotatef(self.rotation[0], 0, 1.0, 0)
        glTranslatef(-(self.voxel_shape[0] - 1) / 2, -(self.voxel_shape[0] - 1) / 2, -(self.voxel_shape[0] - 1) / 2)        

        glClearColor(0.01, 0.01, 0.01, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor4d(1.0, 1.0, 0.0, 1.0)

        if self.vertex_buffer is not None and self.normal_buffer is not None:
            self.shader.use()

            glEnableClientState(GL_VERTEX_ARRAY)
            self.vertex_buffer.bind()
            glVertexPointer(3, GL_FLOAT, 0, self.vertex_buffer)

            glEnableClientState(GL_NORMAL_ARRAY)
            self.normal_buffer.bind()
            glNormalPointer(GL_FLOAT, 0, self.normal_buffer)

            glDrawArrays(GL_TRIANGLES, 0, self.vertex_buffer_size)

    def _initialize_opengl(self):
        pygame.init()
        pygame.display.set_mode(self.size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Voxel Viewer')

        glEnable(GL_CULL_FACE)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        self.shader = Shader()
        self.shader.initShader(open('voxel/viewer/vertex.glsl').read(), open('voxel/viewer/fragment.glsl').read())

    def _run(self):
        self._initialize_opengl()
        self._render()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if self._poll_mouse() or self.request_render:
                self._render()
            pygame.display.flip()
            
            pygame.time.wait(10)

    def stop(self):
        self.running = False


if __name__ == "__main__":    
    from voxels import load_voxels
    
    FILENAME = '/home/marian/shapenet/ShapeNetCore.v2/02691156/1bea1445065705eb37abdc1aa610476c/models/model_normalized.solid.binvox'
    voxels = load_voxels(FILENAME, 48)

    viewer = VoxelViewer()
    viewer.set_voxels(voxels)