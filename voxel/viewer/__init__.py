import pygame
from pygame.locals import *
from OpenGL.arrays import vbo
import pygame.image

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from voxel.viewer.voxels_to_mesh import create_vertices
from voxel.viewer.shader import Shader

import cv2
import skimage

from threading import Thread
import torch

class VoxelViewer():
    def __init__(self, size = (800, 800), start_thread = True, background_color = (0.01, 0.01, 0.01, 1.0)):
        self.size = size
        
        self.mouse = None
        self.rotation = (147, 30)

        self.vertex_buffer = None
        self.normal_buffer = None

        self.model_size = 1

        self.request_render = False

        self.running = True

        self.window = None

        self.background_color = background_color

        if start_thread:
            thread = Thread(target = self._run)
            thread.start()
        else:
            self._initialize_opengl()

    def _update_buffers(self, vertices, normals):
        if self.vertex_buffer is None:
            self.vertex_buffer = vbo.VBO(vertices)
        else:
            self.vertex_buffer.set_array(vertices)

        if self.normal_buffer is None:
            self.normal_buffer = vbo.VBO(normals)
        else:
            self.normal_buffer.set_array(normals)
        
        self.vertex_buffer_size = vertices.shape[0]
        self.request_render = True


    def set_voxels(self, voxels, use_marching_cubes=True, shade_smooth=False):
        if use_marching_cubes:
            if type(voxels) is torch.Tensor:
                voxels = voxels.cpu().numpy()
            voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
            voxel_size = voxels.shape[1]
            try:
                vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(1.0 / voxel_size, 1.0 / voxel_size, 1.0 / voxel_size))
                vertices = vertices[faces, :].astype(np.float32) - 0.5

                if shade_smooth:
                    normals = normals[faces, :].astype(np.float32)
                else:
                    normals = np.cross(vertices[:, 1, :] - vertices[:, 0, :], vertices[:, 2, :] - vertices[:, 0, :])
                    normals = np.repeat(normals, 3, axis=0)

                self._update_buffers(vertices.reshape((-1)), normals.reshape((-1)))            
                self.model_size = 1
            except ValueError:
                pass # Voxel array contains no sign change
        else:
            vertices, normals = create_vertices(voxels)
            vertices -= (voxels.shape[0] + 1) / 2
            self._update_buffers(vertices, normals)         
            self.model_size = max([voxels.shape[0] + 1, voxels.shape[1] + 1, voxels.shape[2] + 1])


    def set_mesh(self, mesh):
        vertices = np.array(mesh.triangles, dtype=np.float32).reshape(-1, 3)
        vertices = vertices.reshape((-1))

        normals = np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)
        
        self._update_buffers(vertices, normals)
        self.model_size = 1.5


    def _poll_mouse(self):
        left_mouse, _, right_mouse = pygame.mouse.get_pressed()
        pressed = left_mouse == 1 or right_mouse == 1
        current_mouse = pygame.mouse.get_pos()
        if self.mouse is not None and pressed:
            movement = (current_mouse[0] - self.mouse[0], current_mouse[1] - self.mouse[1])
            self.rotation = (self.rotation[0] + movement[0], max(-90, min(90, self.rotation[1] + movement[1])))
        self.mouse = current_mouse
        return pressed

    def _render(self):
        self.request_render = False
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.size[0]) / float(self.size[1]), 0.1, self.model_size * 4)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.model_size * 2)
        glRotatef(self.rotation[1], 1.0, 0, 0)
        glRotatef(self.rotation[0], 0, 1.0, 0)
        
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
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
        pygame.display.set_caption('Voxel Viewer')
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.window = pygame.display.set_mode(self.size, pygame.OPENGLBLIT)

        glEnable(GL_CULL_FACE)
        glClearColor(*self.background_color)
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
        
        self.__del__()

    def __del__(self):
        for buffer in [self.normal_buffer, self.vertex_buffer]:
            if buffer is not None:
                buffer.delete()

    def stop(self):
        self.running = False

    def get_image(self, crop = True, output_size = 128):
        if self.request_render:
            self._render()

        string_image = pygame.image.tostring(self.window, 'RGB')
        image = pygame.image.fromstring(string_image, self.size, 'RGB')
        array = np.transpose(pygame.surfarray.array3d(image)[:, :, 0])

        if crop:
            mask = array[:, :] != int(self.background_color[0] * 255)
            coords = np.array(np.nonzero(mask))
            
            if coords.size != 0:
                top_left = np.min(coords, axis=1)
                bottom_right = np.max(coords, axis=1)
            else:
                top_left = np.array((0, 0))
                bottom_right = np.array(array.shape)
                print("Warning: Image contains only white pixels.")
                
            half_size = int(max(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]) / 2)
            center = ((top_left + bottom_right) / 2).astype(int)
            center = (min(max(half_size, center[0]), array.shape[0] - half_size), min(max(half_size, center[1]), array.shape[1] - half_size))
            if half_size > 100:
                array = array[center[0] - half_size : center[0] + half_size, center[1] - half_size : center[1] + half_size]

        array = cv2.resize(array, dsize=(output_size, output_size), interpolation=cv2.INTER_CUBIC)

        return array        