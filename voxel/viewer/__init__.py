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
import trimesh

from scipy.spatial.transform import Rotation
import os
import cv2

CLAMP_TO_EDGE = 33071
SHADOW_TEXTURE_SIZE = 1024

projection_matrix = np.array(
    [[ 1.73205081, 0,           0,           0,         ],
     [ 0,          1.73205081,  0,           0,         ],
     [ 0,          0,          -1.02020202, -0.2020202, ],
     [ 0,          0,          -1,           0,         ]], dtype=float)

def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_dcm()
    return matrix

def get_camera_transform(camera_distance, rotation_y, rotation_x):
    camera_pose = np.identity(4)
    camera_pose[2, 3] = -camera_distance
    camera_pose = np.matmul(camera_pose, get_rotation_matrix(rotation_x, axis='x'))
    camera_pose = np.matmul(camera_pose, get_rotation_matrix(rotation_y, axis='y'))

    camera_pose = np.matmul(projection_matrix, camera_pose)
    return camera_pose

def create_shadow_texture():
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_TEXTURE_SIZE, SHADOW_TEXTURE_SIZE, 0, GL_DEPTH_COMPONENT,
        GL_FLOAT, None
    )
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, CLAMP_TO_EDGE)
    glTexParameterfv(
        GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
        np.ones(4).astype(np.float32)
    )

    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id

class VoxelViewer():
    def __init__(self, size = 800, start_thread = True, background_color = (1, 1, 1, 1)):
        self.size = size
        
        self.mouse = None
        self.rotation = [147, 20]

        self.vertex_buffer = None
        self.normal_buffer = None

        self.model_size = 1

        self.request_render = False

        self.running = True

        self.window = None

        self.background_color = background_color

        self.shadow_framebuffer = None
        self.shadow_texture = None

        if start_thread:
            thread = Thread(target = self._run)
            thread.start()
        else:
            self._initialize_opengl()


        self.floor_vertices = None
        self.floor_normals = None

        self.ground_level = -1

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
                self.ground_level = np.min(vertices[:, 1]).item()         

                if shade_smooth:
                    normals = normals[faces, :].astype(np.float32)
                else:
                    normals = np.cross(vertices[:, 1, :] - vertices[:, 0, :], vertices[:, 2, :] - vertices[:, 0, :])
                    normals = np.repeat(normals, 3, axis=0)

                self._update_buffers(vertices.reshape((-1)), normals.reshape((-1)))
                self.model_size = 0.75
            except ValueError:
                pass # Voxel array contains no sign change
        else:
            vertices, normals = create_vertices(voxels)
            vertices -= (voxels.shape[0] + 1) / 2
            vertices /= voxels.shape[0] + 1
            self._update_buffers(vertices, normals)         
            self.model_size = max([voxels.shape[0] + 1, voxels.shape[1] + 1, voxels.shape[2] + 1])
            self.model_size = 0.75
            self.ground_level = np.min(vertices[1::3]).item()

    def set_mesh(self, mesh, smooth=False, center_and_scale=False):
        vertices = np.array(mesh.triangles, dtype=np.float32).reshape(-1, 3)
        
        if center_and_scale:
            vertices -= mesh.bounding_box.centroid[np.newaxis, :]
            vertices /= np.max(vertices)
        
        self.ground_level = np.min(vertices[:, 1]).item()
        vertices = vertices.reshape((-1))

        if smooth:
            normals = mesh.vertex_normals[mesh.faces.reshape(-1)].astype(np.float32) * -1
        else:
            normals = np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)
        
        self._update_buffers(vertices, normals)
        self.model_size = 1.5

    def _poll_mouse(self):
        left_mouse, _, right_mouse = pygame.mouse.get_pressed()
        pressed = left_mouse == 1 or right_mouse == 1
        current_mouse = pygame.mouse.get_pos()
        if self.mouse is not None and pressed:
            movement = (current_mouse[0] - self.mouse[0], current_mouse[1] - self.mouse[1])
            self.rotation = [self.rotation[0] + movement[0], max(-90, min(90, self.rotation[1] + movement[1]))]
        self.mouse = current_mouse
        return pressed

    def _render_shadow_texture(self, light_vp_matrix):
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_framebuffer)
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.shadow_texture, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, SHADOW_TEXTURE_SIZE, SHADOW_TEXTURE_SIZE)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

        self.depth_shader.use()
        self.depth_shader.set_vp_matrix(light_vp_matrix)
        self._draw_mesh(use_normals=False)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _draw_mesh(self, use_normals=True):
        if self.vertex_buffer is None or self.normal_buffer is None:
            return
        
        glEnableClientState(GL_VERTEX_ARRAY)
        self.vertex_buffer.bind()
        glVertexPointer(3, GL_FLOAT, 0, self.vertex_buffer)

        if use_normals:
            glEnableClientState(GL_NORMAL_ARRAY)
            self.normal_buffer.bind()
            glNormalPointer(GL_FLOAT, 0, self.normal_buffer)

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_buffer_size)

    def _draw_floor(self):
        self.shader.set_y_offset(self.ground_level)

        glEnableClientState(GL_VERTEX_ARRAY)
        self.floor_vertices.bind()
        glVertexPointer(3, GL_FLOAT, 0, self.floor_vertices)

        glEnableClientState(GL_NORMAL_ARRAY)
        self.floor_normals.bind()
        glNormalPointer(GL_FLOAT, 0, self.floor_normals)

        glDrawArrays(GL_TRIANGLES, 0, 6)
    
    def _render(self):
        self.request_render = False

        light_vp_matrix = get_camera_transform(6, 164, 50)
        self._render_shadow_texture(light_vp_matrix)
        
        self.shader.use()
        self.shader.set_floor(False)
        self.shader.set_y_offset(0)
        camera_vp_matrix = get_camera_transform(self.model_size * 2, self.rotation[0], self.rotation[1])
        self.shader.set_vp_matrix(camera_vp_matrix)
        self.shader.set_light_vp_matrix(light_vp_matrix)
        
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glClearDepth(1.0)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, self.size, self.size)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        self.shader.set_shadow_texture(1)
        
        self._draw_mesh()
        self.shader.set_floor(True)
        self._draw_floor()

    def _initialize_opengl(self):
        pygame.init()
        pygame.display.set_caption('Model Viewer')
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.window = pygame.display.set_mode((self.size, self.size), pygame.OPENGLBLIT)

        self.shader = Shader()
        self.shader.initShader(open('voxel/viewer/vertex.glsl').read(), open('voxel/viewer/fragment.glsl').read())

        self.shadow_framebuffer = glGenFramebuffers(1)
        self.shadow_texture = create_shadow_texture()

        self.depth_shader = Shader()
        self.depth_shader.initShader(open('voxel/viewer/depth_vertex.glsl').read(), open('voxel/viewer/depth_fragment.glsl').read())

        self.prepare_floor()

    def prepare_floor(self):
        size = 6
        mesh = trimesh.Trimesh([
            [-size, 0, -size],
            [-size, 0, +size],
            [+size, 0, +size],
            [-size, 0, -size],
            [+size, 0, +size],
            [+size, 0, -size]
            ], faces=[[0, 1, 2], [3, 4, 5]])

        vertices = np.array(mesh.triangles, dtype=np.float32).reshape(-1, 3)
        vertices = vertices.reshape((-1))
        normals = np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)
        
        self.floor_vertices = vbo.VBO(vertices)
        self.floor_normals = vbo.VBO(normals)

    def _run(self):
        self._initialize_opengl()
        self._render()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if pygame.key.get_pressed()[pygame.K_F12]:
                        self.save_screenshot()

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
        image = pygame.image.fromstring(string_image, (self.size, self.size), 'RGB')
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

        if output_size != self.size:
            array = cv2.resize(array, dsize=(output_size, output_size), interpolation=cv2.INTER_CUBIC)

        return array

    def save_screenshot(self):
        FILENAME_FORMAT = "screenshots/{:04d}.png"

        index = 0
        while os.path.isfile(FILENAME_FORMAT.format(index)):
            index += 1
        filename = FILENAME_FORMAT.format(index)
        image = self.get_image(crop=False, output_size=self.size)
        cv2.imwrite(filename, image)
        print("Screenshot saved to " + filename + ".")