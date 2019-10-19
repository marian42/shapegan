from pygame.locals import *
from OpenGL.GL import shaders
import sys

from OpenGL.GL import *
from OpenGL.GLU import *

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

        self.vp_location = None
        self.light_vp_location = None
        self.shadow_texture_location = None
        self.is_floor_location = None
        self.y_offset_location = None
        self.color_location = None

        try:
            glUseProgram(self.program)
        except GLError:
            err = glGetProgramInfoLog(self.program)
            print(err.decode("utf-8"))
            sys.exit()

    def set_light_vp_matrix(self, light_vp_matrix):
        if self.light_vp_location is None:
            self.light_vp_location = glGetUniformLocation(self.program, 'lightVP')

        glUniformMatrix4fv(self.light_vp_location, 1, GL_TRUE, light_vp_matrix)

    def set_vp_matrix(self, vp_matrix):
        if self.vp_location is None:
            self.vp_location = glGetUniformLocation(self.program, 'VP')

        glUniformMatrix4fv(self.vp_location, 1, GL_TRUE, vp_matrix)

    def set_shadow_texture(self, texture):
        if self.shadow_texture_location is None:
            self.shadow_texture_location = glGetUniformLocation(self.program, 'shadow_map')
        glUniform1iv(self.shadow_texture_location, 1, GL_TRUE, texture)

    def set_floor(self, is_floor):
        if self.is_floor_location is None:
            self.is_floor_location = glGetUniformLocation(self.program, 'isFloor')
        glUniform1fv(self.is_floor_location, 1, 1.0 if is_floor else 0.0)

    def set_color(self, color):
        if self.color_location is None:
            self.color_location = glGetUniformLocation(self.program, 'albedo')
        glUniform3fv(self.color_location, 1, color)

    def set_y_offset(self, value):
        if self.y_offset_location is None:
            self.y_offset_location = glGetUniformLocation(self.program, 'yOffset')
        glUniform1fv(self.y_offset_location, 1, value)


    def use(self):
        glUseProgram(self.program)