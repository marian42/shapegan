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

        try:
            glUseProgram(self.program)
        except GLError:
            err = glGetProgramInfoLog(self.program)
            print(err.decode("utf-8"))
            sys.exit()

    def use(self):
        glUseProgram(self.program)