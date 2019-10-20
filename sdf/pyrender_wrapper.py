### Wrapper around the pyrender library that allows to
### 1. disable antialiasing
### 2. render a normal buffer
### This needs to be imported before pyrender is imported anywhere

# Disable antialiasing:
import OpenGL.GL

old_gl_enable = OpenGL.GL.glEnable

def new_gl_enable(value):
    if value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)

OpenGL.GL.glEnable = new_gl_enable

import pyrender

# Render a normal buffer instead of a color buffer
class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("sdf/shaders/mesh.vert", "sdf/shaders/mesh.frag", defines=defines)
        return self.program


def render_normal_and_depth_buffers(mesh, camera, camera_pose, resolution):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth = False))
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    renderer._renderer._program_cache = CustomShaderCache()

    color, depth = renderer.render(scene)
    return color, depth