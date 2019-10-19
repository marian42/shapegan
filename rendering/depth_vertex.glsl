#version 330 core
in vec3 position;

uniform mat4 VP;

void main()
{
    gl_Position = VP * vec4(position, 1.0);
}