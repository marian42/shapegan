varying out vec3 normal;
varying out vec3 position;

varying out vec4 shadowPosition;
varying out vec3 lightPosition;

uniform mat4 VP;
uniform mat4 lightVP;
uniform float yOffset;

void main() {
    vec3 vertexWithOffset = gl_Vertex + vec3(0.0, yOffset, 0.0);
    gl_Position = VP * vec4(vertexWithOffset, 1.0);
    position = gl_Position.xyz;

    shadowPosition = lightVP * vec4(vertexWithOffset, 1.0);
    lightPosition = (VP * inverse(lightVP) * vec4(0.0, 0.0, -1.0, 1.0)).xyz;

    normal = (VP * vec4(gl_Normal, 0.0)).xyz;
}