varying out vec3 normal;
varying out vec3 position;

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    position = gl_Position.xyz / gl_Position.w;
    normal = (gl_ModelViewProjectionMatrix * vec4(gl_Normal, 0.0)).xyz;
}