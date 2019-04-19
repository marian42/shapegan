varying out vec3 normal;

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    normal = (gl_ModelViewProjectionMatrix * vec4(gl_Normal, 0.0)).xyz;
}