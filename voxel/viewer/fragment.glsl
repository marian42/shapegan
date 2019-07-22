in vec3 normal;
in vec3 position;

const vec3 lightPosition = vec3(-5, 10, -20);
const float ambient = 0.3;
const float diffuse = 0.8;
const float specular = 0.6;
const vec3 albedo = vec3(0.9);

void main() {
    normal = normalize(normal);
    vec3 viewDirection = normalize(-position);
    vec3 lightDirection = normalize(lightPosition - position);
    vec3 reflectDirection = -normalize(reflect(lightDirection, normal));

    vec3 color = albedo * (ambient
        + diffuse * max(0.0, dot(lightDirection, normal))
        + specular * pow(max(0.0, dot(reflectDirection, viewDirection)), 20));
    gl_FragColor = vec4(color.r, color.g, color.b, 1.0);
}