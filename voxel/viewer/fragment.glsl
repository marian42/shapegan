in vec3 normal;

const vec3 lightDirection = vec3(0.7, 0.7, -0.14);
const float ambient = 0.2;
const float diffuse = 0.8;
const float specular = 0.3;
const vec3 viewDirection = vec3(0.0, 0.0, 1.0);

const vec3 albedo = vec3(0.4);
void main() {
    vec3 color = albedo * (ambient
        + diffuse * (0.5 + 0.5 * dot(lightDirection, normal))
        + specular * pow(max(0.0, dot(reflect(-lightDirection, normal), viewDirection)), 2.0));
    gl_FragColor = vec4(color.r, color.g, color.b, 1.0);
}