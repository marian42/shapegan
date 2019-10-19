in vec3 normal;
in vec3 position;

in vec4 shadowPosition;
in vec3 lightPosition;

sampler2D shadow_map;

const float ambient = 0.5;
const float diffuse = 0.5;
const float specular = 0.3;

uniform float isFloor;
uniform vec3 albedo;

float isInShadow(vec2 uv, float reference_depth) {
    return reference_depth > texture(shadow_map, uv.xy).r ? 1.0 : 0.0;
}

float texture2DShadowLerp(vec2 uv, float reference_depth, float shadowTextureSize) {
    vec2 texelSize = vec2(1.0) / shadowTextureSize;
    vec2 f = fract(uv * shadowTextureSize + 0.5);
    vec2 centroidUV = floor(uv*shadowTextureSize + 0.5)/shadowTextureSize;

    float lb = isInShadow(centroidUV+texelSize * vec2(0.0, 0.0), reference_depth);
    float lt = isInShadow(centroidUV+texelSize * vec2(0.0, 1.0), reference_depth);
    float rb = isInShadow(centroidUV+texelSize * vec2(1.0, 0.0), reference_depth);
    float rt = isInShadow(centroidUV+texelSize * vec2(1.0, 1.0), reference_depth);
    float a = mix(lb, lt, f.y);
    float b = mix(rb, rt, f.y);
    return mix(a, b, f.x);
}

float getShadow(vec4 shadowPosition, vec3 lightDotNormal){
    vec3 shadow_coords = shadowPosition.xyz / shadowPosition.w;
    shadow_coords = shadow_coords * 0.5 + 0.5;

    if (shadow_coords.z > 1.0) {
        return 0.0;
    }

    float bias = max(0.002 * (1.0 - lightDotNormal), 0.001) / shadowPosition.w;
    float reference_depth = (shadow_coords.z - bias);
    vec2 shadowTextureSize = textureSize(shadow_map, 0);

    float result = 0.0;
    for(int x = -1; x <= 1; x++){
        for(int y= -1; y <= 1; y++){
            vec2 offset = vec2(x, y) / shadowTextureSize;
            result += texture2DShadowLerp(shadow_coords.xy + offset, reference_depth, shadowTextureSize);
        }
    }
    return clamp(result / 9.0, 0.0, 1.0);
}

void main() {
    normal = normalize(normal);
    vec3 viewDirection = normalize(-position);
    vec3 lightDirection = normalize(lightPosition - position);
    vec3 reflectDirection = -normalize(reflect(lightDirection, normal));
    vec3 lightDotNormal = clamp(dot(normal, lightDirection), 0.0, 1.0);

    float shadow = getShadow(shadowPosition, lightDotNormal);
    float rimLight = pow(1.0 - clamp(-normal.z, 0.0, 1.0), 4) * 0.3;
    
    vec3 color = albedo * ambient
        + albedo * diffuse * lightDotNormal * (1.0 - shadow)
        + vec3(1.0) * specular * pow(max(0.0, dot(reflectDirection, viewDirection)), 20) * (1.0 - shadow)
        + vec3(1.0) * rimLight;
    
    if (isFloor == 1.0) {
        color = mix(vec3(1.0), vec3(0.8) * ambient, shadow);
    }
    
    gl_FragColor = vec4(color.rgb, 1.0);
}