#version 450

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(set = 1, binding = 1) uniform MaterialUBO {
    vec4 base_color;
    float metallic;
    float roughness;
    vec2 _padding;
} material;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);
    if (fragColor.a < 0.5) discard;

    vec3 lightDir = normalize(frame.light_pos.xyz - fragWorldPos);
    vec3 normal = normalize(fragNormal);

    float ambient = 0.2;
    float diffuse = max(dot(normal, lightDir), 0.0);
    float lighting = ambient + diffuse * 0.8;

    vec3 finalColor = texColor.rgb * fragColor.rgb * material.base_color.rgb * lighting * frame.light_color.rgb;
    outColor = vec4(finalColor, 1.0);
}
