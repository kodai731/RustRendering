#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragColor;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out uint outObjectID;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(set = 1, binding = 1) uniform MaterialUBO {
    vec4 base_color;
    float metallic;
    float roughness;
    vec2 _padding;
} material;

layout(push_constant) uniform PushConstants {
    uint objectID;
    uint heatmapMode;
} pc;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);
    if (fragColor.a < 0.5) discard;

    vec3 albedoRGB;
    if (pc.heatmapMode == 1u) {
        albedoRGB = fragColor.rgb;
    } else {
        albedoRGB = texColor.rgb * fragColor.rgb * material.base_color.rgb;
    }

    outPosition = vec4(fragWorldPos, 1.0);
    outNormal = vec4(normalize(fragWorldNormal), 1.0);
    outAlbedo = vec4(albedoRGB, 1.0);
    outObjectID = pc.objectID;
}
