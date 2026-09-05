#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragBillboardWorldPos;
layout(location = 2) out float fragBillboardViewDepth;
layout(location = 3) out vec4 fragClipPos;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);

    vec4 viewPos = ubo.view * worldPos;

    gl_Position = ubo.proj * viewPos;

    fragTexCoord = inTexCoord;
    fragBillboardWorldPos = worldPos.xyz;
    fragBillboardViewDepth = -viewPos.z;
    fragClipPos = gl_Position;
}
