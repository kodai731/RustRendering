#version 450

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

layout(set = 2, binding = 0) uniform ObjectUBO {
    mat4 model;
} object;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragColor;

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;

    fragWorldNormal = mat3(object.model) * inNormal;

    fragTexCoord = inTexCoord;
    fragColor = inColor;

    gl_Position = frame.proj * frame.view * worldPos;
}
