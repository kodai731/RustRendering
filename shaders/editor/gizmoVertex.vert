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
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    vec2 gizmoOffset = vec2(0.85, -0.75);
    float gizmoScale = 1.0;

    float aspect = abs(frame.proj[1][1]) / frame.proj[0][0];

    mat3 viewRotation = mat3(frame.view);
    vec3 rotatedPos = viewRotation * inPosition;

    vec3 flippedPos = vec3(rotatedPos.x / aspect, -rotatedPos.y, rotatedPos.z);
    vec4 position = vec4(flippedPos * gizmoScale, 1.0);
    position.xy += gizmoOffset;
    position.z = 0.0;

    gl_Position = position;
    fragColor = inColor;
}
