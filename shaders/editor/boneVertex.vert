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
    vec4 worldPos = object.model * vec4(inPosition, 1.0);
    gl_Position = frame.proj * frame.view * worldPos;
    fragColor = inColor;
}
