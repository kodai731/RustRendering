#version 450

layout(push_constant) uniform PushConstants {
    float alpha;
} pc;

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, pc.alpha);
}
