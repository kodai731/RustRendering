#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // 頂点カラーをそのまま出力
    outColor = vec4(fragColor, 1.0);
}
