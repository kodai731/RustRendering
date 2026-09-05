#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float worldY;

layout(location = 0) out vec4 outColor;

void main() {
    bool isYAxisGrid = (fragColor.g > 0.9 && fragColor.r < 0.1 && fragColor.b < 0.1);

    if (isYAxisGrid) {
        if (worldY >= 0.0) {
            outColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            outColor = vec4(0.0, 1.0, 1.0, 1.0);
        }
    } else {
        outColor = vec4(fragColor, 1.0);
    }
}