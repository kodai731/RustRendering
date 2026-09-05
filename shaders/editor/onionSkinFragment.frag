#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float ghostTintR;
    float ghostTintG;
    float ghostTintB;
    float ghostOpacity;
    int debugMode;
} pc;

void main() {
    if (pc.debugMode == 1) {
        vec3 normalizedPos = fract(fragWorldPos * 0.5 + 0.5);
        outColor = vec4(normalizedPos, 1.0);
        return;
    }

    if (pc.debugMode == 2) {
        float depth = gl_FragCoord.z;
        outColor = vec4(depth, depth, depth, 1.0);
        return;
    }

    if (pc.debugMode == 3) {
        vec3 n = normalize(fragWorldNormal) * 0.5 + 0.5;
        outColor = vec4(n, 1.0);
        return;
    }

    vec3 ghostColor = vec3(pc.ghostTintR, pc.ghostTintG, pc.ghostTintB);
    outColor = vec4(ghostColor, pc.ghostOpacity);
}
