#version 450

// Extrudes the 4-corner base quad into a closed prism shell (rings x stacks + caps).
// Winding is outward-CCW everywhere; the Rust mirror in
// thyllore-render-core/src/flame_shell.rs must stay identical (verified by winding tests).

#include "include/flame_shell_profile.glsl"

layout(lines_adjacency, invocations = 10) in;
layout(triangle_strip, max_vertices = 48) out;

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

layout(set = 1, binding = 0) uniform FlameUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 heightPrimitiveCoefficients[3];
    vec4 radialCoefficients[2];
    vec4 heightCoefficients[2];
    float time;
    float sigmaT;
    float intensity;
    float heightAxisScale;
    float noiseAmplitude;
    float noiseFrequency;
    float noiseScrollSpeed;
    float radialSharpness;
    vec4 colorBase;
    vec4 colorMid;
    vec4 colorTip;
    vec4 temporalData;
    vec4 lightData;
    vec4 styleParams0;
    vec4 styleParams1;
    vec4 styleParams2;
    mat4 trailUnitInverse;
    vec4 trailMeta;
    vec4 trailSamples[16];
    vec4 emitterParams;
    vec4 contourParams;
    vec4 erosionResponse;
    vec4 kernelParams;
    vec4 kernelBlobs[192];
    vec4 boundaryParams;
    vec4 nearFadeParams;
    vec4 radiusCoefficients[2];
    vec4 colorRamp[8];
    vec4 profileParams;
    vec4 heightMonomial[2];
    vec4 envelopeKnots[2];
    vec4 waveParams;
    vec4 waveModes[352];
} flame;

#include "include/flame_shell_support.glsl"

layout(location = 0) in vec3 geomLocalCorner[];

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec2 fragShellUv;

const int RING_SEGMENTS = 16;
const int STACKS = 8;
const float TWO_PI = 6.283185307179586;

vec3 computeRingPosition(vec3 center, float radiusX, float radiusZ, int segment, int stack) {
    float height01 = float(stack) / float(STACKS);
    float radiusScale = flameShellRadiusScale(height01, flameShellSupportScale());
    float angle = TWO_PI * float(segment) / float(RING_SEGMENTS);
    vec3 pos = center + vec3(cos(angle) * radiusX * radiusScale, height01, sin(angle) * radiusZ * radiusScale);
    pos.xz += flame.styleParams2.xy * flame.styleParams2.z * pow(height01, flame.styleParams2.w);
    return pos;
}

void emitShellVertex(vec3 localPos, vec2 shellUv) {
    vec4 worldPos = flame.model * vec4(localPos, 1.0);
    fragWorldPos = worldPos.xyz;
    fragShellUv = shellUv;
    gl_Position = frame.proj * frame.view * worldPos;
    EmitVertex();
}

void main() {
    vec3 center = (geomLocalCorner[0] + geomLocalCorner[1] + geomLocalCorner[2] + geomLocalCorner[3]) * 0.25;
    float radiusX = length(geomLocalCorner[1] - geomLocalCorner[0]) * 0.5;
    float radiusZ = length(geomLocalCorner[3] - geomLocalCorner[0]) * 0.5;

    if (gl_InvocationID < STACKS) {
        int stack = gl_InvocationID;
        float bottom01 = float(stack) / float(STACKS);
        float top01 = float(stack + 1) / float(STACKS);
        for (int i = 0; i <= RING_SEGMENTS; ++i) {
            int segment = i % RING_SEGMENTS;
            float angle01 = float(i) / float(RING_SEGMENTS);
            emitShellVertex(
                computeRingPosition(center, radiusX, radiusZ, segment, stack),
                vec2(angle01, bottom01));
            emitShellVertex(
                computeRingPosition(center, radiusX, radiusZ, segment, stack + 1),
                vec2(angle01, top01));
        }
        EndPrimitive();
    } else if (gl_InvocationID == 8) {
        for (int i = 0; i < RING_SEGMENTS; ++i) {
            int next = (i + 1) % RING_SEGMENTS;
            emitShellVertex(center, vec2(0.0, 0.0));
            emitShellVertex(computeRingPosition(center, radiusX, radiusZ, i, 0), vec2(float(i) / float(RING_SEGMENTS), 0.0));
            emitShellVertex(computeRingPosition(center, radiusX, radiusZ, next, 0), vec2(float(i + 1) / float(RING_SEGMENTS), 0.0));
            EndPrimitive();
        }
    } else {
        vec3 topCenter = center + vec3(0.0, 1.0, 0.0);
        for (int i = 0; i < RING_SEGMENTS; ++i) {
            int next = (i + 1) % RING_SEGMENTS;
            emitShellVertex(topCenter, vec2(0.0, 1.0));
            emitShellVertex(computeRingPosition(center, radiusX, radiusZ, next, STACKS), vec2(float(i + 1) / float(RING_SEGMENTS), 1.0));
            emitShellVertex(computeRingPosition(center, radiusX, radiusZ, i, STACKS), vec2(float(i) / float(RING_SEGMENTS), 1.0));
            EndPrimitive();
        }
    }
}
