#version 450

// F1 thickness pass: signed boundary integration into two RTs.
// A0 (additive): x = sum(sign), y = sum(sign*t), z = sum(sign*H1(h)/h_d), w = reserved
// A1 (MIN blend): x = min(t) = t_near, y = min(-t) = -t_far

#include "include/chebyshev.glsl"

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
} flame;

layout(set = 1, binding = 1) uniform sampler2D gbufferPositionSampler;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragShellUv;

layout(location = 0) out vec4 outAccumulate;
layout(location = 1) out vec2 outRayInterval;

const float H_DIR_EPSILON = 1e-3;

void main() {
    vec3 toFragment = fragWorldPos - frame.camera_pos.xyz;
    float tBoundary = length(toFragment);
    vec3 rayDir = toFragment / tBoundary;

    vec2 screenUv = gl_FragCoord.xy / vec2(textureSize(gbufferPositionSampler, 0));
    vec4 scenePosition = texture(gbufferPositionSampler, screenUv);
    float tClamped = tBoundary;
    if (scenePosition.w > 0.5) {
        float tScene = length(scenePosition.xyz - frame.camera_pos.xyz);
        tClamped = min(tBoundary, tScene);
    }

    float boundarySign = gl_FrontFacing ? -1.0 : 1.0;

    vec3 boundaryPoint = frame.camera_pos.xyz + rayDir * tClamped;
    float hBoundary = clamp((flame.inverseModel * vec4(boundaryPoint, 1.0)).y, 0.0, 1.0);
    float hDir = (flame.inverseModel * vec4(rayDir, 0.0)).y;

    float heightIntegralTerm;
    if (abs(hDir) > H_DIR_EPSILON) {
        float primitive = evaluateChebyshev12(
            flame.heightPrimitiveCoefficients[0],
            flame.heightPrimitiveCoefficients[1],
            flame.heightPrimitiveCoefficients[2],
            hBoundary);
        heightIntegralTerm = boundarySign * primitive / hDir;
    } else {
       float falloff = evaluateChebyshev8(
            flame.heightCoefficients[0],
            flame.heightCoefficients[1],
            hBoundary);
        heightIntegralTerm = boundarySign * falloff * tClamped;
    }

    outAccumulate = vec4(boundarySign, boundarySign * tClamped, heightIntegralTerm, 0.0);
    outRayInterval = vec2(tClamped, -tClamped);
}
