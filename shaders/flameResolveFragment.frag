#version 450

// F2 shading pass: reads F1 boundary integrals and composites premultiplied
// emission into the HDR buffer. push.mode: 0 = analytic height integral,
// 1 = reference raymarch (regression check for F1's z channel), 2 = delta-t debug view.

#include "include/chebyshev.glsl"
#include "include/flame_ray.glsl"

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
    float paddingReserved;
    vec4 colorBase;
    vec4 colorTip;
} flame;

layout(set = 1, binding = 2) uniform sampler2D flameAccumSampler;
layout(set = 1, binding = 3) uniform sampler2D flameIntervalSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform FlamePush {
    int mode;
    int stepCount;
} push;

const float INTERVAL_CLEAR_THRESHOLD = 1e37;
const float H_DIR_EPSILON = 1e-4;

float evaluateHeightFalloff(float height01) {
    return evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1], height01);
}

float evaluateHeightPrimitive(float height01) {
    return evaluateChebyshev12(
        flame.heightPrimitiveCoefficients[0],
        flame.heightPrimitiveCoefficients[1],
        flame.heightPrimitiveCoefficients[2],
        height01);
}

float integrateHeightByRaymarch(float tNear, float tFar, float hOrigin, float hDir) {
    float dt = (tFar - tNear) / float(push.stepCount);
    if (dt <= 0.0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < push.stepCount; ++i) {
        float t = tNear + (float(i) + 0.5) * dt;
        float h = clamp(evaluateHeightAlongRay(t, hOrigin, hDir), 0.0, 1.0);
        sum += evaluateHeightFalloff(h);
    }
    return sum * dt;
}

void main() {
    vec4 accum = texture(flameAccumSampler, fragTexCoord);
    vec2 interval = texture(flameIntervalSampler, fragTexCoord).xy;

    float coverage = accum.x;
    float deltaT = max(accum.y, 0.0);
    float heightIntegral = accum.z;

    if (push.mode == 2) {
        outColor = vec4(vec3(deltaT), 1.0);
        return;
    }

    if (coverage == 0.0 && deltaT <= 0.0) {
        discard;
    }

    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);
    float hOrigin = (flame.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).y;
    float hDir = (flame.inverseModel * vec4(rayDir, 0.0)).y;

    float tNear = interval.x > INTERVAL_CLEAR_THRESHOLD ? 0.0 : interval.x;
    float tFar = interval.y > INTERVAL_CLEAR_THRESHOLD ? tNear : -interval.y;
    tFar = max(tFar, tNear);

    // Camera inside the shell: front boundary terms at t = 0 were never
    // rasterized, so coverage = +N and the missing (-1) * H1(h_o) / h_d
    // terms are restored here (their t and delta-t contributions are zero).
    if (coverage > 0.5 && abs(hDir) > H_DIR_EPSILON) {
        float missingCount = round(coverage);
        heightIntegral -= missingCount * evaluateHeightPrimitive(clamp(hOrigin, 0.0, 1.0)) / hDir;
        tNear = 0.0;
    }

    float emission;
    if (push.mode == 1) {
        emission = integrateHeightByRaymarch(tNear, tFar, hOrigin, hDir);
    } else {
        emission = max(heightIntegral, 0.0);
    }

    float heightMid = clamp(evaluateHeightAlongRay(0.5 * (tNear + tFar), hOrigin, hDir), 0.0, 1.0);
    vec3 rampColor = mix(flame.colorBase.rgb, flame.colorTip.rgb, heightMid);

    vec3 radiance = rampColor * flame.intensity * emission;
    float alpha = 1.0 - exp(-flame.sigmaT * deltaT);
    outColor = vec4(radiance, alpha);
}
