#version 450

// F2 shading pass. push.mode swaps HOW emission is integrated and nothing else:
// ray/interval decoding, camera-inside correction, color ramp and alpha live in
// the shared FlameRaySegment path, so analytic vs raymarch comparisons isolate
// the integration method alone regardless of mesh, colors, or future noise.
// push.mode: 0 = analytic boundary integral, 1 = reference raymarch,
// 2 = delta-t debug view, 3 = styled raymarch (IGN jitter + noise erosion).

#include "include/chebyshev.glsl"
#include "include/flame_ray.glsl"
#include "include/flame_noise.glsl"

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
    vec4 colorMid;
    vec4 colorTip;
    vec4 temporalData;
} flame;

layout(set = 1, binding = 2) uniform sampler2D flameAccumSampler;
layout(set = 1, binding = 3) uniform sampler2D flameIntervalSampler;
layout(set = 1, binding = 4) uniform sampler2D flameHistorySampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

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

struct FlameRaySegment {
    float tNear;
    float tFar;
    vec3 localOrigin;
    vec3 localDir;
    float boundaryHeightIntegral;
};

FlameRaySegment buildRaySegment(float coverage, float heightIntegral, vec2 interval) {
    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    FlameRaySegment segment;
    segment.localOrigin = (flame.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    segment.localDir = (flame.inverseModel * vec4(rayDir, 0.0)).xyz;
    segment.tNear = interval.x > INTERVAL_CLEAR_THRESHOLD ? 0.0 : interval.x;
    float tFar = interval.y > INTERVAL_CLEAR_THRESHOLD ? segment.tNear : -interval.y;
    segment.tFar = max(tFar, segment.tNear);
    segment.boundaryHeightIntegral = heightIntegral;

    // Camera inside the shell: front boundary terms at t = 0 were never
    // rasterized, so coverage = +N and the missing (-1) * H1(h_o) / h_d
    // terms are restored here (their t and delta-t contributions are zero).
    if (coverage > 0.5 && abs(segment.localDir.y) > H_DIR_EPSILON) {
        float missingCount = round(coverage);
        segment.boundaryHeightIntegral -= missingCount
            * evaluateHeightPrimitive(clamp(segment.localOrigin.y, 0.0, 1.0)) / segment.localDir.y;
        segment.tNear = 0.0;
    }
    return segment;
}

float integrateEmissionAnalytic(FlameRaySegment segment) {
    return max(segment.boundaryHeightIntegral, 0.0);
}

float integrateEmissionRaymarch(FlameRaySegment segment, int stepCount) {
    float dt = (segment.tFar - segment.tNear) / float(stepCount);
    if (dt <= 0.0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + 0.5) * dt;
        float h = clamp(
            evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        sum += evaluateHeightFalloff(h);
    }
    return sum * dt;
}

float sampleNoiseErodedDensity(vec3 localPos, float height01) {
    vec3 samplePos = localPos * flame.noiseFrequency
        - vec3(0.0, flame.time * flame.noiseScrollSpeed, 0.0);
    float turbulence = fbm3(samplePos);
    float erosion = flame.noiseAmplitude * mix(0.2, 1.0, height01) * turbulence;
    return max(evaluateHeightFalloff(height01) - erosion, 0.0);
}

float integrateEmissionNoiseRaymarch(FlameRaySegment segment, int stepCount) {
    float dt = (segment.tFar - segment.tNear) / float(stepCount);
    if (dt <= 0.0) {
        return 0.0;
    }
    float jitter = interleavedGradientNoise(gl_FragCoord.xy + vec2(flame.temporalData.y * 5.588238));
    float sum = 0.0;
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + jitter) * dt;
        vec3 localPos = segment.localOrigin + t * segment.localDir;
        float h = clamp(localPos.y, 0.0, 1.0);
        sum += sampleNoiseErodedDensity(localPos, h);
    }
    return sum * dt;
}

vec4 shadeEmission(FlameRaySegment segment, float emission, float deltaT) {
    float heightMid = clamp(
        evaluateHeightAlongRay(
            0.5 * (segment.tNear + segment.tFar), segment.localOrigin.y, segment.localDir.y),
        0.0, 1.0);
    vec3 rampColor;
    if (heightMid < 0.5) {
        rampColor = mix(flame.colorBase.rgb, flame.colorMid.rgb, heightMid * 2.0);
    } else {
        rampColor = mix(flame.colorMid.rgb, flame.colorTip.rgb, (heightMid - 0.5) * 2.0);
    }

    vec3 radiance = rampColor * flame.intensity * emission;
    float alpha = 1.0 - exp(-flame.sigmaT * emission);
    return vec4(radiance, alpha);
}

void main() {
    vec4 accum = texture(flameAccumSampler, fragTexCoord);
    vec2 interval = texture(flameIntervalSampler, fragTexCoord).xy;

    float coverage = accum.x;
    float deltaT = max(accum.y, 0.0);

    if (push.mode == 2) {
        outColor = vec4(max(accum.z, 0.0), deltaT, max(-accum.z, 0.0), 1.0);
        return;
    }

    if (coverage == 0.0 && deltaT <= 0.0) {
        discard;
    }

    FlameRaySegment segment = buildRaySegment(coverage, accum.z, interval);

    float emission;
    if (push.mode == 3) {
        emission = integrateEmissionNoiseRaymarch(segment, push.stepCount);
    } else if (push.mode == 1) {
        emission = integrateEmissionRaymarch(segment, push.stepCount);
    } else {
        emission = integrateEmissionAnalytic(segment);
    }

    vec4 shaded = shadeEmission(segment, emission, deltaT);
    vec4 blended = mix(shaded, texture(flameHistorySampler, fragTexCoord), flame.temporalData.x);
    outColor = blended;
    outHistory = blended;
}
