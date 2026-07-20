#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/depth.glsl"

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(depth_any) out float gl_FragDepth;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D bloomSampler;
layout(binding = 2) uniform sampler2D positionSampler;

layout(binding = 3) uniform SceneData {
    vec4 lightPosition;
    vec4 lightColor;
    mat4 view;
    mat4 proj;
    int debugMode;
    float shadowStrength;
    int enableDistanceAttenuation;
    float exposureValue;
} sceneData;

layout(push_constant) uniform PushConstants {
    int toneMapOperator;
    float gamma;
    float exposureValue;
    float vignetteIntensity;
    float chromaticAberrationIntensity;
    float bloomIntensity;
} pc;

vec3 acesFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 reinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

vec3 applyToneMapOperator(vec3 x) {
    if (pc.toneMapOperator == 1) {
        return acesFilmic(x);
    }
    if (pc.toneMapOperator == 2) {
        return reinhard(x);
    }
    return clamp(x, 0.0, 1.0);
}

const vec3 BACKGROUND_LINEAR = vec3(0.051);

vec3 sampleWithChromaticAberration(vec2 uv, float intensity) {
    vec2 center = vec2(0.5);
    vec2 offset = (uv - center) * intensity;

    float r = texture(hdrSampler, uv + offset).r;
    float g = texture(hdrSampler, uv).g;
    float b = texture(hdrSampler, uv - offset).b;

    return vec3(r, g, b);
}

float computeVignette(vec2 uv, float intensity) {
    vec2 d = uv - vec2(0.5);
    return 1.0 - intensity * dot(d, d) * 4.0;
}

void main() {
    vec4 positionData = texture(positionSampler, fragTexCoord);
    if (positionData.w < 0.5) {
        gl_FragDepth = DEPTH_FAR;
    } else {
        gl_FragDepth = worldToClipDepth(positionData.xyz, sceneData.view, sceneData.proj);
    }

    bool isBackground = positionData.w < 0.5;

    // Background pixels still carry volumetric effects (flame) blended into
    // the HDR buffer as premultiplied (radiance, coverage): attenuate the
    // background by the coverage and add the tonemapped radiance on top.
    if (isBackground) {
        vec4 volumetric = texture(hdrSampler, fragTexCoord);
        float volumetricOpacity = clamp(volumetric.a, 0.0, 1.0);
        vec3 volumetricGlow = applyToneMapOperator(volumetric.rgb * pc.exposureValue);
        volumetricGlow = pow(volumetricGlow, vec3(1.0 / pc.gamma));

        vec3 bg = BACKGROUND_LINEAR * (1.0 - volumetricOpacity);
        bg = pow(bg, vec3(1.0 / pc.gamma)) + volumetricGlow;

        if (pc.bloomIntensity > 0.0) {
            vec3 bloomColor = texture(bloomSampler, fragTexCoord).rgb;
            vec3 bloomGlow = bloomColor * pc.bloomIntensity * pc.exposureValue;
            bloomGlow = applyToneMapOperator(bloomGlow);
            bg += pow(bloomGlow, vec3(1.0 / pc.gamma));
        }

        outColor = vec4(bg, 1.0);
        return;
    }

    vec3 hdrColor;
    if (pc.chromaticAberrationIntensity > 0.0) {
        hdrColor = sampleWithChromaticAberration(fragTexCoord, pc.chromaticAberrationIntensity);
    } else {
        hdrColor = texture(hdrSampler, fragTexCoord).rgb;
    }

    if (pc.bloomIntensity > 0.0) {
        vec3 bloomColor = texture(bloomSampler, fragTexCoord).rgb;
        hdrColor += bloomColor * pc.bloomIntensity;
    }

    hdrColor *= pc.exposureValue;

    vec3 mapped = applyToneMapOperator(hdrColor);
    mapped = pow(mapped, vec3(1.0 / pc.gamma));

    if (pc.vignetteIntensity > 0.0) {
        float vignette = computeVignette(fragTexCoord, pc.vignetteIntensity);
        mapped *= vignette;
    }

    outColor = vec4(mapped, 1.0);
}
