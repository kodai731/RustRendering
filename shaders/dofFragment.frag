#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D depthSampler;

layout(push_constant) uniform PushConstants {
    float focalLengthMm;
    float apertureFStops;
    float sensorHeightMm;
    float focusDistance;
    float nearPlane;
    float maxBlurRadius;
    float viewportHeight;
    int enabled;
} pc;

float linearizeDepth(float depth) {
    return pc.nearPlane / max(depth, 0.0001);
}

float computeCoC(float depth) {
    float focalLength = pc.focalLengthMm * 0.001;
    float apertureDiameter = focalLength / max(pc.apertureFStops, 0.01);

    float numerator = apertureDiameter * focalLength * (pc.focusDistance - depth);
    float denominator = depth * (pc.focusDistance - focalLength);

    float cocWorld = abs(numerator / max(abs(denominator), 0.0001));
    float cocPixels = cocWorld * (pc.viewportHeight / (pc.sensorHeightMm * 0.001));

    return clamp(cocPixels, 0.0, pc.maxBlurRadius);
}

void main() {
    vec4 hdrColor = texture(hdrSampler, fragTexCoord);

    if (pc.enabled == 0) {
        outColor = hdrColor;
        return;
    }

    float depthValue = texture(depthSampler, fragTexCoord).r;
    float linearDepth = linearizeDepth(depthValue);
    float coc = computeCoC(linearDepth);

    if (coc < 0.5) {
        outColor = hdrColor;
        return;
    }

    vec2 texelSize = 1.0 / textureSize(hdrSampler, 0);
    float radius = min(coc, pc.maxBlurRadius);
    int sampleCount = int(clamp(radius * 2.0, 4.0, 32.0));

    vec3 accumColor = vec3(0.0);
    float totalWeight = 0.0;
    float angleStep = 6.28318530718 / float(sampleCount);

    for (int ring = 1; ring <= 3; ring++) {
        float ringRadius = radius * float(ring) / 3.0;
        for (int i = 0; i < sampleCount; i++) {
            float angle = float(i) * angleStep + float(ring) * 0.5;
            vec2 offset = vec2(cos(angle), sin(angle)) * ringRadius * texelSize;
            vec2 sampleUV = fragTexCoord + offset;

            float sampleDepth = texture(depthSampler, sampleUV).r;
            float sampleLinearDepth = linearizeDepth(sampleDepth);
            float sampleCoC = computeCoC(sampleLinearDepth);

            float weight = smoothstep(0.0, 1.0, sampleCoC / max(radius, 0.001));
            accumColor += texture(hdrSampler, sampleUV).rgb * weight;
            totalWeight += weight;
        }
    }

    float centerWeight = 1.0 - smoothstep(0.0, 2.0, coc);
    accumColor += hdrColor.rgb * centerWeight;
    totalWeight += centerWeight;

    vec3 blurredColor = accumColor / max(totalWeight, 0.001);
    float blendFactor = smoothstep(0.0, 2.0, coc);

    outColor = vec4(mix(hdrColor.rgb, blurredColor, blendFactor), hdrColor.a);
}
