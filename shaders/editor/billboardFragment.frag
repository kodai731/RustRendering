#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/common.glsl"

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D positionSampler;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragBillboardWorldPos;
layout(location = 2) in float fragBillboardViewDepth;
layout(location = 3) in vec4 fragClipPos;

layout(location = 0) out vec4 outColor;

const float LINE_WIDTH = 0.045;
const float BULB_RADIUS = 0.30;
const vec2  BULB_CENTER = vec2(0.0, 0.12);
const float NECK_HALF_WIDTH = 0.11;
const float BASE_Y = -0.48;
const float RAY_START_OFFSET = 0.10;
const float RAY_LENGTH = 0.16;
const vec3  ICON_COLOR = vec3(0.0);

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdArc(vec2 p, vec2 center, float radius, float gapAngle) {
    vec2 rel = p - center;
    float ang = atan(rel.y, rel.x);

    float gapCenter = -PI / 2.0;
    float halfGap = gapAngle * 0.5;
    float diff = ang - gapCenter;
    if (diff > PI) diff -= TWO_PI;
    if (diff < -PI) diff += TWO_PI;

    if (abs(diff) < halfGap) {
        float edgeAngle1 = gapCenter + halfGap;
        float edgeAngle2 = gapCenter - halfGap;
        vec2 tip1 = center + radius * vec2(cos(edgeAngle1), sin(edgeAngle1));
        vec2 tip2 = center + radius * vec2(cos(edgeAngle2), sin(edgeAngle2));
        return min(length(p - tip1), length(p - tip2));
    }

    return abs(length(rel) - radius);
}

float generateBulbShape(vec2 p) {
    float d = 1e10;

    float gapAngle = 2.0 * asin(NECK_HALF_WIDTH / BULB_RADIUS);
    d = min(d, sdArc(p, BULB_CENTER, BULB_RADIUS, gapAngle));

    float junctionDy = sqrt(BULB_RADIUS * BULB_RADIUS - NECK_HALF_WIDTH * NECK_HALF_WIDTH);
    float junctionY = BULB_CENTER.y - junctionDy;

    vec2 leftTop  = vec2(-NECK_HALF_WIDTH, junctionY);
    vec2 leftBot  = vec2(-NECK_HALF_WIDTH * 0.82, BASE_Y);
    vec2 rightTop = vec2( NECK_HALF_WIDTH, junctionY);
    vec2 rightBot = vec2( NECK_HALF_WIDTH * 0.82, BASE_Y);
    d = min(d, sdSegment(p, leftTop, leftBot));
    d = min(d, sdSegment(p, rightTop, rightBot));

    float bw = NECK_HALF_WIDTH * 0.82;
    d = min(d, sdSegment(p, vec2(-bw, BASE_Y), vec2(bw, BASE_Y)));

    float lineStep = 0.07;
    for (int i = 1; i <= 2; i++) {
        float y = BASE_Y - lineStep * float(i);
        float w = bw * (1.0 - 0.2 * float(i));
        d = min(d, sdSegment(p, vec2(-w, y), vec2(w, y)));
    }

    return d;
}

float generateRays(vec2 p) {
    float d = 1e10;
    float rayStart = BULB_RADIUS + RAY_START_OFFSET;
    float rayEnd = rayStart + RAY_LENGTH;

    float angles[5] = float[5](
        1.57079632,
        2.35619449,
        0.78539816,
        PI,
        0.0
    );

    for (int i = 0; i < 5; i++) {
        float a = angles[i];
        vec2 dir = vec2(cos(a), sin(a));
        d = min(d, sdSegment(p, BULB_CENTER + dir * rayStart, BULB_CENTER + dir * rayEnd));
    }

    return d;
}

void main() {
    vec2 uv = fragTexCoord - 0.5;
    uv.y = -uv.y;
    vec2 p = uv * 2.0;

    float dBulb = generateBulbShape(p);
    float dRays = generateRays(p);
    float d = min(dBulb, dRays);

    float fw = fwidth(d) * 1.2;
    float halfW = LINE_WIDTH * 0.5;
    float shapeAlpha = 1.0 - smoothstep(halfW - fw, halfW + fw, d);

    if (shapeAlpha < 0.01) {
        discard;
    }

    vec3 ndc = fragClipPos.xyz / fragClipPos.w;
    vec2 screenUV = (ndc.xy + 1.0) * 0.5;

    vec4 gbufferPosition = texture(positionSampler, screenUV);

    vec4 gbufferViewPos = ubo.view * vec4(gbufferPosition.xyz, 1.0);
    float gbufferDepth = -gbufferViewPos.z;

    float maxRelevantDepth = fragBillboardViewDepth * 3.0;
    bool hasRelevantGeometry = gbufferPosition.w > 0.5 && gbufferDepth < maxRelevantDepth && gbufferDepth > 0.0;

    float alpha = shapeAlpha;
    if (hasRelevantGeometry) {
        bool billboardBehind = fragBillboardViewDepth > gbufferDepth;
        if (billboardBehind) {
            alpha *= 0.3;
        }
    }

    outColor = vec4(ICON_COLOR, alpha);
}
