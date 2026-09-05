#version 450

#extension GL_GOOGLE_include_directive : require

#include "include/flame_ray.glsl"

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

#include "wind/include/wind_component.glsl"
#include "wind/include/wind_shell_field.glsl"
#include "wind/include/wind_shell_integral.glsl"
#include "wind/include/wind_reference_quadrature.glsl"

layout(set = 1, binding = 1) uniform sampler2D sceneDepthSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WindPush {
    int mode;
    int stepCount;
    int debugView;
} push;

const int WIND_MODE_CLOSED_FORM = 0;
const int WIND_MODE_REFERENCE_QUADRATURE = 1;
const int WIND_DEBUG_OFF = 0;
const int WIND_DEBUG_OPTICAL_DEPTH = 1;
const int WIND_DEBUG_KNOT_COUNT = 2;
const float SEGMENT_T_MAX = 1e4;

// Scene depth projected onto the view ray cuts the interval where an opaque surface occludes.
bool clampToSceneDepth(vec3 rayDir, inout float tFar, float tNear) {
    float sceneDepth = texture(sceneDepthSampler, fragTexCoord).r;
    if (sceneDepth == DEPTH_FAR) {
        return true;
    }
    vec4 surfaceClip = wind.invViewProj * vec4(fragTexCoord * 2.0 - 1.0, sceneDepth, 1.0);
    vec3 surfaceWorld = surfaceClip.xyz / surfaceClip.w;
    float tDepth = dot(surfaceWorld - frame.camera_pos.xyz, rayDir);
    if (tNear >= tDepth) {
        return false;
    }
    tFar = min(tFar, tDepth);
    return true;
}

void main() {
    vec3 rayDir = reconstructRayDirection(fragTexCoord, wind.invViewProj, frame.camera_pos.xyz);
    vec3 localOrigin = (wind.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    vec3 localDir = (wind.inverseModel * vec4(rayDir, 0.0)).xyz;

    float tNear = 0.0;
    float tFar = SEGMENT_T_MAX;
    if (!clampToWindCone(localOrigin, localDir, tNear, tFar)) {
        discard;
    }
    tNear = max(tNear, 0.0);
    if (!clampToSceneDepth(rayDir, tFar, tNear) || tFar <= tNear) {
        discard;
    }

    int knotCount = 0;
    float opticalDepth = push.mode == WIND_MODE_REFERENCE_QUADRATURE
        ? windReferenceOpticalDepth(localOrigin, localDir, tNear, tFar, push.stepCount)
        : windOpticalDepth(localOrigin, localDir, tNear, tFar, knotCount);
    float transmittance = rteTransmittanceFromOpticalDepth(opticalDepth);
    float coverage = 1.0 - transmittance;

    if (push.debugView == WIND_DEBUG_OPTICAL_DEPTH) {
        outColor = vec4(vec3(coverage), 1.0);
        return;
    }
    if (push.debugView == WIND_DEBUG_KNOT_COUNT) {
        outColor = vec4(float(knotCount) / float(WIND_MAX_KNOTS), 0.0, 0.0, 1.0);
        return;
    }

    vec3 scattered = wind.albedo.rgb * windSkyBrightness() * coverage;
    outColor = vec4(scattered, coverage);
}
