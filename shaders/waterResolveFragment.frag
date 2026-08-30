#version 450

#include "include/flame_ray.glsl"

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

#include "include/water_component.glsl"
#include "include/water_torus_intersect.glsl"

layout(set = 1, binding = 1) uniform sampler2D sceneDepthSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPush {
    int secondaryRays;
    int debugView;
} push;

void main() {
    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    // Transform to local space: origin w=1, dir w=0
    vec3 pLocalOrigin = (water.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    pLocalOrigin /= water.radii.x;
    vec3 dLocal = (water.inverseModel * vec4(rayDir, 0.0)).xyz;
    dLocal = normalize(dLocal);

    // Intersect ray with torus
    float roots[4];
    int hitCount = intersectTorus(pLocalOrigin, dLocal, water.radii.y / water.radii.x, roots);

    if (hitCount == 0) {
        discard;
    }

    // Debug view: color by root count
    if (push.debugView == 1) {
        if (hitCount == 2) {
            outColor = vec4(0.0, 1.0, 0.0, 1.0);
        } else if (hitCount == 4) {
            outColor = vec4(0.0, 0.0, 1.0, 1.0);
        } else {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        return;
    }

    // First hit time in world units
    float t1 = roots[0] * water.radii.x;
    vec3 p1 = frame.camera_pos.xyz + t1 * rayDir;

    // Depth test against scene depth
    float sceneDepth = texture(sceneDepthSampler, fragTexCoord).r;
    float waterDepth = worldToClipDepth(p1, frame.view, frame.proj);
    if (sceneDepth > waterDepth) {
        discard;
    }

    // Compute chord length in world units
    float chord;
    if (hitCount >= 4) {
        chord = (roots[1] - roots[0]) * water.radii.x + (roots[3] - roots[2]) * water.radii.x;
    } else {
        chord = (roots[1] - roots[0]) * water.radii.x;
    }

    // Surface normal at first hit
    vec3 pLocal1 = pLocalOrigin + roots[0] * dLocal;
    vec3 n = normalize(mat3(water.model) * torusGradient(pLocal1, water.radii.y / water.radii.x));

    // Fresnel: Aqoole Reflectance P/S (average of parallel and perpendicular)
    float eta = water.absorption.w;
    float cosThetaI = -dot(rayDir, n);
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));

    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    float F = (rPar * rPar + rPerp * rPerp) * 0.5;

    // Reflection: constant environment + specular highlight
    vec3 reflDir = reflect(rayDir, n);
    vec3 lightDir = normalize(frame.light_pos.xyz - p1);
    float spec = pow(max(dot(reflDir, lightDir), 0.0), 64.0);
    vec3 reflection = vec3(0.6, 0.7, 0.8) + frame.light_color.rgb * spec;

    // Transmission: tint * exp(-absorption * chord)
    vec3 transmission = water.tint.rgb * exp(-water.absorption.rgb * chord);

    // Composite output
    outColor = vec4(F * reflection * water.composite.x + (1.0 - F) * transmission * water.composite.y, 1.0);
    gl_FragDepth = waterDepth;
}
