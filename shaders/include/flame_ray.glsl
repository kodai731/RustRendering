// flame_ray.glsl - ray reconstruction and emission segment integration for flame passes

#ifndef FLAME_RAY_GLSL
#define FLAME_RAY_GLSL

#include "depth.glsl"
#include "radiative_transfer.glsl"

vec3 reconstructRayDirection(vec2 uv, mat4 invViewProj, vec3 cameraPos) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = invViewProj * vec4(ndc, DEPTH_NEAR, 1.0);
    return normalize(world.xyz / world.w - cameraPos);
}

float evaluateHeightAlongRay(float t, float hOrigin, float hDir) {
    return hOrigin + t * hDir;
}

#endif
