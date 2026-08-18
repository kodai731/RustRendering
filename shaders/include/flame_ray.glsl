// flame_ray.glsl - ray reconstruction and emission segment integration for flame passes

#ifndef FLAME_RAY_GLSL
#define FLAME_RAY_GLSL

#include "depth.glsl"

vec3 reconstructRayDirection(vec2 uv, mat4 invViewProj, vec3 cameraPos) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = invViewProj * vec4(ndc, DEPTH_NEAR, 1.0);
    return normalize(world.xyz / world.w - cameraPos);
}

// S * (1 - exp(-sigma*dt)) / sigma with Taylor fallback so sigma -> 0 stays continuous.
// Mirrored in thyllore-render-core/src/flame.rs (integrate_emission_segment) for tests.
float integrateEmissionSegment(float source, float sigmaT, float dt) {
    float x = sigmaT * dt;
    if (x < 1e-3) {
        return source * dt * (1.0 - 0.5 * x + x * x * (1.0 / 6.0));
    }
    return source * (1.0 - exp(-x)) / sigmaT;
}

float evaluateHeightAlongRay(float t, float hOrigin, float hDir) {
    return hOrigin + t * hDir;
}

#endif
