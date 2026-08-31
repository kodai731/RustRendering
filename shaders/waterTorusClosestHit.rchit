#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#define WATER_UBO_SET 0
#define WATER_UBO_BINDING 2
#include "include/water_component.glsl"
#include "include/water_flow.glsl"
#include "include/water_surface.glsl"

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    vec3 pLocal = (gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT) / water.radii.x;
    float rHat = water.radii.y / water.radii.x;
    vec2 uv = torusUV(pLocal);
    float h, hu, hv, slopeVariance;
    waterHeightAndGradient(uv, water.flow.z, water.flow.xy, int(water.composite.z), vec2(0.002), h, hu, hv, slopeVariance);
    vec3 nLocal = waterPerturbedNormal(uv.x, uv.y, h, hu, hv, rHat);
    vec3 n = normalize(vec3(nLocal * gl_WorldToObjectEXT));
    vec3 lightDir = normalize(vec3(0.4, 0.8, 0.4));
    float ndotl = max(dot(n, lightDir), 0.0);
    payload = vec4(water.tint.rgb * (0.25 + 0.75 * ndotl), 1.0);
}
