#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#define WATER_UBO_SET 0
#define WATER_UBO_BINDING 2
#include "include/water_component.glsl"
#include "include/water_flow.glsl"
#include "include/water_surface.glsl"
#include "include/water_lb.glsl"
#include "include/water_torus_intersect.glsl"
#include "include/water_trace_payload.glsl"

layout(location = 0) rayPayloadInEXT WaterTracePayload payload;

void main() {
    float rHat = water.radii.y / water.radii.x;
    vec3 oLocal = gl_ObjectRayOriginEXT / water.radii.x;
    vec3 dLocal = normalize(gl_ObjectRayDirectionEXT);
    float roots[4];
    bool fallbackUsed;
    int hitCount = intersectTorus(oLocal, dLocal, rHat, roots, fallbackUsed);
    if (hitCount <= 0) { payload.color = vec4(0.0); payload.exitOrigin = vec4(0.0); return; }
    float chord = (hitCount >= 2) ? (roots[1] - roots[0]) * water.radii.x : 0.0;
    if (hitCount >= 4) {
        chord += (roots[3] - roots[2]) * water.radii.x;
    }
    vec3 pLocal1 = oLocal + roots[0] * dLocal;
    vec2 uv = torusUV(pLocal1);
    float h, hu, hv, slopeVariance;
    waterHeightAndGradient(uv, water.flow.z, water.flow.xy, int(water.composite.z), vec2(0.002), h, hu, hv, slopeVariance);
    waterLbHeightAndGradient(uv, water.flow.z, water.flow.xy, h, hu, hv);
    vec3 nLocal = waterPerturbedNormal(uv.x, uv.y, h, hu, hv, rHat);
    float eta = water.absorption.w;
    vec3 rayDirWorld = normalize(mat3(water.model) * dLocal);
    vec3 n = normalize(mat3(water.model) * nLocal);
    float cosThetaI = -dot(rayDirWorld, n);
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));
    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    float F = (rPar * rPar + rPerp * rPerp) * 0.5;

    vec3 dRefr = refract(dLocal, nLocal, 1.0 / eta);
    if (length(dRefr) < 1e-4) { dRefr = reflect(dLocal, nLocal); }
    float tExit = torusExitFromInside(pLocal1 + dRefr * 1e-3, dRefr, rHat);
    vec3 pExitLocal = (tExit > 0.0) ? pLocal1 + dRefr * (1e-3 + tExit) : pLocal1;
    vec3 nExit = normalize(torusGradient(pExitLocal, rHat));
    vec3 dExit = refract(dRefr, -nExit, eta);
    if (length(dExit) < 1e-4) { dExit = reflect(dRefr, nExit); }

    vec3 p1World = (water.model * vec4(pLocal1 * water.radii.x, 1.0)).xyz;
    vec3 pExitWorld = (water.model * vec4(pExitLocal * water.radii.x, 1.0)).xyz;
    payload.color = vec4(0.0, 0.0, 0.0, 1.0);
    payload.reflOrigin = vec4(p1World, F);
    payload.reflDir = vec4(normalize(mat3(water.model) * reflect(dLocal, nLocal)), chord);
    payload.exitOrigin = vec4(pExitWorld, 1.0);
    payload.exitDir = vec4(normalize(mat3(water.model) * dExit), slopeVariance);
}
