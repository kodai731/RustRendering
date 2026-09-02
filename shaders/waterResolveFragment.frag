#version 460

#extension GL_GOOGLE_include_directive : require
#ifdef WATER_RAY_QUERY
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

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
#include "include/water_flow.glsl"
#include "include/water_surface.glsl"
#include "include/water_lb.glsl"
#include "include/water_lighting.glsl"

layout(set = 1, binding = 1) uniform sampler2D sceneColorSampler;

#ifdef WATER_RAY_QUERY
layout(set = 1, binding = 2) uniform accelerationStructureEXT sceneTlas;

struct HitShadingRecord { uint64_t vertexAddress; uint64_t indexAddress; mat4 model; mat4 normalMatrix; vec4 baseColor; vec4 params; };
layout(set = 1, binding = 3, std430) readonly buffer HitShadingTable { HitShadingRecord records[]; } hitTable;

layout(set = 1, binding = 4) uniform sampler2D waterHistorySampler;
layout(set = 1, binding = 5) uniform sampler2D waterTraceSampler;
#endif

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
#ifdef WATER_RAY_QUERY
layout(location = 1) out vec4 outHistory;
#endif

layout(push_constant) uniform WaterPush {
    int secondaryRays;
    int debugView;
} push;
#ifdef WATER_RAY_QUERY
#include "include/water_secondary.glsl"
#endif
void main() {
   mat4 invViewProj = water.invViewProj;
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    // Transform to local space: origin w=1, dir w=0
    vec3 pLocalOrigin = (water.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    pLocalOrigin /= water.radii.x;
    vec3 dLocal = (water.inverseModel * vec4(rayDir, 0.0)).xyz;
    dLocal = normalize(dLocal);

    // Intersect ray with torus
    float roots[4];
    bool fallbackUsed;
    int hitCount = intersectTorus(pLocalOrigin, dLocal, water.radii.y / water.radii.x, roots, fallbackUsed);

    if (hitCount == 0) {
        discard;
    }

    // First hit time in world units
    float t1 = roots[0] * water.radii.x;
    vec3 p1 = frame.camera_pos.xyz + t1 * rayDir;
    float waterDepth = worldToClipDepth(p1, frame.view, frame.proj);
    gl_FragDepth = waterDepth;

    // Debug view: color by root count
    if (push.debugView == 1) {
        if (hitCount == 2) {
            outColor = vec4(0.0, 1.0, 0.0, 1.0);
        } else if (hitCount == 4) {
            outColor = vec4(0.0, 0.0, 1.0, 1.0);
        } else {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
#ifdef WATER_RAY_QUERY
        outHistory = outColor;
#endif
        return;
   }

#ifdef WATER_RAY_QUERY
    if (push.debugView == 5) {
        outColor = vec4(texture(waterTraceSampler, fragTexCoord).rgb, 1.0);
#ifdef WATER_RAY_QUERY
        outHistory = outColor;
#endif
        return;
    }
#endif

   // Debug view: torus intersection probe (nearest root, high-precision encoding)
    if (push.debugView == 3 || push.debugView == 4) {
        float t = (push.debugView == 3) ? roots[0] * water.radii.x : roots[1] * water.radii.x;
        float hi = floor(t);
        float mid = floor(fract(t) * 1024.0);
        float lo = fract(t * 1024.0);
        float marker = -(float(hitCount) + (fallbackUsed ? 10.0 : 0.0));
        outColor = vec4(hi, mid, lo, marker);
#ifdef WATER_RAY_QUERY
        outHistory = outColor;
#endif
        return;
    }

    // Compute chord length in world units
    float chord;
    if (hitCount >= 4) {
        chord = (roots[1] - roots[0]) * water.radii.x + (roots[3] - roots[2]) * water.radii.x;
    } else {
        chord = (roots[1] - roots[0]) * water.radii.x;
    }

    // Surface normal at first hit via analytic wave gradient
    vec3 pLocal1 = pLocalOrigin + roots[0] * dLocal;
    float rHat = water.radii.y / water.radii.x;
    vec2 uv = torusUV(pLocal1);

    float du_dx = dFdx(uv.x);
    float du_dy = dFdy(uv.x);
    float dv_dx = dFdx(uv.y);
    float dv_dy = dFdy(uv.y);
    vec2 footprint = vec2(length(vec2(du_dx, du_dy)), length(vec2(dv_dx, dv_dy)));

    if (abs(du_dx) > 3.0) {
        footprint.x = 0.0;
    }
    if (any(isnan(footprint)) || any(isinf(footprint))) { footprint = vec2(0.0); }

    float h, hu, hv, var;
    waterHeightAndGradient(uv, water.flow.z, water.flow.xy, int(water.composite.z), footprint, h, hu, hv, var);
    waterLbHeightAndGradient(uv, water.flow.z, water.flow.xy, h, hu, hv);
    vec3 nLocal = waterPerturbedNormal(uv.x, uv.y, h, hu, hv, rHat);
    vec3 n = normalize(mat3(water.model) * nLocal);

    // Debug view: normal visualization
   if (push.debugView == 2) {
        outColor = vec4(n * 0.5 + 0.5, 1.0);
#ifdef WATER_RAY_QUERY
        outHistory = outColor;
#endif
        return;
    }

    // Fresnel: Aqoole Reflectance P/S (average of parallel and perpendicular)
    float eta = water.absorption.w;
    float cosThetaI = -dot(rayDir, n);
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));

    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    float F = (rPar * rPar + rPerp * rPerp) * 0.5;

  // Reflection
    vec3 reflDir = reflect(rayDir, n);
    vec3 reflection;
#ifdef WATER_RAY_QUERY
    if (push.secondaryRays == 0) {
        // RayQuery path: compute tTorusNext (next torus intersection along reflection ray)
        vec3 reflDirLocal = normalize((water.inverseModel * vec4(reflDir, 0.0)).xyz);
        vec3 reflOriginLocal = pLocal1 + reflDirLocal * 1e-3;

        float reflRoots[4];
        bool reflFallback;
        int reflCount = intersectTorus(reflOriginLocal, reflDirLocal, rHat, reflRoots, reflFallback);
        float tTorusNext = (reflCount > 0) ? reflRoots[0] * water.radii.x : 1e30;

        vec3 rayColor;
        float tScene;
        if (traceScene(p1, reflDir, tTorusNext, rayColor, tScene)) {
            reflection = rayColor;
        } else if (tTorusNext < 1e30) {
            // Depth-2: reflection ray re-entered the torus — Fresnel probabilistic selection
            vec3 p2 = p1 + reflDir * tTorusNext;
            vec3 pLocal2 = (water.inverseModel * vec4(p2, 1.0)).xyz / water.radii.x;
            vec2 uv2 = torusUV(pLocal2);
            float h2, hu2, hv2, var2;
            waterHeightAndGradient(uv2, water.flow.z, water.flow.xy, int(water.composite.z), footprint, h2, hu2, hv2, var2);
            waterLbHeightAndGradient(uv2, water.flow.z, water.flow.xy, h2, hu2, hv2);
            vec3 nLocal2 = waterPerturbedNormal(uv2.x, uv2.y, h2, hu2, hv2, rHat);
            vec3 n2 = normalize(mat3(water.model) * nLocal2);
            float cosThetaI2 = -dot(reflDir, n2);
            float sinThetaT2_2 = (1.0 - cosThetaI2 * cosThetaI2) / (eta * eta);
            float cosThetaT2 = sqrt(max(1.0 - sinThetaT2_2, 0.0));
            float rPar2 = (eta * cosThetaI2 - cosThetaT2) / (eta * cosThetaI2 + cosThetaT2);
            float rPerp2 = (cosThetaI2 - eta * cosThetaT2) / (cosThetaI2 + eta * cosThetaT2);
            float F2 = (rPar2 * rPar2 + rPerp2 * rPerp2) * 0.5;
            float u = waterJitter(gl_FragCoord.xy, water.temporal.y);
            vec3 d2;
            if (u < F2) {
                d2 = reflect(reflDir, n2);
            } else {
                d2 = refract(reflDir, n2, 1.0 / eta);
                if (length(d2) < 1e-4) d2 = reflect(reflDir, n2);
            }
            vec3 c;
            float tScene;
            if (traceScene(p2, d2, 1e30, c, tScene)) {
                reflection = c;
            } else {
                vec3 lightDir = normalize(frame.light_pos.xyz - p2);
                reflection = waterEnvironmentReflection(d2, lightDir, frame.light_color.rgb, var2);
            }
        } else {
            vec3 lightDir = normalize(frame.light_pos.xyz - p1);
            reflection = waterEnvironmentReflection(reflDir, lightDir, frame.light_color.rgb, var);
        }
   } else
#endif
    {
        vec3 lightDir = normalize(frame.light_pos.xyz - p1);
        reflection = waterEnvironmentReflection(reflDir, lightDir, frame.light_color.rgb, var);
    }

#ifdef WATER_RAY_QUERY
    if (push.debugView == 7) {
        outColor = (waterTraceLastHitKind == 1) ? vec4(0.0, 1.0, 0.0, 1.0) : (waterTraceLastHitKind == 2) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 1.0, 1.0);
        outHistory = outColor;
        return;
    }
#endif

    // Transmission: exit-point refraction
    vec3 dRefr = refract(dLocal, nLocal, 1.0 / eta);
    if (length(dRefr) < 1e-4) {
        dRefr = reflect(dLocal, nLocal);
    }

    float exitRoots[4];
    bool exitFallback;
    int exitCount = intersectTorus(pLocal1 + dRefr * 1e-3, dRefr, rHat, exitRoots, exitFallback);
    vec3 pExitLocal;
    if (exitCount > 0) {
        pExitLocal = pLocal1 + dRefr * (1e-3 + exitRoots[0]);
    } else {
        pExitLocal = pLocal1;
    }

    vec3 nExit = normalize(torusGradient(pExitLocal, rHat));
    vec3 dExitLocal = refract(dRefr, -nExit, eta);
    if (length(dExitLocal) < 1e-4) {
        dRefr = reflect(dRefr, nExit);
        float reRoots[4];
        bool reFallback;
        int reCount = intersectTorus(pLocal1 + dRefr * 1e-3, dRefr, rHat, reRoots, reFallback);
        if (reCount > 0) {
            pExitLocal = pLocal1 + dRefr * (1e-3 + reRoots[0]);
        }
        nExit = normalize(torusGradient(pExitLocal, rHat));
        dExitLocal = refract(dRefr, -nExit, eta);
        if (length(dExitLocal) < 1e-4) {
            dExitLocal = dRefr;
        }
    }
    vec3 dExit = normalize(mat3(water.model) * dExitLocal);

    vec4 pExitWorld = water.model * vec4(pExitLocal * water.radii.x, 1.0);
    vec3 background;
    float tBackground = 1e30;
#ifdef WATER_RAY_QUERY
    if (push.secondaryRays == 0) {
        // RayQuery path: traceScene at exit point
        vec3 rayColor;
        if (traceScene(pExitWorld.xyz, dExit, 1e30, rayColor, tBackground)) {
            background = rayColor;
        } else {
            // Depth-2: check if exit ray re-enters the torus
            vec3 pExitLocalCheck = pExitLocal + dExitLocal * 1e-3;
            float reRoots[4];
            bool reFallback;
            int reCount = intersectTorus(pExitLocalCheck, dExitLocal, rHat, reRoots, reFallback);
            if (reCount > 0) {
                // Re-entry into torus — Fresnel probabilistic selection at re-entry point
                float reT = reRoots[0] * water.radii.x;
                vec3 pReEntryWorld = pExitWorld.xyz + dExit * reT;
                vec3 pLocalRe = (water.inverseModel * vec4(pReEntryWorld, 1.0)).xyz / water.radii.x;
                vec2 uvRe = torusUV(pLocalRe);
                float hRe, huRe, hvRe, varRe;
                waterHeightAndGradient(uvRe, water.flow.z, water.flow.xy, int(water.composite.z), footprint, hRe, huRe, hvRe, varRe);
                waterLbHeightAndGradient(uvRe, water.flow.z, water.flow.xy, hRe, huRe, hvRe);
                vec3 nLocalRe = waterPerturbedNormal(uvRe.x, uvRe.y, hRe, huRe, hvRe, rHat);
                vec3 nRe = normalize(mat3(water.model) * nLocalRe);
                float cosThetaIRe = -dot(dExit, nRe);
                float sinThetaT2Re = (1.0 - cosThetaIRe * cosThetaIRe) / (eta * eta);
                float cosThetaTRe = sqrt(max(1.0 - sinThetaT2Re, 0.0));
                float rParRe = (eta * cosThetaIRe - cosThetaTRe) / (eta * cosThetaIRe + cosThetaTRe);
                float rPerpRe = (cosThetaIRe - eta * cosThetaTRe) / (cosThetaIRe + eta * cosThetaTRe);
                float FRe = (rParRe * rParRe + rPerpRe * rPerpRe) * 0.5;
                float u = waterJitter(gl_FragCoord.xy, water.temporal.y);
                vec3 dRe;
                if (u < FRe) {
                    dRe = reflect(dExit, nRe);
                } else {
                    dRe = refract(dExit, nRe, 1.0 / eta);
                    if (length(dRe) < 1e-4) dRe = reflect(dExit, nRe);
                }
                vec3 c;
                float tScene;
                if (traceScene(pReEntryWorld, dRe, 1e30, c, tScene)) {
                    background = c;
                } else {
                    vec4 clip = frame.proj * frame.view * vec4(pReEntryWorld, 1.0);
                    if (clip.w > 0) {
                        vec2 uvExit = clamp((clip.xy / clip.w) * 0.5 + 0.5, 0.0, 1.0);
                        background = texture(sceneColorSampler, uvExit).rgb;
                    } else {
                        background = texture(sceneColorSampler, fragTexCoord).rgb;
                    }
                }
            } else {
                // No re-entry — screen-space fallback
                vec4 clip = frame.proj * frame.view * pExitWorld;
                if (clip.w > 0) {
                    vec2 uvExit = clamp((clip.xy / clip.w) * 0.5 + 0.5, 0.0, 1.0);
                    background = texture(sceneColorSampler, uvExit).rgb;
                } else {
                    background = texture(sceneColorSampler, fragTexCoord).rgb;
                }
            }
        }
    } else if (push.secondaryRays == 2) {
        background = texture(waterTraceSampler, fragTexCoord).rgb;
    } else
#endif
    {
        // ScreenSpace path: project exit point to screen space
        vec4 clip = frame.proj * frame.view * pExitWorld;
        if (clip.w > 0) {
            vec2 uvExit = clamp((clip.xy / clip.w) * 0.5 + 0.5, 0.0, 1.0);
            background = texture(sceneColorSampler, uvExit).rgb;
        } else {
            background = texture(sceneColorSampler, fragTexCoord).rgb;
        }
    }

#ifdef WATER_RAY_QUERY
    if (push.secondaryRays == 0 && water.radii.z > 0.0 && tBackground < 1e29) {
        float R = water.radii.x;
        float cosV = clamp((length(pExitLocal.xz) - 1.0) / rHat, -1.0, 1.0);
        float kappa1 = 1.0 / (rHat * R);
        float kappa2 = cosV / ((1.0 + rHat * cosV) * R);
        float eta2 = water.absorption.w;
        float k1t = eta2 * kappa1;
        float k2t = eta2 * kappa2;
        float d = tBackground;
        float focus = 1.0 / (abs((1.0 - d * k1t) * (1.0 - d * k2t)) + 0.05);
        float caustic = mix(1.0, clamp(focus, 0.0, 4.0), clamp(water.radii.z, 0.0, 2.0) * 0.5);
        background *= caustic;
    }
#endif

    vec3 transmission = mix(background, water.tint.rgb, clamp(water.tint.a, 0.0, 1.0)) * rteTransmittance(waterExtinctionCoefficient(), chord);
    vec3 lightDirExit = normalize(frame.light_pos.xyz - pExitWorld.xyz);
    transmission += waterTransmittedHighlight(dExit, lightDirExit, frame.light_color.rgb, chord);

    float scatterPath = length(pExitWorld.xyz - p1);
    if (scatterPath > 1e-6) {
        vec3 viewDirWater = (pExitWorld.xyz - p1) / scatterPath;
        float ds = scatterPath / float(WATER_SCATTER_SAMPLES);
        for (int i = 0; i < WATER_SCATTER_SAMPLES; ++i) {
            WaterScatterSample smp = waterScatterSampleAt(p1, pExitWorld.xyz, i, frame.light_pos.xyz);
#ifdef WATER_RAY_QUERY
            if (push.secondaryRays == 0 && waterLightOccluded(smp.lightExitPoint, frame.light_pos.xyz)) {
                continue;
            }
#endif
            transmission += waterScatterSampleRadiance(smp, viewDirWater, frame.light_color.rgb, ds);
        }
    }

  // Composite output
   outColor = vec4(F * reflection * water.composite.x + (1.0 - F) * transmission * water.composite.y, 1.0);

#ifdef WATER_RAY_QUERY
    if (push.secondaryRays == 2) {
        outColor = vec4(texture(waterTraceSampler, fragTexCoord).rgb, 1.0);
    }
#endif

#ifdef WATER_RAY_QUERY
    vec4 current = outColor;
    bool currentNonFinite = any(isnan(outColor)) || any(isinf(outColor));
    if (currentNonFinite) {
        current = vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec4 blended = current;
    if (water.temporal.x > 0.0) {
        vec4 history = texture(waterHistorySampler, fragTexCoord);
        if (!any(isnan(history)) && !any(isinf(history))) {
            blended = mix(current, history, water.temporal.x);
        }
    }
    if (push.debugView == 6) {
        if (currentNonFinite) {
            outColor = vec4(0.0, 1.0, 0.0, 1.0);
        } else if (any(isnan(blended)) || any(isinf(blended))) {
            outColor = vec4(1.0, 0.0, 1.0, 1.0);
        } else {
            outColor = vec4(0.0);
        }
        outHistory = blended;
    } else {
        outColor = blended;
        outHistory = blended;
    }
#else
    // No history in Blender: output directly
#endif
}
