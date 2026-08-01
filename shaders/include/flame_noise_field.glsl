#ifndef FLAME_NOISE_FIELD_GLSL
#define FLAME_NOISE_FIELD_GLSL

// Octagonal shell inscribed radius: 0.5 * cos(pi/8), derived from RING_SEGMENTS=8

// Must be included after FlameUBO, chebyshev.glsl, and flame_noise.glsl

// Internal helper: compute the advect vector from style params and time.
vec3 flameNoiseAdvect() {
    return vec3(flame.styleParams2.x, flame.styleParams0.z, flame.styleParams2.y) * flame.time;
}

// Internal helper: anisotropic compression along the advection axis.
vec3 flameAnisoCompress(vec3 v, float axialScale) {
    vec3 adv = vec3(flame.styleParams2.x, flame.styleParams0.z, flame.styleParams2.y);
    vec3 axis = vec3(0.0, 1.0, 0.0);
    if (flame.contourParams.y > 0.0 && dot(adv, adv) > 1e-8) {
        axis = normalize(mix(axis, normalize(adv), clamp(flame.contourParams.y, 0.0, 1.0)));
    }
    return v - dot(v, axis) * axis * (1.0 - axialScale);
}

// Horizontal displacement of the density centerline from wind bend at height h.
// Shared by the domain warp below and the radial band integrals, so the
// analytic path bends exactly like the sampled field.
vec2 flameBendOffsetAt(float h) {
    return flame.styleParams2.xy * flame.styleParams2.z * pow(h, flame.styleParams2.w);
}

// Turbulence may add density where erosion goes negative. Without support
// gating that addition floods regions where the smooth envelope is zero
// (above the tip, outside the radius), and the flooded volume gets sliced
// flat by the shell proxy — visible as a "culled" flame from above.
const float FLAME_FIELD_SUPPORT_EPS = 0.02;
float flameFieldSupport(float dSmooth) {
    return smoothstep(0.0, FLAME_FIELD_SUPPORT_EPS, dSmooth);
}

// Internal helper: compute the warped coordinate q from world position p and height h.
// This is the single source of truth for the domain-warp chain:
//   bendOffset -> pb -> advect -> aniso -> wp -> w -> q
vec3 flameNoiseWarpedCoordinate(vec3 p, float h) {
    // Wind bend deformation (horizontal-only)
    vec2 bendOffset = flameBendOffsetAt(h);
    vec3 pb = vec3(p.x - bendOffset.x, p.y, p.z - bendOffset.y);

    // Domain warp with upward advection
    vec3 wp = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    vec2 w = vec2(fbm3(wp), fbm3(wp + vec3(19.1, 7.7, 3.3))) * 2.0 - 1.0;
    float wy = fbm3(wp + vec3(41.3, 23.7, 11.9)) * 2.0 - 1.0;
    vec3 q = pb + flame.styleParams0.x * mix(0.15, 1.0, h) * vec3(w.x, wy * flame.temporalData.w, w.y);

    return q;
}
// Internal helper: compute erosion value from warped coordinate q and height h.
float flameNoiseErosionAt(vec3 q, float h) {
    return flame.noiseAmplitude * mix(0.2, 1.0, h) * (fbm3(flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency - flameNoiseAdvect()) - 0.35);
}

float flameNoiseFieldDensity(vec3 p, float h, out float dSmooth) {
    vec3 q = flameNoiseWarpedCoordinate(p, h);

    // Tapered radial density
    float taperR = mix(1.0, flame.styleParams1.x, pow(h, flame.styleParams0.w));
    float rn = length(q.xz) / max(taperR, 1e-4);
    dSmooth = evaluateHeightFalloff(h) * exp(-flame.radialSharpness * rn * rn);
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth - erosion)
        * flameFieldSupport(dSmooth);
}

float flameNoiseErosionFactor(vec3 p, float h) {
    if (flame.noiseAmplitude == 0.0) {
        return 1.0;
    }
    vec3 q = flameNoiseWarpedCoordinate(p, h);
    float erosion = flameNoiseErosionAt(q, h);
    return max(1.0 - erosion, 0.0);
}

// Ring emitter: a flame cross-section swept around a circle of normalized major radius
// Rm = emitterParams.y in unit-local XZ. Noise advects around the ring by rotating the
// sample point about Y before field evaluation (seamless, no angular unwrap).
float flameRingFieldDensity(vec3 p, float h, out float dSmooth) {
    float rm = flame.emitterParams.y;
    float minorScale = max(1.0 - rm, 1e-3);
    float ang = flame.emitterParams.z * flame.time;
    float c = cos(ang);
    float s = sin(ang);
    vec3 pr = vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    vec3 q = flameNoiseWarpedCoordinate(pr, h);
    float taperR = mix(1.0, flame.styleParams1.x, pow(h, flame.styleParams0.w));
    float rho = (length(q.xz) - rm) / minorScale;
    float rn = abs(rho) / max(taperR, 1e-4);
    dSmooth = evaluateHeightFalloff(h) * exp(-flame.radialSharpness * rn * rn);
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth - erosion)
        * flameFieldSupport(dSmooth);
}
// MeshSdf emitter: density from a baked 2D silhouette SDF sampled as a billboard in
// unit-local XY. Texel encodes d = r - 0.5 (negative inside), normalized by image height.
float flameSdfFieldDensity(vec3 p, float h, out float dSmooth) {
    vec2 uv = vec2(p.x + 0.5, 1.0 - clamp(p.y, 0.0, 1.0));
    float d = textureLod(flameSdfSampler, uv, 0.0).r - 0.5;
    float shell = 0.06;
    float zn = p.z / max(flame.emitterParams.w, 1e-3);
    float thickness = exp(-zn * zn);
    vec3 q = flameNoiseWarpedCoordinate(p, h);
    dSmooth = clamp(1.0 - max(d, 0.0) / shell, 0.0, 1.0) * thickness;
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth - erosion)
        * flameFieldSupport(dSmooth);
}

float flameEmitterDensity(vec3 p, float h, out float dSmooth) {
    if (flame.emitterParams.x >= 1.5) {
        return flameSdfFieldDensity(p, h, dSmooth);
    }
    if (flame.emitterParams.x >= 0.5) {
        return flameRingFieldDensity(p, h, dSmooth);
    }
    return flameNoiseFieldDensity(p, h, dSmooth);
}

#endif
