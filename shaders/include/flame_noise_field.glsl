#ifndef FLAME_NOISE_FIELD_GLSL
#define FLAME_NOISE_FIELD_GLSL

// Octagonal shell inscribed radius: 0.5 * cos(pi/8), derived from RING_SEGMENTS=8

// Must be included after FlameUBO, chebyshev.glsl, flame_noise.glsl, and
// flame_shell_profile.glsl (FLAME_SHELL_SUPPORT_HEADROOM).

// Compact-support biweight radial profile: (1 - u^2)^2, exactly zero outside u >= 1.
float flameBiweight(float uSquared) {
    float inside = max(1.0 - uSquared, 0.0);
    return inside * inside;
}

// Support radius S of the biweight profile in R(h) units. The curvature at the axis
// matches the former Gaussian exp(-radialSharpness * u^2), so the sharpness lever
// keeps its direction; the shell headroom bounds the support so the proxy never cuts.
// Mirrored in thyllore-render-core/src/flame_radial.rs (flame_radial_support_radius).
float flameRadialSupportRadius() {
    return min(sqrt(2.0 / max(flame.radialSharpness, 1e-3)), FLAME_SHELL_SUPPORT_HEADROOM);
}

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

// Turbulence may add density where erosion goes negative. The field is defined as
// zero outside the smooth envelope's support (dSmooth == 0: above the tip, outside
// the compact radius), so that addition is masked by exact membership — otherwise
// the flooded volume gets sliced flat by the shell proxy, visible as a "culled"
// flame from above.
float flameFieldSupportMask(float dSmooth) {
    return dSmooth > 0.0 ? 1.0 : 0.0;
}

// Envelope fade toward the support boundary, shared by the flooded-erosion
// argument below and the unresolved-noise sigma of the analytic band integrals:
// both the mean shift and the fluctuation of the erosion must shrink with the
// envelope, or the smoothed response keeps a positive floor at the support
// surface (a flat swirling ceiling where the integration domain is clipped).
// Mirrored in thyllore-render-core/src/flame_radial.rs (envelope_fade).
float flameEnvelopeFade(float dSmooth) {
    return min(dSmooth / max(flame.styleParams1.z, 1e-3), 1.0);
}

// Threshold argument with the flooded (negative) erosion faded by the envelope.
// Turbulence may only add density where the envelope still has support, so the
// argument goes to zero together with dSmooth and the field stays continuous
// across the support boundary — a bare [dSmooth > 0] cut used to slice the
// flooded volume, a sharp world-space seam most visible looking down the flame.
// Positive erosion (carving tongues) is untouched.
// Mirrored in thyllore-render-core/src/flame_radial.rs (eroded_argument).
float flameErodedArgument(float dSmooth, float erosion) {
    return dSmooth - (max(erosion, 0.0) + min(erosion, 0.0) * flameEnvelopeFade(dSmooth));
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
// Single source of truth for the smooth (pre-threshold) emitter density.
// `c` is the prepared density coordinate: the warped q for cylinder/ring in the
// styled field, the plain local p for the mode 0/1 parity pair (which stays
// unwarped) and for the SDF billboard (whose dSmooth never warps). The styled
// field passes wiggle = 1.0; the closed-form pair folds the contour wiggle in.
float flameEmitterSmoothDensityAt(vec3 c, float h, float wiggle) {
    if (flame.emitterParams.x >= 1.5) {
        vec2 uv = vec2(c.x + 0.5, 1.0 - clamp(c.y, 0.0, 1.0));
        float d = textureLod(flameSdfSampler, uv, 0.0).r - 0.5;
        float shell = 0.06;
        float zn = c.z / max(flame.emitterParams.w, 1e-3);
        float thickness = exp(-zn * zn);
        return clamp(1.0 - max(d, 0.0) / shell, 0.0, 1.0) * thickness;
    }
    float taperR = mix(1.0, flame.styleParams1.x, pow(h, flame.styleParams0.w));
    float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
    float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
    float rho = (length(c.xz) - rm) / minorScale;
    float rn = abs(rho) / max(taperR * wiggle, 1e-4);
    float u = rn / flameRadialSupportRadius();
    return evaluateHeightFalloff(h) * flameBiweight(u * u);
}

bool flameKernelModelActive() {
    return flame.kernelParams.x >= 0.5;
}

// Biweight kernel sum Σ aᵢ(1-|p-xᵢ|²/rᵢ²)₊². Mirrored in
// thyllore-render-core/src/flame_kernel.rs (evaluate_kernel_blob_density).
float flameKernelBlobDensityAt(vec3 p) {
    float total = 0.0;
    for (int i = 0; i < 96; ++i) {
        vec4 blob = flame.kernelBlobs[2 * i];
        float amp = flame.kernelBlobs[2 * i + 1].x;
        if (amp <= 0.0 || blob.w <= 0.0) {
            continue;
        }
        vec3 rel = p - blob.xyz;
        float u2 = dot(rel, rel) / (blob.w * blob.w);
        float inside = max(1.0 - u2, 0.0);
        total += amp * inside * inside;
    }
    return total;
}

float flameNoiseErosionFromValue(float noise, float h) {
    return flame.noiseAmplitude * mix(0.2, 1.0, h) * (noise - 0.35);
}

// Internal helper: compute erosion value from warped coordinate q and height h.
float flameNoiseErosionAt(vec3 q, float h) {
    float noise = flameKernelModelActive()
        ? flameKernelBlobDensityAt(q)
        : fbm3(flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency - flameNoiseAdvect());
    return flameNoiseErosionFromValue(noise, h);
}

float flameNoiseFieldDensity(vec3 p, float h, out float dSmooth) {
    vec3 q = flameNoiseWarpedCoordinate(p, h);
    dSmooth = flameEmitterSmoothDensityAt(q, h, 1.0);
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion))
        * flameFieldSupportMask(dSmooth);
}

// Raw erosion value at a point, sampled at the warped coordinate like every
// erosion consumer (band freeze, legacy factor, occupancy field).
float flameNoiseErosionValue(vec3 p, float h) {
    vec3 q = flameNoiseWarpedCoordinate(p, h);
    return flameNoiseErosionAt(q, h);
}

float flameNoiseErosionFactor(vec3 p, float h) {
    if (flame.noiseAmplitude == 0.0) {
        return 1.0;
    }
    return max(1.0 - flameNoiseErosionValue(p, h), 0.0);
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
    dSmooth = flameEmitterSmoothDensityAt(q, h, 1.0);
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion))
        * flameFieldSupportMask(dSmooth);
}
// MeshSdf emitter: density from a baked 2D silhouette SDF sampled as a billboard in
// unit-local XY. Texel encodes d = r - 0.5 (negative inside), normalized by image height.
float flameSdfFieldDensity(vec3 p, float h, out float dSmooth) {
    vec3 q = flameNoiseWarpedCoordinate(p, h);
    dSmooth = flameEmitterSmoothDensityAt(p, h, 1.0);
    float erosion = flameNoiseErosionAt(q, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion))
        * flameFieldSupportMask(dSmooth);
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
