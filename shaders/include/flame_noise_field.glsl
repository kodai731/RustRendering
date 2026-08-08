#ifndef FLAME_NOISE_FIELD_GLSL
#define FLAME_NOISE_FIELD_GLSL

// Octagonal shell inscribed radius: 0.5 * cos(pi/8), derived from RING_SEGMENTS=8

// Fixed rotation of the noise lattice for E2 experiments.
// Defined as a compile-time constant from FLAME_NOISE_ROT_DEG_OVERRIDE (default 0).
// When angle is 0, returns v unchanged (bit-identical to current behavior).
// When nonzero, rotates v around axis (1,1,1)/sqrt(3) by the specified angle
// using Rodrigues' rotation formula.
#ifndef FLAME_NOISE_ROT_DEG_OVERRIDE
#define FLAME_NOISE_ROT_DEG_OVERRIDE 0.0
#endif
vec3 flameNoiseLatticeRotate(vec3 v) {
    const float angle = radians(FLAME_NOISE_ROT_DEG_OVERRIDE);
    if (angle == 0.0) {
        return v;
    }
    vec3 axis = normalize(vec3(1.0, 1.0, 1.0));
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

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

// Internal helper: inverse of flameAnisoCompress — anisotropic expansion along the advection axis.
vec3 flameAnisoExpand(vec3 v, float axialScale) {
    vec3 adv = vec3(flame.styleParams2.x, flame.styleParams0.z, flame.styleParams2.y);
    vec3 axis = vec3(0.0, 1.0, 0.0);
    if (flame.contourParams.y > 0.0 && dot(adv, adv) > 1e-8) {
        axis = normalize(mix(axis, normalize(adv), clamp(flame.contourParams.y, 0.0, 1.0)));
    }
    return v + dot(v, axis) * axis * (1.0 / axialScale - 1.0);
}

// Horizontal displacement of the density centerline from wind bend at height h.
// Shared by the domain warp below and the radial band integrals, so the
// analytic path bends exactly like the sampled field.
vec2 flameBendOffsetAt(float h) {
    return flame.styleParams2.xy * flame.styleParams2.z * pow(h, flame.styleParams2.w);
}

// Turbulence basis switch (kernelParams.x = turbulence_model):
// 0 = fbm lattice, 1 = kernel blob sum, 2 = wave mode sum.
bool flameKernelModelActive() {
    return flame.kernelParams.x >= 0.5 && flame.kernelParams.x < 1.5;
}

bool flameWaveModelActive() {
    return flame.kernelParams.x >= 1.5;
}

// Closed-form (family-T) wave variant: pseudo-FM carriers + at most two
// single-frequency shear layers replacing the 16-shear transport
// (geometric_replacement_plan.md「厳密閉形式化」). kernelParams.y = active flag,
// kernelParams.z = shear layer count; layers sit at slot FLAME_WAVE_CF_SHEAR_BASE.
// Mirrored in thyllore-render-core/src/flame_wave.rs (WaveCfContext/WaveCfSample).
const float FLAME_WAVE_CF_CAP = 2.0;
const float FLAME_WAVE_CF_PSI_BOUND = 11.313708;
const int FLAME_WAVE_CF_CHEB_COEFFS = 21;
const int FLAME_WAVE_CF_SHEAR_BASE = 176;

bool flameWaveCfActive() {
    return flame.kernelParams.y > 0.5;
}

// Modulator displacement field at the UNWARPED warp coordinate, pushed through
// the erosion coordinate transform: per mode psi_n = capScale_n * dot(k_n, psiVec)
// and its ray phase rate capScale_n * dot(k_n, rateVec). ampDisp is the warp
// displacement amplitude the per-mode depth (wavePhase.w) is scaled by.
void flameWaveCfPsiVectors(
    vec3 pb, vec3 dir, float h, out vec3 psiVec, out vec3 rateVec, out float ampDisp) {
    ampDisp = flame.styleParams0.x * mix(0.15, 1.0, h);
    vec3 m0 = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    vec3 dm0 = flameAnisoCompress(dir, 0.35) * flame.styleParams0.y;
    int base = int(flame.waveParams.x);
    int count = int(flame.waveParams.y);
    vec3 v = vec3(0.0);
    vec3 vdot = vec3(0.0);
    for (int j = 0; j < count; ++j) {
        vec4 waveVector = flame.waveModes[2 * (base + j)];
        vec4 direction = flame.waveModes[2 * (base + j) + 1];
        float angle = dot(waveVector.xyz, m0) + direction.x;
        v += direction.yzw * (waveVector.w * sin(angle));
        vdot += direction.yzw * (waveVector.w * cos(angle) * dot(waveVector.xyz, dm0));
    }
    float scale = flame.noiseFrequency * ampDisp;
    psiVec = flameAnisoCompress(v, flame.temporalData.z) * scale;
    rateVec = flameAnisoCompress(vdot, flame.temporalData.z) * scale;
}

// Chebyshev tables of sin/cos over [-PSI_BOUND, PSI_BOUND], packed in the
// detail-mode spare slots (wavePhase.z/.w of the first 21 detail modes).
void flameWaveCfLoadCheb(
    out float chebSin[FLAME_WAVE_CF_CHEB_COEFFS],
    out float chebCos[FLAME_WAVE_CF_CHEB_COEFFS]) {
    int base = int(flame.waveParams.x) + int(flame.waveParams.y);
    for (int i = 0; i < FLAME_WAVE_CF_CHEB_COEFFS; ++i) {
        vec4 wavePhase = flame.waveModes[2 * (base + i) + 1];
        chebSin[i] = wavePhase.z;
        chebCos[i] = wavePhase.w;
    }
}

float flameWaveCfClenshaw(float coeffs[FLAME_WAVE_CF_CHEB_COEFFS], float x) {
    float t = 2.0 * x;
    float b1 = 0.0;
    float b2 = 0.0;
    for (int k = FLAME_WAVE_CF_CHEB_COEFFS - 1; k >= 1; --k) {
        float b0 = t * b1 - b2 + coeffs[k];
        b2 = b1;
        b1 = b0;
    }
    return x * b1 - b2 + coeffs[0];
}

// sin(angle + psi) in family T: sinA*T_c(psi) + cosA*T_s(psi), with the
// per-mode RMS depth cap folded into psi.
float flameWaveCfCarrier(
    vec4 waveVector, float depthScale, float angle, vec3 psiVec, float ampDisp,
    float chebSin[FLAME_WAVE_CF_CHEB_COEFFS], float chebCos[FLAME_WAVE_CF_CHEB_COEFFS]) {
    float depth = ampDisp * depthScale;
    float capScale = depth > FLAME_WAVE_CF_CAP ? FLAME_WAVE_CF_CAP / depth : 1.0;
    float x = clamp(
        capScale * dot(waveVector.xyz, psiVec) / FLAME_WAVE_CF_PSI_BOUND, -1.0, 1.0);
    return sin(angle) * flameWaveCfClenshaw(chebCos, x)
        + cos(angle) * flameWaveCfClenshaw(chebSin, x);
}

// Turbulence may add density where erosion goes negative. The field is defined as
// zero outside the smooth envelope's support (dSmooth == 0: above the tip, outside
// the compact radius), so that addition is masked by exact membership — otherwise
// the flooded volume gets sliced flat by the shell proxy, visible as a "culled"
// flame from above.
float flameFieldSupportMask(float dSmooth) {
    return dSmooth > 0.0 ? 1.0 : 0.0;
}

// Column-wise (heightScale, radiusScale) displacing the envelope support — the only
// perturbation that survives tangent-view path averaging (plate-silhouette-diagnosis).
const int FLAME_WAVE_DETAIL_MODE_COUNT = 64;

// Lattice-free replacement for the fbm behind the contour wiggle and the
// boundary displacement. Normalized to the same distribution as the
// expression it replaces: fbm3(p) * (2/0.875) - 1, mean 0, std 0.2422.
float flameDetailNoise(vec3 p) {
    int base = int(flame.waveParams.x) + int(flame.waveParams.y);
    float z = 0.0;
    for (int n = 0; n < FLAME_WAVE_DETAIL_MODE_COUNT; ++n) {
        vec4 waveVector = flame.waveModes[2 * (base + n)];
        vec4 wavePhase = flame.waveModes[2 * (base + n) + 1];
        z += waveVector.w * sin(dot(waveVector.xyz, p) + wavePhase.x);
    }
    return z * (2.0 / 0.875);
}

vec2 flameBoundaryDisplacement(vec2 xz) {
    float amp = flame.boundaryParams.x;
    if (amp == 0.0) {
        return vec2(1.0);
    }
    vec3 q = vec3(
        xz.x * flame.boundaryParams.y,
        -flame.boundaryParams.z * flame.time,
        xz.y * flame.boundaryParams.y);
    // 3.0 ≈ 1/(2·fbm std): amp becomes the typical fractional displacement.
    // Raise capped at +amp so lifted tips stay clear of the y=1 cap; dips stay deep.
    float heightNoise = min((fbm3(flameNoiseLatticeRotate(q)) * (2.0 / 0.875) - 1.0) * 3.0, 1.0);
    float radiusNoise = (fbm3(flameNoiseLatticeRotate(q + vec3(13.7, 41.3, 7.9))) * (2.0 / 0.875) - 1.0) * 3.0;
    return max(
        vec2(1.0 + amp * heightNoise, 1.0 + amp * flame.boundaryParams.w * radiusNoise),
        vec2(0.2));
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

// Fraction of the medium that turbulence does not carve. The field becomes the
// two-component mixture
//   occ = (1 - beta) * phi(D - e) + beta * phi(D),
// i.e. a parcel is a blend of soot the turbulence can carve away and a residual
// that it cannot. Without it the carved field can be EXACTLY zero over a whole
// ray span (e >= D - edgeLow everywhere) and the Beer-Lambert composite returns
// alpha = 1 - exp(0) = 0: a hard hole showing the background through the middle
// of the flame. The residual term uses the SAME threshold on the un-eroded
// density, which is what makes it safe:
//   * D < edgeLow (the outer skirt and the tip taper): both terms vanish, so the
//     silhouette and the support boundary keep their exact geometry — a floor on
//     the raw density instead would haze the whole support and fatten the flame.
//   * e = 0: the two terms coincide and the mixture is the identity, so the
//     noise-free look is preserved exactly for any beta.
//   * D > edgeHigh with any erosion: occ >= beta > 0, so a hole is impossible.
// Mirrored in thyllore-render-core/src/flame_radial.rs (carve_residual_mix).
float flameCarveResidualStrength() {
    return clamp(flame.nearFadeParams.y, 0.0, 1.0);
}

float flameApplyCarveResidual(float carvedOccupancy, float dSmooth) {
    float beta = flameCarveResidualStrength();
    if (beta <= 0.0) {
        return carvedOccupancy;
    }
    float plain = smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth);
    return mix(carvedOccupancy, plain, beta);
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

// Analytic mode-sum replacement of the fbm domain warp for the wave basis: a
// low-wavenumber vector displacement field over the same warp coordinate chain
// (aniso * warp_freq - advect), calibrated to the fbm warp statistics. The
// billowing shear of the flame look lives in this nonlinear composition — a
// purely spectral anisotropy of the erosion modes cannot reproduce it.
// Warp modes sit in the waveModes pairs after the erosion modes
// (waveParams.x = erosion count, waveParams.y = warp count).
// Mirrored in thyllore-render-core/src/flame_wave.rs (evaluate_wave_warp).
vec3 flameWaveWarpOffset(vec3 pb, float h) {
    float amp = flame.styleParams0.x * mix(0.15, 1.0, h);
    int warpCount = int(flame.waveParams.y);
    if (amp == 0.0 || warpCount == 0) {
        return vec3(0.0);
    }
    vec3 wp = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    int base = int(flame.waveParams.x);
    vec3 displacement = vec3(0.0);
    for (int m = 0; m < warpCount; ++m) {
        vec4 waveVector = flame.waveModes[2 * (base + m)];
        vec4 direction = flame.waveModes[2 * (base + m) + 1];
        float value = waveVector.w * sin(dot(waveVector.xyz, wp) + direction.x);
        displacement +=
            vec3(direction.y, direction.z * flame.temporalData.w, direction.w) * value;
    }
    return amp * displacement;
}

const float FLAME_WAVE_SHEAR_STRENGTH_SCALE = 0.96;

// Flow-map warp: apply each mode's shear sequentially (z += curl_dir * strength * a * cos(k·z + φ)).
// Each mode is a pure shear (k·curl_dir = 0), so the flow map at time 1 equals one step of shear.
// Modes are applied in index order (shear composition is non-commutative).
vec3 flameWaveFlowWarp(vec3 pb, float h) {
    float amp = flame.styleParams0.x * mix(0.15, 1.0, h);
    bool cf = flameWaveCfActive();
    int warpCount = cf ? int(flame.kernelParams.z) : int(flame.waveParams.y);
    if (amp == 0.0 || warpCount == 0) {
        return pb;
    }

    // M: warp space (same chain as flameWaveWarpOffset)
    vec3 z = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    float strength = amp * FLAME_WAVE_SHEAR_STRENGTH_SCALE;

    // Sequential shear: each mode updates z, and the next mode's angle uses the updated z
    int base = cf ? FLAME_WAVE_CF_SHEAR_BASE : int(flame.waveParams.x);
    for (int m = 0; m < warpCount; ++m) {
        vec4 waveVector = flame.waveModes[2 * (base + m)];
        vec4 direction = flame.waveModes[2 * (base + m) + 1];
        float angle = dot(waveVector.xyz, z) + direction.x;
        z += direction.yzw * (strength * waveVector.w * cos(angle));
    }

    // M^-1: back to physical space
    return flameAnisoExpand(z / flame.styleParams0.y + flameNoiseAdvect() / flame.styleParams0.y, 0.35);
}

// Flow-map warp rate: same shear composition as flameWaveFlowWarp but also
// accumulates the Jacobian-vector product on v. z is updated identically to
// flameWaveFlowWarp; v starts as the linear part of M^-1 applied to dir and is
// updated by v += curlDir * (-strength * a * sin(angle)) * dot(k, v) each step.
// Returns the final v transformed by the linear part of M^-1 (no translation).
vec3 flameWaveFlowWarpRate(vec3 pb, vec3 dir, float h, out vec3 warpedPoint) {
    float amp = flame.styleParams0.x * mix(0.15, 1.0, h);
    bool cf = flameWaveCfActive();
    int warpCount = cf ? int(flame.kernelParams.z) : int(flame.waveParams.y);
    if (amp == 0.0 || warpCount == 0) {
        warpedPoint = pb;
        return dir;
    }

    // M: warp space (same chain as flameWaveFlowWarp)
    vec3 z = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    // v starts as dir transformed by linear part of M^-1 (no translation for a direction vector)
    vec3 v = flameAnisoCompress(dir, 0.35) * flame.styleParams0.y;
    float strength = amp * FLAME_WAVE_SHEAR_STRENGTH_SCALE;

    // Sequential shear: each mode updates both z and v
    int base = cf ? FLAME_WAVE_CF_SHEAR_BASE : int(flame.waveParams.x);
    for (int m = 0; m < warpCount; ++m) {
        vec4 waveVector = flame.waveModes[2 * (base + m)];
        vec4 direction = flame.waveModes[2 * (base + m) + 1];
        float angle = dot(waveVector.xyz, z) + direction.x;
        float shearValue = strength * waveVector.w * cos(angle);
        float fPrime = -strength * waveVector.w * sin(angle);
        vec3 curlDir = direction.yzw;
        z += curlDir * shearValue;
        v += curlDir * (fPrime * dot(waveVector.xyz, v));
    }

    // M^-1: back to physical space
    warpedPoint = flameAnisoExpand(z / flame.styleParams0.y + flameNoiseAdvect() / flame.styleParams0.y, 0.35);
    // Return v transformed by linear part of M^-1 (no translation for a direction vector)
    return flameAnisoExpand(v / flame.styleParams0.y, 0.35);
}

// Internal helper: compute the warped coordinate q from world position p and height h.
// This is the single source of truth for the domain-warp chain:
//   bendOffset -> pb -> advect -> aniso -> wp -> w -> q
vec3 flameNoiseBendRemoved(vec3 p, float h) {
    vec2 bendOffset = flameBendOffsetAt(h);
    return vec3(p.x - bendOffset.x, p.y, p.z - bendOffset.y);
}

vec3 flameNoiseWarpedCoordinateFromPb(vec3 pb, float h) {
    // Wave basis: the fbm domain warp (3 fbm3 per point) is replaced by the
    // analytic mode-sum warp below — same coordinate chain, no lattice.
    if (flameWaveModelActive()) {
        return flameWaveFlowWarp(pb, h);
    }

    // Domain warp with upward advection
    vec3 wp = flameAnisoCompress(pb, 0.35) * flame.styleParams0.y - flameNoiseAdvect();
    vec2 w = vec2(fbm3(flameNoiseLatticeRotate(wp)), fbm3(flameNoiseLatticeRotate(wp + vec3(19.1, 7.7, 3.3)))) * 2.0 - 1.0;
    float wy = fbm3(flameNoiseLatticeRotate(wp + vec3(41.3, 23.7, 11.9))) * 2.0 - 1.0;
    vec3 q = pb + flame.styleParams0.x * mix(0.15, 1.0, h) * vec3(w.x, wy * flame.temporalData.w, w.y);

    return q;
}

vec3 flameNoiseWarpedCoordinate(vec3 p, float h) {
    return flameNoiseWarpedCoordinateFromPb(flameNoiseBendRemoved(p, h), h);
}
// Single source of truth for the smooth (pre-threshold) emitter density.
// `c` is the prepared density coordinate: the warped q for cylinder/ring in the
// styled field, the plain local p for the mode 0/1 parity pair (which stays
// unwarped) and for the SDF billboard (whose dSmooth never warps). The styled
// field passes wiggle = 1.0; the closed-form pair folds the contour wiggle in.
float flameEmitterSmoothDensityDisplacedAt(vec3 c, float h, float wiggle, vec2 boundary) {
    if (flame.emitterParams.x >= 1.5) {
        float hSdf = clamp(clamp(c.y, 0.0, 1.0) / boundary.x, 0.0, 1.0);
        vec2 uv = vec2(c.x + 0.5, 1.0 - hSdf);
        float d = textureLod(flameSdfSampler, uv, 0.0).r - 0.5;
        float shell = 0.06;
        float zn = c.z / max(flame.emitterParams.w, 1e-3);
        float thickness = exp(-zn * zn);
        return clamp(1.0 - max(d, 0.0) / shell, 0.0, 1.0) * thickness;
    }
    float hb = clamp(h / boundary.x, 0.0, 1.0);
    float taperR = mix(1.0, flame.styleParams1.x, pow(hb, flame.styleParams0.w));
    float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
    float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
    float rho = (length(c.xz) - rm) / minorScale;
    float rn = abs(rho) / max(taperR * wiggle * boundary.y, 1e-4);
    float u = rn / flameRadialSupportRadius();
    // raised columns must vanish before the y=1 cap or the slab cuts them flat
    float capFade = flame.boundaryParams.x != 0.0 ? smoothstep(1.0, 0.94, h) : 1.0;
    return evaluateHeightFalloff(hb) * flameBiweight(u * u) * capFade;
}

float flameEmitterSmoothDensityAt(vec3 c, float h, float wiggle) {
    return flameEmitterSmoothDensityDisplacedAt(c, h, wiggle, flameBoundaryDisplacement(c.xz));
}

// Wave-basis erosion noise: analytic sum of random wave modes over the same
// warped coordinate the fbm samples — no lattice, no cells, so the ray
// restriction is exact 1D and the closed form needs no radial bands. Advection
// translates the coordinate (sweeping omega = k . U falls out); the per-mode
// eddy-turnover rate ~ |k|^(2/3) is scaled by noiseScrollSpeed. Mean-matched to
// fbm3 (0.4375) so flameNoiseErosionFromValue keeps its calibration.
// Mirrored in thyllore-render-core/src/flame_wave.rs (evaluate_wave_noise_cf).
// `pb` is the pre-warp (bend-removed) coordinate the closed-form pseudo-FM
// modulators sample; it is unused when the cf variant is off.
float flameWaveNoiseSum(vec3 w, vec3 pb, float h) {
    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = int(flame.waveParams.x);
    bool cf = flameWaveCfActive();
    vec3 psiVec = vec3(0.0);
    float ampDisp = 0.0;
    float chebSin[FLAME_WAVE_CF_CHEB_COEFFS];
    float chebCos[FLAME_WAVE_CF_CHEB_COEFFS];
    if (cf) {
        vec3 rateVecUnused;
        flameWaveCfPsiVectors(pb, vec3(0.0), h, psiVec, rateVecUnused, ampDisp);
        flameWaveCfLoadCheb(chebSin, chebCos);
    }
    // Low-octave pass first (wavePhase.z == 0): its sum drives the envelope
    // 1 + coeff * zLow that ties the higher octaves to the coarse structure.
    float zLow = 0.0;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z != 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime;
        float carrier = cf
            ? flameWaveCfCarrier(waveVector, wavePhase.w, angle, psiVec, ampDisp, chebSin, chebCos)
            : sin(angle);
        zLow += waveVector.w * carrier;
    }
    float z = zLow;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z == 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime;
        float carrier = cf
            ? flameWaveCfCarrier(waveVector, wavePhase.w, angle, psiVec, ampDisp, chebSin, chebCos)
            : sin(angle);
        z += (1.0 + wavePhase.z * zLow) * waveVector.w * carrier;
    }
    float invScale = flame.waveParams.z;
    float amp = flame.waveParams.w;
    return invScale > 0.0 ? 0.4375 + amp * tanh(z * invScale) : 0.4375 + z;
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
// `pb` is the pre-warp (bend-removed) coordinate the closed-form modulators need.
float flameNoiseErosionAt(vec3 q, vec3 pb, float h) {
    float noise;
    if (flameKernelModelActive()) {
        noise = flameKernelBlobDensityAt(q);
    } else if (flameWaveModelActive()) {
        noise = flameWaveNoiseSum(
            flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency - flameNoiseAdvect(),
            pb, h);
    } else {
        noise = fbm3(flameNoiseLatticeRotate(flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency - flameNoiseAdvect()));
    }
    return flameNoiseErosionFromValue(noise, h);
}

float flameNoiseFieldDensity(vec3 p, float h, out float dSmooth) {
    vec3 pb = flameNoiseBendRemoved(p, h);
    vec3 q = flameNoiseWarpedCoordinateFromPb(pb, h);
    dSmooth = flameEmitterSmoothDensityAt(q, h, 1.0);
    float erosion = flameNoiseErosionAt(q, pb, h);
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
        dSmooth) * flameFieldSupportMask(dSmooth);
}

// Raw erosion value at a point, sampled at the warped coordinate like every
// erosion consumer (band freeze, legacy factor, occupancy field).
float flameNoiseErosionValue(vec3 p, float h) {
    vec3 pb = flameNoiseBendRemoved(p, h);
    vec3 q = flameNoiseWarpedCoordinateFromPb(pb, h);
    return flameNoiseErosionAt(q, pb, h);
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
    vec3 pb = flameNoiseBendRemoved(pr, h);
    vec3 q = flameNoiseWarpedCoordinateFromPb(pb, h);
    dSmooth = flameEmitterSmoothDensityAt(q, h, 1.0);
    float erosion = flameNoiseErosionAt(q, pb, h);
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
        dSmooth) * flameFieldSupportMask(dSmooth);
}
// MeshSdf emitter: density from a baked 2D silhouette SDF sampled as a billboard in
// unit-local XY. Texel encodes d = r - 0.5 (negative inside), normalized by image height.
float flameSdfFieldDensity(vec3 p, float h, out float dSmooth) {
    vec3 pb = flameNoiseBendRemoved(p, h);
    vec3 q = flameNoiseWarpedCoordinateFromPb(pb, h);
    dSmooth = flameEmitterSmoothDensityAt(p, h, 1.0);
    float erosion = flameNoiseErosionAt(q, pb, h);
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
        dSmooth) * flameFieldSupportMask(dSmooth);
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

// カメラが emitter の support に入るとリムレイが 45/45 -> 0/45 に全滅し、これが
// 壁ルックへの二値切替になる。カメラ近傍の密度を 0 へ落として光学的にカメラを場の
// 外に置き、リムのシルエットが常に画面に残るようにする。
float flameNearCameraFade(vec3 pLocal) {
    float radius = flame.nearFadeParams.x;
    if (radius <= 0.0) {
        return 1.0;
    }
    vec3 pWorld = (flame.model * vec4(pLocal, 1.0)).xyz;
    return smoothstep(0.0, radius, length(pWorld - frame.camera_pos.xyz));
}

#endif
