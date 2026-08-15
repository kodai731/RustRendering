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

// S1 plateaued radial factor: max(biweight(u^2 * margin^2), plateau(u^2, margin)).
// margin comes from flame.supportMotion.supportMargin (support_margin).
// For margin == 1.0, plateau is zero and biweight(u^2 * 1) = biweight(u^2), so bit-identical.
float flamePlateauRadialFactor(float uSquared) {
    float margin = flame.supportMotion.supportMargin;
    float core = flameBiweight(uSquared * margin * margin);
    if (margin <= 1.0) { return core; }
    const float PLATEAU_LEVEL = 0.35;
    const float PLATEAU_EDGE_DELTA = 0.1;
    float plateau = PLATEAU_LEVEL * (1.0 - smoothstep(1.0 - PLATEAU_EDGE_DELTA, 1.0, uSquared));
    return max(core, plateau);
}

// Support radius S of the biweight profile in R(h) units. The curvature at the axis
// matches the former Gaussian exp(-radialSharpness * u^2), so the sharpness parameter
// keeps its direction; the shell headroom bounds the support so the proxy never cuts.
// Mirrored in thyllore-render-core/src/flame_radial.rs (flame_radial_support_radius).
float flameRadialSupportRadius() {
  return flame.supportMotion.supportMargin * min(sqrt(2.0 / max(flame.radialSharpness, 1e-3)), FLAME_SHELL_SUPPORT_HEADROOM);
}

// Internal helper: compute the advect vector from style params and time.
vec3 flameNoiseAdvect() {
    return vec3(flame.windBend.windDirection.x, flame.warpStyle.riseSpeed, flame.windBend.windDirection.y) * flame.time;
}

// Internal helper: anisotropic compression along the advection axis.
vec3 flameAnisoCompress(vec3 v, float axialScale) {
    vec3 adv = vec3(flame.windBend.windDirection.x, flame.warpStyle.riseSpeed, flame.windBend.windDirection.y);
    vec3 axis = vec3(0.0, 1.0, 0.0);
    if (flame.contourParams.anisoAxisAdvect > 0.0 && dot(adv, adv) > 1e-8) {
        axis = normalize(mix(axis, normalize(adv), clamp(flame.contourParams.anisoAxisAdvect, 0.0, 1.0)));
    }
    return v - dot(v, axis) * axis * (1.0 - axialScale);
}

// Internal helper: inverse of flameAnisoCompress — anisotropic expansion along the advection axis.
vec3 flameAnisoExpand(vec3 v, float axialScale) {
    vec3 adv = vec3(flame.windBend.windDirection.x, flame.warpStyle.riseSpeed, flame.windBend.windDirection.y);
    vec3 axis = vec3(0.0, 1.0, 0.0);
    if (flame.contourParams.anisoAxisAdvect > 0.0 && dot(adv, adv) > 1e-8) {
        axis = normalize(mix(axis, normalize(adv), clamp(flame.contourParams.anisoAxisAdvect, 0.0, 1.0)));
    }
    return v + dot(v, axis) * axis * (1.0 / axialScale - 1.0);
}

// Horizontal displacement of the density centerline from wind bend at height h.
// Shared by the domain warp below and the radial band integrals, so the
// analytic path bends exactly like the sampled field.
vec2 flameBendOffsetAt(float h) {
    return flame.windBend.windDirection * flame.windBend.bendAmount * pow(h, flame.windBend.bendPower);
}

// Animated meander: time-varying horizontal displacement of the centerline at
// height h, with g(h) = h (base fixed, tip moves more). The mode table comes
// from Rust (flame/constants.rs) through the UBO.
vec2 flameMeanderOffsetAt(float h) {
    vec2 offset = vec2(0.0);
    for (int j = 0; j < 2; ++j) {
        FlameMeanderMode mode = flame.meanderModes[j];
        offset += sin(mode.kappa * h - mode.omega * flame.time + mode.phase) * mode.direction;
    }
    return flame.supportMotion.meanderAmp * h * offset;
}

// Meander-only shifted position for density/support evaluation.
// Total centerline offset = static bend + animated meander, so
// p - total_offset = p - bend - meander = noise_bend_removed_position - meander.
// This is the position where density/support should be evaluated:
// the noise sees the bend-removed position (p - bend), but density/support
// must also account for meander displacement.
vec3 flameMeanderShifted(vec3 p, float h) {
    vec2 off = flameMeanderOffsetAt(h);
    return vec3(p.x - off.x, p.y, p.z - off.y);
}

// Total centerline offset = static bend + animated meander.
// Single source of truth for all density/support centerline displacement.
vec2 flameCenterlineOffsetAt(float h) {
    return flameBendOffsetAt(h) + flameMeanderOffsetAt(h);
}

const float FLAME_WAVE_SHEAR_STRENGTH_SCALE = 0.96;

// Remaining luminous fraction mu(h) of the height envelope — the flame's own
// height scale (1 at the base, 0 at the luminous tip), shared by the tip-carve
// lambda and the warp strain profile so both asymptote toward the same "tip".
float flameEnvelopeRemainingMu(float h) {
    float primitive = evaluateChebyshev12(
        flame.heightPrimitiveCoefficients[0],
        flame.heightPrimitiveCoefficients[1],
        flame.heightPrimitiveCoefficients[2],
        h);
    return clamp((flame.tipCarveParams.primitiveTop - primitive) * flame.tipCarveParams.invPrimitiveRange, 0.0, 1.0);
}

// Asymptotic warp strain (20260809_warp_asymptotic_strain_redesign.md):
// bounded, C-inf, monotone toward the luminous tip, so the composed shear map
// never folds and the noise cannot laminate into height-chirped fringes.
// Mirrored in flame_wave.rs (warp_strain_at) and flame_field_trace.rs.
float flameWarpStrain(float h) {
    return mix(flame.warpStrainParams.strainBase, flame.warpStrainParams.strainTip,
        exp(-flameEnvelopeRemainingMu(h) * flame.warpStrainParams.invReach));
}

// Shear strength strength(h) = strain(h) / K: every mode's dimensionless
// strain strength * a_m * |k_m| stays <= strain(h) <= cap < 1.
float flameWarpStrength(float h) {
    return flameWarpStrain(h) * flame.warpStrainParams.invStrainNorm;
}

// Closed-form (family-T) wave variant: pseudo-FM carriers + at most two
// single-frequency shear layers replacing the 16-shear transport
// (geometric_replacement_plan.md「厳密閉形式化」); cf shear layers sit at slot
// FLAME_WAVE_CF_SHEAR_BASE.
// Mirrored in thyllore-render-core/src/flame_wave.rs (WaveCfContext/WaveCfSample).
const float FLAME_WAVE_CF_CAP = 2.0;

// Fixed slot layout of waveModes (mirrors flame_wave.rs WAVE_MODE_COUNT /
// WAVE_WARP_MODE_COUNT): erosion region first, then warp, then detail.
// waveParams.trackedCount is the TRACKED erosion mode count (5.1 probabilistic
// reduction), not the region size, so the bases are compile-time constants.
const int FLAME_WAVE_EROSION_SLOTS = 128;
const int FLAME_WAVE_WARP_BASE = FLAME_WAVE_EROSION_SLOTS;
const int FLAME_WAVE_WARP_COUNT = 16;
const int FLAME_WAVE_DETAIL_BASE = FLAME_WAVE_WARP_BASE + FLAME_WAVE_WARP_COUNT;
const float FLAME_WAVE_CF_PSI_BOUND = 11.313708;
const int FLAME_WAVE_CF_CHEB_COEFFS = 21;
const int FLAME_WAVE_CF_SHEAR_BASE = 208;
const int FLAME_WAVE_MEDIUM_SWIRL_BASE = 210;
const int FLAME_WAVE_MEDIUM_SWIRL_COUNT = 4;

// Rank-M phase jitter of the erosion carriers: M shared low-wavenumber fields
// Psi_m(w) = sin(kappa_m . w + phi_m) in the warped erosion coordinate; mode n
// adds dot(waveJitter[n].xyz, Psi) to its carrier phase. Pair (i, j) beats then
// carry the pair-specific modulation sum_m (c_i - c_j)_m Psi_m, so the dense
// bottom octave's difference frequencies cannot align into one fringe family
// (fringe_beat_analysis.md; a single shared field — M = 1 — cannot break them).
// Phase-only: |k| and per-mode power are untouched, the erf closed form stays
// exact; the jitter ray rate joins the node low-pass beta below.
// Mirrored in flame_wave.rs (WAVE_JITTER_K / wave_jitter_state).
const int FLAME_WAVE_JITTER_RANK = 3;
const vec3 FLAME_WAVE_JITTER_K[3] = vec3[3](
    vec3(1.017876, 1.357168, -2.261947),
    vec3(-3.116460, 1.402407, 1.869876),
    vec3(2.563540, -4.272566, 1.922655));
const vec3 FLAME_WAVE_JITTER_PHASE = vec3(1.234568, 3.456790, 5.678901);

// waveJitter[0].w carries the runtime kappa scale (0 reads as 1 so a zeroed
// UBO stays neutral); larger = finer jitter, decohering close-up fringes.
float flameWaveJitterKappaScale() {
    float scale = flame.waveJitter[0].w;
    return scale > 0.0 ? scale : 1.0;
}

vec3 flameWaveJitterFields(vec3 w) {
    float scale = flameWaveJitterKappaScale();
    return vec3(
        sin(scale * dot(FLAME_WAVE_JITTER_K[0], w) + FLAME_WAVE_JITTER_PHASE.x),
        sin(scale * dot(FLAME_WAVE_JITTER_K[1], w) + FLAME_WAVE_JITTER_PHASE.y),
        sin(scale * dot(FLAME_WAVE_JITTER_K[2], w) + FLAME_WAVE_JITTER_PHASE.z));
}

void flameWaveJitterState(vec3 w, vec3 rate, out vec3 psi, out vec3 psiRate) {
    float scale = flameWaveJitterKappaScale();
    for (int m = 0; m < FLAME_WAVE_JITTER_RANK; ++m) {
        float angle = scale * dot(FLAME_WAVE_JITTER_K[m], w) + FLAME_WAVE_JITTER_PHASE[m];
        psi[m] = sin(angle);
        psiRate[m] = cos(angle) * scale * dot(FLAME_WAVE_JITTER_K[m], rate);
    }
}

bool flameWaveCfActive() {
    return flame.waveCfParams.enabled > 0.5;
}

// Modulator displacement field at the UNWARPED warp coordinate, pushed through
// the erosion coordinate transform: per mode psi_n = capScale_n * dot(k_n, psiVec)
// and its ray phase rate capScale_n * dot(k_n, rateVec). ampDisp is the warp
// displacement amplitude the per-mode depth (wavePhase.w) is scaled by.
void flameWaveCfPsiVectors(
    vec3 pb, vec3 dir, float h, out vec3 psiVec, out vec3 rateVec, out float ampDisp) {
    ampDisp = flameWarpStrength(h) / FLAME_WAVE_SHEAR_STRENGTH_SCALE;
    vec3 m0 = flameAnisoCompress(pb, 0.35) * flame.warpStyle.warpFreq - flameNoiseAdvect();
    vec3 dm0 = flameAnisoCompress(dir, 0.35) * flame.warpStyle.warpFreq;
    int base = FLAME_WAVE_WARP_BASE;
    int count = FLAME_WAVE_WARP_COUNT;
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
    psiVec = flameAnisoCompress(v, flame.temporalData.noiseAnisoY) * scale;
    rateVec = flameAnisoCompress(vdot, flame.temporalData.noiseAnisoY) * scale;
}

// Chebyshev tables of sin/cos over [-PSI_BOUND, PSI_BOUND], packed in the
// detail-mode spare slots (wavePhase.z/.w of the first 21 detail modes).
void flameWaveCfLoadCheb(
    out float chebSin[FLAME_WAVE_CF_CHEB_COEFFS],
    out float chebCos[FLAME_WAVE_CF_CHEB_COEFFS]) {
    int base = FLAME_WAVE_DETAIL_BASE;
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
    int base = FLAME_WAVE_DETAIL_BASE;
    float z = 0.0;
    for (int n = 0; n < FLAME_WAVE_DETAIL_MODE_COUNT; ++n) {
        vec4 waveVector = flame.waveModes[2 * (base + n)];
        vec4 wavePhase = flame.waveModes[2 * (base + n) + 1];
        z += waveVector.w * sin(dot(waveVector.xyz, p) + wavePhase.x);
    }
    return z * (2.0 / 0.875);
}

vec2 flameBoundaryDisplacement(vec2 xz) {
    float amp = flame.boundaryParams.amp;
    if (amp == 0.0 || flame.unifiedParams.enabled > 0.5) {
        return vec2(1.0);
    }
    vec3 q = vec3(
        xz.x * flame.boundaryParams.freq,
        -flame.boundaryParams.speed * flame.time,
        xz.y * flame.boundaryParams.freq);
    // 3.0 ≈ 1/(2·fbm std): amp becomes the typical fractional displacement.
    // Raise capped at +amp so lifted tips stay clear of the y=1 cap; dips stay deep.
    float heightNoise = min((fbm3(flameNoiseLatticeRotate(q)) * (2.0 / 0.875) - 1.0) * 3.0, 1.0);
    float radiusNoise = (fbm3(flameNoiseLatticeRotate(q + vec3(13.7, 41.3, 7.9))) * (2.0 / 0.875) - 1.0) * 3.0;
    return max(
        vec2(1.0 + amp * heightNoise, 1.0 + amp * flame.boundaryParams.radiusRatio * radiusNoise),
        vec2(0.2));
}

// Raised columns must vanish before the y=1 proxy cap or the slab cuts them
// flat. The fade follows the displaced tip (P4): only columns lifted past the
// cap (bx > 1) fade, over their own overhang [2 - bx, 1] — no fixed world-h
// band, so nothing aligns across columns.
float flameCapFade(float h, float bx) {
    if (flame.boundaryParams.amp == 0.0 || bx <= 1.0) {
        return 1.0;
    }
    return smoothstep(1.0, 2.0 - bx, h);
}

// Envelope fade toward the support boundary, shared by the flooded-erosion
// argument below and the unresolved-noise sigma of the analytic band integrals:
// both the mean shift and the fluctuation of the erosion must shrink with the
// envelope, or the smoothed response keeps a positive floor at the support
// surface (a flat swirling ceiling where the integration domain is clipped).
// Mirrored in thyllore-render-core/src/flame_radial.rs (envelope_fade).
float flameEnvelopeFade(float dSmooth) {
    return min(dSmooth / max(flame.edgeStyle.edgeHigh, 1e-3), 1.0);
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
float flameCarveResidualOuterGate(float uSquared) {
    float margin = flame.supportMotion.supportMargin;
    if (margin <= 1.0) { return 0.0; }
    float w = uSquared * margin * margin;
    return smoothstep(1.0, 1.5, w);
}

float flamePlateauCarveReach(float uSquared) {
    float margin = flame.supportMotion.supportMargin;
    float w = uSquared * margin * margin;
    return max(w - 1.0, 0.0);
}

float flameCarveResidualStrength(float uSquared) {
    return clamp(flame.nearFadeParams.carveResidual, 0.0, 1.0) * (1.0 - flameCarveResidualOuterGate(uSquared));
}

float flameApplyCarveResidual(float carvedOccupancy, float dSmooth, float uSquared) {
    float beta = flameCarveResidualStrength(uSquared);
    if (beta <= 0.0) {
        return carvedOccupancy;
    }
    float plain = smoothstep(flame.edgeStyle.edgeLow, flame.edgeStyle.edgeHigh, dSmooth);
    return mix(carvedOccupancy, plain, beta);
}


// Threshold argument with the flooded (negative) erosion faded by the envelope.
// Turbulence may only add density where the envelope still has support, so the
// argument goes to zero together with dSmooth and the field stays continuous
// across the support boundary — a bare [dSmooth > 0] cut used to slice the
// flooded volume, a sharp world-space seam most visible looking down the flame.
// Positive erosion (carving tongues) is untouched.
// Mirrored in thyllore-render-core/src/flame_radial.rs (eroded_argument).
const float FLAME_EROSION_REMAP_FLOOR = 0.15;
const float FLAME_EROSION_REMAP_STRENGTH = 0.0;   // 0 = off (default look), 1 = full Nubis-style remap
float flameErosionRemapScale(float erosion) {
    return mix(1.0, 1.0 / max(1.0 - max(erosion, 0.0), FLAME_EROSION_REMAP_FLOOR),
               FLAME_EROSION_REMAP_STRENGTH);
}
float flameErodedArgument(float dSmooth, float erosion) {
    float base = dSmooth - (max(erosion, 0.0) + min(erosion, 0.0) * flameEnvelopeFade(dSmooth));
    return base * flameErosionRemapScale(erosion);
}

// Analytic mode-sum replacement of the fbm domain warp for the wave basis: a
// low-wavenumber vector displacement field over the same warp coordinate chain
// (aniso * warp_freq - advect), calibrated to the fbm warp statistics. The
// billowing shear of the flame look lives in this nonlinear composition — a
// purely spectral anisotropy of the erosion modes cannot reproduce it.
// Warp modes sit in the waveModes pairs after the erosion modes
// (bases fixed by the slot-layout constants above).
// Mirrored in thyllore-render-core/src/flame_wave.rs (evaluate_wave_warp).
vec3 flameWaveWarpOffset(vec3 pb, float h) {
    float amp = flameWarpStrength(h) / FLAME_WAVE_SHEAR_STRENGTH_SCALE;
    int warpCount = FLAME_WAVE_WARP_COUNT;
    if (amp == 0.0 || warpCount == 0) {
        return vec3(0.0);
    }
    vec3 wp = flameAnisoCompress(pb, 0.35) * flame.warpStyle.warpFreq - flameNoiseAdvect();
    int base = FLAME_WAVE_WARP_BASE;
    vec3 displacement = vec3(0.0);
    for (int m = 0; m < warpCount; ++m) {
        vec4 waveVector = flame.waveModes[2 * (base + m)];
        vec4 direction = flame.waveModes[2 * (base + m) + 1];
        float value = waveVector.w * sin(dot(waveVector.xyz, wp) + direction.x);
        displacement +=
            vec3(direction.y, direction.z * flame.temporalData.warpYScale, direction.w) * value;
    }
    return amp * displacement;
}

// S1 — the warp map proper, isolated so the evaluation form lives in these
// two functions and nowhere else. Two forms, selected by the displacement-form flag
// (CPU env THYLLORE_FLAME_WARP_FORM, K in warpStrainParams matches the form):
//   displacement (default): every mode evaluated at the INPUT z, so the
//     Jacobian is the bounded sum I + sum c_m f'_m k_m^T — no multiplicative
//     stretch growth, hence no height-chirped lamination (追補2);
//   sequential: the legacy composition — mode m+1 sees the z mode m moved,
//     the Jacobian is a product (kept for A/B).
// Each mode is a pure shear (k·curl_dir = 0):
// z += curl_dir * strength * a * cos(k·z + φ).
bool flameWarpFormDisplacement() {
    return flame.warpFormParams.displacementForm > 0.5;
}

// Medium swirl shear (motion_design L2): azimuthal transport of the RTE
// medium density — the same pure shear form as the warp modes, with packed
// amplitudes already carrying the swirl share of the strain budget (gain 0
// packs zero amplitudes, so the added terms are exactly 0.0). The direction
// slot is [phase, c.x, drift_rate, c.z] (displacement is horizontal): the
// drift rate makes the shear pattern move relative to the advected material,
// so the layers visibly counter-rotate instead of scrolling rigidly. Time is
// a frame uniform, so the drift never enters the ray-rate JVP.
void flameMediumSwirlShear(vec3 z, vec3 v, float strength,
        inout vec3 displacement, inout vec3 rate) {
    float driftTime = flame.noiseScrollSpeed * flame.time;
    for (int m = 0; m < FLAME_WAVE_MEDIUM_SWIRL_COUNT; ++m) {
        vec4 waveVector = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m)];
        vec4 direction = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m) + 1];
        vec3 curlDir = vec3(direction.y, 0.0, direction.w);
        float angle = dot(waveVector.xyz, z) + direction.x + direction.z * driftTime;
        displacement += curlDir * (strength * waveVector.w * cos(angle));
        rate += curlDir
            * (-strength * waveVector.w * sin(angle) * dot(waveVector.xyz, v));
    }
}

vec3 flameWarpMapZ(vec3 z, int base, int warpCount, float strength) {
    if (flameWarpFormDisplacement()) {
        vec3 z0 = z;
        vec3 displacement = vec3(0.0);
        for (int m = 0; m < warpCount; ++m) {
            vec4 waveVector = flame.waveModes[2 * (base + m)];
            vec4 direction = flame.waveModes[2 * (base + m) + 1];
            float angle = dot(waveVector.xyz, z0) + direction.x;
            displacement += direction.yzw * (strength * waveVector.w * cos(angle));
        }
        vec3 swirlRate = vec3(0.0);
        flameMediumSwirlShear(z0, vec3(0.0), strength, displacement, swirlRate);
        return z0 + displacement;
    }
    for (int m = 0; m < warpCount; ++m) {
        vec4 waveVector = flame.waveModes[2 * (base + m)];
        vec4 direction = flame.waveModes[2 * (base + m) + 1];
        float angle = dot(waveVector.xyz, z) + direction.x;
        z += direction.yzw * (strength * waveVector.w * cos(angle));
    }
    float driftTime = flame.noiseScrollSpeed * flame.time;
    for (int m = 0; m < FLAME_WAVE_MEDIUM_SWIRL_COUNT; ++m) {
        vec4 waveVector = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m)];
        vec4 direction = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m) + 1];
        float angle = dot(waveVector.xyz, z) + direction.x + direction.z * driftTime;
        z += vec3(direction.y, 0.0, direction.w) * (strength * waveVector.w * cos(angle));
    }
    return z;
}

// S1 with the Jacobian-vector product J·v, exact for either form:
// displacement: v += sum c_m (f'_m dot(k_m, v0)) with every f' at z0;
// sequential:   v updated through each shear in composition order.
void flameWarpMapJvp(inout vec3 z, inout vec3 v, int base, int warpCount, float strength) {
    if (flameWarpFormDisplacement()) {
        vec3 z0 = z;
        vec3 v0 = v;
        vec3 displacement = vec3(0.0);
        vec3 rate = vec3(0.0);
        for (int m = 0; m < warpCount; ++m) {
            vec4 waveVector = flame.waveModes[2 * (base + m)];
            vec4 direction = flame.waveModes[2 * (base + m) + 1];
            float angle = dot(waveVector.xyz, z0) + direction.x;
            float shearValue = strength * waveVector.w * cos(angle);
            float fPrime = -strength * waveVector.w * sin(angle);
            vec3 curlDir = direction.yzw;
            displacement += curlDir * shearValue;
            rate += curlDir * (fPrime * dot(waveVector.xyz, v0));
        }
        flameMediumSwirlShear(z0, v0, strength, displacement, rate);
        z = z0 + displacement;
        v = v0 + rate;
        return;
    }
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
    float driftTime = flame.noiseScrollSpeed * flame.time;
    for (int m = 0; m < FLAME_WAVE_MEDIUM_SWIRL_COUNT; ++m) {
        vec4 waveVector = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m)];
        vec4 direction = flame.waveModes[2 * (FLAME_WAVE_MEDIUM_SWIRL_BASE + m) + 1];
        float angle = dot(waveVector.xyz, z) + direction.x + direction.z * driftTime;
        float shearValue = strength * waveVector.w * cos(angle);
        float fPrime = -strength * waveVector.w * sin(angle);
        vec3 curlDir = vec3(direction.y, 0.0, direction.w);
        z += curlDir * shearValue;
        v += curlDir * (fPrime * dot(waveVector.xyz, v));
    }
}

// Flow-map warp: M (aniso * warp_freq - advect) -> S1 -> M^-1.
vec3 flameWaveFlowWarp(vec3 pb, float h) {
    float strength = flameWarpStrength(h);
    bool cf = flameWaveCfActive();
    int warpCount = cf ? int(flame.waveCfParams.shearLayerCount) : FLAME_WAVE_WARP_COUNT;
    if (strength == 0.0 || warpCount == 0) {
        return pb;
    }
    vec3 z = flameAnisoCompress(pb, 0.35) * flame.warpStyle.warpFreq - flameNoiseAdvect();
    int base = cf ? FLAME_WAVE_CF_SHEAR_BASE : FLAME_WAVE_WARP_BASE;
    z = flameWarpMapZ(z, base, warpCount, strength);
    return flameAnisoExpand(z / flame.warpStyle.warpFreq + flameNoiseAdvect() / flame.warpStyle.warpFreq, 0.35);
}

// Flow-map warp rate: same shear composition as flameWaveFlowWarp but also
// accumulates the Jacobian-vector product on v. z is updated identically to
// flameWaveFlowWarp; v starts as the linear part of M^-1 applied to dir and is
// updated by v += curlDir * (-strength * a * sin(angle)) * dot(k, v) each step.
// Returns the final v transformed by the linear part of M^-1 (no translation).
vec3 flameWaveFlowWarpRate(vec3 pb, vec3 dir, float h, out vec3 warpedPoint) {
    float strength = flameWarpStrength(h);
    bool cf = flameWaveCfActive();
    int warpCount = cf ? int(flame.waveCfParams.shearLayerCount) : FLAME_WAVE_WARP_COUNT;
    if (strength == 0.0 || warpCount == 0) {
        warpedPoint = pb;
        return dir;
    }

    // M: warp space (same chain as flameWaveFlowWarp); v starts as dir
    // transformed by the linear part (no translation for a direction vector).
    vec3 z = flameAnisoCompress(pb, 0.35) * flame.warpStyle.warpFreq - flameNoiseAdvect();
    vec3 v = flameAnisoCompress(dir, 0.35) * flame.warpStyle.warpFreq;
    int base = cf ? FLAME_WAVE_CF_SHEAR_BASE : FLAME_WAVE_WARP_BASE;
    flameWarpMapJvp(z, v, base, warpCount, strength);

    // M^-1: back to physical space
    warpedPoint = flameAnisoExpand(z / flame.warpStyle.warpFreq + flameNoiseAdvect() / flame.warpStyle.warpFreq, 0.35);
    return flameAnisoExpand(v / flame.warpStyle.warpFreq, 0.35);
}

// Internal helper: compute the warped coordinate q from world position p and height h.
// This is the single source of truth for the domain-warp chain:
//   bendOffset -> pb -> advect -> aniso -> wp -> w -> q
vec3 flameNoiseBendRemoved(vec3 p, float h) {
    vec2 centerlineOffset = flameCenterlineOffsetAt(h);
    return vec3(p.x - centerlineOffset.x, p.y, p.z - centerlineOffset.y);
}

// The complete coordinate chain bundled once (single source of truth):
//   p -> pb (bend removed) -> q (flow warp) -> w (aniso * noiseFrequency - advect)
// plus the ray rate dw/dt for the node low-pass. Build with d = vec3(0.0) for
// point evaluation (the rate is then exactly zero and unused).
struct FlameWarpFrame {
    vec3 pb;    // bend-removed coordinate (cf modulators sample this)
    vec3 q;     // warped physical coordinate (styled density samples this)
    vec3 w;     // noise coordinate
    vec3 rate;  // dw/dt along the ray direction
};

// Medium spread (motion_design L3): age-coordinate volume-preserving opening
// of the RTE medium. The sampling contracts in xz (scale s < 1 toward the
// luminous tip) and stretches in y by 1/s^2 — the det = 1 pair. Rising
// features therefore enlarge laterally, flatten vertically, decelerate (the
// apparent rise speed picks up the factor s^2 through the advect chain) and
// end as sideways puffs the tip carve dissolves, instead of every feature
// streaming upward unchanged. Reach shares the tip carve; gain 0
// evaluates to exactly 1.0 (bit-identical baseline). The scale is frozen per
// node like the strain profile, so it enters the ray-rate chain as plain
// per-axis factors.
float flameMediumSpreadScale(float h) {
    float sigma =
        flame.spreadParams.gain * exp(-flameEnvelopeRemainingMu(h) * flame.tipCarveParams.invReach);
    return exp(-sigma * h);
}

// Medium twist (V design): node-frozen azimuthal rotation of the noise
// coordinate about the bend-removed axis; the Lamb-Oseen radial profile keeps
// the angle finite on the axis, and a rotation never folds so the gain
// (spreadParams.twistGain) is not strain-capped. Counter-rotating mode table and core
// radius come from Rust (flame/constants.rs) through the UBO.
// Mirrored in thyllore-render-debug/src/flame_field_trace.rs (twist_angle).
float flameMediumTwistAngle(float rSquared, float h) {
    float radial = flame.twistCoreRadiusSq / (rSquared + flame.twistCoreRadiusSq);
    float wave = 0.0;
    for (int j = 0; j < 2; ++j) {
        FlameTwistMode mode = flame.twistModes[j];
        wave += mode.amp * cos(mode.kappa * h + mode.omega * flame.time + mode.phase);
    }
    return flame.spreadParams.twistGain * radial * h * wave;
}

FlameWarpFrame flameBuildWarpFrame(vec3 p, vec3 d, float h) {
    FlameWarpFrame frame;
    frame.pb = flameNoiseBendRemoved(p, h);
    vec3 dt = d;
    if (flame.spreadParams.twistGain != 0.0) {
        float rSquared = frame.pb.x * frame.pb.x + frame.pb.z * frame.pb.z;
        float phi = flameMediumTwistAngle(rSquared, h);
        float cp = cos(phi);
        float sp = sin(phi);
        frame.pb = vec3(
            cp * frame.pb.x - sp * frame.pb.z,
            frame.pb.y,
            sp * frame.pb.x + cp * frame.pb.z);
        dt = vec3(cp * d.x - sp * d.z, d.y, sp * d.x + cp * d.z);
    }
    float spread = flameMediumSpreadScale(h);
    float spreadY = 1.0 / (spread * spread);
    frame.pb = vec3(frame.pb.x * spread, frame.pb.y * spreadY, frame.pb.z * spread);
    vec3 ds = vec3(dt.x * spread, dt.y * spreadY, dt.z * spread);
    vec3 rateRaw = flameWaveFlowWarpRate(frame.pb, ds, h, frame.q);
    frame.w = flameAnisoCompress(frame.q, flame.temporalData.noiseAnisoY) * flame.noiseFrequency
        - flameNoiseAdvect();
    frame.rate = flameAnisoCompress(rateRaw, flame.temporalData.noiseAnisoY) * flame.noiseFrequency;
    return frame;
}
// Single source of truth for the smooth (pre-threshold) emitter density.
// `c` is the prepared density coordinate: the warped q for cylinder/ring in the
// styled field, the plain local p for the mode 0/1 parity pair (which stays
// unwarped) and for the SDF billboard (whose dSmooth never warps). The styled
// field passes wiggle = 1.0; the closed-form pair folds the contour wiggle in.
float flameEmitterSmoothDensityDisplacedAt(vec3 c, float h, float wiggle, vec2 boundary, out float uSquared) {
    if (flame.emitterParams.kind >= 1.5) {
        float hSdf = clamp(clamp(c.y, 0.0, 1.0) / boundary.x, 0.0, 1.0);
        vec2 uv = vec2(c.x + 0.5, 1.0 - hSdf);
        float d = textureLod(flameSdfSampler, uv, 0.0).r - 0.5;
        float shell = 0.06;
        float zn = c.z / max(flame.emitterParams.sdfSlabDepth, 1e-3);
        float thickness = exp(-zn * zn);
        uSquared = 0.0;
        return clamp(1.0 - max(d, 0.0) / shell, 0.0, 1.0) * thickness;
    }
    float hb = clamp(h / boundary.x, 0.0, 1.0);
    float taperR = mix(1.0, flame.edgeStyle.radiusTipRatio, pow(hb, flame.warpStyle.taperPower));
    float rm = flame.emitterParams.kind >= 0.5 ? flame.emitterParams.ringMajorRatio : 0.0;
    float minorScale = flame.emitterParams.kind >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
    float rho = (length(c.xz) - rm) / minorScale;
    float rn = abs(rho) / max(taperR * wiggle * boundary.y, 1e-4);
    float u = rn / flameRadialSupportRadius();
    uSquared = u * u;
   return evaluateHeightFalloff(hb) * flamePlateauRadialFactor(uSquared) * flameCapFade(h, boundary.x);
}

float flameEmitterSmoothDensityDisplacedAt(vec3 c, float h, float wiggle, vec2 boundary) {
    float uSquared;
    return flameEmitterSmoothDensityDisplacedAt(c, h, wiggle, boundary, uSquared);
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
// Shared two-pass mode sum (low octave first, then high octaves riding the
// low-octave envelope 1 + coeff * zLow). `dt = 0` (with rate = vec3(0.0))
// evaluates the field at a point: every low-pass weight is then exactly 1 and
// unresolvedPower accumulates exactly 0, so the node quadrature variant and
// the point variant share this single loop.
struct FlameWaveModeSumResult {
    float z;
    float zLow;
    float unresolvedPower;
};

FlameWaveModeSumResult flameWaveModeSum(
    vec3 w, vec3 rate, vec3 pb, vec3 d, float h, float dt, int count, float eddyTime) {
    bool cf = flameWaveCfActive();
    vec3 psiVec = vec3(0.0);
    vec3 psiRateVec = vec3(0.0);
    float ampDisp = 0.0;
    float chebSin[FLAME_WAVE_CF_CHEB_COEFFS];
    float chebCos[FLAME_WAVE_CF_CHEB_COEFFS];
    if (cf) {
        flameWaveCfPsiVectors(pb, d, h, psiVec, psiRateVec, ampDisp);
        flameWaveCfLoadCheb(chebSin, chebCos);
    }
    vec3 jitterPsi;
    vec3 jitterPsiRate;
    flameWaveJitterState(w, rate, jitterPsi, jitterPsiRate);

    FlameWaveModeSumResult result;
    result.zLow = 0.0;
    result.unresolvedPower = 0.0;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z != 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime
            + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsi);
        float betaPhase = dot(waveVector.xyz, rate) + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsiRate);
        float carrier;
        if (cf) {
            float depth = ampDisp * wavePhase.w;
            float capScale = depth > FLAME_WAVE_CF_CAP ? FLAME_WAVE_CF_CAP / depth : 1.0;
            betaPhase += capScale * dot(waveVector.xyz, psiRateVec);
            carrier = flameWaveCfCarrier(
                waveVector, wavePhase.w, angle, psiVec, ampDisp, chebSin, chebCos);
        } else {
            carrier = sin(angle);
        }
        float beta = betaPhase * dt / 3.14159265;
        float b2 = beta * beta;
        float weight = exp(-b2 * b2);
        result.zLow += weight * waveVector.w * carrier;
        result.unresolvedPower += 0.5 * waveVector.w * waveVector.w * (1.0 - weight * weight);
    }
    result.z = result.zLow;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z == 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime
            + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsi);
        float betaPhase = dot(waveVector.xyz, rate) + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsiRate);
        float carrier;
        if (cf) {
            float depth = ampDisp * wavePhase.w;
            float capScale = depth > FLAME_WAVE_CF_CAP ? FLAME_WAVE_CF_CAP / depth : 1.0;
            betaPhase += capScale * dot(waveVector.xyz, psiRateVec);
            carrier = flameWaveCfCarrier(
                waveVector, wavePhase.w, angle, psiVec, ampDisp, chebSin, chebCos);
        } else {
            carrier = sin(angle);
        }
        float beta = betaPhase * dt / 3.14159265;
        float b2 = beta * beta;
        float weight = exp(-b2 * b2);
        float envelope = 1.0 + wavePhase.z * result.zLow;
        result.z += envelope * weight * waveVector.w * carrier;
        result.unresolvedPower +=
            envelope * envelope * 0.5 * waveVector.w * waveVector.w * (1.0 - weight * weight);
    }
    return result;
}

// tanh-shaped noise value at a warp frame (point evaluation: dt = 0).
float flameWaveNoiseSum(vec3 w, vec3 pb, float h) {
    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = int(flame.waveParams.trackedCount);
    FlameWaveModeSumResult sum = flameWaveModeSum(
        w, vec3(0.0), pb, vec3(0.0), h, 0.0, count, eddyTime);
    float invScale = flame.waveParams.inverseScale;
    float amp = flame.waveParams.amplitude;
    return invScale > 0.0 ? 0.4375 + amp * tanh(sum.z * invScale) : 0.4375 + sum.z;
}

// Asymptotic tip-carve depth profile. mu(h) is the remaining luminous fraction
// of the height envelope, so "tip" means the flame's own luminous top for any
// height/envelope — no fixed-h switch exists anywhere in the modulation.
// tipCarveParams: x = kappa (depth), y = 1/mu0 (reach), z = envelope primitive
// at h=1, w = 1 / total envelope mass.
float flameTipCarveLambda(float h) {
    return 1.0 + flame.tipCarveParams.depth * exp(-flameEnvelopeRemainingMu(h) * flame.tipCarveParams.invReach);
}

// Relative (multiplicative) modulation: the stochastic term is zero-mean and
// scales with the local density, so no mean drift ever crosses the response
// window and the visibility isolines follow D, not h (no horizontal onset).
// The deterministic mean shrink that (noise - 0.35) used to carry stays as an
// explicit smooth term so the average silhouette keeps its calibration.
const float FLAME_EROSION_MEAN_SHRINK = 0.0875;
const float FLAME_PLATEAU_CARVE_BOOST = 1.0;
const float FLAME_EROSION_SHELL_REF = 0.30;

// Burnout (D design): age-driven deepening of the deterministic mean shrink,
// mean' = mean * (1 + burnout_gain * B(h)) with B = exp(-mu(h) / mu0) — the
// same remaining-luminous-fraction asymptote as the tip carve and the L3
// spread, sharing tip_carve_reach (mu0 = tipCarveParams.y inverse). Anchoring
// on mu instead of raw h keeps the boost inside the flame's own luminous top
// for any envelope (an h-threshold form sat above the visible column and did
// nothing, measured 2026-08-13). The boost lifts the mean toward the cut
// threshold so the existing stochastic troughs actually sever the support
// (base shedding) and detached parcels rising into smaller mu burn away
// (dissipation). The exp asymptote is smooth — the mean alone never steps
// (fringe gate E(h) condition); severing only happens where a noise trough
// adds on top. burnout_gain = 0 is bit-identical to the pre-burnout field.
// Mirrored in thyllore-render-debug/src/flame_field_trace.rs (burnout_boost).
float flameBurnoutBoost(float h) {
    return flame.warpFormParams.burnoutGain
        * exp(-flameEnvelopeRemainingMu(h) * flame.tipCarveParams.invReach);
}

float flameNoiseErosionFromValue(float noise, float h, float density, float uSquared) {
    float relative = flame.spreadParams.erosionNoiseGain * flameTipCarveLambda(h) * (density / FLAME_EROSION_SHELL_REF)
        * (noise - 0.4375);
    return flame.noiseAmplitude
        * (mix(0.2, 1.0, h) * FLAME_EROSION_MEAN_SHRINK * (1.0 + flameBurnoutBoost(h))
            + FLAME_PLATEAU_CARVE_BOOST * flamePlateauCarveReach(uSquared) + relative);
}

// Response through the (optionally relative) window: unified keeps the
// half-width >= the local modulation sigma so the rectifier never binarizes.
float flameResponseOccupancy(float dSmooth, float erosion, float h, float uSquared) {
    float lo = flame.nearFadeParams.edgeLow;
    float hi = flame.nearFadeParams.edgeHigh;
    if (flame.unifiedParams.enabled > 0.5) {
        float sigmaFloor = flame.unifiedParams.sigmaFloor * flameTipCarveLambda(h)
            * dSmooth / FLAME_EROSION_SHELL_REF;
        float c = 0.5 * (lo + hi);
        float hw = max(0.5 * (hi - lo), 2.24 * sigmaFloor);
        hw *= mix(1.0, 1.0 - flame.spreadParams.edgeOuterSharpen, flameCarveResidualOuterGate(uSquared));
        lo = c - hw;
        hi = c + hw;
    }
    return smoothstep(lo, hi, flameErodedArgument(dSmooth, erosion));
}

// Internal helper: compute erosion value from a warp frame and height h.
float flameNoiseErosionAt(FlameWarpFrame frame, float h, float density, float uSquared) {
    float noise = flameWaveNoiseSum(frame.w, frame.pb, h);
    return flameNoiseErosionFromValue(noise, h, density, uSquared);
}

float flameNoiseFieldDensity(vec3 p, float h, out float dSmooth) {
    FlameWarpFrame frame = flameBuildWarpFrame(p, vec3(0.0), h);
    float uSquared;
   dSmooth = flameEmitterSmoothDensityDisplacedAt(frame.q, h, 1.0, flameBoundaryDisplacement(frame.q.xz), uSquared);
   float erosion = flameNoiseErosionAt(frame, h, dSmooth, uSquared);
    return flameApplyCarveResidual(
      flameResponseOccupancy(dSmooth, erosion, h, uSquared),
        dSmooth, uSquared) * flameFieldSupportMask(dSmooth);
}

// Raw erosion value at a point, sampled at the warped coordinate like every
// erosion consumer (band freeze, legacy factor, occupancy field).
float flameNoiseErosionValue(vec3 p, float h, float density, float uSquared) {
    FlameWarpFrame frame = flameBuildWarpFrame(p, vec3(0.0), h);
    return flameNoiseErosionAt(frame, h, density, uSquared);
}

// Legacy multiplicative factor (trail domain): no local dSmooth exists there,
// so the relative term is evaluated at the shell reference density.
float flameNoiseErosionFactor(vec3 p, float h) {
    if (flame.noiseAmplitude == 0.0) {
        return 1.0;
    }
   return max(1.0 - flameNoiseErosionValue(p, h, FLAME_EROSION_SHELL_REF, 0.0), 0.0);
}

// Ring emitter: a flame cross-section swept around a circle of normalized major radius
// Rm = emitterParams.y in unit-local XZ. Noise advects around the ring by rotating the
// sample point about Y before field evaluation (seamless, no angular unwrap).
float flameRingFieldDensity(vec3 p, float h, out float dSmooth) {
    float rm = flame.emitterParams.ringMajorRatio;
    float minorScale = max(1.0 - rm, 1e-3);
    float ang = flame.emitterParams.ringAngularSpeed * flame.time;
    float c = cos(ang);
    float s = sin(ang);
    vec3 pr = vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    FlameWarpFrame frame = flameBuildWarpFrame(pr, vec3(0.0), h);
    float uSquared;
   dSmooth = flameEmitterSmoothDensityDisplacedAt(frame.q, h, 1.0, flameBoundaryDisplacement(frame.q.xz), uSquared);
  float erosion = flameNoiseErosionAt(frame, h, dSmooth, uSquared);
    return flameApplyCarveResidual(
     flameResponseOccupancy(dSmooth, erosion, h, uSquared),
        dSmooth, uSquared) * flameFieldSupportMask(dSmooth);
}

// MeshSdf emitter: density from a baked 2D silhouette SDF sampled as a billboard in
// unit-local XY. Texel encodes d = r - 0.5 (negative inside), normalized by image height.
float flameSdfFieldDensity(vec3 p, float h, out float dSmooth) {
    FlameWarpFrame frame = flameBuildWarpFrame(p, vec3(0.0), h);
    float uSquared;
  dSmooth = flameEmitterSmoothDensityDisplacedAt(p, h, 1.0, flameBoundaryDisplacement(p.xz), uSquared);
  float erosion = flameNoiseErosionAt(frame, h, dSmooth, uSquared);
    return flameApplyCarveResidual(
    flameResponseOccupancy(dSmooth, erosion, h, uSquared),
        dSmooth, uSquared) * flameFieldSupportMask(dSmooth);
}

float flameEmitterDensity(vec3 p, float h, out float dSmooth) {
    if (flame.emitterParams.kind >= 1.5) {
        return flameSdfFieldDensity(p, h, dSmooth);
    }
    if (flame.emitterParams.kind >= 0.5) {
        return flameRingFieldDensity(p, h, dSmooth);
    }
    return flameNoiseFieldDensity(p, h, dSmooth);
}

// カメラが emitter の support に入るとリムレイが 45/45 -> 0/45 に全滅し、これが
// 壁ルックへの二値切替になる。カメラ近傍の密度を 0 へ落として光学的にカメラを場の
// 外に置き、リムのシルエットが常に画面に残るようにする。
float flameNearCameraFade(vec3 pLocal) {
    float radius = flame.nearFadeParams.radius;
    if (radius <= 0.0) {
        return 1.0;
    }
    vec3 pWorld = (flame.model * vec4(pLocal, 1.0)).xyz;
    return smoothstep(0.0, radius, length(pWorld - frame.camera_pos.xyz));
}

#endif
