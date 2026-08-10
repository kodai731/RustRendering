#ifndef FLAME_RADIAL_INTEGRAL_GLSL
#define FLAME_RADIAL_INTEGRAL_GLSL

// Closed-form emission integral of the compact-support radial density
//   rho(p) = F(h) * (1 - u^2)^2,  u = |p.xz| / (S * R(h)),  zero for u >= 1
// over a ray segment. Along the ray u^2(s) is a quadratic g(s); the height
// envelope F is a degree-7 polynomial (Chebyshev series) and capFade a piecewise
// cubic in raw h, so within one piece the whole integrand is a single polynomial
// in the piece-local variable and every band integral is exact power-rule
// moments — no within-band approximation, hence no band-resolution limit on the
// envelope (the upper-band stripe artifact of the former 3-point quadratic).
// Pieces and occupancy nodes are cut at field-fixed knots (the capFade bounds,
// the displaced tip and the CPU-computed envelope edge crossings), which enter
// and leave the support interval with zero contribution — continuous per ray,
// no per-ray integer switches.
//
// Must be included after FlameUBO, evaluateHeightFalloff, and flame_noise_field.glsl
// (flameBiweight / flameRadialSupportRadius live there).
// Mirrored in thyllore-render-core/src/flame_radial.rs; the accuracy tests live there.

// Beer-Powder (Schneider 2015, non-physical): darkens the rim of carved
// notches so interior structure reads on an optically thick body.
// 0 = off (default look), 1 = full powder.
const float FLAME_POWDER_STRENGTH = 0.0;

// Radius R(h) of the radial density profile, in flame-local units. Texture fit
// bakes an arbitrary R(h) curve (profileParams.x flags it); the parametric taper
// stays the default.
float flameRadialRadiusScale(float height01) {
    if (flame.profileParams.x > 0.5) {
        return FLAME_SHELL_BASE_RADIUS
            * max(evaluateChebyshev8(flame.radiusCoefficients[0], flame.radiusCoefficients[1], height01), 0.05);
    }
    return FLAME_SHELL_BASE_RADIUS
        * mix(1.0, flame.styleParams1.x, pow(height01, flame.styleParams0.w));
}

// Squared reciprocal of the support radius S * R(h), which is what u^2 scales by.
float flameRadialSupportInvSq(float height01) {
    float scale = max(flameRadialSupportRadius() * flameRadialRadiusScale(height01), 1e-4);
    return 1.0 / (scale * scale);
}

float flameRadialDensityFactor(vec3 p, float height01) {
    float radiusSquared = dot(p.xz, p.xz);
    return flameBiweight(flameRadialSupportInvSq(height01) * radiusSquared);
}

// Contour wiggle: per-band field quantity from fbm at the actual 3D point.
// Returns 1.0 when contourParams.x == 0 (identity, matches old path).
float flameContourWiggle(vec3 p, float h) {
    if (flame.contourParams.x == 0.0 || flame.unifiedParams.x > 0.5) { return 1.0; }
    vec3 q = vec3(p.x, h - flame.styleParams0.z * flame.time, p.z) * flame.noiseFrequency;
    return 1.0 + flame.contourParams.x * flameDetailNoise(q);
}

// Pointwise field for the reference raymarch (mode 1): the same eroded threshold
// field the closed form approximates — true smoothstep, exact support membership.
float flamePointOccupancyDensity(vec3 p, float h, float wiggle) {
    vec2 boundary = flameBoundaryDisplacement(p.xz);
    float hb = clamp(h / boundary.x, 0.0, 1.0);
    float wb = wiggle * boundary.y;
    float dSmooth = evaluateHeightFalloff(hb) * flameCapFade(h, boundary.x)
        * flameRadialDensityFactor(vec3(p.x / wb, p.y, p.z / wb), hb)
        * flameNearCameraFade(p);
    float erosion = flameNoiseErosionValue(p, h, dSmooth);
    return flameApplyCarveResidual(
       flameResponseOccupancy(dSmooth, erosion, h),
        dSmooth) * flameFieldSupportMask(dSmooth);
}

// ---- Emitter-generic occupancy (ring / SDF analytic path) ----
// Every non-trail emitter shares the eroded threshold field
//   smoothstep(edgeLow, edgeHigh, dSmooth - erosion) * [dSmooth > 0];
// only dSmooth differs per emitter. The closed form needs dSmooth only at a few
// nodes per band: the argument is piecewise-linearized between exact node values
// and the response integral stays closed form — the sharpness lives in the
// response, not in the nodes. The cylinder keeps its specialized band integral
// (exact support intervals); this generic path serves ring and SDF, unwarped
// like the cylinder pair so mode 0 and mode 1 stay a parity pair.
// Mirrored in thyllore-render-core/src/flame_radial.rs (evaluate_occupancy_node_band).

// Pointwise field for the reference raymarch (mode 1) on ring/SDF emitters:
// the same field the node-based closed form approximates.
float flamePointEmitterOccupancy(vec3 p, float h, float wiggle) {
    float dSmooth = flameEmitterSmoothDensityAt(p, h, wiggle) * flameNearCameraFade(p);
    float erosion = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(p, h, dSmooth) : 0.0;
    return flameApplyCarveResidual(
      flameResponseOccupancy(dSmooth, erosion, h),
        dSmooth) * flameFieldSupportMask(dSmooth);
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar);
float integrateWaveOccupancy(vec3 o, vec3 d, float tNear, float tFar);
vec4 integrateWaveOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar);

// Rays with a near-constant height cannot be parameterized by h, so the moment is taken along t.
float integrateRadialEmission(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }
    return integrateWaveOccupancy(o, d, tNear, tFar);
}

vec3 flameRampColor(float h) {
    if (flame.profileParams.z > 0.5) {
        float u = clamp(h, 0.0, 1.0) * 8.0 - 0.5;
        int i0 = int(clamp(floor(u), 0.0, 7.0));
        int i1 = min(i0 + 1, 7);
        float f = clamp(u - float(i0), 0.0, 1.0);
        return mix(flame.colorRamp[i0].rgb, flame.colorRamp[i1].rgb, f);
    }
    if (h < 0.5) {
        return mix(flame.colorBase.rgb, flame.colorMid.rgb, h * 2.0);
    }
    return mix(flame.colorMid.rgb, flame.colorTip.rgb, (h - 0.5) * 2.0);
}

vec4 integrateRadialRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }
    return integrateWaveOccupancyRTE(o, d, tNear, tFar);
}

// Narrow [tNear, tFar] to the ray's crossing of the ring's outer support
// cylinder, conservative over height (widest taper) and contour wiggle.
// Trimming keeps the band layout attached to the flame body: the raw proxy
// interval of an inside-the-ring ray extends past the support to wherever the
// proxy exit lands, so the band layout would jump wherever that exit switches
// between the top slab and the cone side — the visible horizontal seam where
// the frozen-turbulence texture changes character with the view angle.
// The outer cylinder is convex, so the trimmed span is a single interval whose
// endpoints vary continuously with the ray — no per-pixel topology switches
// (subtracting the inner hole would split the span in two and re-allocating
// integer band counts between the parts creates new arc-shaped seams; the hole
// only costs resolution, its bands stay empty by density gating). Returns
// false when the ray misses the support entirely.
// Mirrored in thyllore-render-core/src/flame_radial.rs (ring_support_span).
bool flameRingSupportSpan(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    float rm = flame.emitterParams.y;
    float minorScale = max(1.0 - rm, 1e-3);
    float wTrim = (1.0 + max(flame.contourParams.x, 0.0))
        * (1.0 + 3.0 * abs(flame.boundaryParams.x) * max(flame.boundaryParams.w, 0.0));
    float taperMax = max(1.0, flame.styleParams1.x);
    float rOut = rm + minorScale * flameRadialSupportRadius() * taperMax * wTrim;

    float a = dot(d.xz, d.xz);
    float b = 2.0 * dot(o.xz, d.xz);
    float c = dot(o.xz, o.xz) - rOut * rOut;
    if (a < 1e-12) {
        return c <= 0.0;
    }
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant <= 0.0) {
        return false;
    }
    float root = sqrt(discriminant);
    tNear = max((-b - root) / (2.0 * a), tNear);
    tFar = min((-b + root) / (2.0 * a), tFar);
    return tFar > tNear;
}

// ---- Wave-basis band-free occupancy (mode-sum turbulence) ----
// The erosion field is an analytic sum of wave modes (flameWaveNoiseSum), so
// the closed form needs no radial bands: one support crossing, uniform
// segments whose positions are affine in the interval (endpoints continuous in
// the ray — no per-ray integer switches), density AND erosion exact at every
// node (both analytic, nothing sampled from a lattice realization), the
// argument linear between nodes, each segment closed-form erf response.
// Modes are split by a ray-fixed physical cutoff g_n = exp(-(omega_n
// alpha_ref)^2 / 2) on the transition-shell crossing time alpha_ref: captured
// (slow) modes enter the argument as exact values (their frequency is far
// below the segment Nyquist, so value sampling cannot alias — the lattice
// fringe source is structurally absent), while the uncaptured share and the
// tanh distortion fold into the response sigma through conditional statistics,
// adapted per segment via log2|omega| band powers.
// Mirrored in thyllore-render-debug/src/flame_field_trace.rs
// (faddeeva_segment_estimate / carrier_slow_state / solve_reference_cutoff).

#ifdef FLAME_WAVE_SEGMENTS_OVERRIDE
const int FLAME_WAVE_SEGMENTS = FLAME_WAVE_SEGMENTS_OVERRIDE;
#else
const int FLAME_WAVE_SEGMENTS = 64;
#endif
// Exact node density of the wave path. The cylinder keeps its density
// convention (support radius with FLAME_SHELL_BASE_RADIUS and the baked R(h)
// curve, exactly the flamePointOccupancyDensity smooth part) so switching the
// noise basis never changes the flame silhouette; ring and SDF use the shared
// emitter density like their raymarch pair.
float flameWaveNodeDensity(vec3 p, float h) {
    float wiggle = flameContourWiggle(p, h);
    vec2 boundary = flameBoundaryDisplacement(p.xz);
    float dens;
    if (flame.emitterParams.x < 0.5) {
        float hb = clamp(h / boundary.x, 0.0, 1.0);
        float wb = max(wiggle * boundary.y, 1e-4);
        dens = evaluateHeightFalloff(hb) * flameCapFade(h, boundary.x)
            * flameRadialDensityFactor(vec3(p.x / wb, p.y, p.z / wb), hb);
    } else {
        dens = flameEmitterSmoothDensityDisplacedAt(p, h, wiggle, boundary);
    }
    return dens * flameNearCameraFade(p);
}
// S3 — support-edge crossing between a dead node (density <= 0) and a live
// node (density > 0): fixed-count bisection on support membership. The count
// is constant, so there is no per-ray integer switch; the returned point moves
// continuously with the ray and locates the edge to (t1 - t0) / 2^steps.
// Mirrored in thyllore-render-debug/src/flame_field_trace.rs.
const int FLAME_SUPPORT_BISECTION_STEPS = 8;
float flameWaveSupportCrossing(vec3 o, vec3 d, float tDead, float tLive) {
    for (int iter = 0; iter < FLAME_SUPPORT_BISECTION_STEPS; ++iter) {
        float tMid = 0.5 * (tDead + tLive);
        vec3 pMid = o + tMid * d;
        if (flameWaveNodeDensity(pMid, clamp(pMid.y, 0.0, 1.0)) > 0.0) {
            tLive = tMid;
        } else {
            tDead = tMid;
        }
    }
    return 0.5 * (tDead + tLive);
}

// ---- Continuous ray integrator (v5) walk pieces ----
// Mirrored in thyllore-render-debug/src/flame_field_trace.rs
// (mean_argument_at / solve_reference_cutoff / faddeeva_segment_estimate).
const int FLAME_CROSSING_SCAN_INTERVALS = 256;
const int FLAME_CROSSING_BISECTION_STEPS = 20;

// Mean (carrier-free) argument: density with only the deterministic shrink.
float flameMeanArgumentAt(vec3 o, vec3 d, float t) {
    vec3 p = o + t * d;
    float h = clamp(p.y, 0.0, 1.0);
    float density = flameWaveNodeDensity(p, h);
    float meanErosion = flame.noiseAmplitude * mix(0.2, 1.0, h) * FLAME_EROSION_MEAN_SHRINK;
    return flameErodedArgument(density, meanErosion);
}

// Ray-fixed reference capture cutoff: the sharpest mean-line shell crossing
// decides which modes the whole ray resolves as values. Density-only scan on
// a lattice-independent grid; conservative full-fold sigma for the width.
float flameSolveReferenceCutoff(
    vec3 o, vec3 d, float t0, float spanTotal,
    FlameCarrierConstants cc, out bool hasCutoff) {
    float center = flame.erosionResponse.x;
    float scanDt = spanTotal / float(FLAME_CROSSING_SCAN_INTERVALS);
    float slopeEps = 1e-3 * spanTotal;
    hasCutoff = false;
    float alphaRef = 0.0;
    float fA = flameMeanArgumentAt(o, d, t0) - center;
    for (int interval = 0; interval < FLAME_CROSSING_SCAN_INTERVALS; ++interval) {
        float tA = t0 + float(interval) * scanDt;
        float tB = tA + scanDt;
        float fB = flameMeanArgumentAt(o, d, tB) - center;
        if (fA * fB < 0.0) {
            float lo = tA;
            float hi = tB;
            bool loNegative = fA < 0.0;
            for (int iter = 0; iter < FLAME_CROSSING_BISECTION_STEPS; ++iter) {
                float mid = 0.5 * (lo + hi);
                if ((flameMeanArgumentAt(o, d, mid) - center < 0.0) == loNegative) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            float tStar = 0.5 * (lo + hi);
            vec3 pStar = o + tStar * d;
            float hStar = clamp(pStar.y, 0.0, 1.0);
            float density = flameWaveNodeDensity(pStar, hStar);
            if (density > 0.0) {
                float fade = flameEnvelopeFade(density);
                float geometry = flame.noiseAmplitude * flameTipCarveLambda(hStar)
                    * (density / FLAME_EROSION_SHELL_REF) * fade;
                float sigmaFloor = flame.unifiedParams.x > 0.5
                    ? flame.unifiedParams.y * flameTipCarveLambda(hStar)
                        * (density / FLAME_EROSION_SHELL_REF) * fade
                    : 0.0;
                float sigmaFull = max(
                    flameFoldedSigmaArgument(geometry, cc, cc.modalPower), sigmaFloor);
                float shellWidth =
                    1.0 / (1.41421356 * flameSmoothErosionResponse(sigmaFull).kappaEff);
                float slope = (flameMeanArgumentAt(o, d, tStar + slopeEps)
                    - flameMeanArgumentAt(o, d, tStar - slopeEps)) / (2.0 * slopeEps);
                if (abs(slope) > 1e-6) {
                    float alpha = shellWidth / abs(slope);
                    alphaRef = hasCutoff ? min(alphaRef, alpha) : alpha;
                    hasCutoff = true;
                }
            }
        }
        fA = fB;
    }
    return alphaRef;
}

// Per-segment estimate: exact slow values in the argument, band-power sigma
// fold adapted to the realized slope, erf closed form.
vec2 flameWaveSegmentCarvedV5(
    vec3 o, vec3 d, float segStart, float segEnd, float span,
    float densityStart, float densityEnd,
    FlameCarrierConstants cc, bool hasCutoff, float alphaRef,
    FlameCarrierState stateStart, FlameCarrierState stateEnd,
    float residual) {
    float hMid = clamp(o.y + (segStart + 0.5 * span) * d.y, 0.0, 1.0);
    float densityAvg = 0.5 * (densityStart + densityEnd);
    float fadeAvg = 0.5 * (flameEnvelopeFade(densityStart) + flameEnvelopeFade(densityEnd));
    float geometry = flame.noiseAmplitude * flameTipCarveLambda(hMid)
        * (densityAvg / FLAME_EROSION_SHELL_REF) * fadeAvg;
    float sigmaFloor = flame.unifiedParams.x > 0.5
        ? flame.unifiedParams.y * flameTipCarveLambda(hMid)
            * (densityAvg / FLAME_EROSION_SHELL_REF) * fadeAvg
        : 0.0;

    float capturedRef = 0.5 * (flameCapturedPower(stateStart, hasCutoff, alphaRef)
        + flameCapturedPower(stateEnd, hasCutoff, alphaRef));
    float foldedRef = max(cc.modalPower - capturedRef, 0.0);
    float sigmaFast = sqrt(cc.sigmaBase * cc.sigmaBase + foldedRef);

    float hStart = clamp(o.y + segStart * d.y, 0.0, 1.0);
    float argStart = flameErodedArgument(densityStart, flameNoiseErosionFromValue(
        0.4375 + flameShapedDeltaMean(stateStart.zSlow, sigmaFast), hStart, densityStart));
    float hEnd = clamp(o.y + segEnd * d.y, 0.0, 1.0);
    float argEnd = flameErodedArgument(densityEnd, flameNoiseErosionFromValue(
        0.4375 + flameShapedDeltaMean(stateEnd.zSlow, sigmaFast), hEnd, densityEnd));
    float slope = (argEnd - argStart) / span;

    float folded = foldedRef;
    float sigmaSmooth = max(flameFoldedSigmaArgument(geometry, cc, folded), sigmaFloor);
    if (hasCutoff) {
        for (int pass = 0; pass < 2; ++pass) {
            sigmaSmooth = max(flameFoldedSigmaArgument(geometry, cc, folded), sigmaFloor);
            float shellWidth =
                1.0 / (1.41421356 * flameSmoothErosionResponse(sigmaSmooth).kappaEff);
            float capturedLocal = 0.0;
            if (abs(slope) > 1e-6) {
                float alphaLocal = max(shellWidth / abs(slope), alphaRef);
                capturedLocal = 0.5 * (flameCapturedPower(stateStart, true, alphaLocal)
                    + flameCapturedPower(stateEnd, true, alphaLocal));
            }
            folded = max(cc.modalPower - min(capturedLocal, capturedRef), 0.0);
        }
    }

    FlameSmoothedResponse response = flameSmoothErosionResponse(sigmaSmooth);
    vec2 carved = flameErosionResponseLinearIntegral(
        response, argStart - slope * segStart, slope, segStart, segEnd);
    if (residual > 0.0) {
        float plainSlope = (densityEnd - densityStart) / span;
        vec2 plain = flameErosionResponseLinearIntegral(
            flameSmoothErosionResponse(0.0),
            densityStart - plainSlope * segStart, plainSlope, segStart, segEnd);
        carved = mix(carved, plain, residual);
    }
    return carved;
}

// Streaming reduction of the segment walk. No per-segment arrays: the walk
// below feeds each segment's (emission, tMean) straight into these
// accumulators, so nothing spills to scratch memory. The RTE booster is a
// scalar on the whole radiance sum, so radiancePre accumulates without it and
// the caller multiplies once at the end (identical math, only the rounding
// order of that product changes).
struct FlameWaveIntegral {
    float total;
    float heightMeanNum;
    vec3 radiancePre;
    vec3 transmittance;
};

FlameWaveIntegral flameWaveOccupancySegments(
    vec3 o, vec3 d, float t0, float t1, bool rte) {
    FlameWaveIntegral acc;
    acc.total = 0.0;
    acc.heightMeanNum = 0.0;
    acc.radiancePre = vec3(0.0);
    acc.transmittance = vec3(1.0);
    float dt = (t1 - t0) / float(FLAME_WAVE_SEGMENTS);
    t0 += (interleavedGradientNoise(gl_FragCoord.xy) - 0.5) * dt;
    if (dt <= 0.0) {
        return acc;
    }
    vec3 sigmaRgb = flame.sigmaT
        * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.w, 0.0, 1.0));

    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = min(int(flame.waveParams.x), FLAME_WAVE_EROSION_SLOTS);

    // Streaming node walk: density first, the mode sum only at nodes touching
    // support (empty segments cost no mode evaluation), one node carried
    // between segments so nothing is evaluated twice. Segments straddling the
    // support edge are cut at the actual crossing (fixed-count bisection on the
    // density, continuous per ray) and their edge argument evaluated at the
    // crossing instead of extrapolated from the dead node, so the silhouette
    // edge resolves independently of the segment count and nothing is emitted
    // outside the support the raymarch pair masks to zero.
    float residual = flameCarveResidualStrength();
    FlameCarrierConstants carrierConstants = flameCarrierConstants(count);
    bool hasCutoff;
    float alphaRef = flameSolveReferenceCutoff(
        o, d, t0, dt * float(FLAME_WAVE_SEGMENTS), carrierConstants, hasCutoff);
    float previousDensity = flameWaveNodeDensity(o + t0 * d, clamp(o.y + t0 * d.y, 0.0, 1.0));
    FlameCarrierState previousState;
    bool previousStateValid = false;
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        float tPrev = t0 + float(segment) * dt;
        float t = tPrev + dt;
        vec3 p = o + t * d;
        float h = clamp(p.y, 0.0, 1.0);
        float density = flameWaveNodeDensity(p, h);
        if (previousDensity <= 0.0 && density <= 0.0) {
            previousDensity = density;
            previousStateValid = false;
            continue;
        }
        bool entering = previousDensity <= 0.0;
        bool exiting = density <= 0.0;
        float segStart = entering ? flameWaveSupportCrossing(o, d, tPrev, t) : tPrev;
        float segEnd = exiting ? flameWaveSupportCrossing(o, d, t, tPrev) : t;
        float span = segEnd - segStart;
        if (span < 1e-4 * dt) {
            previousDensity = density;
            previousStateValid = false;
            continue;
        }
        float densityStart = entering ? 0.0 : previousDensity;
        float densityEnd = exiting ? 0.0 : density;
        if (entering || !previousStateValid) {
            float tEval = entering ? segStart : tPrev;
            vec3 pEval = o + tEval * d;
            previousState = flameCarrierSlowState(
                flameBuildWarpFrame(pEval, d, clamp(pEval.y, 0.0, 1.0)),
                count, eddyTime, carrierConstants, hasCutoff, alphaRef);
        }
        FlameCarrierState currentState;
        {
            float tEval = exiting ? segEnd : t;
            vec3 pEval = o + tEval * d;
            currentState = flameCarrierSlowState(
                flameBuildWarpFrame(pEval, d, clamp(pEval.y, 0.0, 1.0)),
                count, eddyTime, carrierConstants, hasCutoff, alphaRef);
        }
        vec2 carved = flameWaveSegmentCarvedV5(
            o, d, segStart, segEnd, span, densityStart, densityEnd,
            carrierConstants, hasCutoff, alphaRef, previousState, currentState, residual);
        float emission = max(carved.x, 0.0);
        float tMean = carved.x > 1e-6
            ? clamp(carved.y / carved.x, segStart, segEnd)
            : t0 + (float(segment) + 0.5) * dt;
        acc.total += emission;
        if (rte) {
            vec3 pMean = o + tMean * d;
            float hMean = clamp(pMean.y, 0.0, 1.0);
            acc.heightMeanNum += emission * hMean;
            float edge = 0.0;
            if (flame.emitterParams.x < 1.5) {
                float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
                float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
                float taperR = mix(1.0, flame.styleParams1.x, pow(hMean, flame.styleParams0.w));
                float rhoNorm = abs((length(pMean.xz) - rm) / minorScale) / max(taperR, 1e-4);
                edge = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, rhoNorm), 0.0, 1.0);
            }
            vec3 tau = sigmaRgb * emission;
            vec3 absorbed = vec3(1.0) - exp(-tau);
            absorbed = mix(absorbed, absorbed * (vec3(1.0) - exp(-2.0 * tau)), FLAME_POWDER_STRENGTH);
            acc.radiancePre += acc.transmittance
                * mix(flameRampColor(hMean), flame.colorTip.rgb, edge) * absorbed;
            acc.transmittance *= exp(-tau);
        }

        previousDensity = density;
        previousState = currentState;
        previousStateValid = !exiting;
    }
    return acc;
}


// Support crossing of the wave path: the ring trims to its convex outer
// cylinder like the band path; cylinder and SDF keep the proxy interval
// (empty segments are gated by the node densities).
bool flameWaveSupportSpan(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    bool ringEmitter = flame.emitterParams.x >= 0.5 && flame.emitterParams.x < 1.5;
    if (ringEmitter) {
        return flameRingSupportSpan(o, d, tNear, tFar);
    }
    return tFar > tNear;
}

float integrateWaveOccupancy(vec3 o, vec3 d, float tNear, float tFar) {
    if (!flameWaveSupportSpan(o, d, tNear, tFar)) {
        return 0.0;
    }
    return flameWaveOccupancySegments(o, d, tNear, tFar, false).total;
}

// Beer-Lambert composite directly over the wave segments (camera-ordered by
// construction), streamed inside the walk — no per-segment arrays anywhere.
vec4 integrateWaveOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (!flameWaveSupportSpan(o, d, tNear, tFar)) {
        return vec4(0.0);
    }
    FlameWaveIntegral acc = flameWaveOccupancySegments(o, d, tNear, tFar, true);

    float heightMean = acc.total > 1e-6 ? acc.heightMeanNum / acc.total : 0.0;
    float tempNorm = clamp(acc.total * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMean);
    float boost = 1.0 + flame.styleParams1.w * tempNorm * tempNorm;

    vec3 radiance = acc.radiancePre * flame.intensity * boost;
    return vec4(radiance, 1.0 - dot(acc.transmittance, vec3(1.0 / 3.0)));
}

vec4 integrateEmitterOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }
    return integrateWaveOccupancyRTE(o, d, tNear, tFar);
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }
    return integrateWaveOccupancy(o, d, tNear, tFar);
}

#endif
