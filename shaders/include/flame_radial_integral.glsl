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
// bakes an arbitrary R(h) curve (profileParams.radiusActive flags it); the parametric taper
// stays the default.
float flameRadialRadiusScale(float height01) {
    if (flame.profileParams.radiusActive > 0.5) {
        return FLAME_SHELL_BASE_RADIUS
            * max(evaluateChebyshev8(flame.radiusCoefficients[0], flame.radiusCoefficients[1], height01), 0.05);
    }
    return FLAME_SHELL_BASE_RADIUS
        * mix(1.0, flame.edgeStyle.radiusTipRatio, pow(height01, flame.warpStyle.taperPower));
}

// Squared reciprocal of the support radius S * R(h), which is what u^2 scales by.
float flameRadialSupportInvSq(float height01) {
    float scale = max(flameRadialSupportRadius() * flameRadialRadiusScale(height01), 1e-4);
    return 1.0 / (scale * scale);
}

float flameRadialDensityFactor(vec3 p, float height01, out float uSquared) {
    uSquared = flameRadialSupportInvSq(height01) * dot(p.xz, p.xz);
    return flameBiweight(uSquared);
}

float flameRadialDensityFactor(vec3 p, float height01) {
    float _unused;
    return flameRadialDensityFactor(p, height01, _unused);
}

// Contour wiggle: per-band field quantity from fbm at the actual 3D point.
// Returns 1.0 when the contour wiggle amp is 0 (identity, matches old path).
float flameContourWiggle(vec3 p, float h) {
    if (flame.contourParams.wiggleAmp == 0.0 || flame.unifiedParams.enabled > 0.5) { return 1.0; }
    vec3 q = vec3(p.x, h - flame.warpStyle.riseSpeed * flame.time, p.z) * flame.noiseFrequency;
    return 1.0 + flame.contourParams.wiggleAmp * flameDetailNoise(q);
}

// Pointwise field for the reference raymarch (mode 1): the same eroded threshold
// field the closed form approximates — true smoothstep, exact support membership.
float flamePointOccupancyDensity(vec3 p, float h, float wiggle) {
    float hs = h;
    float burnout;
    vec3 ps = flameSupportPositionBurnout(p, hs, burnout);
    vec2 boundary = flameBoundaryDisplacement(ps.xz);
    float hb = clamp(hs / boundary.x, 0.0, 1.0);
    float wb = wiggle * boundary.y;
    float uSquared;
    float radial = flameRadialDensityFactor(vec3(ps.x / wb, ps.y, ps.z / wb), hb, uSquared);
    float dSmooth = evaluateHeightFalloff(hb) * flameCapFade(hs, boundary.x) * radial
        * flameNearCameraFade(p) * burnout;
   float erosion = flameNoiseErosionValue(p, h, dSmooth, uSquared);
    return flameApplyCarveResidual(
       flameResponseOccupancy(dSmooth, erosion, hs, uSquared),
        dSmooth, uSquared) * flameFieldSupportMask(dSmooth);
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
    float hs = h;
    float burnout;
    vec3 ps = flameSupportPositionBurnout(p, hs, burnout);
    float uSquared;
    float dSmooth = flameEmitterSmoothDensityDisplacedAt(ps, hs, wiggle, flameBoundaryDisplacement(ps.xz), uSquared)
        * flameNearCameraFade(p) * burnout;
   float erosion = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(p, h, dSmooth, uSquared) : 0.0;
    return flameApplyCarveResidual(
      flameResponseOccupancy(dSmooth, erosion, hs, uSquared),
        dSmooth, uSquared) * flameFieldSupportMask(dSmooth);
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
    if (flame.profileParams.colorActive > 0.5) {
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

vec3 flameTemperatureColor(float temperatureK) {
    float span = max(flame.thermalParams.tempHotK - flame.thermalParams.tempColdK, 1.0);
    float u = clamp((temperatureK - flame.thermalParams.tempColdK) / span, 0.0, 1.0) * 8.0 - 0.5;
    int i0 = int(clamp(floor(u), 0.0, 7.0));
    int i1 = min(i0 + 1, 7);
    float f = clamp(u - float(i0), 0.0, 1.0);
    return mix(flame.tempRamp[i0].rgb, flame.tempRamp[i1].rgb, f);
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
    float rm = flame.emitterParams.ringMajorRatio;
    float minorScale = max(1.0 - rm, 1e-3);
    float wTrim = (1.0 + max(flame.contourParams.wiggleAmp, 0.0))
        * (1.0 + 3.0 * abs(flame.boundaryParams.amp) * max(flame.boundaryParams.radiusRatio, 0.0));
    float taperMax = max(1.0, flame.edgeStyle.radiusTipRatio);
   float rOut = rm + minorScale * flameRadialSupportRadius() * taperMax * wTrim + 2.0 * flame.supportMotion.meanderAmp;

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
// Modes the node spacing cannot resolve are attenuated by a smooth low-pass in
// beta = k . d and their power routed into the response sigma — the smooth
// analogue of the fbm path's FLAME_EROSION_NOISE_SIGMA, with no h-quantized
// band structure anywhere (band-boundary-coherence.md, E1).
// Mirrored in thyllore-render-core/src/flame_wave.rs
// (evaluate_wave_occupancy_segments / wave_ray_attenuation).

// Exact node density of the wave path. The cylinder keeps its density
// convention (support radius with FLAME_SHELL_BASE_RADIUS and the baked R(h)
// curve, exactly the flamePointOccupancyDensity smooth part) so switching the
// noise basis never changes the flame silhouette; ring and SDF use the shared
// emitter density like their raymarch pair.
float flameWaveNodeDensity(vec3 p, float h) {
    float burnout;
    vec3 ps = flameSupportPositionBurnout(p, h, burnout);
    float wiggle = flameContourWiggle(ps, h);
    vec2 boundary = flameBoundaryDisplacement(ps.xz);
    float dens;
    if (flame.emitterParams.kind < 0.5) {
        float hb = clamp(h / boundary.x, 0.0, 1.0);
        float wb = max(wiggle * boundary.y, 1e-4);
        dens = evaluateHeightFalloff(hb) * flameCapFade(h, boundary.x)
            * flameRadialDensityFactor(vec3(ps.x / wb, ps.y, ps.z / wb), hb);
    } else {
        dens = flameEmitterSmoothDensityDisplacedAt(ps, h, wiggle, boundary);
    }
    return dens * flameNearCameraFade(p) * burnout;
}
// Everything one node contributes to the segment closed form. `mixDensity` and
// `emissivity` are the mixing-degree curves at the node; they scale the segment
// mass and radiance only, so the carved argument and the support are untouched.
struct FlameNodeSample {
    float argument;
    float shapedNoise;
    float sigmaNoise;
    float remapScale;
    float density;
    float mixDensity;
    float temperature;
    float emissivity;
};

FlameNodeSample flameNodeSampleEmpty() {
    FlameNodeSample node;
    node.argument = 0.0;
    node.shapedNoise = 0.4375;
    node.sigmaNoise = 0.0;
    node.remapScale = 1.0;
    node.density = 0.0;
    node.mixDensity = 1.0;
    node.temperature = flame.thermalParams.tempHotK;
    node.emissivity = 1.0;
    return node;
}

// Node-local low-pass: weights come from the warped rate at this node, so a
// locally stretched node does not smooth the whole ray.
FlameNodeSample flameWaveNodeSample(
    vec3 p, vec3 d, float h, float density, float dt, int count, float eddyTime) {
    FlameWarpFrame warpFrame = flameBuildWarpFrame(p, d, h);
    float hs = warpFrame.h;
    FlameWaveModeSumResult sum = flameWaveModeSum(
        warpFrame.w, warpFrame.rate, warpFrame.pb, d, hs, dt, count, eddyTime);

    // 5.1 probabilistic reduction: erosion modes past the tracked count (sorted by
    // |k| ascending on the CPU) are not tracked; their full variance enters the
    // response as blur. The skipped high-octave power rides the same low-octave
    // envelope as the tracked high modes (uniform waveParams.envCoeff).
    float unresolvedPower = sum.unresolvedPower + flame.waveCfParams.skippedPowerPlain;
    float envSkip = 1.0 + flame.waveParams.envCoeff * sum.zLow;
    unresolvedPower += flame.waveCfParams.skippedPowerEnv * envSkip * envSkip;

    FlameNodeSample node;
    node.sigmaNoise = sqrt(unresolvedPower);
    float invScale = flame.waveParams.inverseScale;
    float amp = flame.waveParams.amplitude;
    node.shapedNoise = invScale > 0.0 ? 0.4375 + amp * tanh(sum.z * invScale) : 0.4375 + sum.z;
    node.density = density;

    float uSquared;
    if (flame.emitterParams.kind >= 1.5) {
        uSquared = 0.0;
    } else {
        vec3 ps = flameSupportPosition(p, h);
        float wiggle = flameContourWiggle(ps, h);
        vec2 boundary = flameBoundaryDisplacement(ps.xz);
        float hb = clamp(h / boundary.x, 0.0, 1.0);
        float taperR = mix(1.0, flame.edgeStyle.radiusTipRatio, pow(hb, flame.warpStyle.taperPower));
        float rm = flame.emitterParams.kind >= 0.5 ? flame.emitterParams.ringMajorRatio : 0.0;
        float minorScale = flame.emitterParams.kind >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
        float rho = (length(ps.xz) - rm) / minorScale;
        float rn = abs(rho) / max(taperR * wiggle * boundary.y, 1e-4);
        float u = rn / flameRadialSupportRadius();
        uSquared = u * u;
    }
    float mixing = flameMixingDegree(sum.zMix, hs, uSquared);
    node.mixDensity = flameMixDensityFactor(mixing);
    node.temperature = flameMixTemperature(mixing);
    node.emissivity = flameWienEmissivity(node.temperature);
    float erosion = flameNoiseErosionFromValue(node.shapedNoise, hs, node.density, uSquared);
    node.remapScale = flameErosionRemapScale(erosion);
    node.argument = flameErodedArgument(node.density, erosion);
    return node;
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


// Node endpoint state of one segment, filled by the walk and consumed by the
// pure per-segment math below (start = the walk's carried previous node or the
// entering-edge crossing, end = the current node or the exiting-edge crossing).
struct FlameSegmentNodes {
    float argumentStart;
    float argumentEnd;
    float shapedStart;
    float shapedEnd;
    float sigmaStart;
    float sigmaEnd;
    float remapStart;
    float remapEnd;
    float densityStart;
    float densityEnd;
    float mixDensityStart;
    float mixDensityEnd;
    float temperatureStart;
    float temperatureEnd;
    float emissivityStart;
    float emissivityEnd;
};

// Pure per-segment math (no walk state): sigmaEff composition from the node
// endpoints, the erf closed-form integral of the piecewise-linear argument,
// and the carve-residual floor. Returns carved = (emission, first moment).
vec2 flameWaveSegmentCarved(
    FlameSegmentNodes nodes, vec3 o, vec3 d,
    float segStart, float segEnd, float span,
    float uSquared, float invScale, float amp) {
    // Shaping derivative average: g'(z) = amp * invScale * (1 - t^2) where
    // t = (shapedNoise - 0.4375) / amp; if invScale <= 0 then gPrime = 1.0.
    float shapingDerivAvg;
    if (invScale > 0.0) {
        float tPrevVal = (nodes.shapedStart - 0.4375) / amp;
        float tCurrVal = (nodes.shapedEnd - 0.4375) / amp;
        shapingDerivAvg = 0.5 * amp * invScale
            * ((1.0 - tPrevVal * tPrevVal) + (1.0 - tCurrVal * tCurrVal));
    } else {
        shapingDerivAvg = 1.0;
    }

    float hMid = clamp(o.y + (segStart + 0.5 * span) * d.y, 0.0, 1.0);
    float sigmaEff = flame.spreadParams.erosionNoiseGain * 0.5 * (nodes.sigmaStart + nodes.sigmaEnd) * shapingDerivAvg * abs(flame.noiseAmplitude)
        * flameTipCarveLambda(hMid) * (0.5 * (nodes.densityStart + nodes.densityEnd) / FLAME_EROSION_SHELL_REF)
        * 0.5 * (flameEnvelopeFade(nodes.densityStart) + flameEnvelopeFade(nodes.densityEnd))
        * 0.5 * (nodes.remapStart + nodes.remapEnd);
    float slope = (nodes.argumentEnd - nodes.argumentStart) / span;
    if (flame.unifiedParams.enabled > 0.5) {
        float sigmaFloor = flame.unifiedParams.sigmaFloor * flameTipCarveLambda(hMid)
            * (0.5 * (nodes.densityStart + nodes.densityEnd) / FLAME_EROSION_SHELL_REF)
            * 0.5 * (flameEnvelopeFade(nodes.densityStart) + flameEnvelopeFade(nodes.densityEnd));
        sigmaFloor *= mix(1.0, 1.0 - flame.spreadParams.edgeOuterSharpen, flameCarveResidualOuterGate(uSquared));
        sigmaEff = max(sigmaEff, sigmaFloor);
    }
    FlameSmoothedResponse response = flameSmoothErosionResponse(sigmaEff);
    vec2 carved = flameErosionResponseLinearIntegral(
        response, nodes.argumentStart - slope * segStart, slope, segStart, segEnd);
    float residual = flameCarveResidualStrength(uSquared);
    if (residual > 0.0) {
        float plainSlope = (nodes.densityEnd - nodes.densityStart) / span;
        vec2 plain = flameErosionResponseLinearIntegral(
            flameSmoothErosionResponse(0.0),
            nodes.densityStart - plainSlope * segStart, plainSlope, segStart, segEnd);
        carved = mix(carved, plain, residual);
    }
    return carved;
}

// The segment grid jitter walks with the frame only while the history blend
// can average it (accumWeight > 0): a still batch frame keeps the static grid.
vec2 flameSegmentJitterShift() {
    if (flame.temporalData.accumWeight <= 0.0) {
        return vec2(0.0);
    }
    return vec2(flame.temporalData.frameIndex * 5.588238);
}

// Streaming reduction of the segment walk. No per-segment arrays: the walk
// below feeds each segment's (emission, tMean) straight into these
// accumulators, so nothing spills to scratch memory. The RTE booster is a
// scalar on the whole radiance sum, so radiancePre accumulates without it and
// the caller multiplies once at the end (identical math, only the rounding
// order of that product changes).
struct FlameWaveIntegral {
    float total;
    vec3 radiancePre;
    vec3 transmittance;
};

FlameWaveIntegral flameWaveOccupancySegments(
    vec3 o, vec3 d, float t0, float t1, bool rte) {
    FlameWaveIntegral acc;
    acc.total = 0.0;
    acc.radiancePre = vec3(0.0);
    acc.transmittance = vec3(1.0);
    int segmentCount = int(flame.segmentParams.count);
    float dt = (t1 - t0) * flame.segmentParams.invCount;
    t0 += (interleavedGradientNoise(gl_FragCoord.xy + flameSegmentJitterShift()) - 0.5) * dt;
    if (dt <= 0.0) {
        return acc;
    }
    vec3 sigmaRgb = flame.sigmaT
        * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.sigmaDispersion, 0.0, 1.0));

    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = min(int(flame.waveParams.trackedCount), FLAME_WAVE_EROSION_SLOTS);

    // Streaming node walk: density first, the mode sum only at nodes touching
    // support (empty segments cost no mode evaluation), one node carried
    // between segments so nothing is evaluated twice. Segments straddling the
    // support edge are cut at the actual crossing (fixed-count bisection on the
    // density, continuous per ray) and their edge argument evaluated at the
    // crossing instead of extrapolated from the dead node, so the silhouette
    // edge resolves independently of the segment count and nothing is emitted
    // outside the support the raymarch pair masks to zero.
    float invScale = flame.waveParams.inverseScale;
    float amp = flame.waveParams.amplitude;
    float previousDensity = flameWaveNodeDensity(o + t0 * d, clamp(o.y + t0 * d.y, 0.0, 1.0));
    FlameNodeSample previous = flameNodeSampleEmpty();
    bool previousValid = false;
    for (int segment = 0; segment < segmentCount; ++segment) {
        float tPrev = t0 + float(segment) * dt;
        float t = tPrev + dt;
        vec3 p = o + t * d;
        float h = clamp(p.y, 0.0, 1.0);
        float density = flameWaveNodeDensity(p, h);
        if (previousDensity <= 0.0 && density <= 0.0) {
            previousDensity = density;
            previousValid = false;
            continue;
        }
        bool entering = previousDensity <= 0.0;
        bool exiting = density <= 0.0;
        float segStart = entering ? flameWaveSupportCrossing(o, d, tPrev, t) : tPrev;
        float segEnd = exiting ? flameWaveSupportCrossing(o, d, t, tPrev) : t;
        float span = segEnd - segStart;
        if (span < 1e-4 * dt) {
            previousDensity = density;
            previousValid = false;
            continue;
        }
        if (entering) {
            vec3 pStart = o + segStart * d;
            previous = flameWaveNodeSample(
                pStart, d, clamp(pStart.y, 0.0, 1.0), 0.0, dt, count, eddyTime);
        } else if (!previousValid) {
            vec3 pPrev = o + tPrev * d;
            previous = flameWaveNodeSample(
                pPrev, d, clamp(pPrev.y, 0.0, 1.0), previousDensity, dt, count, eddyTime);
        }
        FlameNodeSample current;
        if (exiting) {
            vec3 pEnd = o + segEnd * d;
            current = flameWaveNodeSample(pEnd, d, clamp(pEnd.y, 0.0, 1.0), 0.0, dt, count, eddyTime);
        } else {
            current = flameWaveNodeSample(p, d, h, density, dt, count, eddyTime);
        }

        FlameSegmentNodes nodes;
        nodes.argumentStart = previous.argument;
        nodes.argumentEnd = current.argument;
        nodes.shapedStart = previous.shapedNoise;
        nodes.shapedEnd = current.shapedNoise;
        nodes.sigmaStart = previous.sigmaNoise;
        nodes.sigmaEnd = current.sigmaNoise;
        nodes.remapStart = previous.remapScale;
        nodes.remapEnd = current.remapScale;
        nodes.densityStart = previous.density;
        nodes.densityEnd = current.density;
        nodes.mixDensityStart = previous.mixDensity;
        nodes.mixDensityEnd = current.mixDensity;
        nodes.temperatureStart = previous.temperature;
        nodes.temperatureEnd = current.temperature;
        nodes.emissivityStart = previous.emissivity;
        nodes.emissivityEnd = current.emissivity;
        float hMid = clamp(o.y + (segStart + 0.5 * span) * d.y, 0.0, 1.0);
        vec3 pMid = o + (segStart + 0.5 * span) * d;
        float uSquared;
        if (flame.emitterParams.kind < 0.5) {
            float wb = max(flameContourWiggle(pMid, hMid) * flameBoundaryDisplacement(pMid.xz).y, 1e-4);
            flameRadialDensityFactor(pMid / wb, hMid, uSquared);
        } else {
            vec2 boundary = flameBoundaryDisplacement(pMid.xz);
            float wiggle = flameContourWiggle(pMid, hMid);
            flameEmitterSmoothDensityDisplacedAt(pMid, hMid, wiggle, boundary, uSquared);
        }
        vec2 carved = flameWaveSegmentCarved(
            nodes, o, d, segStart, segEnd, span, uSquared, invScale, amp);
        float emission = max(carved.x, 0.0) * 0.5 * (nodes.mixDensityStart + nodes.mixDensityEnd);
        acc.total += emission;
        if (rte) {
            vec3 tau = sigmaRgb * emission;
            vec3 absorbed = vec3(1.0) - exp(-tau);
            absorbed = mix(absorbed, absorbed * (vec3(1.0) - exp(-2.0 * tau)), FLAME_POWDER_STRENGTH);
            float emissivity = 0.5 * (nodes.emissivityStart + nodes.emissivityEnd);
            vec3 color = flameTemperatureColor(0.5 * (nodes.temperatureStart + nodes.temperatureEnd));
            acc.radiancePre += acc.transmittance * color * absorbed * emissivity;
            acc.transmittance *= exp(-tau);
        }

        previousDensity = density;
        previous = current;
        previousValid = !exiting;
    }
    return acc;
}


// Support crossing of the wave path: the ring trims to its convex outer
// cylinder like the band path; cylinder and SDF keep the proxy interval
// (empty segments are gated by the node densities).
bool flameWaveSupportSpan(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    bool ringEmitter = flame.emitterParams.kind >= 0.5 && flame.emitterParams.kind < 1.5;
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
    vec3 radiance = acc.radiancePre * flame.intensity;
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
