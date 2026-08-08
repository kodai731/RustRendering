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
    if (flame.contourParams.x == 0.0) { return 1.0; }
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
    float erosion = flameNoiseErosionValue(p, h);
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
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
    float erosion = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(p, h) : 0.0;
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
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
// Modes the node spacing cannot resolve are attenuated by a smooth low-pass in
// beta = k . d and their power routed into the response sigma — the smooth
// analogue of the fbm path's FLAME_EROSION_NOISE_SIGMA, with no h-quantized
// band structure anywhere (band-boundary-coherence.md, E1).
// Mirrored in thyllore-render-core/src/flame_wave.rs
// (evaluate_wave_occupancy_segments / wave_ray_attenuation).

#ifdef FLAME_WAVE_SEGMENTS_OVERRIDE
const int FLAME_WAVE_SEGMENTS = FLAME_WAVE_SEGMENTS_OVERRIDE;
#else
const int FLAME_WAVE_SEGMENTS = 64;
#endif
const int FLAME_WAVE_MODE_SLOTS = 178;
// Warped noise coordinate of the wave basis: wind bend removed, the analytic
// mode-sum warp applied, then the same aniso/frequency/advect chain the fbm
// erosion samples — exactly flameNoiseWarpedCoordinate's wave branch followed
// by the erosion coordinate transform, so mode 0 and mode 1 stay a parity pair.
vec3 flameWaveCoordinate(vec3 p, float h) {
    vec2 bendOffset = flameBendOffsetAt(h);
    vec3 pb = vec3(p.x - bendOffset.x, p.y, p.z - bendOffset.y);
    vec3 q = flameWaveFlowWarp(pb, h);
    return flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency
        - flameNoiseAdvect();
}

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
// Node-local low-pass: weights come from the warped rate at this node, so a
// locally stretched node does not smooth the whole ray.
float flameWaveNodeArgumentLocal(
    vec3 p, vec3 d, float h, float density, float dt,
    int count, float eddyTime, out float shapedNoise, out float sigmaNoise) {
    vec3 pb = flameNoiseBendRemoved(p, h);
    vec3 q;
    vec3 rateRaw = flameWaveFlowWarpRate(pb, d, h, q);
    vec3 w = flameAnisoCompress(q, flame.temporalData.z) * flame.noiseFrequency - flameNoiseAdvect();
    vec3 rate = flameAnisoCompress(rateRaw, flame.temporalData.z) * flame.noiseFrequency;

    // Closed-form variant: pseudo-FM phase psi_n and its ray rate from the
    // modulator field at the unwarped coordinate (parity with flameWaveNoiseSum).
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

    // Low-octave pass first (wavePhase.z == 0): the resolved low sum drives the
    // envelope 1 + coeff * zLow of the higher octaves (cross-scale coupling).
    float zLow = 0.0;
    float unresolvedPower = 0.0;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z != 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime;
        float betaPhase = dot(waveVector.xyz, rate);
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
        zLow += weight * waveVector.w * carrier;
        unresolvedPower += 0.5 * waveVector.w * waveVector.w * (1.0 - weight * weight);
    }
    float z = zLow;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        if (wavePhase.z == 0.0) {
            continue;
        }
        float angle = dot(waveVector.xyz, w) + wavePhase.x + wavePhase.y * eddyTime;
        float betaPhase = dot(waveVector.xyz, rate);
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
        float envelope = 1.0 + wavePhase.z * zLow;
        z += envelope * weight * waveVector.w * carrier;
        unresolvedPower += envelope * envelope * 0.5 * waveVector.w * waveVector.w * (1.0 - weight * weight);
    }

    sigmaNoise = sqrt(unresolvedPower);
    float invScale = flame.waveParams.z;
    float amp = flame.waveParams.w;
    shapedNoise = invScale > 0.0 ? 0.4375 + amp * tanh(z * invScale) : 0.4375 + z;
    return flameErodedArgument(density, flameNoiseErosionFromValue(shapedNoise, h));
}

void flameWaveOccupancySegments(
    vec3 o, vec3 d, float t0, float t1,
    out float segEmission[FLAME_WAVE_SEGMENTS],
    out float segTMean[FLAME_WAVE_SEGMENTS]) {
    float dt = (t1 - t0) / float(FLAME_WAVE_SEGMENTS);
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        segEmission[segment] = 0.0;
        segTMean[segment] = t0 + (float(segment) + 0.5) * dt;
    }
    if (dt <= 0.0) {
        return;
    }

    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = min(int(flame.waveParams.x), FLAME_WAVE_MODE_SLOTS);

    // Streaming node walk: density first, the mode sum only at nodes touching
    // support (empty segments cost no mode evaluation), one node carried
    // between segments so nothing is evaluated twice.
    float residual = flameCarveResidualStrength();
    float invScale = flame.waveParams.z;
    float amp = flame.waveParams.w;
    float previousDensity = flameWaveNodeDensity(o + t0 * d, clamp(o.y + t0 * d.y, 0.0, 1.0));
    float previousArgument = 0.0;
    float previousShapedNoise = 0.4375;
    float previousSigma = 0.0;
    bool previousArgumentValid = false;
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        float tPrev = t0 + float(segment) * dt;
        float t = tPrev + dt;
        vec3 p = o + t * d;
        float h = clamp(p.y, 0.0, 1.0);
        float density = flameWaveNodeDensity(p, h);
        if (previousDensity <= 0.0 && density <= 0.0) {
            previousDensity = density;
            previousArgumentValid = false;
            continue;
        }
        if (!previousArgumentValid) {
            vec3 pPrev = o + tPrev * d;
            float hPrev = clamp(pPrev.y, 0.0, 1.0);
            previousArgument = flameWaveNodeArgumentLocal(
                pPrev, d, hPrev, previousDensity, dt, count, eddyTime,
                previousShapedNoise, previousSigma);
        }
        float currentShapedNoise;
        float currentSigma;
        float argument = flameWaveNodeArgumentLocal(p, d, h, density, dt, count, eddyTime,
            currentShapedNoise, currentSigma);

        // Shaping derivative average: g'(z) = amp * invScale * (1 - t^2) where
        // t = (shapedNoise - 0.4375) / amp; if invScale <= 0 then gPrime = 1.0.
        float shapingDerivAvg;
        if (invScale > 0.0) {
            float tPrevVal = (previousShapedNoise - 0.4375) / amp;
            float tCurrVal = (currentShapedNoise - 0.4375) / amp;
            shapingDerivAvg = 0.5 * amp * invScale
                * ((1.0 - tPrevVal * tPrevVal) + (1.0 - tCurrVal * tCurrVal));
        } else {
            shapingDerivAvg = 1.0;
        }

        float hMid = clamp(o.y + (tPrev + 0.5 * dt) * d.y, 0.0, 1.0);
        float sigmaEff = 0.5 * (previousSigma + currentSigma) * shapingDerivAvg * abs(flame.noiseAmplitude) * mix(0.2, 1.0, hMid)
            * 0.5 * (flameEnvelopeFade(previousDensity) + flameEnvelopeFade(density));
        FlameSmoothedResponse response = flameSmoothErosionResponse(sigmaEff);
        float slope = (argument - previousArgument) / dt;
        vec2 carved = flameErosionResponseLinearIntegral(
            response, previousArgument - slope * tPrev, slope, tPrev, t);
        if (residual > 0.0) {
            float plainSlope = (density - previousDensity) / dt;
            vec2 plain = flameErosionResponseLinearIntegral(
                flameSmoothErosionResponse(0.0),
                previousDensity - plainSlope * tPrev, plainSlope, tPrev, t);
            carved = mix(carved, plain, residual);
        }
        segEmission[segment] = max(carved.x, 0.0);
        if (carved.x > 1e-6) {
            segTMean[segment] = clamp(carved.y / carved.x, tPrev, t);
        }

        previousDensity = density;
        previousArgument = argument;
        previousShapedNoise = currentShapedNoise;
        previousSigma = currentSigma;
        previousArgumentValid = true;
    }
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
    float segEmission[FLAME_WAVE_SEGMENTS];
    float segTMean[FLAME_WAVE_SEGMENTS];
    flameWaveOccupancySegments(o, d, tNear, tFar, segEmission, segTMean);
    float total = 0.0;
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        total += segEmission[segment];
    }
    return total;
}

// Beer-Lambert composite directly over the wave segments (camera-ordered by
// construction) — no fixed band arrays anywhere in the wave pipeline.
vec4 integrateWaveOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (!flameWaveSupportSpan(o, d, tNear, tFar)) {
        return vec4(0.0);
    }
    float segEmission[FLAME_WAVE_SEGMENTS];
    float segTMean[FLAME_WAVE_SEGMENTS];
    flameWaveOccupancySegments(o, d, tNear, tFar, segEmission, segTMean);

    float total = 0.0;
    float heightMean = 0.0;
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        total += segEmission[segment];
        heightMean += segEmission[segment]
            * clamp(o.y + segTMean[segment] * d.y, 0.0, 1.0);
    }
    heightMean = total > 1e-6 ? heightMean / total : 0.0;
    float tempNorm = clamp(total * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMean);
    float boost = 1.0 + flame.styleParams1.w * tempNorm * tempNorm;

    vec3 radiance = vec3(0.0);
    vec3 sigmaRgb = flame.sigmaT
        * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.w, 0.0, 1.0));
    vec3 transmittance = vec3(1.0);
    for (int segment = 0; segment < FLAME_WAVE_SEGMENTS; ++segment) {
        vec3 pMean = o + segTMean[segment] * d;
        float hMean = clamp(pMean.y, 0.0, 1.0);
        float edge = 0.0;
        if (flame.emitterParams.x < 1.5) {
            float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
            float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
            float taperR = mix(1.0, flame.styleParams1.x, pow(hMean, flame.styleParams0.w));
            float rhoNorm = abs((length(pMean.xz) - rm) / minorScale) / max(taperR, 1e-4);
            edge = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, rhoNorm), 0.0, 1.0);
        }
        vec3 tau = sigmaRgb * segEmission[segment];
        vec3 absorbed = vec3(1.0) - exp(-tau);
        radiance += transmittance
            * mix(flameRampColor(hMean), flame.colorTip.rgb, edge)
            * flame.intensity * boost * absorbed;
        transmittance *= exp(-tau);
    }
    return vec4(radiance, 1.0 - dot(transmittance, vec3(1.0 / 3.0)));
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
