// Continuous ray integrator — REFERENCE ONLY, NOT COMPILED ANYWHERE.
//
// Modes below a ray-fixed reference cutoff are resolved as exact values; the
// uncaptured share folds into response sigma through tanh conditional statistics.

// Carrier statistics: non-modal unresolved std, total std, total modal power,
// the smallest wave number (band grid scale), the low-band power feeding the
// statistical envelope, and the tanh statistical-linearization gain and
// distortion residual.
const int FLAME_CAPTURE_BANDS = 8;
const float FLAME_INV_SQRT2 = 0.70710678;

// Positive nodes / doubled weights of the 8-point Gauss-Hermite rule,
// prescaled so E[f(z)] = sum w * f(sqrt(2) sigma x), z ~ N(0, sigma^2).
const vec2 FLAME_GAUSS_HERMITE_8[4] = vec2[4](
    vec2(0.3811870, 0.7460245),
    vec2(1.1571937, 0.2344810),
    vec2(1.9816568, 0.0192712),
    vec2(2.9306374, 0.0002246));

// Ray-constant carrier data (z units): non-modal unresolved std, total std,
// total modal power, the smallest wave number (band grid scale; the mode
// table is sorted by |k| ascending), the low-band power feeding the
// statistical envelope, and the tanh statistical-linearization gain and
// distortion residual.
struct FlameCarrierConstants {
    float sigmaBase;
    float sigmaZ;
    float modalPower;
    float kMin;
    float lowPower;
    float gain;
    float distortion;
};

float flameCarrierEnvelopeRms(vec4 wavePhase, float lowPower) {
    return wavePhase.z != 0.0 ? sqrt(1.0 + wavePhase.z * wavePhase.z * lowPower) : 1.0;
}

FlameCarrierConstants flameCarrierConstants(int count) {
    FlameCarrierConstants cc;
    cc.lowPower = 0.0;
    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        if (flame.waveModes[2 * n + 1].z == 0.0) {
            cc.lowPower += 0.5 * waveVector.w * waveVector.w;
        }
    }
    cc.modalPower = 0.0;
    for (int n = 0; n < count; ++n) {
        float amplitude = flameCarrierEnvelopeRms(flame.waveModes[2 * n + 1], cc.lowPower)
            * flame.waveModes[2 * n].w;
        cc.modalPower += 0.5 * amplitude * amplitude;
    }
    cc.kMin = length(flame.waveModes[0].xyz);
    float envSkipPower = 1.0 + flame.waveParams.y * flame.waveParams.y * cc.lowPower;
    cc.sigmaBase = sqrt(max(flame.waveCfParams.z + flame.waveCfParams.w * envSkipPower, 0.0));
    cc.sigmaZ = sqrt(cc.modalPower + cc.sigmaBase * cc.sigmaBase);

    float invScale = flame.waveParams.z;
    float amp = flame.waveParams.w;
    if (invScale > 0.0) {
        float eSech2 = 0.0;
        float eTanh2 = 0.0;
        for (int i = 0; i < 4; ++i) {
            float t = tanh(invScale * 1.41421356 * cc.sigmaZ * FLAME_GAUSS_HERMITE_8[i].x);
            eSech2 += FLAME_GAUSS_HERMITE_8[i].y * (1.0 - t * t);
            eTanh2 += FLAME_GAUSS_HERMITE_8[i].y * t * t;
        }
        cc.gain = amp * invScale * eSech2;
        cc.distortion = sqrt(max(amp * amp * eTanh2 - cc.gain * cc.gain * cc.sigmaZ * cc.sigmaZ, 0.0));
    } else {
        cc.gain = 1.0;
        cc.distortion = 0.0;
    }
    return cc;
}

// Per-node carrier state: the resolved slow value (exact per-mode capture sum
// at the reference cutoff — any lossy per-node reconstruction turns into
// argument noise at the shell) plus band powers for the segment-local fold.
struct FlameCarrierState {
    float zSlow;
    float powerBand[FLAME_CAPTURE_BANDS];
    float omega0;
};

FlameCarrierState flameCarrierSlowState(
    FlameWarpFrame frame, int count, float eddyTime,
    FlameCarrierConstants cc, bool hasCutoff, float alphaRef) {
    vec3 jitterPsi;
    vec3 jitterPsiRate;
    flameWaveJitterState(frame.w, frame.rate, jitterPsi, jitterPsiRate);

    FlameCarrierState state;
    state.zSlow = 0.0;
    for (int band = 0; band < FLAME_CAPTURE_BANDS; ++band) {
        state.powerBand[band] = 0.0;
    }
    state.omega0 = max(cc.kMin * length(frame.rate), 1e-3) * 0.5;

    for (int n = 0; n < count; ++n) {
        vec4 waveVector = flame.waveModes[2 * n];
        vec4 wavePhase = flame.waveModes[2 * n + 1];
        float angle = dot(waveVector.xyz, frame.w) + wavePhase.x + wavePhase.y * eddyTime
            + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsi);
        float omega = dot(waveVector.xyz, frame.rate)
            + dot(flame.waveJitter[min(n, 95)].xyz, jitterPsiRate);
        float amplitude = flameCarrierEnvelopeRms(wavePhase, cc.lowPower) * waveVector.w;
        float power = 0.5 * amplitude * amplitude;

        if (hasCutoff) {
            float kappa = omega * alphaRef * FLAME_INV_SQRT2;
            state.zSlow += exp(-kappa * kappa) * amplitude * sin(angle);
        }

        float u = clamp(log2(max(abs(omega) / state.omega0, 1e-6)),
            0.0, float(FLAME_CAPTURE_BANDS - 1));
        for (int band = 0; band < FLAME_CAPTURE_BANDS; ++band) {
            float hat = max(1.0 - abs(u - float(band)), 0.0);
            state.powerBand[band] += hat * power;
        }
    }
    return state;
}

float flameCapturedPower(FlameCarrierState state, bool hasCutoff, float alpha) {
    if (!hasCutoff) {
        return 0.0;
    }
    float sum = 0.0;
    for (int band = 0; band < FLAME_CAPTURE_BANDS; ++band) {
        float omegaBand = state.omega0 * float(1 << band);
        float k = omegaBand * alpha * FLAME_INV_SQRT2;
        float g = exp(-k * k);
        sum += state.powerBand[band] * g * g;
    }
    return sum;
}

// Conditional mean of `shaped - 0.4375` given the resolved slow carrier `u`,
// averaging the tanh over the unresolved Gaussian residual of std sigmaFast.
float flameShapedDeltaMean(float u, float sigmaFast) {
    float invScale = flame.waveParams.z;
    float amp = flame.waveParams.w;
    if (invScale <= 0.0) {
        return u;
    }
    float mean = 0.0;
    for (int i = 0; i < 4; ++i) {
        float offset = 1.41421356 * sigmaFast * FLAME_GAUSS_HERMITE_8[i].x;
        mean += FLAME_GAUSS_HERMITE_8[i].y * 0.5
            * (tanh(invScale * (u + offset)) + tanh(invScale * (u - offset)));
    }
    return amp * mean;
}

// Argument-unit sigma of the folded carrier share at the given geometry
// chain: gain-passed base + folded modal power, plus the tanh distortion.
float flameFoldedSigmaArgument(
    float geometry, FlameCarrierConstants cc, float foldedPower) {
    float baseZ = cc.sigmaBase * cc.sigmaBase + foldedPower;
    return sqrt(geometry * geometry
        * (cc.gain * cc.gain * baseZ + cc.distortion * cc.distortion));
}

// Walk pieces: mean argument computation, reference cutoff determination via
// shell crossing detection, and per-segment erf closed-form integration.
const int FLAME_CROSSING_SCAN_INTERVALS = 256;
const int FLAME_CROSSING_BISECTION_STEPS = 20;

// Mean (carrier-free) argument: density with only the deterministic shrink.
float flameMeanArgumentAt(vec3 o, vec3 d, float t) {
    vec3 p = o + t * d;
    float h = clamp(p.y, 0.0, 1.0);
    float density = flameWaveNodeDensity(p, h);
    float meanErosion = flame.noiseAmplitude * mix(0.2, 1.0, h) * FLAME_EROSION_MEAN_SHRINK
        * (1.0 + flameBurnoutBoost(h));
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

    FlameCarrierConstants carrierConstants = flameCarrierConstants(count);
    bool hasCutoff;
    float alphaRef = flameSolveReferenceCutoff(
        o, d, t0, dt * float(FLAME_WAVE_SEGMENTS), carrierConstants, hasCutoff);
    float previousDensity = flameWaveNodeDensity(o + t0 * d, clamp(o.y + t0 * d.y, 0.0, 1.0));
    FlameCarrierState previousState;
    bool previousStateValid = false;

            previousStateValid = false;

            previousStateValid = false;

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

        previousState = currentState;
        previousStateValid = !exiting;
