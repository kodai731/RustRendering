#ifndef FLAME_RADIAL_INTEGRAL_GLSL
#define FLAME_RADIAL_INTEGRAL_GLSL

// Closed-form emission integral of the compact-support radial density
//   rho(p) = F(h) * (1 - u^2)^2,  u = |p.xz| / (S * R(h)),  zero for u >= 1
// over a ray segment. Along the ray u^2(s) is a quadratic g(s), so each height band
// is the exact polynomial integral of (F0 + F1 s + F2 s^2) * (1 - g(s))^2 over the
// interval where g(s) <= 1 — power-rule moments only, no tail and no pedestal.
//
// Must be included after FlameUBO, evaluateHeightFalloff, and flame_noise_field.glsl
// (flameBiweight / flameRadialSupportRadius live there).
// Mirrored in thyllore-render-core/src/flame_radial.rs; the accuracy tests live there.

const int FLAME_RADIAL_BAND_COUNT = 6;
const float FLAME_RADIAL_MIN_DIR_Y = 1e-4;
const float FLAME_RADIAL_MIN_HEIGHT_SPAN = 1e-5;

// Interval where a*s^2 + b*s + c <= 1 clipped to [-halfWidth, halfWidth].
// Citardauq-form roots keep grazing rays (discriminant near zero) precise.
// Returns lo > hi when the support is empty.
vec2 flameSupportInterval(float a, float b, float c, float halfWidth) {
    if (a < 1e-12) {
        return c <= 1.0 ? vec2(-halfWidth, halfWidth) : vec2(1.0, -1.0);
    }
    float discriminant = b * b - 4.0 * a * (c - 1.0);
    if (discriminant <= 0.0) {
        return vec2(1.0, -1.0);
    }
    float root = sqrt(discriminant);
    float gauge = -0.5 * (b + (b >= 0.0 ? root : -root));
    float sFirst = gauge / a;
    float sSecond = (c - 1.0) / gauge;
    return vec2(
        max(min(sFirst, sSecond), -halfWidth),
        min(max(sFirst, sSecond), halfWidth));
}

// Integral (.x) and first moment (.y) of (f.x + f.y s + f.z s^2) * (1 - g(s))^2 over
// the part of [-halfWidth, halfWidth] inside the support g(s) = a s^2 + b s + c <= 1.
vec2 flameBiweightBandEmission(float a, float b, float c, vec3 f, float halfWidth) {
    vec2 interval = flameSupportInterval(a, b, c, halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }

    float m = 1.0 - c;
    float w0 = m * m;
    float w1 = -2.0 * b * m;
    float w2 = b * b - 2.0 * a * m;
    float w3 = 2.0 * a * b;
    float w4 = a * a;
    float e[7] = float[](
        f.x * w0,
        f.x * w1 + f.y * w0,
        f.x * w2 + f.y * w1 + f.z * w0,
        f.x * w3 + f.y * w2 + f.z * w1,
        f.x * w4 + f.y * w3 + f.z * w2,
        f.y * w4 + f.z * w3,
        f.z * w4);

    float powerLo = 1.0;
    float powerHi = 1.0;
    float moments[8];
    for (int n = 0; n < 8; ++n) {
        powerLo *= interval.x;
        powerHi *= interval.y;
        moments[n] = (powerHi - powerLo) / float(n + 1);
    }

    vec2 result = vec2(0.0);
    for (int n = 0; n < 7; ++n) {
        result += e[n] * vec2(moments[n], moments[n + 1]);
    }
    return result;
}

// Std of the unresolved erosion fluctuation inside one band, in argument units.
// FLAME_EROSION_NOISE_SIGMA is the std of e(s) - e(band center) for decorrelated
// fbm3 samples: value-noise std ~0.185, octave amplitudes (0.5, 0.25, 0.125) give
// fbm std ~0.106, and the difference of two decorrelated samples scales by sqrt(2).
// Chords shorter than one noise cell stay correlated, so sigma ramps in linearly.
const float FLAME_EROSION_NOISE_SIGMA = 0.15;

float flameErosionSigma(float height01, float chordLength) {
    float decorrelation = min(chordLength * flame.noiseFrequency, 1.0);
    return abs(flame.noiseAmplitude) * mix(0.2, 1.0, height01)
        * FLAME_EROSION_NOISE_SIGMA * decorrelation;
}

// Smooth field density at band coordinate s:
// d(s) = (f.x + f.y s + f.z s^2) (1 - g(s))^2 with g(s) = a s^2 + b s + c.
float flameOccupancyDensity(float a, float b, float c, vec3 f, float s) {
    float g = (a * s + b) * s + c;
    float inside = max(1.0 - g, 0.0);
    return (f.x + (f.y + f.z * s) * s) * inside * inside;
}

vec2 flameOccupancyPiece(
    FlameSmoothedResponse response, float s0, float s1, float value0, float value1) {
    float span = s1 - s0;
    if (span < 1e-7) {
        return vec2(0.0);
    }
    float slope = (value1 - value0) / span;
    return flameErosionResponseLinearIntegral(response, value0 - slope * s0, slope, s0, s1);
}

// Occupancy integral (.x) and first moment (.y) of the smoothed threshold response
// phi_sigma(x(s)) over the support part of [-halfWidth, halfWidth]. The argument is
// linearized on the two monotone halves split at the vertex of g, sharing the exact
// node values, so the pieces join C0 and band edges agree with their neighbors.
// Sigma fades toward the support boundary per piece (mean of the node envelope
// fades): the unresolved fluctuation is of the *faded* erosion, so it must vanish
// with the envelope like the argument does — a flat band sigma leaves phi_sigma
// with a positive floor at the clipped support surface (the ceiling artifact).
vec2 flameOccupancyBandIntegral(
    float a, float b, float c, vec3 f, float halfWidth, float erosionBand, float sigma) {
    vec2 interval = flameSupportInterval(a, b, c, halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }
    float sSplit = a > 1e-12
        ? clamp(-0.5 * b / a, interval.x, interval.y)
        : 0.5 * (interval.x + interval.y);
    float densityLo = flameOccupancyDensity(a, b, c, f, interval.x);
    float densityMid = flameOccupancyDensity(a, b, c, f, sSplit);
    float densityHi = flameOccupancyDensity(a, b, c, f, interval.y);
    float valueLo = flameErodedArgument(densityLo, erosionBand);
    float valueMid = flameErodedArgument(densityMid, erosionBand);
    float valueHi = flameErodedArgument(densityHi, erosionBand);
    float fadeLo = flameEnvelopeFade(densityLo);
    float fadeMid = flameEnvelopeFade(densityMid);
    float fadeHi = flameEnvelopeFade(densityHi);

    FlameSmoothedResponse responseLo =
        flameSmoothErosionResponse(sigma * 0.5 * (fadeLo + fadeMid));
    FlameSmoothedResponse responseHi =
        flameSmoothErosionResponse(sigma * 0.5 * (fadeMid + fadeHi));
    return flameOccupancyPiece(responseLo, interval.x, sSplit, valueLo, valueMid)
        + flameOccupancyPiece(responseHi, sSplit, interval.y, valueMid, valueHi);
}

// Radius R(h) of the radial density profile, in flame-local units.
float flameRadialRadiusScale(float height01) {
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
    return 1.0 + flame.contourParams.x * (fbm3(vec3(p.x, h - flame.styleParams0.z * flame.time, p.z) * flame.noiseFrequency) * (2.0 / 0.875) - 1.0);
}

// Horizontal offset of the ray at a given height, with q = d.xz / d.y.
vec2 flameRayPointAtHeight(vec3 o, vec2 q, float height) {
    return o.xz + (height - o.y) * q;
}

// Pointwise field for the reference raymarch (mode 1): the same eroded threshold
// field the closed form approximates — true smoothstep, exact support membership.
float flamePointOccupancyDensity(vec3 p, float h, float wiggle) {
    float dSmooth = evaluateHeightFalloff(h)
        * flameRadialDensityFactor(vec3(p.x / wiggle, p.y, p.z / wiggle), h);
    float erosion = flameNoiseErosionValue(p, h);
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion))
        * flameFieldSupportMask(dSmooth);
}

// Occupancy variant of the along-ray fallback: same quadratic setup, threshold
// response instead of the plain biweight emission. The first moment is about tCenter.
vec2 flameOccupancyAlongRay(
    vec3 o, vec3 d, float tNear, float tFar, float invSq, float erosionBand, float sigma) {
    float tCenter = 0.5 * (tNear + tFar);
    vec2 p = o.xz + tCenter * d.xz;
    float halfWidth = 0.5 * (tFar - tNear);
    float falloff = evaluateHeightFalloff(clamp(o.y + tCenter * d.y, 0.0, 1.0));
    return flameOccupancyBandIntegral(
        invSq * dot(d.xz, d.xz), 2.0 * invSq * dot(p, d.xz), invSq * dot(p, p),
        vec3(falloff, 0.0, 0.0), halfWidth, erosionBand, sigma);
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
    float dSmooth = flameEmitterSmoothDensityAt(p, h, wiggle);
    float erosion = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(p, h) : 0.0;
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion))
        * flameFieldSupportMask(dSmooth);
}

const int FLAME_OCCUPANCY_NODE_SEGMENTS = 4;

// Occupancy integral (.x) and first moment in t (.y) of the smoothed threshold
// response over one t band, with dSmooth sampled at the segment nodes and the
// argument linear between them. Segments whose both node densities are zero lie
// outside the support and contribute nothing (exact membership at node resolution);
// densities are sampled before the erosion fbm so empty bands cost no noise at all.
// The frozen erosion and its sigma are sampled at the density-weighted node
// position, not the band midpoint: the midpoint of a band that straddles empty
// space (ring seen from inside) lands away from the wall, so the frozen
// turbulence would be sampled where no density exists and sweep through world
// space as the camera moves — the view-dependent shimmer of the band freeze.
// Mirrored in thyllore-render-core/src/flame_radial.rs (density_weighted_node_t).
vec2 flameOccupancyNodeBand(vec3 o, vec3 d, float t0, float t1, float wiggleBand, float chord) {
    float dt = (t1 - t0) / float(FLAME_OCCUPANCY_NODE_SEGMENTS);
    float density[FLAME_OCCUPANCY_NODE_SEGMENTS + 1];
    float weightSum = 0.0;
    float tWeighted = 0.0;
    for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
        float t = t0 + float(node) * dt;
        vec3 p = o + t * d;
        density[node] = flameEmitterSmoothDensityAt(p, clamp(p.y, 0.0, 1.0), wiggleBand);
        weightSum += density[node];
        tWeighted += density[node] * t;
    }
    if (weightSum <= 0.0) {
        return vec2(0.0);
    }

    vec3 pSample = o + (tWeighted / weightSum) * d;
    float hSample = clamp(pSample.y, 0.0, 1.0);
    float erosionBand = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(pSample, hSample) : 0.0;
    float sigma = flame.noiseAmplitude != 0.0 ? flameErosionSigma(hSample, chord) : 0.0;
    float argument[FLAME_OCCUPANCY_NODE_SEGMENTS + 1];
    for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
        argument[node] = flameErodedArgument(density[node], erosionBand);
    }
    vec2 total = vec2(0.0);
    for (int segment = 1; segment <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++segment) {
        if (density[segment - 1] > 0.0 || density[segment] > 0.0) {
            // Per-segment sigma fade toward the support boundary (see
            // flameOccupancyBandIntegral): the fluctuation vanishes with the envelope.
            FlameSmoothedResponse response = flameSmoothErosionResponse(sigma * 0.5
                * (flameEnvelopeFade(density[segment - 1]) + flameEnvelopeFade(density[segment])));
            float tPrev = t0 + float(segment - 1) * dt;
            float slope = (argument[segment] - argument[segment - 1]) / dt;
            total += flameErosionResponseLinearIntegral(
                response, argument[segment - 1] - slope * tPrev, slope, tPrev, tPrev + dt);
        }
    }
    return total;
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar);

// Rays with a near-constant height cannot be parameterized by h, so the moment is taken along t.
float integrateRadialEmissionAlongRay(vec3 o, vec3 d, float tNear, float tFar, float invSq) {
    float tCenter = 0.5 * (tNear + tFar);
    vec2 p = o.xz + tCenter * d.xz;
    float halfWidth = 0.5 * (tFar - tNear);
    float falloff = evaluateHeightFalloff(clamp(o.y + tCenter * d.y, 0.0, 1.0));
    return flameBiweightBandEmission(
        invSq * dot(d.xz, d.xz), 2.0 * invSq * dot(p, d.xz), invSq * dot(p, p),
        vec3(falloff, 0.0, 0.0), halfWidth).x;
}

float integrateRadialEmission(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }

    float heightNear = clamp(o.y + tNear * d.y, 0.0, 1.0);
    float heightFar = clamp(o.y + tFar * d.y, 0.0, 1.0);
    float heightLo = min(heightNear, heightFar);
    float heightHi = max(heightNear, heightFar);
    if (abs(d.y) < FLAME_RADIAL_MIN_DIR_Y
        || heightHi - heightLo < FLAME_RADIAL_MIN_HEIGHT_SPAN) {
        float midHeight = clamp(o.y + 0.5 * (tNear + tFar) * d.y, 0.0, 1.0);
        vec3 pMid = o + 0.5 * (tNear + tFar) * d;
        float wMid = flameContourWiggle(pMid, midHeight);
        vec2 bendMid = flameBendOffsetAt(midHeight);
        if (flame.noiseAmplitude != 0.0) {
            return flameOccupancyAlongRay(
                vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, tNear, tFar,
                flameRadialSupportInvSq(midHeight) / (wMid * wMid),
                flameNoiseErosionValue(pMid, midHeight),
                flameErosionSigma(midHeight, tFar - tNear)).x;
        }
        return integrateRadialEmissionAlongRay(
            vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, tNear, tFar,
            flameRadialSupportInvSq(midHeight) / (wMid * wMid))
            * flameNoiseErosionFactor(pMid, midHeight);
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    vec2 q = d.xz / d.y;
    float quadratic = dot(q, q);

    // Trim to the widest support (over heights, wiggle and wind bend) so grazing rays
    // do not spend bands on empty range. Bands outside the true support integrate to
    // exactly zero, so the extra margin costs only band resolution, never correctness.
    float wTrim = 1.0 + max(flame.contourParams.x, 0.0);
    float widestRadius = flameRadialSupportRadius() * wTrim
        * max(flameRadialRadiusScale(0.0), flameRadialRadiusScale(1.0));
    float bendMax = length(flame.styleParams2.xy) * flame.styleParams2.z;
    if (quadratic > 1e-12) {
        float support = (widestRadius + bendMax) / sqrt(quadratic);
        float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
        heightLo = max(heightLo, closestApproachHeight - support);
        heightHi = min(heightHi, closestApproachHeight + support);
        if (heightHi <= heightLo) {
            return 0.0;
        }
    }

    float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
    float halfWidth = 0.5 * bandWidth;
    float bandChord = bandWidth * length(vec3(q.x, 1.0, q.y));
    float total = 0.0;
    float falloffLo = evaluateHeightFalloff(heightLo);
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        float center = heightLo + (float(band) + 0.5) * bandWidth;
        float falloffMid = evaluateHeightFalloff(center);
        float falloffHi = evaluateHeightFalloff(center + halfWidth);

        vec2 pTrue = flameRayPointAtHeight(o, q, center);
        vec2 pxz = pTrue - flameBendOffsetAt(center);
        float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
        float invSq = flameRadialSupportInvSq(center) / (wBand * wBand);
        float slope = (falloffHi - falloffLo) / bandWidth;
        float curvature = 2.0 * (falloffHi + falloffLo - 2.0 * falloffMid) / (bandWidth * bandWidth);
        if (flame.noiseAmplitude != 0.0) {
            float erosionBand = flameNoiseErosionValue(vec3(pTrue.x, center, pTrue.y), center);
            total += flameOccupancyBandIntegral(
                invSq * quadratic, 2.0 * invSq * dot(pxz, q), invSq * dot(pxz, pxz),
                vec3(falloffMid, slope, curvature), halfWidth, erosionBand,
                flameErosionSigma(center, bandChord)).x;
        } else {
            float eBand = flameNoiseErosionFactor(vec3(pTrue.x, center, pTrue.y), center);
            total += eBand * flameBiweightBandEmission(
                invSq * quadratic, 2.0 * invSq * dot(pxz, q), invSq * dot(pxz, pxz),
                vec3(falloffMid, slope, curvature), halfWidth).x;
        }

        falloffLo = falloffHi;
    }
    return total / abs(d.y);
}

vec3 flameRampColor(float h) {
    if (h < 0.5) {
        return mix(flame.colorBase.rgb, flame.colorMid.rgb, h * 2.0);
    }
    return mix(flame.colorMid.rgb, flame.colorTip.rgb, (h - 0.5) * 2.0);
}

// Beer-Lambert composite over per-band emissions — the one RTE assembly shared by
// the cylinder bands (camera order given by `reversed`) and the emitter-generic
// t bands (already camera-ordered, reversed = false).
vec4 flameCompositeRteBands(
    float bandEmission[FLAME_RADIAL_BAND_COUNT],
    float bandHeight[FLAME_RADIAL_BAND_COUNT],
    float bandEdge[FLAME_RADIAL_BAND_COUNT],
    bool reversed) {
    float total = 0.0;
    float heightMean = 0.0;
    for (int i = 0; i < FLAME_RADIAL_BAND_COUNT; ++i) {
        total += bandEmission[i];
        heightMean += bandEmission[i] * bandHeight[i];
    }
    heightMean = total > 1e-6 ? heightMean / total : 0.0;
    float tempNorm = clamp(total * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMean);
    float boost = 1.0 + flame.styleParams1.w * tempNorm * tempNorm;

    vec3 radiance = vec3(0.0);
    vec3 sigmaRgb = flame.sigmaT * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.w, 0.0, 1.0));
    vec3 transmittance = vec3(1.0);
    for (int i = 0; i < FLAME_RADIAL_BAND_COUNT; ++i) {
        int idx = reversed ? FLAME_RADIAL_BAND_COUNT - 1 - i : i;
        vec3 tau = sigmaRgb * bandEmission[idx];
        vec3 absorbed = vec3(1.0) - exp(-tau);
        radiance += transmittance * mix(flameRampColor(bandHeight[idx]), flame.colorTip.rgb, bandEdge[idx]) * flame.intensity * boost * absorbed;
        transmittance *= exp(-tau);
    }
    return vec4(radiance, 1.0 - dot(transmittance, vec3(1.0 / 3.0)));
}

vec4 integrateRadialRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }

    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    bool reversed = false;

    float heightNear = clamp(o.y + tNear * d.y, 0.0, 1.0);
    float heightFar = clamp(o.y + tFar * d.y, 0.0, 1.0);
    float heightLo = min(heightNear, heightFar);
    float heightHi = max(heightNear, heightFar);

    if (abs(d.y) < FLAME_RADIAL_MIN_DIR_Y || heightHi - heightLo < FLAME_RADIAL_MIN_HEIGHT_SPAN) {
        float dt = (tFar - tNear) / float(FLAME_RADIAL_BAND_COUNT);
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            float t0 = tNear + float(band) * dt;
            vec3 pMid = o + (t0 + 0.5 * dt) * d;
            float midHeight = clamp(pMid.y, 0.0, 1.0);
            float wMid = flameContourWiggle(pMid, midHeight);
            vec2 bendMid = flameBendOffsetAt(midHeight);
            float c;
            if (flame.noiseAmplitude != 0.0) {
                c = flameOccupancyAlongRay(
                    vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, t0, t0 + dt,
                    flameRadialSupportInvSq(midHeight) / (wMid * wMid),
                    flameNoiseErosionValue(pMid, midHeight),
                    flameErosionSigma(midHeight, dt)).x;
            } else {
                c = integrateRadialEmissionAlongRay(
                    vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, t0, t0 + dt,
                    flameRadialSupportInvSq(midHeight) / (wMid * wMid))
                    * flameNoiseErosionFactor(pMid, midHeight);
            }
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = midHeight;
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(pMid.xz - bendMid) / max(flameRadialRadiusScale(midHeight), 1e-4)), 0.0, 1.0);
        }
    } else {
        vec2 q = d.xz / d.y;
        float quadratic = dot(q, q);

        float wTrim = 1.0 + max(flame.contourParams.x, 0.0);
        float widestRadius = flameRadialSupportRadius() * wTrim
            * max(flameRadialRadiusScale(0.0), flameRadialRadiusScale(1.0));
        float bendMax = length(flame.styleParams2.xy) * flame.styleParams2.z;
        if (quadratic > 1e-12) {
            float support = (widestRadius + bendMax) / sqrt(quadratic);
            float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
            heightLo = max(heightLo, closestApproachHeight - support);
            heightHi = min(heightHi, closestApproachHeight + support);
            if (heightHi <= heightLo) {
                return vec4(0.0);
            }
        }

        float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
        float halfWidth = 0.5 * bandWidth;
        float bandChord = bandWidth * length(vec3(q.x, 1.0, q.y));
        float falloffLo = evaluateHeightFalloff(heightLo);
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            float center = heightLo + (float(band) + 0.5) * bandWidth;
            float falloffMid = evaluateHeightFalloff(center);
            float falloffHi = evaluateHeightFalloff(center + halfWidth);

            vec2 pTrue = flameRayPointAtHeight(o, q, center);
            vec2 p = pTrue - flameBendOffsetAt(center);
            float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
            float invSq = flameRadialSupportInvSq(center) / (wBand * wBand);
            float slope = (falloffHi - falloffLo) / bandWidth;
            float curvature = 2.0 * (falloffHi + falloffLo - 2.0 * falloffMid) / (bandWidth * bandWidth);
            vec2 emission;
            if (flame.noiseAmplitude != 0.0) {
                float erosionBand = flameNoiseErosionValue(vec3(pTrue.x, center, pTrue.y), center);
                emission = flameOccupancyBandIntegral(
                    invSq * quadratic, 2.0 * invSq * dot(p, q), invSq * dot(p, p),
                    vec3(falloffMid, slope, curvature), halfWidth, erosionBand,
                    flameErosionSigma(center, bandChord));
            } else {
                float eBand = flameNoiseErosionFactor(vec3(pTrue.x, center, pTrue.y), center);
                emission = eBand * flameBiweightBandEmission(
                    invSq * quadratic, 2.0 * invSq * dot(p, q), invSq * dot(p, p),
                    vec3(falloffMid, slope, curvature), halfWidth);
            }
            float c = emission.x / abs(d.y);
            float meanOffset = emission.x > 1e-6 ? clamp(emission.y / emission.x, -halfWidth, halfWidth) : 0.0;
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = clamp(center + meanOffset, 0.0, 1.0);
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(p) / max(flameRadialRadiusScale(center), 1e-4)), 0.0, 1.0);

            falloffLo = falloffHi;
        }
        reversed = d.y < 0.0;
    }

    return flameCompositeRteBands(bandEmission, bandHeight, bandEdge, reversed);
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
    float wTrim = 1.0 + max(flame.contourParams.x, 0.0);
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

// Emitter-generic occupancy bands over t — the one band fill shared by the scalar
// emission integral and the RTE composite. The t bands are camera-ordered; for
// the ring they cover the (convex) outer-support crossing instead of the raw
// proxy interval.
void flameEmitterOccupancyBands(
    vec3 o, vec3 d, float tNear, float tFar,
    out float bandEmission[FLAME_RADIAL_BAND_COUNT],
    out float bandHeight[FLAME_RADIAL_BAND_COUNT],
    out float bandEdge[FLAME_RADIAL_BAND_COUNT]) {
    bool ringEmitter = flame.emitterParams.x >= 0.5 && flame.emitterParams.x < 1.5;
    if (ringEmitter && !flameRingSupportSpan(o, d, tNear, tFar)) {
        float hFallback = clamp(o.y + 0.5 * (tNear + tFar) * d.y, 0.0, 1.0);
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            bandEmission[band] = 0.0;
            bandHeight[band] = hFallback;
            bandEdge[band] = 0.0;
        }
        return;
    }
    float dt = (tFar - tNear) / float(FLAME_RADIAL_BAND_COUNT);
    float chord = dt * length(d);
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        float t0 = tNear + float(band) * dt;
        vec3 pMid = o + (t0 + 0.5 * dt) * d;
        float hMid = clamp(pMid.y, 0.0, 1.0);
        float wBand = flameContourWiggle(pMid, hMid);
        vec2 occupancy = flameOccupancyNodeBand(o, d, t0, t0 + dt, wBand, chord);
        float tMean = occupancy.x > 1e-6 ? clamp(occupancy.y / occupancy.x, t0, t0 + dt) : t0 + 0.5 * dt;
        vec3 pMean = o + tMean * d;
        float hMean = clamp(pMean.y, 0.0, 1.0);
        bandEmission[band] = max(occupancy.x, 0.0);
        bandHeight[band] = hMean;
        float edge = 0.0;
        if (flame.emitterParams.x < 1.5) {
            float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
            float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
            float taperR = mix(1.0, flame.styleParams1.x, pow(hMean, flame.styleParams0.w));
            float rhoNorm = abs((length(pMean.xz) - rm) / minorScale) / max(taperR, 1e-4);
            edge = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, rhoNorm), 0.0, 1.0);
        }
        bandEdge[band] = edge;
    }
}

vec4 integrateEmitterOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }
    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    flameEmitterOccupancyBands(o, d, tNear, tFar, bandEmission, bandHeight, bandEdge);
    return flameCompositeRteBands(bandEmission, bandHeight, bandEdge, false);
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }
    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    flameEmitterOccupancyBands(o, d, tNear, tFar, bandEmission, bandHeight, bandEdge);
    float total = 0.0;
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        total += bandEmission[band];
    }
    return total;
}

#endif
