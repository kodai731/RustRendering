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

// Argument of the threshold response at band coordinate s: the eroded smooth field
// x(s) = (f.x + f.y s + f.z s^2) (1 - g(s))^2 - erosionBand with g(s) = a s^2 + b s + c.
float flameOccupancyArgument(float a, float b, float c, vec3 f, float erosionBand, float s) {
    float g = (a * s + b) * s + c;
    float inside = max(1.0 - g, 0.0);
    return (f.x + (f.y + f.z * s) * s) * inside * inside - erosionBand;
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
vec2 flameOccupancyBandIntegral(
    float a, float b, float c, vec3 f, float halfWidth, float erosionBand, float sigma) {
    vec2 interval = flameSupportInterval(a, b, c, halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }
    float sSplit = a > 1e-12
        ? clamp(-0.5 * b / a, interval.x, interval.y)
        : 0.5 * (interval.x + interval.y);
    float valueLo = flameOccupancyArgument(a, b, c, f, erosionBand, interval.x);
    float valueMid = flameOccupancyArgument(a, b, c, f, erosionBand, sSplit);
    float valueHi = flameOccupancyArgument(a, b, c, f, erosionBand, interval.y);

    FlameSmoothedResponse response = flameSmoothErosionResponse(sigma);
    return flameOccupancyPiece(response, interval.x, sSplit, valueLo, valueMid)
        + flameOccupancyPiece(response, sSplit, interval.y, valueMid, valueHi);
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
    return smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth - erosion)
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

#endif
