#ifndef FLAME_RADIAL_INTEGRAL_GLSL
#define FLAME_RADIAL_INTEGRAL_GLSL

// Closed-form emission integral of rho(p) = F(h) * exp(-k * |p.xz|^2 / R(h)^2) over a ray segment.
// The integrand is polynomial x Gaussian, so each height band reduces to the moments
//   M_m = int s^m exp(-(a s^2 + b s + c)) ds,  M_0 from erf and M_1, M_2 from
//   2a*M_{m+1} = m*M_{m-1} - b*M_m - [s^m exp(...)].
//
// Must be included after FlameUBO and evaluateHeightFalloff.
// Mirrored in thyllore-render-core/src/flame_radial.rs; the accuracy tests live there.

const int FLAME_RADIAL_BAND_COUNT = 6;
const float FLAME_RADIAL_MIN_DIR_Y = 1e-4;
const float FLAME_RADIAL_MIN_HEIGHT_SPAN = 1e-5;
// Exponent variation across a band below which the constant-exponent moments are used.
const float FLAME_RADIAL_FLAT_EXPONENT = 2e-2;
const float FLAME_RADIAL_SUPPORT_SIGMA = 4.0;
const float FLAME_RADIAL_PI = 3.14159265358979;
// Maximum radius of the flame support (pedestal subtraction).
const float FLAME_RADIAL_RMAX = FLAME_SHELL_SUPPORT_HEADROOM;

// Abramowitz-Stegun 7.1.26.
float flameErf(float x) {
    float magnitude = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * magnitude);
    float series = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t;
    return sign(x) * (1.0 - series * exp(-magnitude * magnitude));
}

// int_{-halfWidth}^{halfWidth} s^m exp(-(a s^2 + b s + c)) ds for m = 0, 1, 2.
vec3 flameGaussianMoments(float a, float b, float c, float halfWidth) {
    if (a * halfWidth * halfWidth < FLAME_RADIAL_FLAT_EXPONENT
        && abs(b) * halfWidth < FLAME_RADIAL_FLAT_EXPONENT) {
        float constantWeight = exp(-c);
        return constantWeight
            * vec3(2.0 * halfWidth, 0.0, (2.0 / 3.0) * halfWidth * halfWidth * halfWidth);
    }

    float rootA = sqrt(a);
    float center = b / (2.0 * a);
    float peak = exp(-(c - b * b / (4.0 * a)));
    float moment0 = peak * 0.5 * sqrt(FLAME_RADIAL_PI / a)
        * (flameErf(rootA * (halfWidth + center)) - flameErf(rootA * (center - halfWidth)));

    float gaugeHi = exp(-(a * halfWidth * halfWidth + b * halfWidth + c));
    float gaugeLo = exp(-(a * halfWidth * halfWidth - b * halfWidth + c));
    float moment1 = (gaugeLo - gaugeHi - b * moment0) / (2.0 * a);
    float moment2 = (moment0 - b * moment1 - halfWidth * (gaugeHi + gaugeLo)) / (2.0 * a);
    return vec3(moment0, moment1, moment2);
}

// Pedestal interval moments: [int_I(1), int_I(s), int_I(s^2)] over the interval I where
// a*s^2 + b*s + c <= emax, clipped to [-halfWidth, halfWidth].
// If discriminant < 0 (ray always inside support), I is the entire band.
// If ray is outside support (no roots within band), returns zero vector.
vec3 flamePedestalIntervalMoments(float a, float b, float c, float emax, float halfWidth) {
    // If a is near zero, treat as linear: b*s + c <= emax => s <= (emax - c)/b.
    if (a < 1e-12) {
        // Near-flat exponent: if center is inside support, use full band; else empty.
        if (c > emax) {
            return vec3(0.0);
        }
        // Full band [-halfWidth, halfWidth]: I0=2h, I1=0, I2=(2/3)*h^3
        return vec3(2.0 * halfWidth, 0.0, (2.0 / 3.0) * halfWidth * halfWidth * halfWidth);
    }

    float discriminant = b * b - 4.0 * a * (c - emax);
    if (discriminant < 0.0) {
        // Ray always inside support — I is the entire band [-halfWidth, halfWidth].
        return vec3(2.0 * halfWidth, 0.0, (2.0 / 3.0) * halfWidth * halfWidth * halfWidth);
    }

    float root = sqrt(discriminant);
    float s_lo = (-b - root) / (2.0 * a);
    float s_hi = (-b + root) / (2.0 * a);
    // Clip to [-halfWidth, halfWidth]
    s_lo = max(s_lo, -halfWidth);
    s_hi = min(s_hi, halfWidth);
    if (s_hi <= s_lo) {
        return vec3(0.0);
    }
    // Polynomial moments over [s_lo, s_hi]: I0 = s_hi - s_lo, I1 = 0.5*(s_hi^2 - s_lo^2), I2 = (1/3)*(s_hi^3 - s_lo^3)
    return vec3(s_hi - s_lo, 0.5 * (s_hi * s_hi - s_lo * s_lo), (1.0 / 3.0) * (s_hi * s_hi * s_hi - s_lo * s_lo * s_lo));
}

// Gaussian radius R(h) of the radial density, in flame-local units.
float flameRadialGaussianScale(float height01) {
    return FLAME_SHELL_BASE_RADIUS
        * mix(1.0, flame.styleParams1.x, pow(height01, flame.styleParams0.w));
}

float flameRadialDensityFactor(vec3 p, float height01) {
    float radius = length(p.xz) / max(flameRadialGaussianScale(height01), 1e-4);
    float exponent = flame.radialSharpness * radius * radius;
    float emax = flame.radialSharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
    return max(0.0, exp(-exponent) - exp(-emax));
}

// Squared reciprocal of the Gaussian radius, which is what the exponent scales by.
float flameRadialExponentScale(float height01) {
    float scale = max(flameRadialGaussianScale(height01), 1e-4);
    return flame.radialSharpness / (scale * scale);
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

// Rays with a near-constant height cannot be parameterized by h, so the moment is taken along t.
float integrateRadialEmissionAlongRay(vec3 o, vec3 d, float tNear, float tFar, float k) {
    float tCenter = 0.5 * (tNear + tFar);
    vec2 p = o.xz + tCenter * d.xz;
    float halfWidth = 0.5 * (tFar - tNear);
    vec3 moments = flameGaussianMoments(
        k * dot(d.xz, d.xz), 2.0 * k * dot(p, d.xz), k * dot(p, p), halfWidth);
    // Pedestal subtraction: subtract exp(-Emax) * integral over interval I where E(s) <= Emax.
    float emax = flame.radialSharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
    float a = k * dot(d.xz, d.xz);
    float b = 2.0 * k * dot(p, d.xz);
    float c = k * dot(p, p);
    vec3 pedestalMoments = flamePedestalIntervalMoments(a, b, c, emax, halfWidth);
    return evaluateHeightFalloff(clamp(o.y + tCenter * d.y, 0.0, 1.0)) * (moments.x - exp(-emax) * pedestalMoments.x);
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
        return integrateRadialEmissionAlongRay(
            vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, tNear, tFar,
            flameRadialExponentScale(midHeight) / (wMid * wMid))
            * flameNoiseErosionFactor(pMid, midHeight);
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    vec2 q = d.xz / d.y;
    float quadratic = dot(q, q);

    // Trim to the Gaussian support so grazing rays do not spend bands on empty range.
    float wTrim = 1.0 + max(flame.contourParams.x, 0.0);
    float baseScale = flameRadialExponentScale(0.0) / (wTrim * wTrim);
    if (quadratic * baseScale > 1e-12) {
        float support = FLAME_RADIAL_SUPPORT_SIGMA / sqrt(quadratic * baseScale);
        float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
        heightLo = max(heightLo, closestApproachHeight - support);
        heightHi = min(heightHi, closestApproachHeight + support);
        if (heightHi <= heightLo) {
            return 0.0;
        }
    }

    float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
    float halfWidth = 0.5 * bandWidth;
    float total = 0.0;
    float falloffLo = evaluateHeightFalloff(heightLo);
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        float center = heightLo + (float(band) + 0.5) * bandWidth;
        float falloffMid = evaluateHeightFalloff(center);
        float falloffHi = evaluateHeightFalloff(center + halfWidth);

        vec2 pTrue = flameRayPointAtHeight(o, q, center);
        vec2 pxz = pTrue - flameBendOffsetAt(center);
        float eBand = flameNoiseErosionFactor(vec3(pTrue.x, center, pTrue.y), center);
        float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
        float k = flameRadialExponentScale(center) / (wBand * wBand);
        vec3 moments = flameGaussianMoments(
            k * quadratic, 2.0 * k * dot(pxz, q), k * dot(pxz, pxz), halfWidth);
        float slope = (falloffHi - falloffLo) / bandWidth;
        float curvature = 2.0 * (falloffHi + falloffLo - 2.0 * falloffMid) / (bandWidth * bandWidth);

        // Pedestal subtraction: subtract exp(-Emax) * [falloffMid*I0 + slope*I1 + curvature*I2]
        // where I_m = int_I s^m ds over the interval I where a*s^2+b*s+c <= Emax, clipped to [-halfWidth,halfWidth].
        float emax = flame.radialSharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
        float a = k * quadratic;
        float b = 2.0 * k * dot(pxz, q);
        float c = k * dot(pxz, pxz);
        vec3 pedestalMoments = flamePedestalIntervalMoments(a, b, c, emax, halfWidth);
        total += eBand * (falloffMid * moments.x + slope * moments.y + curvature * moments.z
            - exp(-emax) * (falloffMid * pedestalMoments.x + slope * pedestalMoments.y + curvature * pedestalMoments.z));

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
            float c = integrateRadialEmissionAlongRay(
                vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, t0, t0 + dt,
                flameRadialExponentScale(midHeight) / (wMid * wMid))
                * flameNoiseErosionFactor(pMid, midHeight);
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = midHeight;
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(pMid.xz - bendMid) / max(flameRadialGaussianScale(midHeight), 1e-4)), 0.0, 1.0);
        }
    } else {
        vec2 q = d.xz / d.y;
        float quadratic = dot(q, q);

        float wTrim = 1.0 + max(flame.contourParams.x, 0.0);
        float baseScale = flameRadialExponentScale(0.0) / (wTrim * wTrim);
        if (quadratic * baseScale > 1e-12) {
            float support = FLAME_RADIAL_SUPPORT_SIGMA / sqrt(quadratic * baseScale);
            float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
            heightLo = max(heightLo, closestApproachHeight - support);
            heightHi = min(heightHi, closestApproachHeight + support);
            if (heightHi <= heightLo) {
                return vec4(0.0);
            }
        }

        float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
        float halfWidth = 0.5 * bandWidth;
        float falloffLo = evaluateHeightFalloff(heightLo);
        float emax = flame.radialSharpness * FLAME_RADIAL_RMAX * FLAME_RADIAL_RMAX;
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            float center = heightLo + (float(band) + 0.5) * bandWidth;
            float falloffMid = evaluateHeightFalloff(center);
            float falloffHi = evaluateHeightFalloff(center + halfWidth);

            vec2 pTrue = flameRayPointAtHeight(o, q, center);
            vec2 p = pTrue - flameBendOffsetAt(center);
            float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
            float k = flameRadialExponentScale(center) / (wBand * wBand);
            float eBand = flameNoiseErosionFactor(vec3(pTrue.x, center, pTrue.y), center);
            vec3 moments = flameGaussianMoments(
                k * quadratic, 2.0 * k * dot(p, q), k * dot(p, p), halfWidth);
            float slope = (falloffHi - falloffLo) / bandWidth;
            float curvature = 2.0 * (falloffHi + falloffLo - 2.0 * falloffMid) / (bandWidth * bandWidth);
            vec3 pedestalMoments = flamePedestalIntervalMoments(
                k * quadratic, 2.0 * k * dot(p, q), k * dot(p, p), emax, halfWidth);
            float c = eBand
                * (falloffMid * moments.x + slope * moments.y + curvature * moments.z
                    - exp(-emax) * (falloffMid * pedestalMoments.x + slope * pedestalMoments.y
                        + curvature * pedestalMoments.z))
                / abs(d.y);
            float meanOffset = moments.x > 1e-6 ? clamp(moments.y / moments.x, -halfWidth, halfWidth) : 0.0;
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = clamp(center + meanOffset, 0.0, 1.0);
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(p) / max(flameRadialGaussianScale(center), 1e-4)), 0.0, 1.0);

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
