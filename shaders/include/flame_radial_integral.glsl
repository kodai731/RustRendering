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

// Gaussian radius R(h) of the radial density, in flame-local units.
float flameRadialGaussianScale(float height01) {
    return FLAME_SHELL_BASE_RADIUS
        * mix(1.0, flame.styleParams1.x, pow(height01, flame.styleParams0.w));
}

// Per-sample form of the same density factor, for the reference raymarch to integrate.
float flameRadialDensityFactor(vec3 p, float height01) {
    float radius = length(p.xz) / max(flameRadialGaussianScale(height01), 1e-4);
    return exp(-flame.radialSharpness * radius * radius);
}

// Squared reciprocal of the Gaussian radius, which is what the exponent scales by.
float flameRadialExponentScale(float height01) {
    float scale = max(flameRadialGaussianScale(height01), 1e-4);
    return flame.radialSharpness / (scale * scale);
}

// Rays with a near-constant height cannot be parameterized by h, so the moment is taken along t.
float integrateRadialEmissionAlongRay(vec3 o, vec3 d, float tNear, float tFar, float k) {
    float tCenter = 0.5 * (tNear + tFar);
    vec2 p = o.xz + tCenter * d.xz;
    vec3 moments = flameGaussianMoments(
        k * dot(d.xz, d.xz), 2.0 * k * dot(p, d.xz), k * dot(p, p), 0.5 * (tFar - tNear));
    return evaluateHeightFalloff(clamp(o.y + tCenter * d.y, 0.0, 1.0)) * moments.x;
}

// Horizontal offset of the ray at a given height, with q = d.xz / d.y.
vec2 flameRayPointAtHeight(vec3 o, vec2 q, float height) {
    return o.xz + (height - o.y) * q;
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
        return integrateRadialEmissionAlongRay(
            o, d, tNear, tFar, flameRadialExponentScale(midHeight));
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    vec2 q = d.xz / d.y;
    float quadratic = dot(q, q);

    // Trim to the Gaussian support so grazing rays do not spend bands on empty range.
    float baseScale = flameRadialExponentScale(0.0);
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

        float k = flameRadialExponentScale(center);
        vec2 p = flameRayPointAtHeight(o, q, center);
        vec3 moments = flameGaussianMoments(
            k * quadratic, 2.0 * k * dot(p, q), k * dot(p, p), halfWidth);
        float slope = (falloffHi - falloffLo) / bandWidth;
        float curvature = 2.0 * (falloffHi + falloffLo - 2.0 * falloffMid) / (bandWidth * bandWidth);
        total += falloffMid * moments.x + slope * moments.y + curvature * moments.z;

        falloffLo = falloffHi;
    }
    return total / abs(d.y);
}

#endif
