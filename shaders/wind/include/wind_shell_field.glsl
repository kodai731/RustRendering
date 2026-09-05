#ifndef WIND_SHELL_FIELD_GLSL
#define WIND_SHELL_FIELD_GLSL

// Density field of the tornado: compact-support polynomial shells in q = x^2 + z^2
// (wall around P(h) = (base + slope * h)^2, core around the axis) times a height
// envelope. Mirrored in thyllore-effect-core/src/wind/analytic/shell_integral.rs.
// Must be included after wind_component.glsl.

const float WIND_LINEAR_COEFFICIENT_EPSILON = 1e-7;

float windHeight() { return wind.shape.x; }
float windWallRadiusBase() { return wind.shape.y; }
float windWallRadiusSlope() { return wind.shape.z; }
float windWallWidthQ() { return wind.shape.w; }
float windCoreRadiusSq() { return wind.core.x; }
float windCoreStrength() { return wind.core.y; }
float windWallStrength() { return wind.core.z; }
float windTopFade() { return wind.core.w; }
float windSigmaT() { return wind.optics.x; }
float windSkyBrightness() { return wind.optics.y; }

bool windCoreActive() {
    return windCoreRadiusSq() > 1e-8 && windCoreStrength() > 0.0;
}

float windFadeStart() {
    return 1.0 - windTopFade();
}

float windWallRadius(float h) {
    return windWallRadiusBase() + windWallRadiusSlope() * h;
}

float windEnvelopeRadius(float h) {
    return max(max(windWallRadius(h), sqrt(windCoreRadiusSq())), 0.0) + sqrt(windWallWidthQ());
}

float windEnvelopeHeight(float h) {
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }
    float fadeStart = windFadeStart();
    if (h <= fadeStart) {
        return 1.0;
    }
    float v = (h - fadeStart) / windTopFade();
    return 1.0 - v * v * (3.0 - 2.0 * v);
}

float windBiweight(float u) {
    float inside = max(1.0 - u * u, 0.0);
    return inside * inside;
}

float windDensityAt(vec3 p) {
    float h = p.y / windHeight();
    float envelope = windEnvelopeHeight(h);
    if (envelope <= 0.0) {
        return 0.0;
    }
    float q = p.x * p.x + p.z * p.z;

    float wallRadius = windWallRadius(h);
    float wall = windWallStrength() * windBiweight((q - wallRadius * wallRadius) / windWallWidthQ());
    float core = windCoreActive() ? windCoreStrength() * windBiweight(q / windCoreRadiusSq()) : 0.0;
    return windSigmaT() * envelope * (wall + core);
}

// Cone frustum (radius linear in height between the envelope radii) x height slab.
bool clampToWindCone(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    float radiusBase = windEnvelopeRadius(0.0);
    float radiusTop = windEnvelopeRadius(1.0);
    float slopePerUnitY = (radiusTop - radiusBase) / windHeight();
    float m = radiusBase + slopePerUnitY * o.y;
    float n = slopePerUnitY * d.y;
    float a = dot(d.xz, d.xz) - n * n;
    float b = 2.0 * (dot(o.xz, d.xz) - m * n);
    float c = dot(o.xz, o.xz) - m * m;

    if (abs(a) < WIND_LINEAR_COEFFICIENT_EPSILON) {
        if (abs(b) < WIND_LINEAR_COEFFICIENT_EPSILON) {
            if (c > 0.0) return false;
        } else {
            float tRoot = -c / b;
            if (b > 0.0) {
                tFar = min(tFar, tRoot);
            } else {
                tNear = max(tNear, tRoot);
            }
        }
    } else {
        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0) {
            if (a > 0.0) return false;
        } else if (a > 0.0) {
            float sqrtDiscriminant = sqrt(discriminant);
            float t0 = (-b - sqrtDiscriminant) / (2.0 * a);
            float t1 = (-b + sqrtDiscriminant) / (2.0 * a);
            tNear = max(tNear, min(t0, t1));
            tFar = min(tFar, max(t0, t1));
        }
    }

    if (abs(d.y) < WIND_LINEAR_COEFFICIENT_EPSILON) {
        if (o.y < 0.0 || o.y > windHeight()) return false;
    } else {
        float tY0 = -o.y / d.y;
        float tY1 = (windHeight() - o.y) / d.y;
        tNear = max(tNear, min(tY0, tY1));
        tFar = min(tFar, max(tY0, tY1));
    }

    return tNear <= tFar;
}

#endif
