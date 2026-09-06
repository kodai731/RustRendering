#ifndef WIND_SHELL_FIELD_GLSL
#define WIND_SHELL_FIELD_GLSL

// Density field of the tornado: compact-support polynomial shells in q = x^2 + z^2
// (wall around P(h) = (base + slope * h)^2, core around the axis, ground ring around
// Pr faded over its own height) times a height envelope.
// Mirrored in thyllore-effect-core/src/wind/analytic/shell_integral.rs.
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
float windHTop() { return wind.optics.w; }
float windSpreadOffset() { return wind.albedo.w; }
float windRingHeight() { return wind.ring.x; }
float windRingRadiusSq() { return wind.ring.y; }
float windRingWidthQ() { return wind.ring.z; }
float windRingStrength() { return wind.ring.w; }
float windPhaseG() { return wind.lighting.x; }
float windSunIntensity() { return wind.lighting.y; }
float windStreakOrder() { return wind.streak.x; }
float windStreakTwist() { return wind.streak.y; }
float windStreakRiseSpeed() { return wind.streak.z; }
float windStreakAmplitude() { return wind.streak.w; }
float windStreakPhase() { return wind.streak2.x; }
float windStreakRiseTime() { return wind.streak2.y; }

bool windCoreActive() {
    return windCoreRadiusSq() > 1e-8 && windCoreStrength() > 0.0;
}

bool windRingActive() {
    return windRingStrength() > 0.0;
}

float windRingBoundsRadius() {
    return windRingActive() ? sqrt(max(windRingRadiusSq() + windRingWidthQ(), 0.0)) : 0.0;
}

float windRingTopY() {
    return windRingHeight() * windHeight();
}

float windFadeStart() {
    return 1.0 - windTopFade();
}

float windWallRadius(float h) {
    return windWallRadiusBase() + windWallRadiusSlope() * h;
}

float windStreakSigma(vec3 local) {
    float angle = windStreakOrder() * atan(local.z, local.x) - windStreakTwist() * local.y
        - windStreakPhase() + windStreakRiseTime() * local.y;
    return 1.0 + windStreakAmplitude() * cos(angle);
}

float windWallRadiusSq(float h) {
    float radius = windWallRadius(h);
    return radius * radius + windSpreadOffset();
}

float windEnvelopeRadius(float h) {
    return sqrt(max(max(windWallRadiusSq(h), windCoreRadiusSq()), 0.0)) + sqrt(windWallWidthQ());
}

float windEnvelopeHeight(float h) {
    if (h < 0.0 || h > windHTop()) {
        return 0.0;
    }
    float normalizedHeight = h / windHTop();
    float fadeStart = windFadeStart();
    if (normalizedHeight <= fadeStart) {
        return 1.0;
    }
    float v = (normalizedHeight - fadeStart) / windTopFade();
    return 1.0 - v * v * (3.0 - 2.0 * v);
}

float windBiweight(float u) {
    float inside = max(1.0 - u * u, 0.0);
    return inside * inside;
}

float windRingFade(float v) {
    if (v >= 1.0) {
        return 0.0;
    }
    return 1.0 - v * v * (3.0 - 2.0 * v);
}

float windDensityAt(vec3 p) {
    float h = p.y / windHeight();
    float envelope = windEnvelopeHeight(h);
    if (envelope <= 0.0) {
        return 0.0;
    }
    float q = p.x * p.x + p.z * p.z;

    float wall = windWallStrength() * windBiweight((q - windWallRadiusSq(h)) / windWallWidthQ());
    float core = windCoreActive() ? windCoreStrength() * windBiweight(q / windCoreRadiusSq()) : 0.0;
    float ring = windRingActive()
        ? windRingStrength() * windRingFade(h / windRingHeight())
              * windBiweight((q - windRingRadiusSq()) / windRingWidthQ())
        : 0.0;
    return windSigmaT() * envelope * (wall + core + ring);
}

bool clampToConeFrustum(
    float radiusBase, float radiusTop, float topY,
    vec3 o, vec3 d, inout float tNear, inout float tFar) {
    float slopePerUnitY = (radiusTop - radiusBase) / topY;
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
        if (o.y < 0.0 || o.y > topY) return false;
    } else {
        float tY0 = -o.y / d.y;
        float tY1 = (topY - o.y) / d.y;
        tNear = max(tNear, min(tY0, tY1));
        tFar = min(tFar, max(tY0, tY1));
    }

    return tNear <= tFar;
}

// Cone frustum (radius linear in height between the envelope radii) x height slab.
// When ring is active, the cone is the union of the wall frustum and the ring frustum.
bool clampToWindCone(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    float topY = windHTop() * windHeight();
    float radiusBase = windEnvelopeRadius(0.0);
    float radiusTop = windEnvelopeRadius(windHTop());

    float savedNear = tNear;
    float savedFar = tFar;
    bool wallHit = clampToConeFrustum(radiusBase, radiusTop, topY, o, d, tNear, tFar);

    if (!windRingActive()) {
        return wallHit;
    }

    float ringRadius = windRingBoundsRadius();
    float ringTopY = windRingTopY();
    tNear = savedNear;
    tFar = savedFar;
    bool ringHit = clampToConeFrustum(ringRadius, ringRadius, ringTopY, o, d, tNear, tFar);

    if (!wallHit) {
        return ringHit;
    }
    float wallNear = tNear;
    float wallFar = tFar;
    tNear = savedNear;
    tFar = savedFar;
    clampToConeFrustum(radiusBase, radiusTop, topY, o, d, tNear, tFar);
    float finalWallNear = tNear;
    float finalWallFar = tFar;
    tNear = savedNear;
    tFar = savedFar;
    clampToConeFrustum(ringRadius, ringRadius, ringTopY, o, d, tNear, tFar);

    tNear = min(finalWallNear, tNear);
    tFar = max(finalWallFar, tFar);
    return true;
}

#endif
