#ifndef WIND_LIGHTING_GLSL
#define WIND_LIGHTING_GLSL

// Single scattering along the view ray: per closed-form piece the in-scatter source is
// averaged over fixed midpoint nodes, each node shadowed by the same shell field toward
// the sun and toward the zenith.
// Must be included after wind_shell_integral.glsl.

#include "include/radiative_transfer.glsl"

const int WIND_SCATTER_NODES = 4;
const float WIND_SHADOW_RAY_T_MAX = 1e4;
const vec3 WIND_ZENITH_DIRECTION = vec3(0.0, 1.0, 0.0);

float windOpticalDepthToward(vec3 origin, vec3 direction) {
    float tNear = 0.0;
    float tFar = WIND_SHADOW_RAY_T_MAX;
    if (!clampToWindCone(origin, direction, tNear, tFar)) {
        return 0.0;
    }
    tNear = max(tNear, 0.0);
    if (tFar <= tNear) {
        return 0.0;
    }
    int knotCount = 0;
    return windOpticalDepth(origin, direction, tNear, tFar, knotCount);
}

float windInScatterSource(vec3 position, vec3 lightPosition, vec3 viewDir) {
    vec3 lightDir = normalize(lightPosition - position);
    float sunTransmittance = rteTransmittanceFromOpticalDepth(windOpticalDepthToward(position, lightDir));
    float skyTransmittance =
        rteTransmittanceFromOpticalDepth(windOpticalDepthToward(position, WIND_ZENITH_DIRECTION));
    return windSunIntensity() * sunTransmittance
            * rteHenyeyGreenstein(dot(viewDir, lightDir), windPhaseG())
        + windSkyBrightness() * skyTransmittance;
}

float windPieceInScatter(vec3 o, vec3 d, float s0, float s1, vec3 lightPosition, vec3 viewDir) {
    float pieceLength = s1 - s0;
    float sum = 0.0;
    for (int i = 0; i < WIND_SCATTER_NODES; ++i) {
        vec3 node = o + d * (s0 + rteMidpointDistance(i, WIND_SCATTER_NODES, pieceLength));
        sum += windInScatterSource(node, lightPosition, viewDir);
    }
    return sum / float(WIND_SCATTER_NODES);
}

vec3 windSingleScatterRadiance(
    vec3 o, vec3 d, float tNear, float tFar, vec3 lightPosition,
    out float opticalDepth, out int knotCount) {
    opticalDepth = 0.0;
    knotCount = 0;
    if (tFar <= tNear) {
        return vec3(0.0);
    }

    float knots[WIND_MAX_KNOTS];
    knotCount = windRayKnots(o, d, tNear, tFar, knots);
    vec3 viewDir = normalize(d);

    float radiance = 0.0;
    for (int i = 1; i < knotCount; ++i) {
        float pieceDepth = windPieceOpticalDepth(o, d, knots[i - 1], knots[i]);
        float frontTransmittance = rteTransmittanceFromOpticalDepth(opticalDepth);
        float source = windPieceInScatter(o, d, knots[i - 1], knots[i], lightPosition, viewDir);
        radiance += frontTransmittance * source * (1.0 - rteTransmittanceFromOpticalDepth(pieceDepth));
        opticalDepth += pieceDepth;
    }
    return radiance * wind.albedo.rgb;
}

#endif
