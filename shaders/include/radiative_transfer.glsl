#ifndef RADIATIVE_TRANSFER_GLSL
#define RADIATIVE_TRANSFER_GLSL

#include "common.glsl"

// dL/ds = -sigma_t L + sigma_a L_e + sigma_s * integral p(theta) L_in
// Each effect supplies its own coefficients; this file holds only the equation.

float rteTransmittance(float sigmaT, float distance) {
    return exp(-sigmaT * distance);
}

vec3 rteTransmittance(vec3 sigmaT, float distance) {
    return exp(-sigmaT * distance);
}

float rteTransmittanceFromOpticalDepth(float opticalDepth) {
    return exp(-opticalDepth);
}

float rteOpacity(float sigmaT, float opticalDepth) {
    return 1.0 - exp(-sigmaT * opticalDepth);
}

// S * (1 - exp(-sigma*dt)) / sigma with Taylor fallback so sigma -> 0 stays continuous.
// Mirrored in thyllore-render-core/src/flame.rs (integrate_emission_segment) for tests.
float rteIntegrateEmissionSegment(float source, float sigmaT, float dt) {
    float x = sigmaT * dt;
    if (x < 1e-3) {
        return source * dt * (1.0 - 0.5 * x + x * x * (1.0 / 6.0));
    }
    return source * (1.0 - exp(-x)) / sigmaT;
}

float rteHenyeyGreenstein(float cosTheta, float g) {
    float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(max(denom, 1e-6)));
}

float rteMidpointDistance(int index, int sampleCount, float pathLength) {
    return (float(index) + 0.5) * pathLength / float(sampleCount);
}

vec3 rteSingleScatterSample(vec3 sigmaS, vec3 sigmaT, float phase, float viewDistance, float ds, vec3 lightRadiance) {
    return sigmaS * phase * rteTransmittance(sigmaT, viewDistance) * lightRadiance * ds;
}

#endif
