#ifndef FLAME_EROSION_RESPONSE_GLSL
#define FLAME_EROSION_RESPONSE_GLSL

// Additive erf-bridge model of the erosion threshold response (P4):
//   phi(x) = 0.5 (1 + erf(xi)) + c1 psi(xi) + c2 psi(xi / 2),
//   xi = kappa (x - center), psi(t) = t exp(-t^2),
// fit at bake time to smoothstep(edge_low, edge_high, x) and packed into
// flame.erosionResponse = (center, kappa, c1, c2).
// Unresolved band noise of std sigma smooths every term inside its own family
// (tau = kappa sigma): erf keeps its shape with kappa_eff = kappa / sqrt(1 + 2 tau^2),
// the corrections shrink by r^2 — sigma -> 0 restores the fitted sharpness,
// large sigma flattens the response.
// Mirrored in thyllore-math-core/src/erf_response.rs (accuracy tests live there).
// Must be included after FlameUBO and flame_erf_moments.glsl.

struct FlameSmoothedResponse {
    float center;
    float kappaEff;
    float weight1;
    float weight2;
    float kappa2;
};

FlameSmoothedResponse flameSmoothErosionResponse(float sigma) {
    float kappa = flame.erosionResponse.y;
    float tauSquared = kappa * sigma * kappa * sigma;
    float r1 = inversesqrt(1.0 + 2.0 * tauSquared);
    float r2 = inversesqrt(1.0 + 0.5 * tauSquared);
    FlameSmoothedResponse response;
    response.center = flame.erosionResponse.x;
    response.kappaEff = kappa * r1;
    response.weight1 = flame.erosionResponse.z * r1 * r1;
    response.weight2 = flame.erosionResponse.w * r2 * r2;
    response.kappa2 = 0.5 * kappa * r2;
    return response;
}

// Integral (.x) and first moment (.y) of the smoothed response along the linear
// argument x(s) = x0 + x1 s over [s0, s1]. Every term reduces to the E/J moment
// families over the piece, shifted to its own center.
vec2 flameErosionResponseLinearIntegral(
    FlameSmoothedResponse response, float x0, float x1, float s0, float s1) {
    if (s1 <= s0) {
        return vec2(0.0);
    }
    float mid = 0.5 * (s0 + s1);
    float halfWidth = 0.5 * (s1 - s0);
    float y0 = x0 - response.center + x1 * mid;
    float span = s1 - s0;
    vec2 result = 0.5 * vec2(span, span * mid);

    float erfMoments[FLAME_ERF_MOMENT_COUNT];
    float gaussianMoments[FLAME_GAUSSIAN_MOMENT_COUNT];
    float alpha1 = response.kappaEff * x1;
    float beta1 = response.kappaEff * y0;
    flameErfGaussianPowerMoments(alpha1, beta1, halfWidth, erfMoments, gaussianMoments);
    result += 0.5 * vec2(erfMoments[0], mid * erfMoments[0] + erfMoments[1]);
    // psi(alpha s + beta) = (alpha s + beta) exp(-(alpha s + beta)^2).
    float psi1 = alpha1 * gaussianMoments[1] + beta1 * gaussianMoments[0];
    float psi1First = alpha1 * gaussianMoments[2] + beta1 * gaussianMoments[1];
    result += response.weight1 * vec2(psi1, mid * psi1 + psi1First);

    if (abs(response.weight2) > 1e-5) {
        float erfMoments2[FLAME_ERF_MOMENT_COUNT];
        float gaussianMoments2[FLAME_GAUSSIAN_MOMENT_COUNT];
        float alpha2 = response.kappa2 * x1;
        float beta2 = response.kappa2 * y0;
        flameErfGaussianPowerMoments(alpha2, beta2, halfWidth, erfMoments2, gaussianMoments2);
        float psi2 = alpha2 * gaussianMoments2[1] + beta2 * gaussianMoments2[0];
        float psi2First = alpha2 * gaussianMoments2[2] + beta2 * gaussianMoments2[1];
        result += response.weight2 * vec2(psi2, mid * psi2 + psi2First);
    }

    return result;
}

#endif
