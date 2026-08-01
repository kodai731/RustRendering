#ifndef FLAME_ERF_MOMENTS_GLSL
#define FLAME_ERF_MOMENTS_GLSL

// M_erf / M_gauss moment families over a centered interval [-h, h]:
//   E(n) = int s^n erf(alpha s + beta) ds          (n = 0..6)
//   J(m) = int s^m exp(-(alpha s + beta)^2) ds     (m = 0..7)
// Mirrored in thyllore-math-core/src/erf_moments.rs, which carries the accuracy
// tests (f64 internals there; this f32 mirror trades tail precision for speed —
// fine against the erf-bridge fit targets of ~5e-3).
// Regimes: Taylor around the interval center for |alpha| h <= 0.5 (the recurrence
// would divide by alpha^2 and cancel), integration-by-parts recurrence in
// sigma = s / h otherwise.

const int FLAME_ERF_MOMENT_COUNT = 7;
const int FLAME_GAUSSIAN_MOMENT_COUNT = 8;
const int FLAME_ERF_TAYLOR_TERMS = 13;
const float FLAME_ERF_TAYLOR_ALPHA_H = 0.5;
const float FLAME_ERF_SATURATION = 5.5;
const float FLAME_ERF_SQRT_PI = 1.7724538509;

// Complementary error function for any real x, fractional error <= 1.2e-7
// (Numerical Recipes rational-exponential fit). Relative accuracy in the tail is
// what keeps the moment recurrence usable.
float flameErfc(float x) {
    float magnitude = abs(x);
    float t = 1.0 / (1.0 + 0.5 * magnitude);
    float tail = t * exp(-magnitude * magnitude - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587
                                    + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0.0 ? tail : 2.0 - tail;
}

float flameErf(float x) {
    return 1.0 - flameErfc(x);
}

// int_{-1}^{1} sigma^j d sigma.
float flameUnitPowerMoment(int j) {
    return (j % 2 == 1) ? 0.0 : 2.0 / float(j + 1);
}

// Both moment families over [-halfWidth, halfWidth].
void flameErfGaussianPowerMoments(
    float alpha, float beta, float halfWidth,
    out float erfMoments[FLAME_ERF_MOMENT_COUNT],
    out float gaussianMoments[FLAME_GAUSSIAN_MOMENT_COUNT]) {
    for (int n = 0; n < FLAME_ERF_MOMENT_COUNT; ++n) {
        erfMoments[n] = 0.0;
    }
    for (int m = 0; m < FLAME_GAUSSIAN_MOMENT_COUNT; ++m) {
        gaussianMoments[m] = 0.0;
    }
    if (halfWidth <= 0.0) {
        return;
    }

    float alphaH = alpha * halfWidth;

    // Scale factors h^{j+1} shared by both families (sigma = s / h).
    float scale[FLAME_GAUSSIAN_MOMENT_COUNT];
    float power = halfWidth;
    for (int j = 0; j < FLAME_GAUSSIAN_MOMENT_COUNT; ++j) {
        scale[j] = power;
        power *= halfWidth;
    }

    // Saturated bridge: erf is +-1 over the whole interval and the Gaussian vanishes.
    if (beta - abs(alphaH) > FLAME_ERF_SATURATION || beta + abs(alphaH) < -FLAME_ERF_SATURATION) {
        float saturatedSign = beta > 0.0 ? 1.0 : -1.0;
        for (int n = 0; n < FLAME_ERF_MOMENT_COUNT; ++n) {
            erfMoments[n] = saturatedSign * flameUnitPowerMoment(n) * scale[n];
        }
        return;
    }

    float erfSigma[FLAME_ERF_MOMENT_COUNT];
    float gaussianSigma[FLAME_GAUSSIAN_MOMENT_COUNT];

    if (abs(alphaH) <= FLAME_ERF_TAYLOR_ALPHA_H) {
        // Gaussian Taylor coefficients c_k = (-1)^k H_k(beta) exp(-beta^2) / k! via the
        // physicists' Hermite recurrence; erf' = (2/sqrt(pi)) exp(-t^2) chains them.
        float weight = exp(-beta * beta);
        float gaussianTaylor[FLAME_ERF_TAYLOR_TERMS];
        gaussianTaylor[0] = weight;
        float hermitePrev = 1.0;
        float hermite = 2.0 * beta;
        float factorial = 1.0;
        float sign = 1.0;
        for (int k = 1; k < FLAME_ERF_TAYLOR_TERMS; ++k) {
            factorial *= float(k);
            sign = -sign;
            gaussianTaylor[k] = sign * hermite * weight / factorial;
            float hermiteNext = 2.0 * beta * hermite - 2.0 * float(k) * hermitePrev;
            hermitePrev = hermite;
            hermite = hermiteNext;
        }

        float erfSeries[FLAME_ERF_TAYLOR_TERMS];
        float gaussianSeries[FLAME_ERF_TAYLOR_TERMS];
        erfSeries[0] = flameErf(beta);
        gaussianSeries[0] = gaussianTaylor[0];
        float alphaPower = 1.0;
        for (int k = 1; k < FLAME_ERF_TAYLOR_TERMS; ++k) {
            alphaPower *= alphaH;
            erfSeries[k] = (2.0 / FLAME_ERF_SQRT_PI) * gaussianTaylor[k - 1] * alphaPower / float(k);
            gaussianSeries[k] = gaussianTaylor[k] * alphaPower;
        }

        for (int n = 0; n < FLAME_ERF_MOMENT_COUNT; ++n) {
            float total = 0.0;
            for (int k = 0; k < FLAME_ERF_TAYLOR_TERMS; ++k) {
                total += erfSeries[k] * flameUnitPowerMoment(n + k);
            }
            erfSigma[n] = total;
        }
        for (int m = 0; m < FLAME_GAUSSIAN_MOMENT_COUNT; ++m) {
            float total = 0.0;
            for (int k = 0; k < FLAME_ERF_TAYLOR_TERMS; ++k) {
                total += gaussianSeries[k] * flameUnitPowerMoment(m + k);
            }
            gaussianSigma[m] = total;
        }
    } else {
        float tHi = alphaH + beta;
        float tLo = -alphaH + beta;
        float gaugeHi = exp(-tHi * tHi);
        float gaugeLo = exp(-tLo * tLo);
        // erf_hi - erf_lo == erfc(t_lo) - erfc(t_hi): the erfc form keeps relative
        // accuracy in the tails where the direct erf difference cancels.
        float erfDifference = flameErfc(tLo) - flameErfc(tHi);
        float erfSum = 2.0 - flameErfc(tHi) - flameErfc(tLo);

        gaussianSigma[0] = FLAME_ERF_SQRT_PI / (2.0 * alphaH) * erfDifference;
        float invTwoAlphaSq = 1.0 / (2.0 * alphaH * alphaH);
        float betaOverAlpha = beta / alphaH;
        for (int j = 1; j < FLAME_GAUSSIAN_MOMENT_COUNT; ++j) {
            // sigma^{j-1} at the bounds: 1 at sigma = 1, (-1)^{j-1} at sigma = -1.
            float boundary = ((j - 1) % 2 == 0) ? gaugeHi - gaugeLo : gaugeHi + gaugeLo;
            float twoBack = j >= 2 ? gaussianSigma[j - 2] : 0.0;
            gaussianSigma[j] = -invTwoAlphaSq * boundary
                + (float(j) - 1.0) * invTwoAlphaSq * twoBack
                - betaOverAlpha * gaussianSigma[j - 1];
        }

        // E(n) by parts: [sigma^{n+1} erf]_{-1}^{1} feeds the boundary, J(n+1) the rest.
        for (int n = 0; n < FLAME_ERF_MOMENT_COUNT; ++n) {
            float boundary = ((n + 1) % 2 == 0) ? erfDifference : erfSum;
            erfSigma[n] = (boundary
                - (2.0 / FLAME_ERF_SQRT_PI) * alphaH * gaussianSigma[n + 1]) / float(n + 1);
        }
    }

    for (int n = 0; n < FLAME_ERF_MOMENT_COUNT; ++n) {
        erfMoments[n] = erfSigma[n] * scale[n];
    }
    for (int m = 0; m < FLAME_GAUSSIAN_MOMENT_COUNT; ++m) {
        gaussianMoments[m] = gaussianSigma[m] * scale[m];
    }
}

#endif
