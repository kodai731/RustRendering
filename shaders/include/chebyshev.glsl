// chebyshev.glsl - Clenshaw evaluation of Chebyshev series (fully unrolled)
//
// Coefficient layout matches thyllore-math-core pack_coefficients_vec4:
//   c0 = [C0..C3], c1 = [C4..C7], c2 = [C8..C11]
// Series must be fit over domain [0,1]; x01 is normalized to [-1,1] internally.

#ifndef CHEBYSHEV_GLSL
#define CHEBYSHEV_GLSL

float evaluateChebyshev8(vec4 c0, vec4 c1, float x01) {
    float u = 2.0 * x01 - 1.0;
    float t = 2.0 * u;
    float b7 = c1.w;
    float b6 = t * b7 + c1.z;
    float b5 = t * b6 - b7 + c1.y;
    float b4 = t * b5 - b6 + c1.x;
    float b3 = t * b4 - b5 + c0.w;
    float b2 = t * b3 - b4 + c0.z;
    float b1 = t * b2 - b3 + c0.y;
    return u * b1 - b2 + c0.x;
}

float evaluateChebyshev12(vec4 c0, vec4 c1, vec4 c2, float x01) {
    float u = 2.0 * x01 - 1.0;
    float t = 2.0 * u;
    float b11 = c2.w;
    float b10 = t * b11 + c2.z;
    float b9 = t * b10 - b11 + c2.y;
    float b8 = t * b9 - b10 + c2.x;
    float b7 = t * b8 - b9 + c1.w;
    float b6 = t * b7 - b8 + c1.z;
    float b5 = t * b6 - b7 + c1.y;
    float b4 = t * b5 - b6 + c1.x;
    float b3 = t * b4 - b5 + c0.w;
    float b2 = t * b3 - b4 + c0.z;
    float b1 = t * b2 - b3 + c0.y;
    return u * b1 - b2 + c0.x;
}

#endif
