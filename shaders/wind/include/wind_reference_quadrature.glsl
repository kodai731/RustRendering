#ifndef WIND_REFERENCE_QUADRATURE_GLSL
#define WIND_REFERENCE_QUADRATURE_GLSL

// Sample-based reference for the closed-form integral (debug mode only): midpoint
// quadrature of the pointwise density. Never part of the product path.

float windReferenceOpticalDepth(vec3 o, vec3 d, float tNear, float tFar, int stepCount) {
    int steps = max(stepCount, 1);
    float step = (tFar - tNear) / float(steps);
    float total = 0.0;
    for (int i = 0; i < steps; ++i) {
        float t = tNear + (float(i) + 0.5) * step;
        total += windDensityAt(o + d * t);
    }
    return total * step;
}

#endif
