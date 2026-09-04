#ifndef WATER_TORUS_INTERSECT_GLSL
#define WATER_TORUS_INTERSECT_GLSL

#include "include/common.glsl"

// Signed cube root: pow(x, 1.0/3.0) is undefined for x < 0 in GLSL (NaN).
float cbrtSigned(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }

// Torus implicit function: (|p|^2 + 1 - rHat^2)^2 - 4*(x^2 + z^2) = 0
float torusImplicit(vec3 p, float rHat) {
    return pow(dot(p, p) + 1.0 - rHat * rHat, 2.0) - 4.0 * (p.x * p.x + p.z * p.z);
}

// Gradient of the implicit function
vec3 torusGradient(vec3 p, float rHat) {
    float factor = 4.0 * (dot(p, p) + 1.0 - rHat * rHat);
    return vec3(factor * p.x - 8.0 * p.x, factor * p.y, factor * p.z - 8.0 * p.z);
}

// Solve quartic c[4]*t^4 + c[3]*t^3 + c[2]*t^2 + c[1]*t + c[0] = 0
// using Ferrari's method (Graphics Gems Roots3And4).
// Returns number of real roots written to roots[].
int solveQuartic(float c[5], out float roots[4]) {
    float a = c[3] / c[4];
    float b = c[2] / c[4];
    float cc = c[1] / c[4];
    float d = c[0] / c[4];
    float sq_a = a * a;
    float p = -(3.0 / 8.0) * sq_a + b;
    float q = (1.0 / 8.0) * sq_a * a - (1.0 / 2.0) * a * b + cc;
    float r = -(3.0 / 256.0) * sq_a * sq_a + (1.0 / 16.0) * sq_a * b - (1.0 / 4.0) * a * cc + d;

    int count = 0;
    const float EPS = 1e-9;

    if (abs(r) < EPS) {
        // r ≈ 0: t * (t^3 + p*t + q) = 0
        roots[count++] = 0.0;
        float ca = 0.0;
        float cb = p;
        float cc_c = q;
        float sq_ca = ca * ca;
        float pc = (1.0 / 3.0) * (-(1.0 / 3.0) * sq_ca + cb);
        float qc = (1.0 / 2.0) * ((2.0 / 27.0) * ca * sq_ca - (1.0 / 3.0) * ca * cb + cc_c);
        float cb_p = pc * pc * pc;
        float cubic_d = qc * qc + cb_p;

        if (abs(cubic_d) < EPS) {
            if (abs(qc) < EPS) {
                roots[count++] = 0.0 - ca / 3.0;
            } else {
                float u = cbrtSigned(-qc);
                roots[count++] = 2.0 * u - ca / 3.0;
                roots[count++] = -u - ca / 3.0;
            }
        } else if (cubic_d < 0.0) {
            float phi = acos(-qc / sqrt(-cb_p)) / 3.0;
            float t = 2.0 * sqrt(-pc);
            roots[count++] = t * cos(phi) - ca / 3.0;
            roots[count++] = -t * cos(phi + PI / 3.0) - ca / 3.0;
            roots[count++] = -t * cos(phi - PI / 3.0) - ca / 3.0;
        } else {
            float sqrt_disc = sqrt(cubic_d);
            float u = cbrtSigned(sqrt_disc - qc);
            float v = -cbrtSigned(sqrt_disc + qc);
            roots[count++] = u + v - ca / 3.0;
        }
    } else if (abs(q) < EPS) {
        // q ≈ 0: biquadratic t^4 + p*t^2 + r = 0
        float disc = p * p - 4.0 * r;
        if (disc >= 0.0) {
            float sqrt_disc = sqrt(disc);
            float sq1 = (-p - sqrt_disc) * 0.5;
            float sq2 = (-p + sqrt_disc) * 0.5;
            if (sq1 >= 0.0) {
                float root = sqrt(sq1);
                roots[count++] = -root;
                roots[count++] = root;
            }
            if (sq2 >= 0.0 && sq2 > EPS) {
                float root = sqrt(sq2);
                roots[count++] = -root;
                roots[count++] = root;
            }
        }
    } else {
        // General case: resolvent cubic z^3 + (-p/2)*z^2 + (-r)*z + (r*p/2 - q^2/8) = 0
        // Coefficients for solve_cubic: [r*p/2 - q^2/8, -r, -p/2, 1]
        float ca = (-p / 2.0) / 1.0;
        float cb = (-r) / 1.0;
        float cc_c = (r * p / 2.0 - q * q / 8.0) / 1.0;
        float sq_ca = ca * ca;
        float pc = (1.0 / 3.0) * (-(1.0 / 3.0) * sq_ca + cb);
        float qc = (1.0 / 2.0) * ((2.0 / 27.0) * ca * sq_ca - (1.0 / 3.0) * ca * cb + cc_c);
        float cb_p = pc * pc * pc;
        float cubic_d = qc * qc + cb_p;

        float z;
        if (abs(cubic_d) < EPS) {
            if (abs(qc) < EPS) {
                z = -ca / 3.0;
            } else {
                float u = cbrtSigned(-qc);
                z = 2.0 * u - ca / 3.0;
            }
        } else if (cubic_d < 0.0) {
            float phi = acos(-qc / sqrt(-cb_p)) / 3.0;
            float t = 2.0 * sqrt(-pc);
            z = t * cos(phi) - ca / 3.0;
        } else {
            float sqrt_disc = sqrt(cubic_d);
            float u = cbrtSigned(sqrt_disc - qc);
            float v = -cbrtSigned(sqrt_disc + qc);
            z = u + v - ca / 3.0;
        }

        // Ferrari decomposition: need sqrt(z^2 - r) and sqrt(2*z - p)
        float z2r = z * z - r;
        float twozp = 2.0 * z - p;
        if (z2r < -EPS || twozp < -EPS) {
            return 0;
        }
        float u = sqrt(max(z2r, 0.0));
        float v = sqrt(max(twozp, 0.0));

        // Two quadratics: t^2 + sign*t + (z - u) = 0 and t^2 - sign*t + (z + u) = 0
        float sign = (q < 0.0) ? -v : v;

        // First quadratic
        float disc1 = sign * sign - 4.0 * (z - u);
        if (disc1 >= 0.0) {
            float sqrt_disc1 = sqrt(disc1);
            roots[count++] = (-sign - sqrt_disc1) * 0.5;
            roots[count++] = (-sign + sqrt_disc1) * 0.5;
        }

        // Second quadratic
        float sign2 = (q < 0.0) ? v : -v;
        float disc2 = sign2 * sign2 - 4.0 * (z + u);
        if (disc2 >= 0.0) {
            float sqrt_disc2 = sqrt(disc2);
            roots[count++] = (-sign2 - sqrt_disc2) * 0.5;
            roots[count++] = (-sign2 + sqrt_disc2) * 0.5;
        }
    }

    // Shift back: t -= a/4
    for (int i = 0; i < count; ++i) {
        roots[i] -= a / 4.0;
    }

    return count;
}

// Sphere-tracing fallback from the bounding-sphere entry point; a chord through the
// bounding sphere is at most its diameter, so marching stops there.
bool torusSphereTraceFallback(vec3 o, vec3 d, float rHat, out float t) {
    float tMax = 2.0 * (1.0 + rHat);
    t = 0.0;
    for (int i = 0; i < 48; ++i) {
        vec3 p = o + d * t;
        float sdf = length(vec2(length(p.xz) - 1.0, p.y)) - rHat;
        if (abs(sdf) < 1e-4) {
            if (t > 1e-6) return true;
        }
        if (t > tMax) return false;
        t += max(sdf, 1e-4);
    }
    return false;
}

// Intersect ray with torus. o is origin normalized by major radius, d is unit direction.
// Returns number of valid (t > 1e-6) ascending roots written to roots[].
// fallbackUsed is true if SDF sphere-tracing fallback was used instead of quartic roots.
int intersectTorus(vec3 o, vec3 d, float rHat, out float roots[4], out bool fallbackUsed) {
    fallbackUsed = false;
    // Bounding sphere early-out: sphere of radius (1 + rHat) centered at origin
    float o_mag_sq = dot(o, o);
    float bounding_radius = 1.0 + rHat;
    float oc = dot(o, d);
    float disc = oc * oc - (o_mag_sq - bounding_radius * bounding_radius);
    if (disc < 0.0 && o_mag_sq > bounding_radius * bounding_radius) {
        return 0;
    }

    // Origin re-basing: compute bounding sphere entry point tEnter and shift origin to o' = o + tEnter*d
    // This keeps |o'| <= 1 + rHat, so quartic coefficients are O(1) instead of O(|o|^4).
    float tEnter;
    if (o_mag_sq <= bounding_radius * bounding_radius) {
        // Camera inside sphere: tEnter = 0
        tEnter = 0.0;
    } else {
        // Camera outside sphere: use the near intersection
        tEnter = -oc - sqrt(disc);
    }
    vec3 oPrime = o + tEnter * d;

    // Compute quartic coefficients from ray-torus intersection using re-based origin o'
    float coeff_a = d.x * d.x + d.y * d.y + d.z * d.z;
    float coeff_b = 2.0 * (oPrime.x * d.x + oPrime.y * d.y + oPrime.z * d.z);
    float coeff_c = oPrime.x * oPrime.x + oPrime.y * oPrime.y + oPrime.z * oPrime.z;
    float coeff_d = coeff_c + 1.0 - rHat * rHat;

    float a4 = coeff_a * coeff_a;
    float a3 = 2.0 * coeff_a * coeff_b;
    float a2 = 2.0 * coeff_a * coeff_d + coeff_b * coeff_b - 4.0 * coeff_a + 4.0 * d.y * d.y;
    float a1 = 2.0 * coeff_b * coeff_d - 4.0 * coeff_b + 8.0 * oPrime.y * d.y;
    float a0 = coeff_d * coeff_d - 4.0 * coeff_c + 4.0 * oPrime.y * oPrime.y;

    // Solve quartic: a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0 = 0
    float c[5] = float[](a0, a1, a2, a3, a4);
    int count = solveQuartic(c, roots);

    // Newton-Raphson refinement using implicit function (3 iterations)
    // g(t) = (|p|^2 + 1 - rHat^2)^2 - 4*(p.x^2 + p.z^2), p = o' + t*d
    // g'(t) = grad(g)(p) . d
    for (int iter = 0; iter < 3; ++iter) {
        for (int i = 0; i < count; ++i) {
            float t = roots[i];
            vec3 p = oPrime + d * t;
            float magSq = dot(p, p);
            float rHat2 = rHat * rHat;
            float gVal = (magSq + 1.0 - rHat2) * (magSq + 1.0 - rHat2) - 4.0 * (p.x * p.x + p.z * p.z);
            float factor = 4.0 * (magSq + 1.0 - rHat2);
            float gx = factor * p.x - 8.0 * p.x;
            float gy = factor * p.y;
            float gz = factor * p.z - 8.0 * p.z;
            float gPrime = gx * d.x + gy * d.y + gz * d.z;
            if (abs(gPrime) < 1e-12) continue;
            float correction = gVal / gPrime;
            roots[i] -= correction;
        }
    }

    // Add tEnter back to all roots
    for (int i = 0; i < count; ++i) {
        roots[i] += tEnter;
    }

    // Filter: keep only t > 1e-6
    int validCount = 0;
    for (int i = 0; i < count; ++i) {
        if (roots[i] > 1e-6) {
            roots[validCount++] = roots[i];
        }
    }

    // Sphere-tracing fallback: if quartic found no valid roots but bounding sphere hit,
    // use SDF sphere tracing to catch grazing intersections the quartic misses.
    if (validCount == 0) {
        float t_out;
        if (torusSphereTraceFallback(oPrime, d, rHat, t_out)) {
            roots[0] = t_out + tEnter;
            validCount = 1;
            fallbackUsed = true;
        }
    }

    // Sort ascending (bubble sort, max 4 elements)
    for (int i = 1; i < validCount; ++i) {
        int j = i;
        while (j > 0 && roots[j - 1] > roots[j]) {
            float tmp = roots[j];
            roots[j] = roots[j - 1];
            roots[j - 1] = tmp;
            --j;
        }
    }

    return validCount;
}

// First exit of a ray starting inside the tube: sign change of the implicit function,
// bracketed and refined by bisection. Independent of the quartic discriminant, so grazing
// exits stay continuous. Returns 0 when no exit is found.
// The longest straight chord inside the tube runs tangent to the hole: 2*sqrt((1+r)^2-(1-r)^2) = 4*sqrt(r).
float torusExitFromInside(vec3 o, vec3 d, float rHat) {
    const int BRACKET_STEPS = 32;
    const int BISECT_STEPS = 12;
    float tMax = 4.0 * sqrt(rHat) + 1e-3;
    float tInside = 0.0;
    bool seenInside = torusImplicit(o, rHat) < 0.0;

    for (int i = 1; i <= BRACKET_STEPS; ++i) {
        float t = tMax * float(i) / float(BRACKET_STEPS);
        bool inside = torusImplicit(o + d * t, rHat) < 0.0;
        if (inside) {
            tInside = t;
            seenInside = true;
            continue;
        }
        if (!seenInside) {
            continue;
        }
        float tOutside = t;
        for (int k = 0; k < BISECT_STEPS; ++k) {
            float mid = 0.5 * (tInside + tOutside);
            if (torusImplicit(o + d * mid, rHat) < 0.0) {
                tInside = mid;
            } else {
                tOutside = mid;
            }
        }
        return 0.5 * (tInside + tOutside);
    }
    return 0.0;
}

// First entry of a ray starting outside the tube, by the same bracketing. Thin grazing
// crossings narrower than a bracket step are skipped on purpose. Returns 0 when none.
float torusEntryFromOutside(vec3 o, vec3 d, float rHat) {
    const int BRACKET_STEPS = 32;
    const int BISECT_STEPS = 12;
    float oc = dot(o, d);
    float boundingRadius = 1.0 + rHat;
    float disc = oc * oc - (dot(o, o) - boundingRadius * boundingRadius);
    if (disc < 0.0) {
        return 0.0;
    }
    float tStart = max(-oc - sqrt(disc), 0.0);
    float tEnd = -oc + sqrt(disc);
    if (tEnd <= tStart) {
        return 0.0;
    }

    float tOutside = tStart;
    for (int i = 1; i <= BRACKET_STEPS; ++i) {
        float t = mix(tStart, tEnd, float(i) / float(BRACKET_STEPS));
        if (torusImplicit(o + d * t, rHat) >= 0.0) {
            tOutside = t;
            continue;
        }
        float tInside = t;
        for (int k = 0; k < BISECT_STEPS; ++k) {
            float mid = 0.5 * (tOutside + tInside);
            if (torusImplicit(o + d * mid, rHat) < 0.0) {
                tInside = mid;
            } else {
                tOutside = mid;
            }
        }
        return 0.5 * (tOutside + tInside);
    }
    return 0.0;
}

// Grazing re-entries fade out instead of switching on the root count.
float torusReentryWeight(float cosTheta) {
    return smoothstep(0.0, 0.25, cosTheta);
}

#endif
