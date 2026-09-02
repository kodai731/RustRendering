
#extension GL_GOOGLE_include_directive : require

// flame_ray.glsl - ray reconstruction and emission segment integration for flame passes


// depth.glsl - Reverse-Z depth utilities
//
// This engine uses Reverse-Z depth mapping:
//   Near plane = 1.0, Far plane = 0.0
//   Depth comparison for "closer wins" = GREATER_OR_EQUAL
//
// Shader authors should use these constants and helpers instead of
// hardcoding depth values. This keeps reverse-Z knowledge in one place.


// Canonical depth constants (reverse-Z)
const float DEPTH_FAR  = 0.0;
const float DEPTH_NEAR = 1.0;

// Convert world position to clip-space depth value suitable for gl_FragDepth.
//
// Usage:
//   gl_FragDepth = worldToClipDepth(worldPos, view, proj);
float worldToClipDepth(vec3 worldPos, mat4 view, mat4 proj) {
    vec4 clipPos = proj * view * vec4(worldPos, 1.0);
    return clipPos.z / clipPos.w;
}

// Convert raw depth buffer value to linear eye-space distance.
//
// With reverse-Z the projection maps:
//   z_near -> 1.0,  z_far -> 0.0
// The relationship is:  rawDepth = (near / z) for infinite far plane,
// or more generally:     rawDepth = near * far / (far - z * (far - near))
//
// This function inverts that mapping.
//
// Usage:
//   float linearDist = linearizeDepth(gl_FragCoord.z, nearPlane, farPlane);
float linearizeDepth(float rawDepth, float nearPlane, float farPlane) {
    return nearPlane * farPlane / (farPlane - rawDepth * (farPlane - nearPlane));
}



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
    return (1.0 - g * g) / (4.0 * 3.141592653589793 * denom * sqrt(max(denom, 1e-6)));
}

float rteMidpointDistance(int index, int sampleCount, float pathLength) {
    return (float(index) + 0.5) * pathLength / float(sampleCount);
}

vec3 rteSingleScatterSample(vec3 sigmaS, vec3 sigmaT, float phase, float viewDistance, float ds, vec3 lightRadiance) {
    return sigmaS * phase * rteTransmittance(sigmaT, viewDistance) * lightRadiance * ds;
}



vec3 reconstructRayDirection(vec2 uv, mat4 invViewProj, vec3 cameraPos) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = invViewProj * vec4(ndc, DEPTH_NEAR, 1.0);
    return normalize(world.xyz / world.w - cameraPos);
}

float evaluateHeightAlongRay(float t, float hOrigin, float hDir) {
    return hOrigin + t * hDir;
}


struct FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
};


#ifndef WATER_UBO_SET
#define WATER_UBO_SET 1
#define WATER_UBO_BINDING 0
#endif

struct WaterUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 radii;
    vec4 absorption;
    vec4 flow;
    vec4 composite;
    vec4 tint;
    vec4 lighting;
    vec4 scattering;
    vec4 temporal;
    vec4 waveModes[16];
    mat4 invViewProj;
    vec4 lbModes[20];
};



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
        // r ≈ 0: roots are 0 and the cubic q*t^3 + p*t^2 + t = 0
        // Solve cubic [q, p, 0, 1]
        float ca = p / q;
        float cb = 0.0 / q;
        float cc_c = 1.0 / q;
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
            roots[count++] = -t * cos(phi + 3.141592653589793 / 3.0) - ca / 3.0;
            roots[count++] = -t * cos(phi - 3.141592653589793 / 3.0) - ca / 3.0;
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

// Sphere-tracing fallback: SDF of torus in normalized coordinates.
// Returns true if hit found within max steps (t > 1e-6), false otherwise.
bool torusSphereTraceFallback(vec3 o, vec3 d, float rHat, out float t) {
    t = 0.0;
    for (int i = 0; i < 48; ++i) {
        vec3 p = o + d * t;
        float sdf = length(vec2(length(p.xz) - 1.0, p.y)) - rHat;
        if (abs(sdf) < 1e-4) {
            if (t > 1e-6) return true;
        }
        if (t > 4.0) return false;
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
        if (torusSphereTraceFallback(o, d, rHat, t_out)) {
            roots[0] = t_out;
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
// bracketed at tube-diameter scale and refined by bisection. Independent of the quartic
// discriminant, so grazing exits stay continuous. Returns 0 when no exit is found.
float torusExitFromInside(vec3 o, vec3 d, float rHat) {
    const int BRACKET_STEPS = 16;
    const int BISECT_STEPS = 12;
    float tMax = 2.0 * rHat + 1e-3;
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



vec2 torusUV(vec3 pLocalNormalized) {
    return vec2(atan(pLocalNormalized.z, pLocalNormalized.x),
                atan(pLocalNormalized.y, length(pLocalNormalized.xz) - 1.0));
}

vec2 advectUV(vec2 uv, vec2 flowRate, float time) {
    return uv + flowRate * time;
}



float sinc(float x) {
    if (abs(x) < 1e-6) return 1.0;
    return sin(x) / x;
}

void waterHeightAndGradient(vec2 uv, float time, vec2 flowRate, int modeCount, vec2 footprint, out float h, out float hu, out float hv, out float slopeVariance) {
    h = 0.0;
    hu = 0.0;
    hv = 0.0;
    slopeVariance = 0.0;

    float u = uv.x;
    float v = uv.y;
    float a = flowRate.x;
    float b = flowRate.y;
    float rHat = water.radii.y / water.radii.x;
    float rho = 1.0 + rHat * cos(v);

    for (int k = 0; k < 8; k++) {
        if (k >= modeCount) break;

        int m = int(water.waveModes[k * 2].x);
        int n = int(water.waveModes[k * 2].y);
        float amp = water.waveModes[k * 2].z;
        float omega = water.waveModes[k * 2].w;
        float phase = water.waveModes[k * 2 + 1].x;
        float ampN = amp / water.radii.x;

        float phasePrime = m * (u + a * time) + n * (v + b * time) - omega * time + phase;
        float cosVal = cos(phasePrime);
        float sinVal = sin(phasePrime);

        float sincM = sinc(m * footprint.x);
        float sincN = sinc(n * footprint.y);
        float sincProduct = sincM * sincN;

        h += amp * cosVal * sincProduct;
        hu -= amp * m * sinVal * sincProduct;
        hv -= amp * n * sinVal * sincProduct;

        slopeVariance += ampN * ampN * (m * m / (rho * rho) + n * n / (rHat * rHat)) * (1.0 - sincProduct * sincProduct) * 0.5;
    }
}

vec3 waterPerturbedNormal(float u, float v, float h, float hu, float hv, float rHat) {
    float cosU = cos(u);
    float sinU = sin(u);
    float cosV = cos(v);
    float sinV = sin(v);

    vec3 e_u = vec3(-sinU, 0.0, cosU);
    vec3 e_v = vec3(-sinV * cosU, cosV, -sinV * sinU);
    vec3 n = vec3(cosV * cosU, sinV, cosV * sinU);

    float kappa1 = 1.0 / rHat;
    float kappa2 = cosV / (1.0 + rHat * cosV);

    float scaledH = h / water.radii.x;
    float scaledHu = hu / water.radii.x;
    float scaledHv = hv / water.radii.x;

    vec3 nPrime = (1.0 + scaledH * kappa1) * (1.0 + scaledH * kappa2) * n
        - (1.0 + scaledH * kappa1) * scaledHu / (1.0 + rHat * cosV) * e_u
        - (1.0 + scaledH * kappa2) * scaledHv / rHat * e_v;

    return normalize(nPrime);
}



// chebyshev.glsl - Clenshaw evaluation of Chebyshev series (fully unrolled)
//
// Coefficient layout matches thyllore-math-core pack_coefficients_vec4:
//   c0 = [C0..C3], c1 = [C4..C7], c2 = [C8..C11]
// Series must be fit over domain [0,1]; x01 is normalized to [-1,1] internally.


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



const int LB_MODE_COUNT = 4;
const int LB_SLOTS_PER_MODE = 5;

float waterLbCheb(vec4 lo, vec4 hi, float t) {
    return evaluateChebyshev8(lo, hi, 0.5 * t + 0.5);
}

void waterLbHeightAndGradient(vec2 uv, float time, vec2 flowRate, inout float h, inout float hu, inout float hv) {
    for (int k = 0; k < LB_MODE_COUNT; ++k) {
        int slot = LB_SLOTS_PER_MODE * k;
        vec4 head = water.lbModes[slot];
        float m = head.x;
        float omega = head.y;
        float amplitude = head.z;
        float phase = head.w;

        if (amplitude <= 0.0) {
            continue;
        }

        float phasePrime = m * (uv.x + flowRate.x * time) - omega * time + phase;
        float vAdvected = mod(uv.y + flowRate.y * time, 6.28318530718);
        float t = (vAdvected - 3.14159265359) / 3.14159265359;

        float phi = waterLbCheb(water.lbModes[slot + 1], water.lbModes[slot + 2], t);
        float dphi = waterLbCheb(water.lbModes[slot + 3], water.lbModes[slot + 4], t);

        h += amplitude * cos(phasePrime) * phi;
        hu += -amplitude * m * sin(phasePrime) * phi;
        hv += amplitude * cos(phasePrime) * dphi;
    }
}




#define WATER_SCATTER_SAMPLES 4

float waterFresnelReflectance(float cosThetaI, float eta) {
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));
    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return (rPar * rPar + rPerp * rPerp) * 0.5;
}

vec3 waterScatteringCoefficient() {
    return water.tint.rgb * water.lighting.w;
}

vec3 waterExtinctionCoefficient() {
    return water.absorption.rgb + waterScatteringCoefficient();
}

vec3 waterEnvironmentReflection(vec3 reflDir, vec3 lightDir, vec3 lightColor, float slopeVariance) {
    float sharpness = water.lighting.y / (1.0 + water.lighting.y * slopeVariance);
    float spec = pow(max(dot(reflDir, lightDir), 0.0), sharpness);
    return vec3(0.6, 0.7, 0.8) * water.lighting.z + lightColor * water.lighting.x * spec;
}

vec3 waterTransmittedHighlight(vec3 exitDir, vec3 lightDir, vec3 lightColor, float chord) {
    float spec = pow(max(dot(exitDir, lightDir), 0.0), water.lighting.y);
    return lightColor * water.lighting.x * spec * rteTransmittance(waterExtinctionCoefficient(), chord);
}

struct WaterScatterSample {
    vec3 position;
    float viewDistance;
    vec3 lightDir;
    vec3 lightExitPoint;
    float waterDistance;
    float surfaceTransmission;
};

// Light path from an interior point to the light: straight line, first exit through the surface,
// plus the ring chord if the line re-enters the torus (self-occlusion); Snell bending is ignored.
WaterScatterSample waterScatterSampleAt(vec3 entry, vec3 exit, int index, vec3 lightPos) {
    WaterScatterSample smp;
    float pathLength = length(exit - entry);
    smp.viewDistance = rteMidpointDistance(index, WATER_SCATTER_SAMPLES, pathLength);
    smp.position = entry + (exit - entry) * (smp.viewDistance / pathLength);
    smp.lightDir = normalize(lightPos - smp.position);

    float rHat = water.radii.y / water.radii.x;
    vec3 originLocal = (water.inverseModel * vec4(smp.position, 1.0)).xyz / water.radii.x;
    vec3 dirLocal = normalize((water.inverseModel * vec4(smp.lightDir, 0.0)).xyz);
    float firstExit = torusExitFromInside(originLocal, dirLocal, rHat);
    vec3 exitLocal = originLocal + dirLocal * firstExit;
    float lastExit = firstExit;
    float insideDistance = firstExit;

    float ringEntry = torusEntryFromOutside(exitLocal + dirLocal * 1e-3, dirLocal, rHat);
    if (ringEntry > 0.0) {
        float ringStart = firstExit + 1e-3 + ringEntry;
        float ringChord = torusExitFromInside(originLocal + dirLocal * (ringStart + 1e-3), dirLocal, rHat);
        insideDistance += ringChord;
        lastExit = ringStart + 1e-3 + ringChord;
    }
    smp.waterDistance = insideDistance * water.radii.x;
    smp.lightExitPoint = smp.position + smp.lightDir * (lastExit * water.radii.x);

    vec3 nExit = normalize(mat3(water.model) * torusGradient(exitLocal, rHat));
    smp.surfaceTransmission = 1.0 - waterFresnelReflectance(max(dot(nExit, smp.lightDir), 0.0), water.absorption.w);
    return smp;
}

vec3 waterScatterSampleRadiance(WaterScatterSample smp, vec3 viewDir, vec3 lightColor, float ds) {
    vec3 sigmaS = waterScatteringCoefficient();
    vec3 sigmaT = waterExtinctionCoefficient();
    vec3 lightRadiance = lightColor * water.lighting.x * smp.surfaceTransmission * rteTransmittance(sigmaT, smp.waterDistance);
    float phase = rteHenyeyGreenstein(dot(smp.lightDir, viewDir), water.scattering.x);
    return rteSingleScatterSample(sigmaS, sigmaT, phase, smp.viewDistance, ds, lightRadiance);
}







// Second-bounce misses read the scene color through a wide box filter so thin screen
// features (grid lines) cannot alias into combs after two refractions.
vec3 sampleSceneColorBlurred(vec2 uv) {
    vec2 texel = 6.0 / vec2(textureSize(sceneColorSampler, 0));
    vec3 sum = vec3(0.0);
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            sum += water.tint.rgb;
        }
    }
    return sum / 25.0;
}

void main() {
   mat4 invViewProj = water.invViewProj;
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    // Transform to local space: origin w=1, dir w=0
    vec3 pLocalOrigin = (water.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    pLocalOrigin /= water.radii.x;
    vec3 dLocal = (water.inverseModel * vec4(rayDir, 0.0)).xyz;
    dLocal = normalize(dLocal);

    // Intersect ray with torus
    float roots[4];
    bool fallbackUsed;
    int hitCount = intersectTorus(pLocalOrigin, dLocal, water.radii.y / water.radii.x, roots, fallbackUsed);

    if (hitCount == 0) {
        discard;
    }

    // First hit time in world units
    float t1 = roots[0] * water.radii.x;
    vec3 p1 = frame.camera_pos.xyz + t1 * rayDir;
    float waterDepth = worldToClipDepth(p1, frame.view, frame.proj);
    gl_FragDepth = waterDepth;

    // Debug view: color by root count
    if (push.debugView == 1) {
        if (hitCount == 2) {
            outColor = vec4(0.0, 1.0, 0.0, 1.0);
        } else if (hitCount == 4) {
            outColor = vec4(0.0, 0.0, 1.0, 1.0);
        } else {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        return;
   }


   // Debug view: torus intersection probe (nearest root, high-precision encoding)
    if (push.debugView == 3 || push.debugView == 4) {
        float t = (push.debugView == 3) ? roots[0] * water.radii.x : roots[1] * water.radii.x;
        float hi = floor(t);
        float mid = floor(fract(t) * 1024.0);
        float lo = fract(t * 1024.0);
        float marker = -(float(hitCount) + (fallbackUsed ? 10.0 : 0.0));
        outColor = vec4(hi, mid, lo, marker);
        return;
    }

    float chord = 0.0;
    if (hitCount >= 2) {
        chord = (roots[1] - roots[0]) * water.radii.x;
    }
    if (hitCount >= 4) {
        chord += (roots[3] - roots[2]) * water.radii.x;
    }

    // Surface normal at first hit via analytic wave gradient
    vec3 pLocal1 = pLocalOrigin + roots[0] * dLocal;
    float rHat = water.radii.y / water.radii.x;
    vec2 uv = torusUV(pLocal1);

    float du_dx = dFdx(uv.x);
    float du_dy = dFdy(uv.x);
    float dv_dx = dFdx(uv.y);
    float dv_dy = dFdy(uv.y);
    vec2 footprint = vec2(length(vec2(du_dx, du_dy)), length(vec2(dv_dx, dv_dy)));

    if (abs(du_dx) > 3.0) {
        footprint.x = 0.0;
    }
    if (any(isnan(footprint)) || any(isinf(footprint))) { footprint = vec2(0.0); }

    float h, hu, hv, var;
    waterHeightAndGradient(uv, water.flow.z, water.flow.xy, int(water.composite.z), footprint, h, hu, hv, var);
    waterLbHeightAndGradient(uv, water.flow.z, water.flow.xy, h, hu, hv);
    vec3 nLocal = waterPerturbedNormal(uv.x, uv.y, h, hu, hv, rHat);
    vec3 n = normalize(mat3(water.model) * nLocal);

    // Debug view: normal visualization
   if (push.debugView == 2) {
        outColor = vec4(n * 0.5 + 0.5, 1.0);
        return;
    }

    // Fresnel: Aqoole Reflectance P/S (average of parallel and perpendicular)
    float eta = water.absorption.w;
    float cosThetaI = -dot(rayDir, n);
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));

    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    float F = (rPar * rPar + rPerp * rPerp) * 0.5;

  // Reflection
    vec3 reflDir = reflect(rayDir, n);
    vec3 reflection;
    {
        vec3 lightDir = normalize(frame.light_pos.xyz - p1);
        reflection = waterEnvironmentReflection(reflDir, lightDir, frame.light_color.rgb, var);
    }


    // Transmission: exit-point refraction
    vec3 dRefr = refract(dLocal, nLocal, 1.0 / eta);
    if (length(dRefr) < 1e-4) {
        dRefr = reflect(dLocal, nLocal);
    }

    float tExit = torusExitFromInside(pLocal1 + dRefr * 1e-3, dRefr, rHat);
    vec3 pExitLocal = (tExit > 0.0) ? pLocal1 + dRefr * (1e-3 + tExit) : pLocal1;

    vec3 nExit = normalize(torusGradient(pExitLocal, rHat));
    vec3 dExitLocal = refract(dRefr, -nExit, eta);
    if (length(dExitLocal) < 1e-4) {
        dRefr = reflect(dRefr, nExit);
        float tReExit = torusExitFromInside(pExitLocal + dRefr * 1e-3, dRefr, rHat);
        if (tReExit > 0.0) {
            pExitLocal = pExitLocal + dRefr * (1e-3 + tReExit);
        }
        nExit = normalize(torusGradient(pExitLocal, rHat));
        dExitLocal = refract(dRefr, -nExit, eta);
        if (length(dExitLocal) < 1e-4) {
            dExitLocal = dRefr;
        }
    }
    vec3 dExit = normalize(mat3(water.model) * dExitLocal);

    vec4 pExitWorld = water.model * vec4(pExitLocal * water.radii.x, 1.0);
    vec3 background;
    float tBackground = 1e30;
    {
        // ScreenSpace path: project exit point to screen space
        vec4 clip = frame.proj * frame.view * pExitWorld;
        if (clip.w > 0) {
            vec2 uvExit = clamp((clip.xy / clip.w) * 0.5 + 0.5, 0.0, 1.0);
            background = water.tint.rgb;
        } else {
            background = water.tint.rgb;
        }
    }


    vec3 transmission = mix(background, water.tint.rgb, clamp(water.tint.a, 0.0, 1.0)) * rteTransmittance(waterExtinctionCoefficient(), chord);
    vec3 lightDirExit = normalize(frame.light_pos.xyz - pExitWorld.xyz);
    transmission += waterTransmittedHighlight(dExit, lightDirExit, frame.light_color.rgb, chord);

    float scatterPath = length(pExitWorld.xyz - p1);
    if (scatterPath > 1e-6) {
        vec3 viewDirWater = (pExitWorld.xyz - p1) / scatterPath;
        float ds = scatterPath / float(WATER_SCATTER_SAMPLES);
        for (int i = 0; i < WATER_SCATTER_SAMPLES; ++i) {
            WaterScatterSample smp = waterScatterSampleAt(p1, pExitWorld.xyz, i, frame.light_pos.xyz);
            transmission += waterScatterSampleRadiance(smp, viewDirWater, frame.light_color.rgb, ds);
        }
    }

  // Composite output
   outColor = vec4(F * reflection * water.composite.x + (1.0 - F) * transmission * water.composite.y, 1.0);


}

