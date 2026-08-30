
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



vec3 reconstructRayDirection(vec2 uv, mat4 invViewProj, vec3 cameraPos) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = invViewProj * vec4(ndc, DEPTH_NEAR, 1.0);
    return normalize(world.xyz / world.w - cameraPos);
}

// S * (1 - exp(-sigma*dt)) / sigma with Taylor fallback so sigma -> 0 stays continuous.
// Mirrored in thyllore-render-core/src/flame.rs (integrate_emission_segment) for tests.
float integrateEmissionSegment(float source, float sigmaT, float dt) {
    float x = sigmaT * dt;
    if (x < 1e-3) {
        return source * dt * (1.0 - 0.5 * x + x * x * (1.0 / 6.0));
    }
    return source * (1.0 - exp(-x)) / sigmaT;
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


struct WaterUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 radii;
    vec4 absorption;
    vec4 flow;
    vec4 composite;
    vec4 tint;
    vec4 temporal;
    vec4 waveModes[16];
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
int intersectTorus(vec3 o, vec3 d, float rHat, out float roots[4]) {
    // Bounding sphere early-out: sphere of radius (1 + rHat) centered at origin
    float o_mag_sq = dot(o, o);
    float bounding_radius = 1.0 + rHat;
    float oc = dot(o, d);
    float disc = oc * oc - (o_mag_sq - bounding_radius * bounding_radius);
    if (disc < 0.0 && o_mag_sq > bounding_radius * bounding_radius) {
        return 0;
    }

    // Compute quartic coefficients from ray-torus intersection
    float coeff_a = d.x * d.x + d.y * d.y + d.z * d.z;
    float coeff_b = 2.0 * (o.x * d.x + o.y * d.y + o.z * d.z);
    float coeff_c = o.x * o.x + o.y * o.y + o.z * o.z;
    float coeff_d = coeff_c + 1.0 - rHat * rHat;

    float a4 = coeff_a * coeff_a;
    float a3 = 2.0 * coeff_a * coeff_b;
    float a2 = 2.0 * coeff_a * coeff_d + coeff_b * coeff_b - 4.0 * coeff_a + 4.0 * d.y * d.y;
    float a1 = 2.0 * coeff_b * coeff_d - 4.0 * coeff_b + 8.0 * o.y * d.y;
    float a0 = coeff_d * coeff_d - 4.0 * coeff_c + 4.0 * o.y * o.y;

    // Solve quartic: a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0 = 0
    float c[5] = float[](a0, a1, a2, a3, a4);
    int count = solveQuartic(c, roots);

    // Newton-Raphson refinement (2 iterations)
    for (int iter = 0; iter < 2; ++iter) {
        for (int i = 0; i < count; ++i) {
            float t = roots[i];
            float f = a0 + t * (a1 + t * (a2 + t * (a3 + t * a4)));
            float df = a1 + t * (2.0 * a2 + t * (3.0 * a3 + t * 4.0 * a4));
            float correction = f / (df + 1e-30);
            roots[i] -= correction;
        }
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







void main() {
    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    // Transform to local space: origin w=1, dir w=0
    vec3 pLocalOrigin = (water.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    pLocalOrigin /= water.radii.x;
    vec3 dLocal = (water.inverseModel * vec4(rayDir, 0.0)).xyz;
    dLocal = normalize(dLocal);

    // Intersect ray with torus
    float roots[4];
    int hitCount = intersectTorus(pLocalOrigin, dLocal, water.radii.y / water.radii.x, roots);

    if (hitCount == 0) {
        discard;
    }

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

    // First hit time in world units
    float t1 = roots[0] * water.radii.x;
    vec3 p1 = frame.camera_pos.xyz + t1 * rayDir;

    float waterDepth = worldToClipDepth(p1, frame.view, frame.proj);

    // Compute chord length in world units
    float chord;
    if (hitCount >= 4) {
        chord = (roots[1] - roots[0]) * water.radii.x + (roots[3] - roots[2]) * water.radii.x;
    } else {
        chord = (roots[1] - roots[0]) * water.radii.x;
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

    float h, hu, hv, var;
    waterHeightAndGradient(uv, water.flow.z, water.flow.xy, int(water.composite.z), footprint, h, hu, hv, var);

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

    // Reflection: constant environment + specular highlight
    vec3 reflDir = reflect(rayDir, n);
    vec3 lightDir = normalize(frame.light_pos.xyz - p1);
    float spec = pow(max(dot(reflDir, lightDir), 0.0), 64.0 / (1.0 + 64.0 * var));
    vec3 reflection = vec3(0.6, 0.7, 0.8) + frame.light_color.rgb * spec;

    // Transmission: exit-point refraction
    vec3 dRefr = refract(dLocal, nLocal, 1.0 / eta);
    if (length(dRefr) < 1e-4) {
        dRefr = reflect(dLocal, nLocal);
    }

    float exitRoots[4];
    int exitCount = intersectTorus(pLocal1 + dRefr * 1e-3, dRefr, rHat, exitRoots);
    vec3 pExitLocal;
    if (exitCount > 0) {
        pExitLocal = pLocal1 + dRefr * (1e-3 + exitRoots[0]);
    } else {
        pExitLocal = pLocal1;
    }

    // Secondary TIR check at exit point
    vec3 nExit = normalize(torusGradient(pExitLocal, rHat));
    vec3 dExit = refract(dRefr, -nExit, eta);
    if (length(dExit) < 1e-4) {
        dRefr = reflect(dRefr, nExit);
        float reRoots[4];
        int reCount = intersectTorus(pLocal1 + dRefr * 1e-3, dRefr, rHat, reRoots);
        if (reCount > 0) {
            pExitLocal = pLocal1 + dRefr * (1e-3 + reRoots[0]);
        }
    }

    vec4 pExitWorld = water.model * vec4(pExitLocal * water.radii.x, 1.0);
    vec4 clip = frame.proj * frame.view * pExitWorld;
    vec3 background;
    if (clip.w > 0) {
        vec2 uvExit = clamp((clip.xy / clip.w) * 0.5 + 0.5, 0.0, 1.0);
        background = water.tint.rgb;
    } else {
        background = water.tint.rgb;
    }

    vec3 transmission = mix(background, water.tint.rgb, clamp(water.tint.a, 0.0, 1.0)) * exp(-water.absorption.rgb * chord);

    // Composite output
    outColor = vec4(F * reflection * water.composite.x + (1.0 - F) * transmission * water.composite.y, 1.0);
    gl_FragDepth = waterDepth;
}

