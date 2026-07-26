#version 450

// F2 shading pass. push.mode swaps HOW emission is integrated and nothing else:
// ray/interval decoding, camera-inside correction, color ramp and alpha live in
// the shared FlameRaySegment path, so analytic vs raymarch comparisons isolate
// the integration method alone regardless of mesh, colors, or future noise.
// push.mode: 0 = analytic boundary integral, 1 = reference raymarch,
// 2 = delta-t debug view, 3 = styled raymarch (IGN jitter + noise erosion).

#include "include/chebyshev.glsl"
#include "include/flame_ray.glsl"
#include "include/flame_noise.glsl"

layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_pos;
    vec4 light_color;
} frame;

layout(set = 1, binding = 0) uniform FlameUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 heightPrimitiveCoefficients[3];
    vec4 radialCoefficients[2];
    vec4 heightCoefficients[2];
    float time;
    float sigmaT;
    float intensity;
    float heightAxisScale;
    float noiseAmplitude;
    float noiseFrequency;
    float noiseScrollSpeed;
    float radialSharpness;
    vec4 colorBase;
    vec4 colorMid;
    vec4 colorTip;
    vec4 temporalData;
    vec4 lightData;
    vec4 styleParams0;
    vec4 styleParams1;
    vec4 styleParams2;
    mat4 trailUnitInverse;
    vec4 trailMeta;
    vec4 trailSamples[16];
    vec4 emitterParams;
} flame;

layout(set = 1, binding = 2) uniform sampler2D flameAccumSampler;
layout(set = 1, binding = 3) uniform sampler2D flameIntervalSampler;
layout(set = 1, binding = 4) uniform sampler2D flameHistorySampler;
layout(set = 1, binding = 5) uniform sampler2D flameSdfSampler;
layout(set = 1, binding = 6) uniform sampler2D sceneDepthSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

layout(push_constant) uniform FlamePush {
    int mode;
    int stepCount;
} push;

const float INTERVAL_CLEAR_THRESHOLD = 1e37;
const float H_DIR_EPSILON = 1e-4;
const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

float evaluateHeightFalloff(float height01) {
    return evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1], height01);
}

#include "include/flame_noise_field.glsl"

float evaluateHeightPrimitive(float height01) {
    return evaluateChebyshev12(
        flame.heightPrimitiveCoefficients[0],
        flame.heightPrimitiveCoefficients[1],
        flame.heightPrimitiveCoefficients[2],
        height01);
}

// Self-shadow optical depth: layered concentric cylinders (3 layers) with Chebyshev density.
float computeSelfShadowTau(vec3 p, vec3 l) {
    // Layer radii S = [1/3, 2/3, 1], midpoints m = [1/6, 0.5, 5/6]
    float s[3] = float[](1.0/3.0, 2.0/3.0, 1.0);
    float m[3] = float[](1.0/6.0, 0.5, 5.0/6.0);

    // Evaluate density at each layer midpoint using Chebyshev coefficients
    float dens[4];
    for (int k = 0; k < 3; ++k) {
        dens[k] = evaluateChebyshev8(flame.radialCoefficients[0], flame.radialCoefficients[1], m[k]);
    }
    dens[3] = 0.0;

    // Compute weights w_k = dens_k - dens_{k+1}
    float w[3];
    for (int k = 0; k < 3; ++k) {
        w[k] = dens[k] - dens[k + 1];
    }

    float px = p.x, py = p.y, pz = p.z;
    float lx = l.x, ly = l.y, lz = l.z;

    float total = 0.0;

    for (int k = 0; k < 3; ++k) {
        float sk = s[k];
        float a = lx * lx + lz * lz;

        // Find intersection of cylinder (x^2 + z^2 = S_k^2) and ray p + s*L
        float s0, s1;
        if (a < 1e-6) {
            // Ray is parallel to cylinder axis
            if (px * px + pz * pz <= sk * sk) {
                s0 = 0.0;
                s1 = 1e4;
            } else {
                continue;
            }
        } else {
            // Solve quadratic: a*s^2 + 2*(px*lx + pz*lz)*s + (px^2 + pz^2 - sk^2) = 0
            float b = 2.0 * (px * lx + pz * lz);
            float c = px * px + pz * pz - sk * sk;
            float disc = b * b - 4.0 * a * c;

            if (disc <= 0.0) {
                continue;
            }

            float sqrt_disc = sqrt(disc);
            s0 = (-b - sqrt_disc) / (2.0 * a);
            s1 = (-b + sqrt_disc) / (2.0 * a);

            // Clip to s >= 0
            if (s1 < 0.0) {
                continue;
            }
            if (s0 < 0.0) {
                s0 = 0.0;
            }
        }

        // Clip interval by height h(s) = p.y + s*L.y in [0, 1]
        float lo = s0;
        float hi = s1;

        if (abs(ly) < 1e-4) {
            // h is approximately constant
            if (py < 0.0 || py > 1.0) {
                continue;
            }
            // F is coefficients.height evaluated at p.y
            float f_val = evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1], py);
            total += w[k] * f_val * (hi - lo);
        } else {
            // h(s) = py + s*ly, find where h in [0, 1]
            float s_lo = (0.0 - py) / ly;
            float s_hi = (1.0 - py) / ly;

            if (s_lo > s_hi) {
                float tmp = s_lo;
                s_lo = s_hi;
                s_hi = tmp;
            }

            // Clip [lo, hi] by [s_lo, s_hi]
            lo = max(lo, s_lo);
            hi = min(hi, s_hi);

            if (lo >= hi) {
                continue;
            }

            // I_k = (H1(h(s1)) - H1(h(s0))) / L.y
            float h_s0 = py + lo * ly;
            float h_s1 = py + hi * ly;

            float h1_s0 = evaluateHeightPrimitive(h_s0);
            float h1_s1 = evaluateHeightPrimitive(h_s1);

            total += w[k] * (h1_s1 - h1_s0) / ly;
        }
    }

    return max(0.0, flame.sigmaT * total);
}

struct FlameRaySegment {
    float tNear;
    float tFar;
    vec3 localOrigin;
    vec3 localDir;
    float boundaryHeightIntegral;
};

bool clampToShellCone(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    // Shell cone: |p.xz| <= R0 * mix(1.0, TAPER, p.y), R0 = 0.5, TAPER = 0.25 (must match flameShellGeometry.geom TAPER_TIP_SCALE)
    const float CONE_R0 = 0.5;
    const float CONE_TAPER = 0.25;
    // f(t) = m + n*t, m = CONE_R0 * (1.0 - (1.0-CONE_TAPER) * o.y), n = -CONE_R0 * (1.0-CONE_TAPER) * d.y
    // Condition |o.xz + t*d.xz|^2 - f(t)^2 <= 0 -> a = dot(d.xz,d.xz) - n*n, b = 2.0*(dot(o.xz,d.xz) - m*n), c = dot(o.xz,o.xz) - m*m
    float m = CONE_R0 * (1.0 - (1.0 - CONE_TAPER) * o.y);
    float n = -CONE_R0 * (1.0 - CONE_TAPER) * d.y;
    float a = dot(d.xz, d.xz) - n * n;
    float b = 2.0 * (dot(o.xz, d.xz) - m * n);
    float c = dot(o.xz, o.xz) - m * m;

    if (abs(a) < 1e-6) {
        // Linear case: b*t + c <= 0
        if (abs(b) < 1e-6) {
            // Constant: either always inside (c <= 0) or never (c > 0)
            if (c > 0.0) return false;
        } else {
            float tRoot = -c / b;
            if (b > 0.0) {
                // b*t + c <= 0 -> t <= tRoot
                if (tFar > tRoot) tFar = tRoot;
            } else {
                // b < 0: b*t + c <= 0 -> t >= tRoot
                if (tNear < tRoot) tNear = tRoot;
            }
        }
    } else {
        float disc = b * b - 4.0 * a * c;
        if (disc < 0.0) {
            // No real roots: if a > 0, quadratic is always positive -> empty
            // If a < 0, quadratic is always negative -> conservative (inside everywhere)
            if (a > 0.0) return false;
            // a < 0: conservative, no clamping needed
        } else {
            float sqrtDisc = sqrt(disc);
            float tCone0 = (-b - sqrtDisc) / (2.0 * a);
            float tCone1 = (-b + sqrtDisc) / (2.0 * a);
            if (tCone0 > tCone1) { float tmp = tCone0; tCone0 = tCone1; tCone1 = tmp; }
            if (a > 0.0) {
                // Quadratic opens upward: interval [tCone0, tCone1] is inside
                if (tNear < tCone0) tNear = tCone0;
                if (tFar > tCone1) tFar = tCone1;
            } else {
                // a < 0: conservative, no clamping (outside intervals would be (-inf,t0] U [t1,inf))
            }
        }
    }

    // Y-slab: 0 <= y <= 1
    if (abs(d.y) < 1e-6) {
        // Horizontal ray: check if origin is within slab
        if (o.y < 0.0 || o.y > 1.0) return false;
    } else {
        float tY0 = -o.y / d.y;
        float tY1 = (1.0 - o.y) / d.y;
        if (tY0 > tY1) { float tmp = tY0; tY0 = tY1; tY1 = tmp; }
        // Clamp to y-slab interval
        if (tNear < tY0) tNear = tY0;
        if (tFar > tY1) tFar = tY1;
    }

    return tNear <= tFar;
}

FlameRaySegment buildRaySegment(float coverage, float heightIntegral, vec2 interval) {
    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    FlameRaySegment segment;
    segment.localOrigin = (flame.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    segment.localDir = (flame.inverseModel * vec4(rayDir, 0.0)).xyz;
    segment.tNear = interval.x > INTERVAL_CLEAR_THRESHOLD ? 0.0 : interval.x;
    float tFar = interval.y > INTERVAL_CLEAR_THRESHOLD ? segment.tNear : -interval.y;
    segment.tFar = max(tFar, segment.tNear);
    segment.boundaryHeightIntegral = heightIntegral;

    // Geometric camera-inside test: local origin must be within the shell cone
    bool cameraInside = segment.localOrigin.y >= 0.0 && segment.localOrigin.y <= 1.0
        && length(segment.localOrigin.xz) <= 0.5 * mix(1.0, 0.25, segment.localOrigin.y);

    // Camera inside the shell: front boundary terms at t = 0 were never
    // rasterized, so coverage = +N and the missing (-1) * H1(h_o) / h_d
    // terms are restored here (their t and delta-t contributions are zero).
    if (coverage > 0.5 && cameraInside && abs(segment.localDir.y) > H_DIR_EPSILON) {
        float missingCount = round(coverage);
        segment.boundaryHeightIntegral -= missingCount
            * evaluateHeightPrimitive(clamp(segment.localOrigin.y, 0.0, 1.0)) / segment.localDir.y;
        segment.tNear = 0.0;
    }

    // Clamp to shell cone for non-trail domains (cylinder emitter only, bend off)
    if (flame.emitterParams.x < 0.5 && flame.trailMeta.x < 1.0 && abs(flame.styleParams2.z) < 1e-4) {
        // For unpaired coverage pixels (abs(round(coverage)) > 0.5 && !cameraInside), the interval
        // buffer is degenerate (tNear == tFar). Reset to wide interval before cone clamping so
        // clampToShellCone correctly finds the intersection with the shell cone/y-slab.
        if (abs(round(coverage)) > 0.5 && !cameraInside) {
            segment.tNear = 0.0;
            segment.tFar = 1e4;
        }
        if (!clampToShellCone(segment.localOrigin, segment.localDir, segment.tNear, segment.tFar)) {
            // Empty interval: return with tNear > tFar so caller can detect it
            segment.tNear = 1.0;
            segment.tFar = 0.0;
        } else if (abs(segment.localDir.y) > H_DIR_EPSILON) {
            // Recalculate boundaryHeightIntegral using analytical form for clamped interval
            vec3 o = segment.localOrigin;
            vec3 d = segment.localDir;
            segment.boundaryHeightIntegral = (evaluateHeightPrimitive(clamp(o.y + segment.tFar * d.y, 0.0, 1.0))
                - evaluateHeightPrimitive(clamp(o.y + segment.tNear * d.y, 0.0, 1.0))) / segment.localDir.y;
        } else if (abs(round(coverage)) > 0.5 && !cameraInside) {
            // Camera outside but coverage non-zero (boundary artifact): use mid-point rule
            vec3 o = segment.localOrigin;
            vec3 d = segment.localDir;
            segment.boundaryHeightIntegral = evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1],
                clamp(o.y + 0.5 * (segment.tNear + segment.tFar) * d.y, 0.0, 1.0)) * (segment.tFar - segment.tNear);
        }
    }

    float depthRaw = texture(sceneDepthSampler, fragTexCoord).r;
    if (depthRaw != DEPTH_FAR) {
        vec4 depthWorld = invViewProj * vec4(fragTexCoord * 2.0 - 1.0, depthRaw, 1.0);
        vec3 depthPos = depthWorld.xyz / depthWorld.w;
        float tDepth = dot(depthPos - frame.camera_pos.xyz, rayDir);
        if (segment.tNear >= tDepth) {
            segment.tFar = segment.tNear - 1.0;
        } else {
            segment.tFar = min(segment.tFar, tDepth);
        }
    }

    return segment;
}

float integrateEmissionAnalytic(FlameRaySegment segment) {
    return max(segment.boundaryHeightIntegral, 0.0);
}

float integrateEmissionRaymarch(FlameRaySegment segment, int stepCount) {
    float dt = (segment.tFar - segment.tNear) / float(stepCount);
    if (dt <= 0.0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + 0.5) * dt;
        vec3 p = segment.localOrigin + t * segment.localDir;
        float h = clamp(
            evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        sum += evaluateHeightFalloff(h) * flameNoiseErosionFactor(p, h);
    }
    return sum * dt;
}


vec4 shadeEmission(FlameRaySegment segment, float emission, float deltaT) {
    float heightMid = clamp(
        evaluateHeightAlongRay(
            0.5 * (segment.tNear + segment.tFar), segment.localOrigin.y, segment.localDir.y),
        0.0, 1.0);
    vec3 rampColor;
    if (heightMid < 0.5) {
        rampColor = mix(flame.colorBase.rgb, flame.colorMid.rgb, heightMid * 2.0);
    } else {
        rampColor = mix(flame.colorMid.rgb, flame.colorTip.rgb, (heightMid - 0.5) * 2.0);
    }

    float tempNorm = clamp(emission * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMid);
    vec3 radiance = rampColor * flame.intensity * (1.0 + flame.styleParams1.w * pow(tempNorm, 2.0)) * emission;
    float alpha = 1.0 - exp(-flame.sigmaT * emission);
    return vec4(radiance, alpha);
}

void main() {
    vec4 accum = texture(flameAccumSampler, fragTexCoord);
    vec2 interval = texture(flameIntervalSampler, fragTexCoord).xy;

    float coverage = accum.x;
    float deltaT = max(accum.y, 0.0);

    if (push.mode == 4) {
        FlameRaySegment segment = buildRaySegment(coverage, accum.z, interval);
        outColor = vec4(0.5 + 0.25 * clamp(coverage, -2.0, 2.0), 0.1 * deltaT, segment.tFar > segment.tNear ? 0.2 * clamp(segment.tFar - segment.tNear, 0.0, 5.0) : 0.0, 1.0);
        outHistory = outColor;
        return;
    }

    if (push.mode == 2) {
        outColor = vec4(max(accum.z, 0.0), deltaT, max(-accum.z, 0.0), 1.0);
        outHistory = outColor;
        return;
    }

    if (coverage == 0.0 && deltaT <= 0.0) {
        discard;
    }

    FlameRaySegment segment = buildRaySegment(coverage, accum.z, interval);

    // Discard if clamping produced an empty interval (tNear > tFar)
    if (segment.tNear > segment.tFar) {
        discard;
    }

    vec3 L;
    float trans;

    if (push.mode == 3) {
        // Per-step Beer-Lambert integrator with tapered radial density
        float dt = (segment.tFar - segment.tNear) / float(push.stepCount);
        float jitter = interleavedGradientNoise(gl_FragCoord.xy + vec2(flame.temporalData.y * 5.588238));
        L = vec3(0.0);
        trans = 1.0;
        for (int i = 0; i < push.stepCount; ++i) {
            float t = segment.tNear + (float(i) + jitter) * dt;
            vec3 p = segment.localOrigin + t * segment.localDir;
            float h = clamp(p.y, 0.0, 1.0);

            float dSmooth;
            float density;
            if (flame.trailMeta.x >= 1.0) {
                vec3 pWorld = (flame.model * vec4(p, 1.0)).xyz;
                vec3 baseUnit = (flame.trailUnitInverse * vec4(pWorld, 1.0)).xyz;
                density = 0.0;
                dSmooth = 0.0;
                float hBest = h;
                for (int s = 0; s < int(flame.trailMeta.x); ++s) {
                    vec3 pu = baseUnit - flame.trailSamples[s].xyz;
                    float hu = clamp(pu.y, 0.0, 1.0);
                    float ds;
                    float d = flameEmitterDensity(pu, hu, ds) * flame.trailSamples[s].w;
                    if (d > density) { density = d; dSmooth = ds; hBest = hu; }
                }
                h = hBest;
            } else {
                density = flameEmitterDensity(p, h, dSmooth);
            }

            // Beer-Lambert step
            float sigma = flame.sigmaT * density;
            float a = 1.0 - exp(-sigma * dt);

            // Temperature-driven ramp color
            float tempNorm = clamp(dSmooth, 0.0, 1.0) * (1.0 - 0.55 * h);
            vec3 rampColor;
            float u = 1.0 - tempNorm;
            if (u < 0.5) {
                rampColor = mix(flame.colorBase.rgb, flame.colorMid.rgb, u * 2.0);
            } else {
                rampColor = mix(flame.colorMid.rgb, flame.colorTip.rgb, (u - 0.5) * 2.0);
            }

            L += trans * rampColor * flame.intensity * (1.0 + flame.styleParams1.w * pow(tempNorm, 4.0)) * a;
            trans *= 1.0 - a;
        }
    } else {
        // Modes 0/1: legacy path via emission scalar + shadeEmission
        float emission;
        if (push.mode == 1) {
            emission = integrateEmissionRaymarch(segment, push.stepCount);
        } else {
            emission = integrateEmissionAnalytic(segment);
            // Mode 0: multiply emission by weighted average of flameNoiseErosionFactor
            // evaluated at tNear, midpoint, and tFar (weights 0.25, 0.5, 0.25)
            float tMid = 0.5 * (segment.tNear + segment.tFar);
            vec3 pNear = segment.localOrigin + segment.tNear * segment.localDir;
            vec3 pMid = segment.localOrigin + tMid * segment.localDir;
            vec3 pFar = segment.localOrigin + segment.tFar * segment.localDir;
            float hNear = clamp(pNear.y, 0.0, 1.0);
            float hMid = clamp(pMid.y, 0.0, 1.0);
            float hFar = clamp(pFar.y, 0.0, 1.0);
            emission *= 0.25 * flameNoiseErosionFactor(pNear, hNear)
                   + 0.5 * flameNoiseErosionFactor(pMid, hMid)
                    + 0.25 * flameNoiseErosionFactor(pFar, hFar);
        }
        if (flame.lightData.w > 0.0) {
            vec3 pMid = segment.localOrigin + 0.5 * (segment.tNear + segment.tFar) * segment.localDir;
            emission *= mix(1.0, exp(-computeSelfShadowTau(pMid, normalize(flame.lightData.xyz))), flame.lightData.w);
        }
        vec4 shaded = shadeEmission(segment, emission, deltaT);
        L = shaded.rgb;
        trans = 1.0 - shaded.a;
    }

    // Self-shadow midpoint multiply on L (mode 3)
    if (push.mode == 3 && flame.lightData.w > 0.0) {
        vec3 pMid = segment.localOrigin + 0.5 * (segment.tNear + segment.tFar) * segment.localDir;
        L *= mix(1.0, exp(-computeSelfShadowTau(pMid, normalize(flame.lightData.xyz))), flame.lightData.w);
    }

    vec4 shaded = vec4(L, 1.0 - trans);
    // Occlusion must track displayed luminance: a dim flame adds little light and must not darken the background.
    shaded.a *= smoothstep(0.0, flame.colorBase.a, dot(shaded.rgb, LUMA_WEIGHTS));
    vec4 blended = mix(shaded, texture(flameHistorySampler, fragTexCoord), flame.temporalData.x);
    outColor = blended;
    outHistory = blended;
}
