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
} flame;

layout(set = 1, binding = 2) uniform sampler2D flameAccumSampler;
layout(set = 1, binding = 3) uniform sampler2D flameIntervalSampler;
layout(set = 1, binding = 4) uniform sampler2D flameHistorySampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

layout(push_constant) uniform FlamePush {
    int mode;
    int stepCount;
} push;

const float INTERVAL_CLEAR_THRESHOLD = 1e37;
const float H_DIR_EPSILON = 1e-4;

float evaluateHeightFalloff(float height01) {
    return evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1], height01);
}

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

    // Camera inside the shell: front boundary terms at t = 0 were never
    // rasterized, so coverage = +N and the missing (-1) * H1(h_o) / h_d
    // terms are restored here (their t and delta-t contributions are zero).
    if (coverage > 0.5 && abs(segment.localDir.y) > H_DIR_EPSILON) {
        float missingCount = round(coverage);
        segment.boundaryHeightIntegral -= missingCount
            * evaluateHeightPrimitive(clamp(segment.localOrigin.y, 0.0, 1.0)) / segment.localDir.y;
        segment.tNear = 0.0;
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
        float h = clamp(
            evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        sum += evaluateHeightFalloff(h);
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

    vec3 radiance = rampColor * flame.intensity * emission;
    float alpha = 1.0 - exp(-flame.sigmaT * emission);
    return vec4(radiance, alpha);
}

void main() {
    vec4 accum = texture(flameAccumSampler, fragTexCoord);
    vec2 interval = texture(flameIntervalSampler, fragTexCoord).xy;

    float coverage = accum.x;
    float deltaT = max(accum.y, 0.0);

    if (push.mode == 2) {
        outColor = vec4(max(accum.z, 0.0), deltaT, max(-accum.z, 0.0), 1.0);
        return;
    }

    if (coverage == 0.0 && deltaT <= 0.0) {
        discard;
    }

    FlameRaySegment segment = buildRaySegment(coverage, accum.z, interval);

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

            // Wind bend deformation (horizontal-only)
            vec2 bendOffset = flame.styleParams2.xy * flame.styleParams2.z * pow(h, flame.styleParams2.w);
            vec3 pb = vec3(p.x - bendOffset.x, p.y, p.z - bendOffset.y);

            // Domain warp with upward advection
            vec3 advect = vec3(flame.styleParams2.x, flame.styleParams0.z, flame.styleParams2.y) * flame.time;
            vec3 aniso = vec3(1.0, 0.35, 1.0);
            vec3 wp = (pb * aniso) * flame.styleParams0.y - advect;
            vec2 w = vec2(fbm3(wp), fbm3(wp + vec3(19.1, 7.7, 3.3))) * 2.0 - 1.0;
            vec3 q = pb + flame.styleParams0.x * mix(0.15, 1.0, h) * vec3(w.x, 0.0, w.y);

            // Tapered radial density
            float taperR = mix(1.0, flame.styleParams1.x, pow(h, flame.styleParams0.w));
            float rn = length(q.xz) / max(taperR, 1e-4);
            float dSmooth = evaluateHeightFalloff(h) * exp(-flame.radialSharpness * rn * rn);
            float erosion = flame.noiseAmplitude * mix(0.2, 1.0, h)
                * (fbm3((q * aniso) * flame.noiseFrequency - advect) - 0.35);
            float density = smoothstep(flame.styleParams1.y, flame.styleParams1.z, dSmooth - erosion);

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
    vec4 blended = mix(shaded, texture(flameHistorySampler, fragTexCoord), flame.temporalData.x);
    outColor = blended;
    outHistory = blended;
}
