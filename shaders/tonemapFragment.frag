#version 450
#extension GL_GOOGLE_include_directive : require

#include "include/depth.glsl"

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(depth_any) out float gl_FragDepth;

layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform sampler2D bloomSampler;
layout(binding = 2) uniform sampler2D positionSampler;

layout(binding = 3) uniform SceneData {
    vec4 lightPosition;
    vec4 lightColor;
    mat4 view;
    mat4 proj;
    int debugMode;
    float shadowStrength;
    int enableDistanceAttenuation;
    float exposureValue;
} sceneData;

layout(push_constant) uniform PushConstants {
    int toneMapOperator;
    float gamma;
    float exposureValue;
    float vignetteIntensity;
    float chromaticAberrationIntensity;
    float bloomIntensity;
    vec4 plumePosition;
    vec4 plumeParams0;
    vec4 plumeParams1;
    vec4 plumeParams2;
} pc;

vec3 acesFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 reinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

vec3 applyToneMapOperator(vec3 x) {
    if (pc.toneMapOperator == 1) {
        return acesFilmic(x);
    }
    if (pc.toneMapOperator == 2) {
        return reinhard(x);
    }
    return clamp(x, 0.0, 1.0);
}

vec3 sampleWithChromaticAberration(vec2 uv, float intensity) {
    vec2 center = vec2(0.5);
    vec2 offset = (uv - center) * intensity;

    float r = texture(hdrSampler, uv + offset).r;
    float g = texture(hdrSampler, uv).g;
    float b = texture(hdrSampler, uv - offset).b;

    return vec3(r, g, b);
}

float computeVignette(vec2 uv, float intensity) {
    vec2 d = uv - vec2(0.5);
    return 1.0 - intensity * dot(d, d) * 4.0;
}

// Heat plume refraction: closed-form deflection matching flame_plume.rs exactly.
const float PLUME_REFRACTIVITY_AIR = 2.77e-4;
const float PLUME_AMBIENT_TEMPERATURE_K = 293.0;
const int PLUME_BAND_COUNT = 6;
const float PLUME_FLAT_EXPONENT = 2e-2;
const float PLUME_PI = 3.14159265358979;

// Abramowitz-Stegun 7.1.26 (same as flameErf in flame_radial_integral.glsl).
float plumeErf(float x) {
    float magnitude = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * magnitude);
    float series = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t;
    return sign(x) * (1.0 - series * exp(-magnitude * magnitude));
}

// int_{-halfWidth}^{halfWidth} s^m exp(-(a s^2 + b s + c)) ds for m = 0, 1, 2.
// Same formula as flameGaussianMoments in flame_radial_integral.glsl.
vec3 plumeGaussianMoments(float a, float b, float c, float halfWidth) {
    if (a * halfWidth * halfWidth < PLUME_FLAT_EXPONENT
        && abs(b) * halfWidth < PLUME_FLAT_EXPONENT) {
        float constantWeight = exp(-c);
        return constantWeight
            * vec3(2.0 * halfWidth, 0.0, (2.0 / 3.0) * halfWidth * halfWidth * halfWidth);
    }

    float rootA = sqrt(a);
    float center = b / (2.0 * a);
    float peak = exp(-(c - b * b / (4.0 * a)));
    float moment0 = peak * 0.5 * sqrt(PLUME_PI / a)
        * (plumeErf(rootA * (halfWidth + center)) - plumeErf(rootA * (center - halfWidth)));

    float gaugeHi = exp(-(a * halfWidth * halfWidth + b * halfWidth + c));
    float gaugeLo = exp(-(a * halfWidth * halfWidth - b * halfWidth + c));
    float moment1 = (gaugeLo - gaugeHi - b * moment0) / (2.0 * a);
    float moment2 = (moment0 - b * moment1 - halfWidth * (gaugeHi + gaugeLo)) / (2.0 * a);
    return vec3(moment0, moment1, moment2);
}

// Temperature difference at height h: (T - 293) * min((h + 0.2)^(-5/3), 1.0).
float plumeDeltaT(float h) {
    float base = pc.plumeParams0.x - PLUME_AMBIENT_TEMPERATURE_K;
    float factor = pow(h + 0.2, -5.0 / 3.0);
    return base * min(factor, 1.0);
}

// Plume width at height h: width_base + width_slope * max(h, 0.0).
float plumeWidth(float h) {
    return pc.plumeParams0.y + pc.plumeParams0.z * max(h, 0.0);
}

// Closed-form deflection using 6 height bands with frozen parameters.
// Returns [dx, dz] deflection matching flame_plume.rs exactly.
vec2 plumeDeflection(vec3 o, vec3 d, float tNear, float tFar) {
    float bandDt = (tFar - tNear) / float(PLUME_BAND_COUNT);
    float totalDx = 0.0;
    float totalDz = 0.0;

    for (int band = 0; band < PLUME_BAND_COUNT; band++) {
        float t0 = tNear + float(band) * bandDt;
        float tc = t0 + bandDt * 0.5;

        // Midpoint position
        vec3 pc_pos = o + tc * d;
        float hc = pc_pos.y;
        float b_hc = plumeWidth(hc);
        float k = 1.0 / (b_hc * b_hc);

        // Gaussian coefficients
        float a = k * (d.x * d.x + d.z * d.z);
        float b_lin = 2.0 * k * (pc_pos.x * d.x + pc_pos.z * d.z);
        float c = k * (pc_pos.x * pc_pos.x + pc_pos.z * pc_pos.z);

        vec3 moments = plumeGaussianMoments(a, b_lin, c, bandDt * 0.5);

        // Amplitude: -REFRACTIVITY_AIR * delta_t(hc) / AMBIENT_TEMPERATURE_K
        float amp = -PLUME_REFRACTIVITY_AIR * (plumeDeltaT(hc) / PLUME_AMBIENT_TEMPERATURE_K);

        // x contribution: -(2k) * amp * (pc.x*M0 + d.x*M1)
        totalDx += -(2.0 * k) * amp * (pc_pos.x * moments.x + d.x * moments.y);
        // z contribution: -(2k) * amp * (pc.z*M0 + d.z*M1)
        totalDz += -(2.0 * k) * amp * (pc_pos.z * moments.x + d.z * moments.y);
    }

   return vec2(totalDx, totalDz);
}

float plumeHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123); }
float plumeValueNoise(vec3 p) { vec3 i = floor(p); vec3 f = fract(p); f = f * f * (3.0 - 2.0 * f); float n000 = plumeHash(i); float n100 = plumeHash(i + vec3(1,0,0)); float n010 = plumeHash(i + vec3(0,1,0)); float n110 = plumeHash(i + vec3(1,1,0)); float n001 = plumeHash(i + vec3(0,0,1)); float n101 = plumeHash(i + vec3(1,0,1)); float n011 = plumeHash(i + vec3(0,1,1)); float n111 = plumeHash(i + vec3(1,1,1)); return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y), mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z); }
float plumeFbm(vec3 p) { float v = 0.0; float a = 0.5; for (int i = 0; i < 3; ++i) { v += a * plumeValueNoise(p); p *= 2.03; a *= 0.5; } return v; }

void main() {
    vec4 positionData = texture(positionSampler, fragTexCoord);
    if (positionData.w < 0.5) {
        gl_FragDepth = DEPTH_FAR;
    } else {
        gl_FragDepth = worldToClipDepth(positionData.xyz, sceneData.view, sceneData.proj);
    }
    // Heat plume refraction: if active, distort UV before HDR sampling.
    vec2 uv = fragTexCoord;
    if (pc.plumePosition.w > 0.5) {
        // Camera position C from view matrix: C = -transpose(R) * t
        vec3 C = -(sceneData.view[0].xyz * sceneData.view[3].x
                 + sceneData.view[1].xyz * sceneData.view[3].y
                 + sceneData.view[2].xyz * sceneData.view[3].z);

        // Sample background position
        vec3 Pbg = texture(positionSampler, fragTexCoord).xyz;
        if (Pbg != vec3(0.0)) {
            vec3 rayDir = normalize(Pbg - C);
            float dBg = length(Pbg - C);

            // Plume origin in local coords (C relative to plume base)
            vec3 originLocal = C - pc.plumePosition.xyz;
            float plumeHeight = pc.plumeParams1.x;

            // Clip t interval to y-slab [0, plume_height]
            float tNear = 0.0;
            float tFar = dBg;

            if (rayDir.y > 1e-6) {
                float ty0 = -originLocal.y / rayDir.y;
                float ty1 = (plumeHeight - originLocal.y) / rayDir.y;
                tNear = max(tNear, ty0);
                tFar = min(tFar, ty1);
            } else if (rayDir.y < -1e-6) {
                float ty0 = -originLocal.y / rayDir.y;
                float ty1 = (plumeHeight - originLocal.y) / rayDir.y;
                tNear = max(tNear, ty1);
                tFar = min(tFar, ty0);
            } else {
                // Ray parallel to y-axis: check if origin is within slab
                if (originLocal.y < 0.0 || originLocal.y > plumeHeight) {
                    tFar = -1.0; // no intersection
                }
            }

            // Clip by radius |xz| < 4 * b(plume_height)
            float maxRadius = 4.0 * plumeWidth(plumeHeight);
            if (maxRadius > 0.0) {
                // Solve |originLocal.xz + t * rayDir.xz|^2 < maxRadius^2
                float ax = rayDir.x * rayDir.x + rayDir.z * rayDir.z;
                float bx = 2.0 * (originLocal.x * rayDir.x + originLocal.z * rayDir.z);
                float cx = originLocal.x * originLocal.x + originLocal.z * originLocal.z - maxRadius * maxRadius;

                if (ax > 1e-8) {
                    float disc = bx * bx - 4.0 * ax * cx;
                    if (disc > 0.0) {
                        float sqrtDisc = sqrt(disc);
                        float rt0 = (-bx - sqrtDisc) / (2.0 * ax);
                        float rt1 = (-bx + sqrtDisc) / (2.0 * ax);
                        tNear = max(tNear, rt0);
                        tFar = min(tFar, rt1);
                    } else {
                        // Ray never enters radius: no intersection
                        tFar = -1.0;
                    }
                } else {
                    // Near-zero xz direction: check if origin is within radius
                    if (cx > 0.0) {
                        tFar = -1.0; // no intersection
                    }
                }
           }


            // If interval is valid and non-empty, compute deflection
            if (tNear < tFar && tFar >= 0.0) {
               vec2 theta = plumeDeflection(originLocal, rayDir, tNear, tFar) * pc.plumeParams0.w;

                vec3 pMid = C + rayDir * (tNear + tFar) * 0.5;
                float turb = 1.0 + pc.plumeParams1.z * (plumeFbm(pMid * 6.0 - vec3(pc.plumeParams2.x, max(pc.plumeParams2.y, 1.0), pc.plumeParams2.z) * (2.0 * pc.plumeParams1.y)) * 2.0 - 1.0);
                theta *= turb;
              float deltaD = dBg - tFar;
                if (deltaD > 0.0) {
                    vec3 Pdisp = Pbg + vec3(theta.x, 0.0, theta.y) * deltaD;

                    vec4 clipD = sceneData.proj * sceneData.view * vec4(Pdisp, 1.0);
                    vec4 clipO = sceneData.proj * sceneData.view * vec4(Pbg, 1.0);
                    vec2 uvD = (clipD.xy / clipD.w) * 0.5 + 0.5;
                    vec2 uvO = (clipO.xy / clipO.w) * 0.5 + 0.5;
                    uv = clamp(uv + (uvD - uvO), 0.0, 1.0);
                }
            }
        }
    }

    // Every pixel takes one path: the HDR buffer already holds scene radiance for
    // background, opaque surfaces, grid and volumetrics alike, so a single exposure
    // and tone map curve applies to all of them.
   vec3 hdrColor;
    if (pc.chromaticAberrationIntensity > 0.0) {
        hdrColor = sampleWithChromaticAberration(uv, pc.chromaticAberrationIntensity);
    } else {
        hdrColor = texture(hdrSampler, uv).rgb;
    }

    if (pc.bloomIntensity > 0.0) {
        vec3 bloomColor = texture(bloomSampler, fragTexCoord).rgb;
        hdrColor += bloomColor * pc.bloomIntensity;
    }

    hdrColor *= pc.exposureValue;

    vec3 mapped = applyToneMapOperator(hdrColor);
    mapped = pow(mapped, vec3(1.0 / pc.gamma));

    if (pc.vignetteIntensity > 0.0) {
        float vignette = computeVignette(fragTexCoord, pc.vignetteIntensity);
        mapped *= vignette;
    }

    outColor = vec4(mapped, 1.0);
}
