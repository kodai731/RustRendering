#version 450

// F2 shading pass. push.mode swaps HOW emission is integrated and nothing else:
// the ray interval, camera-inside handling, color ramp and alpha live in the
// shared FlameRaySegment path, so analytic vs raymarch comparisons isolate
// the integration method alone regardless of colors or future noise.
// The interval [tNear, tFar] is derived in closed form from the shell envelope
// (cone x y-slab, clampToShellCone); no rasterized proxy geometry exists.
// push.mode: 0 = analytic boundary integral, 1 = reference raymarch,
// 2 = delta-t debug view, 3 = styled raymarch (IGN jitter + noise erosion).

#include "include/chebyshev.glsl"
#include "include/flame_ray.glsl"
#include "include/flame_noise.glsl"
#include "include/flame_shell_profile.glsl"

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
    vec4 trail_coefficients[4];
    vec4 emitterParams;
    vec4 contourParams;
    vec4 erosionResponse;
    vec4 waveCfParams;
    vec4 boundaryParams;
    vec4 nearFadeParams;
    vec4 radiusCoefficients[2];
    vec4 colorRamp[8];
    vec4 profileParams;
    vec4 waveParams;
    vec4 tipCarveParams;
    vec4 warpStrainParams;
    vec4 warpFormParams;
    vec4 unifiedParams;
    vec4 spreadParams;
    vec4 waveModes[428];
    vec4 waveJitter[96];
} flame;

#include "include/flame_shell_support.glsl"

layout(set = 1, binding = 4) uniform sampler2D flameHistorySampler;
layout(set = 1, binding = 5) uniform sampler2D flameSdfSampler;
layout(set = 1, binding = 6) uniform sampler2D sceneDepthSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

layout(push_constant) uniform FlamePush {
    int mode;
    int stepCount;
    int debugView;
} push;

const float SEGMENT_T_MAX = 1e4;
const float H_DIR_EPSILON = 1e-4;
const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

float evaluateHeightFalloff(float height01) {
    return evaluateChebyshev8(flame.heightCoefficients[0], flame.heightCoefficients[1], height01);
}

#include "include/flame_noise_field.glsl"
#include "include/flame_erf_moments.glsl"
#include "include/flame_erosion_response.glsl"
#include "include/flame_radial_integral.glsl"

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

const int DEPTH_CLAMP_NO_SURFACE = 0;
const int DEPTH_CLAMP_SURFACE_BEHIND = 1;
const int DEPTH_CLAMP_TRUNCATES = 2;
const int DEPTH_CLAMP_OCCLUDES = 3;

struct FlameDepthClamp {
    float sceneDepth;
    float tDepth;
    int state;
};

struct FlameRaySegment {
   float tNear;
    float tFar;
    vec3 localOrigin;
    vec3 localDir;
    float boundaryHeightIntegral;
    FlameDepthClamp depthClamp;
    bool cylinderDomain;
};

// The closed form and the shell clamp both assume the plain cylinder domain.
// Cylinder emitter without a trail: the radial band integral applies wind
// bend per band (flameBendOffsetAt), so bent flames stay on this path instead
// of falling back to the boundary integral, which cuts flat from above.
bool isCylinderDomain() {
    return flame.emitterParams.x < 0.5 && flame.trailMeta.x < 1.0;
}

bool clampToShellCone(vec3 o, vec3 d, float radiusPad, inout float tNear, inout float tFar) {
    // Shell cone: |p.xz| <= flameShellOuterRadius(p.y) + radiusPad, which is linear in y:
    // f(t) = m + n*t with m = flameShellOuterRadius(o.y) + radiusPad and n = its slope along the ray.
    // Condition |o.xz + t*d.xz|^2 - f(t)^2 <= 0 -> a = dot(d.xz,d.xz) - n*n, b = 2.0*(dot(o.xz,d.xz) - m*n), c = dot(o.xz,o.xz) - m*m
    float supportScale = flameShellSupportScale();
    float coneSlope =
        flameShellOuterRadius(1.0, supportScale) - flameShellOuterRadius(0.0, supportScale);
    float m = flameShellOuterRadius(o.y, supportScale) + radiusPad;
    float n = coneSlope * d.y;
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

// Scene depth (grid + opaque meshes) projected onto the view ray, so the emission
// interval can be cut where a surface occludes the flame.
FlameDepthClamp resolveSceneDepthClamp(mat4 invViewProj, vec3 rayDir, float tNear, float tFar) {
    FlameDepthClamp depthClamp;
    depthClamp.sceneDepth = texture(sceneDepthSampler, fragTexCoord).r;
    depthClamp.tDepth = 0.0;
    depthClamp.state = DEPTH_CLAMP_NO_SURFACE;

    if (depthClamp.sceneDepth == DEPTH_FAR) {
        return depthClamp;
    }

    vec4 surfaceClip = invViewProj * vec4(fragTexCoord * 2.0 - 1.0, depthClamp.sceneDepth, 1.0);
    vec3 surfaceWorld = surfaceClip.xyz / surfaceClip.w;
    depthClamp.tDepth = dot(surfaceWorld - frame.camera_pos.xyz, rayDir);

    if (tNear >= depthClamp.tDepth) {
        depthClamp.state = DEPTH_CLAMP_OCCLUDES;
    } else if (tFar > depthClamp.tDepth) {
        depthClamp.state = DEPTH_CLAMP_TRUNCATES;
    } else {
        depthClamp.state = DEPTH_CLAMP_SURFACE_BEHIND;
    }
    return depthClamp;
}

FlameRaySegment buildRaySegment() {
    mat4 invViewProj = inverse(frame.proj * frame.view);
    vec3 rayDir = reconstructRayDirection(fragTexCoord, invViewProj, frame.camera_pos.xyz);

    FlameRaySegment segment;
    segment.localOrigin = (flame.inverseModel * vec4(frame.camera_pos.xyz, 1.0)).xyz;
    segment.localDir = (flame.inverseModel * vec4(rayDir, 0.0)).xyz;
    segment.tNear = 0.0;
    segment.tFar = SEGMENT_T_MAX;
    segment.boundaryHeightIntegral = 0.0;
    segment.cylinderDomain = isCylinderDomain();
    segment.depthClamp.sceneDepth = DEPTH_FAR;
    segment.depthClamp.tDepth = 0.0;
    segment.depthClamp.state = DEPTH_CLAMP_NO_SURFACE;

    // Wind bend shifts the density sideways by at most |wind| * bendAmount (h^p <= 1);
    // the trail proxy must not cut it, so its cone is padded by that bound. Non-trail
    // emitters keep the unpadded cone (their integrators bend per evaluation).
    float radiusPad = flame.trailMeta.x >= 1.0
        ? length(flame.styleParams2.xy) * flame.styleParams2.z
        : 0.0;
    if (!clampToShellCone(segment.localOrigin, segment.localDir, radiusPad, segment.tNear, segment.tFar)) {
        segment.tNear = 1.0;
        segment.tFar = 0.0;
        return segment;
    }
    segment.tNear = max(segment.tNear, 0.0);
    if (segment.tFar < segment.tNear) {
        segment.tNear = 1.0;
        segment.tFar = 0.0;
        return segment;
    }

    // Scene depth must cut the interval before the emission integral is derived from it,
    // otherwise the integral and the shaded midpoint describe different segments.
    segment.depthClamp = resolveSceneDepthClamp(invViewProj, rayDir, segment.tNear, segment.tFar);
    if (segment.depthClamp.state == DEPTH_CLAMP_OCCLUDES) {
        segment.tNear = 1.0;
        segment.tFar = 0.0;
        return segment;
    }
    if (segment.depthClamp.state != DEPTH_CLAMP_NO_SURFACE) {
        segment.tFar = min(segment.tFar, segment.depthClamp.tDepth);
    }

    vec3 o = segment.localOrigin;
    vec3 d = segment.localDir;
    if (abs(d.y) > H_DIR_EPSILON) {
        segment.boundaryHeightIntegral = (evaluateHeightPrimitive(clamp(o.y + segment.tFar * d.y, 0.0, 1.0))
            - evaluateHeightPrimitive(clamp(o.y + segment.tNear * d.y, 0.0, 1.0))) / d.y;
    } else {
        // Near-horizontal ray: h is constant over the interval, use the mid-point rule.
        segment.boundaryHeightIntegral = evaluateHeightFalloff(
            clamp(o.y + 0.5 * (segment.tNear + segment.tFar) * d.y, 0.0, 1.0))
            * (segment.tFar - segment.tNear);
    }

    return segment;
}

float integrateEmissionAnalytic(FlameRaySegment segment) {
    // The wave basis keeps the cylinder density convention; both dispatchers
    // route to the band-free wave segment integrator.
    if (segment.cylinderDomain) {
        return max(integrateRadialEmission(
            segment.localOrigin, segment.localDir, segment.tNear, segment.tFar), 0.0);
    }
    if (flame.trailMeta.x < 1.0) {
        return integrateEmitterOccupancy(
            segment.localOrigin, segment.localDir, segment.tNear, segment.tFar);
    }
    return max(segment.boundaryHeightIntegral, 0.0);
}

#include "include/flame_reference_march.glsl"


// Debug mode 4: hue = clamp decision, 1-unit brightness bands = tDepth magnitude.
// dark blue: no scene surface / cyan: tDepth is NaN / magenta: tDepth behind camera
// red: segment fully occluded (flame discarded) / yellow: tFar truncated / green: surface behind segment
vec3 visualizeDepthClamp(FlameDepthClamp depthClamp) {
    if (depthClamp.state == DEPTH_CLAMP_NO_SURFACE) {
        return vec3(0.0, 0.0, 0.25);
    }
    if (isnan(depthClamp.tDepth)) {
        return vec3(0.0, 1.0, 1.0);
    }
    if (depthClamp.tDepth < 0.0) {
        return vec3(1.0, 0.0, 1.0);
    }

    float band = 0.35 + 0.65 * fract(depthClamp.tDepth);
    if (depthClamp.state == DEPTH_CLAMP_OCCLUDES) {
        return vec3(band, 0.0, 0.0);
    }
    if (depthClamp.state == DEPTH_CLAMP_TRUNCATES) {
        return vec3(band, band, 0.0);
    }
    return vec3(0.0, band, 0.0);
}

vec4 shadeEmission(FlameRaySegment segment, float emission) {
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

// ---- Numeric debug views (push.debugView > 0) ----
// The flame color is replaced by a colormap of the selected intermediate,
// sampled at the max-density node along the ray (Emission Total integrates).
// History blending is bypassed so the display is the raw current-frame value.

vec3 flameDebugHeat(float v) {
    v = clamp(v, 0.0, 1.0);
    return vec3(clamp(v * 3.0, 0.0, 1.0), clamp(v * 3.0 - 1.0, 0.0, 1.0), clamp(v * 3.0 - 2.0, 0.0, 1.0));
}

vec3 flameDebugDiverging(float v) {
    float p = clamp(v, 0.0, 1.0);
    float n = clamp(-v, 0.0, 1.0);
    return vec3(p, 0.08 + 0.5 * p * p, n);
}

vec4 flameDebugViewColor(FlameRaySegment segment) {
    vec3 o = segment.localOrigin;
    vec3 d = segment.localDir;
    float tNear = segment.tNear;
    float tFar = segment.tFar;
    if (!flameWaveSupportSpan(o, d, tNear, tFar)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    if (push.debugView == 6) {
        float total = integrateWaveOccupancy(o, d, tNear, tFar);
        return vec4(flameDebugHeat(total * 2.0), 1.0);
    }
    if (push.debugView == 9) {
        // Segment grid geometry: R = node-grid phase (level sets are the
        // integrator's t-lattice — compare its arcs against the fringes),
        // G = segment length dt, B = interval entry phase in world t.
        float dt = (tFar - tNear) / float(FLAME_WAVE_SEGMENTS);
        return vec4(fract(tNear / max(dt, 1e-6)), clamp(dt * 8.0, 0.0, 1.0),
            fract(tNear * 8.0), 1.0);
    }
    float dt = (tFar - tNear) / float(FLAME_WAVE_SEGMENTS);
    float bestT = tNear;
    float bestDensity = 0.0;
    for (int i = 0; i <= FLAME_WAVE_SEGMENTS; ++i) {
        float t = tNear + float(i) * dt;
        vec3 pNode = o + t * d;
        float density = flameWaveNodeDensity(pNode, clamp(pNode.y, 0.0, 1.0));
        if (density > bestDensity) {
            bestDensity = density;
            bestT = t;
        }
    }
    if (bestDensity <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec3 p = o + bestT * d;
    float h = clamp(p.y, 0.0, 1.0);
    if (push.debugView == 4) {
        return vec4(flameDebugHeat(bestDensity), 1.0);
    }
    if (push.debugView == 7 || push.debugView == 8) {
        vec3 w = flameBuildWarpFrame(p, vec3(0.0), h).w;
        if (push.debugView == 7) {
            return vec4(flameWaveJitterFields(w) * 0.5 + 0.5, 1.0);
        }
        return vec4(fract(w * 0.15915494), 1.0);
    }
    if (push.debugView == 10) {
        // Warp Strain: the designed profile strain(h) against the fold-free cap.
        // Green (0) -> yellow (cap/2) -> red (cap); any pixel above the cap is
        // magenta = fold-free violation (regression detector). Cap mirrors
        // flame_wave.rs WARP_STRAIN_CAP.
        float v = flameWarpStrain(h) / 0.9;
        if (v > 1.0) {
            return vec4(1.0, 0.0, 1.0, 1.0);
        }
        vec3 c = v < 0.5
            ? mix(vec3(0.0, 0.7, 0.1), vec3(1.0, 0.9, 0.0), v * 2.0)
            : mix(vec3(1.0, 0.9, 0.0), vec3(0.9, 0.05, 0.0), v * 2.0 - 1.0);
        return vec4(c, 1.0);
    }
    if (push.debugView == 11) {
        // Warp Stretch: realized local stretch |J . d| / |d| of the composed
        // shear map along the ray direction, log2-mapped blue (compressed)
        // -> white (1) -> red (stretched), saturating at +-3 octaves. Fringes
        // over strong compression = strain-driven lamination.
        vec3 pb = flameNoiseBendRemoved(p, h);
        vec3 q;
        vec3 rate = flameWaveFlowWarpRate(pb, d, h, q);
        float stretch = length(rate) / max(length(d), 1e-6);
        float oct = clamp(log2(max(stretch, 1e-6)) / 3.0, -1.0, 1.0);
        vec3 c = oct >= 0.0
            ? mix(vec3(1.0), vec3(0.85, 0.1, 0.05), oct)
            : mix(vec3(1.0), vec3(0.1, 0.25, 0.9), -oct);
        return vec4(c, 1.0);
    }
    float eddyTime = flame.noiseScrollSpeed * flame.time;
    int count = min(int(flame.waveParams.x), FLAME_WAVE_EROSION_SLOTS);
    float shapedNoise;
    float sigmaNoise;
    float remapScale;
    float argument = flameWaveNodeArgumentLocal(
        p, d, h, bestDensity, dt, count, eddyTime, shapedNoise, sigmaNoise, remapScale);
    if (push.debugView == 1) {
        float v = (shapedNoise - 0.4375) / max(flame.waveParams.w, 1e-4);
        return vec4(flameDebugDiverging(v), 1.0);
    }
    if (push.debugView == 2) {
        return vec4(flameDebugDiverging(flameNoiseErosionFromValue(shapedNoise, h, bestDensity)), 1.0);
    }
    if (push.debugView == 3) {
        return vec4(flameDebugDiverging(argument * 2.0), 1.0);
    }
    if (push.debugView == 5) {
        return vec4(flameDebugHeat(sigmaNoise * 4.0), 1.0);
    }
    return vec4(flameDebugHeat(bestDensity), 1.0);
}

void main() {
    FlameRaySegment segment = buildRaySegment();

    if (push.debugView > 0 && flame.trailMeta.x < 1.0 && segment.tNear <= segment.tFar) {
        outColor = flameDebugViewColor(segment);
        outHistory = outColor;
        return;
    }

    if (push.mode == 4) {
        outColor = vec4(visualizeDepthClamp(segment.depthClamp), 1.0);
        outHistory = outColor;
        return;
    }

    if (push.mode == 2) {
        float deltaT = max(segment.tFar - segment.tNear, 0.0);
        outColor = vec4(
            max(segment.boundaryHeightIntegral, 0.0),
            deltaT,
            max(-segment.boundaryHeightIntegral, 0.0),
            1.0);
        outHistory = outColor;
        return;
    }

    // Discard if the analytic envelope produced an empty interval (tNear > tFar)
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

                // Cubic curve coefficients: c0, c1, c2, c3 (each vec4, xyz = coefficient)
                vec3 c0 = flame.trail_coefficients[0].xyz;
                vec3 c1 = flame.trail_coefficients[1].xyz;
                vec3 c2 = flame.trail_coefficients[2].xyz;
                vec3 c3 = flame.trail_coefficients[3].xyz;

                // Chord projection: u_chord = clamp(dot(p - c0, c1 - c0) / dot(c1 - c0, c1 - c0), 0.0, 1.0)
                vec3 chord = c1 - c0;
                float chordLenSq = dot(chord, chord);
                float u = 0.0;
                if (chordLenSq > 1e-6) {
                    u = clamp(dot(baseUnit - c0, chord) / chordLenSq, 0.0, 1.0);
                }

                // 3 Newton steps to refine u on f(u) = |p - C(u)|^2
                for (int step = 0; step < 3; ++step) {
                    float u2 = u * u;
                    float u3 = u2 * u;
                    vec3 curvePos = c0 + c1 * u + c2 * u2 + c3 * u3;
                    vec3 curveDeriv = c1 + c2 * 2.0 * u + c3 * 3.0 * u2;
                    vec3 residual = baseUnit - curvePos;
                    float fp = 2.0 * dot(residual, -curveDeriv);
                    float fpp = 2.0 * dot(curveDeriv, curveDeriv);
                    if (fpp > 1e-8) {
                        u -= fp / fpp;
                    }
                    u = clamp(u, 0.0, 1.0);
                }

                // Evaluate density at u* with weight (1.0 - u)
                float u2 = u * u;
                float u3 = u2 * u;
                vec3 curvePos = c0 + c1 * u + c2 * u2 + c3 * u3;
                vec3 pu = baseUnit - curvePos;
                float hu = clamp(pu.y, 0.0, 1.0);
                density = flameEmitterDensity(pu, hu, dSmooth) * (1.0 - u);
                h = hu;
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
        if (flame.trailMeta.x < 1.0 && flame.contourParams.z >= 2.0) {
            vec4 rte;
            if (push.mode == 1) {
                rte = integrateRTERaymarch(segment, push.stepCount);
            } else if (segment.cylinderDomain) {
                rte = integrateRadialRTE(segment.localOrigin, segment.localDir, segment.tNear, segment.tFar);
            } else {
                rte = integrateEmitterOccupancyRTE(segment.localOrigin, segment.localDir, segment.tNear, segment.tFar);
            }
            if (flame.lightData.w > 0.0) {
                vec3 pMid = segment.localOrigin + 0.5 * (segment.tNear + segment.tFar) * segment.localDir;
                rte.rgb *= mix(1.0, exp(-computeSelfShadowTau(pMid, normalize(flame.lightData.xyz))), flame.lightData.w);
            }
            L = rte.rgb;
            trans = 1.0 - rte.a;
        } else {
            // Modes 0/1: legacy path via emission scalar + shadeEmission
            float emission;
            if (push.mode == 1) {
                emission = integrateEmissionRaymarch(segment, push.stepCount);
            } else {
                emission = integrateEmissionAnalytic(segment);
                // Mode 0: multiply emission by weighted average of flameNoiseErosionFactor
                // evaluated at tNear, midpoint, and tFar (weights 0.25, 0.5, 0.25)
                // Only for the trail domain — cylinder and emitter occupancy paths own their erosion
                if (flame.trailMeta.x >= 1.0) {
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
            }
            if (flame.lightData.w > 0.0) {
                vec3 pMid = segment.localOrigin + 0.5 * (segment.tNear + segment.tFar) * segment.localDir;
                emission *= mix(1.0, exp(-computeSelfShadowTau(pMid, normalize(flame.lightData.xyz))), flame.lightData.w);
            }
            vec4 shaded = shadeEmission(segment, emission);
            L = shaded.rgb;
            trans = 1.0 - shaded.a;
        }
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
