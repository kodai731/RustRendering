#ifndef WATER_SECONDARY_GLSL
#define WATER_SECONDARY_GLSL

#include "flame_noise.glsl"

// waterJitter: interleaved gradient noise for depth-2 probabilistic sampling.
float waterJitter(vec2 fragCoord, float frameIndex) {
    return interleavedGradientNoise(fragCoord + vec2(frameIndex * 5.588238));
}

// Last hit kind from traceScene: 0 = miss, 1 = in-screen hit, 2 = out-of-screen hit
int waterTraceLastHitKind = 0;

// traceScene: cast a ray against the scene TLAS and shade the hit point.
// Returns true if a triangle intersection was found, false on miss.
// Hybrid: if the hit point projects to screen space within [0,1], returns the
// scene color from sceneColorSampler; otherwise returns hit lighting.
layout(buffer_reference, scalar) buffer VertexBuffer { vec4 v[]; };
layout(buffer_reference, scalar) buffer IndexBuffer { uint i[]; };

bool waterLightOccluded(vec3 origin, vec3 lightPos) {
    vec3 toLight = lightPos - origin;
    float distance = length(toLight);
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, sceneTlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 1e-3, toLight / distance, distance);
    while (rayQueryProceedEXT(rq)) {
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionTriangleEXT;
}

bool traceScene(vec3 o, vec3 d, float tMax, out vec3 color, out float tHit) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, sceneTlas, gl_RayFlagsOpaqueEXT, 0xFF, o, 1e-3, d, tMax);

    while (rayQueryProceedEXT(rq)) {
        // Ignore AABB candidates — water torus self-re-entry is handled by tTorusNext
    }

    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        tHit = tMax;
        waterTraceLastHitKind = 0;
        return false;
    }

    uint idx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    HitShadingRecord rec = hitTable.records[idx];

    // If vertexAddress is 0, this is an inactive instance (e.g. empty TLAS placeholder) — return miss
    if (rec.vertexAddress == 0) {
        tHit = tMax;
        waterTraceLastHitKind = 0;
        return false;
    }

    tHit = rayQueryGetIntersectionTEXT(rq, true);

  // Decode vertex buffer address from uint64_t
    VertexBuffer vb;
    vb = VertexBuffer(rec.vertexAddress);

   uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    int vi0, vi1, vi2;
    if (rec.indexAddress != 0u) {
        IndexBuffer ib = IndexBuffer(rec.indexAddress);
        uint i0 = ib.i[primIdx * 3u];
        uint i1 = ib.i[primIdx * 3u + 1u];
        uint i2 = ib.i[primIdx * 3u + 2u];
        vi0 = int(i0) * 3;
        vi1 = int(i1) * 3;
        vi2 = int(i2) * 3;
    } else {
        vi0 = int(primIdx) * 9;
        vi1 = vi0 + 3;
        vi2 = vi0 + 6;
    }

    // Read 3 vertices from buffer_reference (packed: pos@0, color@12, tex@28, normal@36)

    vec3 p0 = vb.v[vi0].xyz;
    vec3 c0 = vb.v[vi0 + 1].rgb;
    vec3 n0 = vb.v[vi0 + 2].xyz;

    vec3 p1 = vb.v[vi1].xyz;
    vec3 c1 = vb.v[vi1 + 1].rgb;
    vec3 n1 = vb.v[vi1 + 2].xyz;

    vec3 p2 = vb.v[vi2].xyz;
    vec3 c2 = vb.v[vi2 + 1].rgb;
    vec3 n2 = vb.v[vi2 + 2].xyz;

   // Barycentric coordinates
    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;

    // Interpolate position, color, normal
    vec3 P = p0 * w + p1 * u + p2 * v;
    vec3 vertexColor = c0 * w + c1 * u + c2 * v;
    vec3 nLocal = n0 * w + n1 * u + n2 * v;

    // Transform normal to world space
    vec3 N = normalize(rec.normalMatrix[0].xyz * nLocal.x + rec.normalMatrix[1].xyz * nLocal.y + rec.normalMatrix[2].xyz * nLocal.z);

    vec3 L = normalize(frame.light_pos.xyz - P);
    float ndotl = max(dot(N, L), 0.0);
    vec3 hitLighting = rec.baseColor.rgb * vertexColor * (0.15 + 0.85 * ndotl) * frame.light_color.rgb * water.lighting.x;

    // Hybrid: project hit point to screen space
    vec4 clip = frame.proj * frame.view * vec4(P, 1.0);
    if (clip.w > 0.0) {
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            color = texture(sceneColorSampler, uv).rgb;
            waterTraceLastHitKind = 1;
            return true;
        }
    }

    // Screen-space projection failed (outside viewport), use hit lighting
    color = hitLighting;
    waterTraceLastHitKind = 2;
    return true;
}

#endif
