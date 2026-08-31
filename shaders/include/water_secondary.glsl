#ifndef WATER_SECONDARY_GLSL
#define WATER_SECONDARY_GLSL

// traceScene: cast a ray against the scene TLAS and shade the hit point.
// Returns true if a triangle intersection was found, false on miss.
// Hybrid: if the hit point projects to screen space within [0,1], returns the
// scene color from sceneColorSampler; otherwise returns hit lighting.
layout(buffer_reference, scalar) buffer VertexBuffer { vec4 v[]; };

bool traceScene(vec3 o, vec3 d, float tMax, out vec3 color) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, sceneTlas, gl_RayFlagsOpaqueEXT, 0xFF, o, 1e-3, d, tMax);

    while (rayQueryProceedEXT(rq)) {}

    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }

    uint idx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    HitShadingRecord rec = hitTable.records[idx];

    // If vertexAddress is 0, this is an inactive instance (e.g. empty TLAS placeholder) — return miss
    if (rec.vertexAddress == 0) {
        return false;
    }

  // Decode vertex buffer address from uint64_t
    VertexBuffer vb;
    vb = VertexBuffer(rec.vertexAddress);

    uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    int indices[3] = int[](int(primIdx * 3), int(primIdx * 3 + 1), int(primIdx * 3 + 2));

    // Read 3 vertices from buffer_reference (packed: pos@0, color@12, tex@28, normal@36)
    vec4 v0 = vb.v[indices[0]];
    vec4 v1 = vb.v[indices[1]];
    vec4 v2 = vb.v[indices[2]];

    // Extract position (xyz), color (w of first vec4 is unused, color is next vec4)
    // Packed format: vec3 pos @0, vec4 color @12, vec2 tex @28, vec3 normal @36
    // Each vertex = 48 bytes = 12 floats = 3 vec4s
    int vi0 = indices[0] * 3;
    int vi1 = indices[1] * 3;
    int vi2 = indices[2] * 3;

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

    // Hit lighting: baseColor.rgb * vertexColor.rgb * (0.15 + 0.85 * max(dot(N, L), 0)) * light_color.rgb
    vec3 L = normalize(frame.light_pos.xyz - P);
    float ndotl = max(dot(N, L), 0.0);
    vec3 hitLighting = rec.baseColor.rgb * vertexColor * (0.15 + 0.85 * ndotl) * frame.light_color.rgb;

    // Hybrid: project hit point to screen space
    vec4 clip = frame.proj * frame.view * vec4(P, 1.0);
    if (clip.w > 0.0) {
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            color = texture(sceneColorSampler, uv).rgb;
            return true;
        }
    }

    // Screen-space projection failed (outside viewport), use hit lighting
    color = hitLighting;
    return true;
}

#endif
