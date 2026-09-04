#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "include/trace_payload.glsl"

struct HitShadingRecord { uint64_t vertexAddress; uint64_t indexAddress; mat4 model; mat4 normalMatrix; vec4 baseColor; vec4 params; };
layout(set = 0, binding = 3, std430) readonly buffer HitShadingTable { HitShadingRecord records[]; } hitTable;
layout(buffer_reference, scalar) buffer VertexBuffer { vec4 v[]; };
layout(buffer_reference, scalar) buffer IndexBuffer { uint i[]; };
layout(push_constant) uniform WaterTraceLight {
    layout(offset = 96) vec4 lightPos;
    layout(offset = 112) vec4 lightColor;
} light;
layout(location = 0) rayPayloadInEXT TracePayload payload;
hitAttributeEXT vec2 attribs;

void main() {
    HitShadingRecord rec = hitTable.records[gl_InstanceCustomIndexEXT];
    if (rec.vertexAddress == 0) { payload.color = vec4(0.0); payload.exitOrigin = vec4(0.0); return; }
   VertexBuffer vb = VertexBuffer(rec.vertexAddress);
    uint primIdx = gl_PrimitiveID;
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
    vec3 p0 = vb.v[vi0].xyz;    vec3 c0 = vb.v[vi0 + 1].rgb; vec3 n0 = vb.v[vi0 + 2].xyz;
    vec3 p1 = vb.v[vi1].xyz;    vec3 c1 = vb.v[vi1 + 1].rgb; vec3 n1 = vb.v[vi1 + 2].xyz;
    vec3 p2 = vb.v[vi2].xyz;    vec3 c2 = vb.v[vi2 + 1].rgb; vec3 n2 = vb.v[vi2 + 2].xyz;
    float u = attribs.x;
    float v = attribs.y;
    float w = 1.0 - u - v;
    vec3 P = gl_ObjectToWorldEXT * vec4(p0 * w + p1 * u + p2 * v, 1.0);
    vec3 vertexColor = c0 * w + c1 * u + c2 * v;
    vec3 N = n0 * w + n1 * u + n2 * v;
    vec3 norm = normalize((N * gl_WorldToObjectEXT).xyz);
    vec3 L = light.lightPos.xyz - P;
    float dist = length(L);
    L /= dist;
    float ndotl = max(dot(norm, L), 0.0);
    float atten = 1.0 / (1.0 + 0.05 * dist * dist);
    vec3 shaded = rec.baseColor.rgb * vertexColor * (0.15 + 0.85 * ndotl) * light.lightColor.rgb * atten;
    payload.color = vec4(shaded, 1.0);
    payload.exitOrigin = vec4(0.0);
}
