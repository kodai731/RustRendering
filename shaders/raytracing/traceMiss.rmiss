#version 460
#extension GL_EXT_ray_tracing : require

#include "include/trace_payload.glsl"

layout(location = 0) rayPayloadInEXT TracePayload payload;

void main() {
    payload.color = vec4(0.0);
    payload.exitOrigin = vec4(0.0);
}
