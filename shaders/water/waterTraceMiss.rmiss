#version 460
#extension GL_EXT_ray_tracing : require

#include "water/include/water_trace_payload.glsl"

layout(location = 0) rayPayloadInEXT WaterTracePayload payload;

void main() {
    payload.color = vec4(0.0);
    payload.exitOrigin = vec4(0.0);
}
