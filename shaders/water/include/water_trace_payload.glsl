#ifndef WATER_TRACE_PAYLOAD_GLSL
#define WATER_TRACE_PAYLOAD_GLSL
struct WaterTracePayload {
    vec4 color;      // rgb shaded color, a = 1 hit / 0 miss
    vec4 reflOrigin; // xyz entry point (world), w = F
    vec4 reflDir;    // xyz reflection dir (world), w = chord
    vec4 exitOrigin; // xyz exit point (world), w = 1 water hit / 0 otherwise
    vec4 exitDir;    // xyz exit dir (world), w = slope variance
};
#endif
