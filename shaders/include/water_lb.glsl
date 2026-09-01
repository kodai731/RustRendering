#ifndef WATER_LB_GLSL
#define WATER_LB_GLSL

#include "chebyshev.glsl"

const int LB_MODE_COUNT = 4;
const int LB_SLOTS_PER_MODE = 5;

float waterLbCheb(vec4 lo, vec4 hi, float t) {
    return evaluateChebyshev8(lo, hi, 0.5 * t + 0.5);
}

void waterLbHeightAndGradient(vec2 uv, float time, vec2 flowRate, inout float h, inout float hu, inout float hv) {
    for (int k = 0; k < LB_MODE_COUNT; ++k) {
        int slot = LB_SLOTS_PER_MODE * k;
        vec4 head = water.lbModes[slot];
        float m = head.x;
        float omega = head.y;
        float amplitude = head.z;
        float phase = head.w;

        if (amplitude <= 0.0) {
            continue;
        }

        float phasePrime = m * (uv.x + flowRate.x * time) - omega * time + phase;
        float vAdvected = mod(uv.y + flowRate.y * time, 6.28318530718);
        float t = (vAdvected - 3.14159265359) / 3.14159265359;

        float phi = waterLbCheb(water.lbModes[slot + 1], water.lbModes[slot + 2], t);
        float dphi = waterLbCheb(water.lbModes[slot + 3], water.lbModes[slot + 4], t);

        h += amplitude * cos(phasePrime) * phi;
        hu += -amplitude * m * sin(phasePrime) * phi;
        hv += amplitude * cos(phasePrime) * dphi;
    }
}

#endif
