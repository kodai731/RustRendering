#ifndef WATER_FLOW_GLSL
#define WATER_FLOW_GLSL

vec2 torusUV(vec3 pLocalNormalized) {
    return vec2(atan(pLocalNormalized.z, pLocalNormalized.x),
                atan(pLocalNormalized.y, length(pLocalNormalized.xz) - 1.0));
}

vec2 advectUV(vec2 uv, vec2 flowRate, float time) {
    return uv + flowRate * time;
}

#endif
