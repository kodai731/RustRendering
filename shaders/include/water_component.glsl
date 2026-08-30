#ifndef WATER_COMPONENT_GLSL
#define WATER_COMPONENT_GLSL

layout(set = 1, binding = 0) uniform WaterUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 radii;
    vec4 absorption;
    vec4 flow;
    vec4 composite;
    vec4 tint;
    vec4 temporal;
} water;

#endif
