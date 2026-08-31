#ifndef WATER_COMPONENT_GLSL
#define WATER_COMPONENT_GLSL

#ifndef WATER_UBO_SET
#define WATER_UBO_SET 1
#define WATER_UBO_BINDING 0
#endif

layout(set = WATER_UBO_SET, binding = WATER_UBO_BINDING) uniform WaterUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 radii;
    vec4 absorption;
    vec4 flow;
    vec4 composite;
    vec4 tint;
    vec4 temporal;
    vec4 waveModes[16];
    mat4 invViewProj;
} water;

#endif
