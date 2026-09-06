#ifndef WIND_COMPONENT_GLSL
#define WIND_COMPONENT_GLSL

// WindUBO is generated into thyllore-effect-core/src/wind/gpu/components/generated.rs
// (cargo run -p thyllore-shader-manifest --bin generate_gpu_blocks).

#ifndef WIND_UBO_SET
#define WIND_UBO_SET 1
#define WIND_UBO_BINDING 0
#endif

layout(set = WIND_UBO_SET, binding = WIND_UBO_BINDING) uniform WindUBO {
    mat4 model;
    mat4 inverseModel;
    vec4 shape;
    vec4 core;
    vec4 optics;
    vec4 albedo;
    vec4 ring;
    vec4 lighting;
    vec4 streak;
    vec4 streak2;
    mat4 invViewProj;
} wind;

#endif
