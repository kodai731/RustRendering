#version 450

#include "include/flame_shell_profile.glsl"

layout(location = 0) out vec3 geomLocalCorner;

const float R = FLAME_SHELL_BASE_RADIUS;
const vec3 QUAD_CORNERS[4] = vec3[4](
    vec3(-R, 0.0, -R),
    vec3(R, 0.0, -R),
    vec3(R, 0.0, R),
    vec3(-R, 0.0, R)
);

void main() {
    geomLocalCorner = QUAD_CORNERS[gl_VertexIndex & 3];
    gl_Position = vec4(geomLocalCorner, 1.0);
}
