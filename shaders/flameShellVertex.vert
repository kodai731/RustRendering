#version 450

layout(location = 0) out vec3 geomLocalCorner;

const vec3 QUAD_CORNERS[4] = vec3[4](
    vec3(-0.5, 0.0, -0.5),
    vec3(0.5, 0.0, -0.5),
    vec3(0.5, 0.0, 0.5),
    vec3(-0.5, 0.0, 0.5)
);

void main() {
    geomLocalCorner = QUAD_CORNERS[gl_VertexIndex & 3];
    gl_Position = vec4(geomLocalCorner, 1.0);
}
