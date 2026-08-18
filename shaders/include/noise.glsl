// noise.glsl - hash / value noise / fbm / IGN jitter (ALU only, no textures)

#ifndef NOISE_GLSL
#define NOISE_GLSL

uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

float hash13(vec3 p) {
    uvec3 h = pcg3d(floatBitsToUint(p));
    return float(h.x) * (1.0 / 4294967296.0);
}

float valueNoise3(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = p - cell;
    vec3 w = f * f * (3.0 - 2.0 * f);

    float n000 = hash13(cell + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(cell + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(cell + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(cell + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(cell + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(cell + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(cell + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(cell + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, w.x);
    float nx10 = mix(n010, n110, w.x);
    float nx01 = mix(n001, n101, w.x);
    float nx11 = mix(n011, n111, w.x);
    float nxy0 = mix(nx00, nx10, w.y);
    float nxy1 = mix(nx01, nx11, w.y);
    return mix(nxy0, nxy1, w.z);
}

float fbm3(vec3 p) {
    float sum = 0.5 * valueNoise3(p);
    sum += 0.25 * valueNoise3(2.0 * p + vec3(17.3, 9.1, 4.7));
    sum += 0.125 * valueNoise3(4.0 * p + vec3(31.7, 2.9, 12.3));
    return sum * (1.0 / 0.875);
}

float interleavedGradientNoise(vec2 fragCoord) {
    return fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
}

#endif
