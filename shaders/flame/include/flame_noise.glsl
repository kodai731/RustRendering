// flame_noise.glsl - procedural noise for the styled flame raymarch (mode 3).
// IGN (Jimenez 2014, "Next Generation Post Processing in Call of Duty") jitters
// the per-pixel ray start so 8-step marching dithers instead of banding.

#ifndef FLAME_NOISE_GLSL
#define FLAME_NOISE_GLSL

float interleavedGradientNoise(vec2 pixelCoord) {
    return fract(52.9829189 * fract(dot(pixelCoord, vec2(0.06711056, 0.00583715))));
}

float hashCell(vec3 cell) {
    vec3 p = fract(cell * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise3(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = hashCell(cell + vec3(0.0, 0.0, 0.0));
    float n100 = hashCell(cell + vec3(1.0, 0.0, 0.0));
    float n010 = hashCell(cell + vec3(0.0, 1.0, 0.0));
    float n110 = hashCell(cell + vec3(1.0, 1.0, 0.0));
    float n001 = hashCell(cell + vec3(0.0, 0.0, 1.0));
    float n101 = hashCell(cell + vec3(1.0, 0.0, 1.0));
    float n011 = hashCell(cell + vec3(0.0, 1.0, 1.0));
    float n111 = hashCell(cell + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

float fbm3(vec3 p) {
    float sum = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 3; ++octave) {
        sum += amplitude * valueNoise3(p);
        p = p * 2.02 + vec3(13.7);
        amplitude *= 0.5;
    }
    return sum;
}

#endif
