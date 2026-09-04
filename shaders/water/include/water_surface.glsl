#ifndef WATER_SURFACE_GLSL
#define WATER_SURFACE_GLSL

float sinc(float x) {
    if (abs(x) < 1e-6) return 1.0;
    return sin(x) / x;
}

void waterHeightAndGradient(vec2 uv, float time, vec2 flowRate, int modeCount, vec2 footprint, out float h, out float hu, out float hv, out float slopeVariance) {
    h = 0.0;
    hu = 0.0;
    hv = 0.0;
    slopeVariance = 0.0;

    float u = uv.x;
    float v = uv.y;
    float a = flowRate.x;
    float b = flowRate.y;
    float rHat = water.radii.y / water.radii.x;
    float rho = 1.0 + rHat * cos(v);

    for (int k = 0; k < 8; k++) {
        if (k >= modeCount) break;

        int m = int(water.waveModes[k * 2].x);
        int n = int(water.waveModes[k * 2].y);
        float amp = water.waveModes[k * 2].z;
        float omega = water.waveModes[k * 2].w;
        float phase = water.waveModes[k * 2 + 1].x;
        float ampN = amp / water.radii.x;

        float phasePrime = m * (u + a * time) + n * (v + b * time) - omega * time + phase;
        float cosVal = cos(phasePrime);
        float sinVal = sin(phasePrime);

        float sincM = sinc(m * footprint.x);
        float sincN = sinc(n * footprint.y);
        float sincProduct = sincM * sincN;

        h += amp * cosVal * sincProduct;
        hu -= amp * m * sinVal * sincProduct;
        hv -= amp * n * sinVal * sincProduct;

        slopeVariance += ampN * ampN * (m * m / (rho * rho) + n * n / (rHat * rHat)) * (1.0 - sincProduct * sincProduct) * 0.5;
    }
}

vec3 waterPerturbedNormal(float u, float v, float h, float hu, float hv, float rHat) {
    float cosU = cos(u);
    float sinU = sin(u);
    float cosV = cos(v);
    float sinV = sin(v);

    vec3 e_u = vec3(-sinU, 0.0, cosU);
    vec3 e_v = vec3(-sinV * cosU, cosV, -sinV * sinU);
    vec3 n = vec3(cosV * cosU, sinV, cosV * sinU);

    float kappa1 = 1.0 / rHat;
    float kappa2 = cosV / (1.0 + rHat * cosV);

    float scaledH = h / water.radii.x;
    float scaledHu = hu / water.radii.x;
    float scaledHv = hv / water.radii.x;

    vec3 nPrime = (1.0 + scaledH * kappa1) * (1.0 + scaledH * kappa2) * n
        - (1.0 + scaledH * kappa1) * scaledHu / (1.0 + rHat * cosV) * e_u
        - (1.0 + scaledH * kappa2) * scaledHv / rHat * e_v;

    return normalize(nPrime);
}

#endif
