#ifndef WATER_LIGHTING_GLSL
#define WATER_LIGHTING_GLSL

#include "include/radiative_transfer.glsl"
#include "include/torus_intersect.glsl"

#define WATER_SCATTER_SAMPLES 4

float waterFresnelReflectance(float cosThetaI, float eta) {
    float sinThetaT2 = (1.0 - cosThetaI * cosThetaI) / (eta * eta);
    float cosThetaT = sqrt(max(1.0 - sinThetaT2, 0.0));
    float rPar = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return (rPar * rPar + rPerp * rPerp) * 0.5;
}

vec3 waterScatteringCoefficient() {
    return water.tint.rgb * water.lighting.w;
}

vec3 waterExtinctionCoefficient() {
    return water.absorption.rgb + waterScatteringCoefficient();
}

vec3 waterEnvironmentReflection(vec3 reflDir, vec3 lightDir, vec3 lightColor, float slopeVariance) {
    float sharpness = water.lighting.y / (1.0 + water.lighting.y * slopeVariance);
    float spec = pow(max(dot(reflDir, lightDir), 0.0), sharpness);
    return vec3(0.6, 0.7, 0.8) * water.lighting.z + lightColor * water.lighting.x * spec;
}

vec3 waterTransmittedHighlight(vec3 exitDir, vec3 lightDir, vec3 lightColor, float chord) {
    float spec = pow(max(dot(exitDir, lightDir), 0.0), water.lighting.y);
    return lightColor * water.lighting.x * spec * rteTransmittance(waterExtinctionCoefficient(), chord);
}

struct WaterScatterSample {
    vec3 position;
    float viewDistance;
    vec3 lightDir;
    vec3 lightExitPoint;
    float waterDistance;
    float surfaceTransmission;
};

// Light path from an interior point to the light: straight line, first exit through the surface,
// plus the ring chord if the line re-enters the torus (self-occlusion); Snell bending is ignored.
WaterScatterSample waterScatterSampleAt(vec3 entry, vec3 exit, int index, vec3 lightPos) {
    WaterScatterSample smp;
    float pathLength = length(exit - entry);
    smp.viewDistance = rteMidpointDistance(index, WATER_SCATTER_SAMPLES, pathLength);
    smp.position = entry + (exit - entry) * (smp.viewDistance / pathLength);
    smp.lightDir = normalize(lightPos - smp.position);

    float rHat = water.radii.y / water.radii.x;
    vec3 originLocal = (water.inverseModel * vec4(smp.position, 1.0)).xyz / water.radii.x;
    vec3 dirLocal = normalize((water.inverseModel * vec4(smp.lightDir, 0.0)).xyz);
    float firstExit = torusExitFromInside(originLocal, dirLocal, rHat);
    vec3 exitLocal = originLocal + dirLocal * firstExit;
    float lastExit = firstExit;
    float insideDistance = firstExit;

    float ringEntry = torusEntryFromOutside(exitLocal + dirLocal * 1e-3, dirLocal, rHat);
    if (ringEntry > 0.0) {
        float ringStart = firstExit + 1e-3 + ringEntry;
        float ringChord = torusExitFromInside(originLocal + dirLocal * (ringStart + 1e-3), dirLocal, rHat);
        insideDistance += ringChord;
        lastExit = ringStart + 1e-3 + ringChord;
    }
    smp.waterDistance = insideDistance * water.radii.x;
    smp.lightExitPoint = smp.position + smp.lightDir * (lastExit * water.radii.x);

    vec3 nExit = normalize(mat3(water.model) * torusGradient(exitLocal, rHat));
    smp.surfaceTransmission = 1.0 - waterFresnelReflectance(max(dot(nExit, smp.lightDir), 0.0), water.absorption.w);
    return smp;
}

vec3 waterScatterSampleRadiance(WaterScatterSample smp, vec3 viewDir, vec3 lightColor, float ds) {
    vec3 sigmaS = waterScatteringCoefficient();
    vec3 sigmaT = waterExtinctionCoefficient();
    vec3 lightRadiance = lightColor * water.lighting.x * smp.surfaceTransmission * rteTransmittance(sigmaT, smp.waterDistance);
    float phase = rteHenyeyGreenstein(dot(smp.lightDir, viewDir), water.scattering.x);
    return rteSingleScatterSample(sigmaS, sigmaT, phase, smp.viewDistance, ds, lightRadiance);
}

#endif
