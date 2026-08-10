// Reference raymarch integrators (debug / A-B comparison only, push.mode 1 and 3).
// Deliberately quarantined from the analytic includes: these are the only
// sample-based integrators allowed to exist, and the closed-form guard checks
// that no other file grows a t-lattice loop. They call the same point
// evaluators as the analytic path so the pair stays comparable.
float integrateEmissionRaymarch(FlameRaySegment segment, int stepCount) {
    float dt = (segment.tFar - segment.tNear) / float(stepCount);
    if (dt <= 0.0) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + 0.5) * dt;
        vec3 p = segment.localOrigin + t * segment.localDir;
        float h = clamp(
            evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        float w = flameContourWiggle(p, h);
        if (segment.cylinderDomain && flame.noiseAmplitude != 0.0) {
            sum += flamePointOccupancyDensity(p, h, w);
        } else if (!segment.cylinderDomain && flame.trailMeta.x < 1.0) {
            sum += flamePointEmitterOccupancy(p, h, w);
        } else {
            float radial = segment.cylinderDomain ? flameRadialDensityFactor(vec3(p.x / w, p.y, p.z / w), h) : 1.0;
            sum += evaluateHeightFalloff(h) * radial * flameNoiseErosionFactor(p, h);
        }
    }
    return sum * dt;
}

vec4 integrateRTERaymarch(FlameRaySegment segment, int stepCount) {
    float dt = (segment.tFar - segment.tNear) / float(stepCount);
    if (dt <= 0.0) {
        return vec4(0.0);
    }
    float total = 0.0;
    float heightMean = 0.0;
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + 0.5) * dt;
        vec3 p = segment.localOrigin + t * segment.localDir;
        float h = clamp(evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        float w = flameContourWiggle(p, h);
        float rho;
        if (!segment.cylinderDomain) {
            rho = flamePointEmitterOccupancy(p, h, w);
        } else if (flame.noiseAmplitude != 0.0) {
            rho = flamePointOccupancyDensity(p, h, w);
        } else {
            rho = evaluateHeightFalloff(h)
                * flameRadialDensityFactor(vec3(p.x / w, p.y, p.z / w), h)
                * flameNoiseErosionFactor(p, h);
        }
        total += rho * dt;
        heightMean += rho * dt * h;
    }
    heightMean = total > 1e-6 ? heightMean / total : 0.0;
    float tempNorm = clamp(total * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMean);
    float boost = 1.0 + flame.styleParams1.w * tempNorm * tempNorm;

    vec3 radiance = vec3(0.0);
    vec3 sigmaRgb = flame.sigmaT * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.w, 0.0, 1.0));
    vec3 transmittance = vec3(1.0);
    for (int i = 0; i < stepCount; ++i) {
        float t = segment.tNear + (float(i) + 0.5) * dt;
        vec3 p = segment.localOrigin + t * segment.localDir;
        float h = clamp(evaluateHeightAlongRay(t, segment.localOrigin.y, segment.localDir.y), 0.0, 1.0);
        float w = flameContourWiggle(p, h);
        float rho;
        if (!segment.cylinderDomain) {
            rho = flamePointEmitterOccupancy(p, h, w);
        } else if (flame.noiseAmplitude != 0.0) {
            rho = flamePointOccupancyDensity(p, h, w);
        } else {
            rho = evaluateHeightFalloff(h)
                * flameRadialDensityFactor(vec3(p.x / w, p.y, p.z / w), h)
                * flameNoiseErosionFactor(p, h);
        }
        vec3 tau = sigmaRgb * rho * dt;
        radiance += transmittance * flameRampColor(h) * flame.intensity * boost * (vec3(1.0) - exp(-tau));
        transmittance *= exp(-tau);
    }
    return vec4(radiance, 1.0 - dot(transmittance, vec3(1.0 / 3.0)));
}
