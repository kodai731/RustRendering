#ifndef FLAME_BRANCH_TRANSPORT_GLSL
#define FLAME_BRANCH_TRANSPORT_GLSL

#include "include/common.glsl"

// Branch element layer (A: vortex transport): every live element is a (tilted)
// vortex line; each perpendicular slice rotates about it by a windowed Lamb-Oseen
// angle, compact inside rho < reach, so the map is a bijection for any gain.
// Mirrored in thyllore-effect-core/src/flame/branch.rs.

bool flameBranchActive() {
    return flame.branchField.count > 0.5;
}

float flameBranchSmoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Ease-out winding over the first WIND_FRACTION of the life (fastest at birth,
// decelerating to rest), hold, then an unwind over envelopeTime so the map is the
// identity at death; the unwind is hidden outside the trunk by the burnout mask.
float flameBranchEnvelope(float age) {
    float life = flame.branchField.life;
    float envelopeTime = flame.branchField.envelopeTime;
    float windFraction = flame.branchField.ageProfile.windFraction;
    float t = clamp(age / max(windFraction * life, 1e-3), 0.0, 1.0);
    float easeOut = 1.0 - (1.0 - t) * (1.0 - t);
    return easeOut * (1.0 - flameBranchSmoothstep(life - envelopeTime, life, age));
}

// Burnout strength: rises from half the life to 1 when the unwind starts and
// releases in the last part of the unwind, when the remaining rotation is
// negligible, so the mask never jumps at death.
float flameBranchBurnout(float age) {
    float life = flame.branchField.life;
    float envelopeTime = flame.branchField.envelopeTime;
    FlameBranchAgeProfile profile = flame.branchField.ageProfile;
    float unwindStart = life - envelopeTime;
    float releaseStart = life - profile.burnoutReleaseFraction * envelopeTime;
    return flameBranchSmoothstep(profile.burnoutStartFraction * life, unwindStart, age)
        * (1.0 - flameBranchSmoothstep(releaseStart, life, age));
}

// (1 - exp(-rho^2 / rc^2)) / (2 pi rho^2) and its derivative in rho^2.
vec2 flameBranchLambOseen(float rhoSq, float coreRadius) {
    float coreSq = coreRadius * coreRadius;
    float x = rhoSq / coreSq;
    if (x < 1e-3) {
        return vec2(
            (1.0 - 0.5 * x + x * x / 6.0) / (TWO_PI * coreSq),
            (-0.5 + x / 3.0) / (TWO_PI * coreSq * coreSq));
    }
    float decay = exp(-x);
    return vec2(
        (1.0 - decay) / (TWO_PI * rhoSq),
        (decay * x - (1.0 - decay)) / (TWO_PI * rhoSq * rhoSq));
}

bool flameVortexElementAt(int index, out FlameVortexElement element) {
    FlameBranchElement spawn = flame.branchField.elements[index];
    float age = flame.time - spawn.spawnTime;
    if (age < 0.0 || age >= flame.branchField.life) {
        return false;
    }
    float sinAz = sin(spawn.azimuth);
    float cosAz = cos(spawn.azimuth);
    float lateral = spawn.side * spawn.trunkRadius
        * (flame.branchField.coreOffset + flame.branchField.driftRate * age);
    element.center = vec3(
        lateral * cosAz,
        spawn.spawnHeight + flame.branchField.riseRate * age,
        lateral * sinAz);
    float sinTilt = sin(spawn.tilt);
    float cosTilt = cos(spawn.tilt);
    vec3 horizontalLine = vec3(-sinAz, 0.0, cosAz);
    element.outward = vec3(cosAz, 0.0, sinAz);
    element.line = vec3(cosTilt * horizontalLine.x, sinTilt, cosTilt * horizontalLine.z);
    element.up = vec3(-sinTilt * horizontalLine.x, cosTilt, -sinTilt * horizontalLine.z);

    float progress = age / flame.branchField.life;
    float reachRatio = flame.branchField.reachStart
        + (flame.branchField.reachEnd - flame.branchField.reachStart) * progress;
    float scale = spawn.trunkRadius * spawn.size;
    element.reach = reachRatio * scale;
    element.coreRadius = flame.branchField.coreRadius * scale;
    element.circulation = spawn.side * flame.branchField.gain * TWO_PI
        * element.coreRadius * element.coreRadius * flameBranchEnvelope(age);
    element.alongOffset = spawn.alongOffset;
    return true;
}

// Isotropic offset of p from the element center (y scaled by aspect).
vec3 flameVortexIsotropicOffset(FlameVortexElement element, vec3 p) {
    return vec3(
        p.x - element.center.x,
        (p.y - element.center.y) * flame.branchField.aspect,
        p.z - element.center.z);
}

// (u, along, v) frame coordinates of an isotropic offset.
vec3 flameVortexFrameCoordinates(FlameVortexElement element, vec3 q) {
    return vec3(
        dot(q, element.outward),
        dot(q, element.line) - element.alongOffset * element.reach,
        dot(q, element.up));
}

// Each slice perpendicular to the vortex line rotates about the line by the
// Lamb-Oseen angle gated by a ball rho^2 + along^2 < reach^2 around the element
// center; unit determinant, and the tongue's boundary stays round from every view.
vec3 flameVortexPullBackJvp(FlameVortexElement element, vec3 p, inout vec3 dir) {
    float aspect = flame.branchField.aspect;
    vec3 frameCoords = flameVortexFrameCoordinates(element, flameVortexIsotropicOffset(element, p));
    float u = frameCoords.x;
    float along = frameCoords.y;
    float v = frameCoords.z;
    float reach = element.reach;
    float reachSq = reach * reach;
    float rhoSq = u * u + v * v;
    float s = (rhoSq + along * along) / reachSq;
    if (s >= 1.0) {
        return p;
    }

    float gate = (1.0 - s) * (1.0 - s);
    vec2 profile = flameBranchLambOseen(rhoSq, element.coreRadius);
    float circulation = element.circulation;
    float psi = circulation * gate * profile.x;

    vec3 dq = vec3(dir.x, dir.y * aspect, dir.z);
    float du = dot(dq, element.outward);
    float dAlong = dot(dq, element.line);
    float dv = dot(dq, element.up);
    float dRhoSq = 2.0 * (u * du + v * dv);
    float dS = (dRhoSq + 2.0 * along * dAlong) / reachSq;
    float dGate = -2.0 * (1.0 - s) * dS;
    float dPsi = circulation * (dGate * profile.x + gate * profile.y * dRhoSq);

    float sn = sin(psi);
    float cs = cos(psi);
    float u1 = u * cs - v * sn;
    float v1 = u * sn + v * cs;
    float du1 = du * cs - dv * sn - dPsi * v1;
    float dv1 = du * sn + dv * cs + dPsi * u1;
    float alongTotal = along + element.alongOffset * reach;
    vec3 moved = u1 * element.outward + alongTotal * element.line + v1 * element.up;
    vec3 movedDir = du1 * element.outward + dAlong * element.line + dv1 * element.up;
    dir = vec3(movedDir.x, movedDir.y / aspect, movedDir.z);
    return element.center + vec3(moved.x, moved.y / aspect, moved.z);
}

// Density mask of one element at trunk-local p (before the pull-back): a plateau
// over the element's disc that only bites the medium outside the trunk, so the
// tongue dims away in place while the trunk keeps its material.
float flameVortexBurnoutMask(FlameVortexElement element, float burnout, float trunkRadius, vec3 p) {
    vec3 frameCoords = flameVortexFrameCoordinates(element, flameVortexIsotropicOffset(element, p));
    float u = frameCoords.x;
    float along = frameCoords.y;
    float v = frameCoords.z;
    FlameBranchAgeProfile profile = flame.branchField.ageProfile;
    float reach = max(element.reach, 1e-4);
    float outer = 1.0 + profile.burnoutMargin;
    float radius = sqrt(u * u + v * v + along * along) / reach;
    float plateau = 1.0 - flameBranchSmoothstep(1.0, outer, radius);

    float axisRadius = length(p.xz) / max(trunkRadius, 1e-4);
    float outsideTrunk = flameBranchSmoothstep(profile.burnoutTrunkInner, 1.0, axisRadius);
    return 1.0 - burnout * plateau * outsideTrunk;
}

// Product of the burnout masks of every live element at trunk-local p.
float flameBranchBurnoutMask(vec3 p) {
    int count = min(int(flame.branchField.count), FLAME_BRANCH_MAX_ELEMENTS);
    float mask = 1.0;
    for (int i = 0; i < count; ++i) {
        FlameVortexElement element;
        if (flameVortexElementAt(i, element)) {
            FlameBranchElement spawn = flame.branchField.elements[i];
            float burnout = flameBranchBurnout(flame.time - spawn.spawnTime);
            mask *= flameVortexBurnoutMask(element, burnout, spawn.trunkRadius, p);
        }
    }
    return mask;
}

// Composite pull-back through the live elements (table is newest first) with
// the Jacobian-vector product of `dir` carried along.
vec3 flameBranchPullBackJvp(vec3 p, inout vec3 dir) {
    int count = min(int(flame.branchField.count), FLAME_BRANCH_MAX_ELEMENTS);
    for (int i = 0; i < count; ++i) {
        FlameVortexElement element;
        if (flameVortexElementAt(i, element)) {
            p = flameVortexPullBackJvp(element, p, dir);
        }
    }
    return p;
}

vec3 flameBranchPullBack(vec3 p) {
    vec3 dir = vec3(0.0);
    return flameBranchPullBackJvp(p, dir);
}

vec3 flameBranchDebugHue(float t) {
    return clamp(abs(fract(t + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

// Debug view: the element displacing this trunk-local sample the most, hued by
// its stable hash, brightened by the displacement (in core radii) and whitened
// inside the core; dimmed where the smooth density stays below the occupancy
// threshold (transported tail that the render never shows); untouched samples
// show the smooth density in grey.
vec3 flameBranchDebugColor(vec3 ps, float density) {
    int count = min(int(flame.branchField.count), FLAME_BRANCH_MAX_ELEMENTS);
    float bestDisplacement = 0.0;
    float bestHash = 0.0;
    float bestCoreRadius = 1.0;
    bool insideCore = false;
    for (int i = 0; i < count; ++i) {
        FlameVortexElement element;
        if (!flameVortexElementAt(i, element)) {
            continue;
        }
        vec3 dir = vec3(0.0);
        float displacement = length(flameVortexPullBackJvp(element, ps, dir) - ps);
        if (displacement > bestDisplacement) {
            bestDisplacement = displacement;
            bestHash = flame.branchField.elements[i].hash01;
            bestCoreRadius = element.coreRadius;
            vec3 frameCoords = flameVortexFrameCoordinates(element, flameVortexIsotropicOffset(element, ps));
            insideCore = frameCoords.x * frameCoords.x + frameCoords.z * frameCoords.z
                < element.coreRadius * element.coreRadius;
        }
    }
    if (bestDisplacement <= 1e-5) {
        return vec3(0.35 * clamp(density, 0.0, 1.0));
    }
    float strength = clamp(bestDisplacement / bestCoreRadius, 0.0, 1.0);
    vec3 color = flameBranchDebugHue(bestHash) * mix(0.3, 1.0, strength);
    color = insideCore ? mix(color, vec3(1.0), 0.6) : color;
    float visible = flameBranchSmoothstep(
        flame.nearFadeParams.edgeLow, flame.nearFadeParams.edgeHigh, density);
    return color * mix(0.12, 1.0, visible);
}

#endif
