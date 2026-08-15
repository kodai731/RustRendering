#ifndef FLAME_BRANCH_TRANSPORT_GLSL
#define FLAME_BRANCH_TRANSPORT_GLSL

// Branch element layer (A: vortex transport): every live element is a horizontal
// vortex line; each perpendicular slice rotates about it by a windowed Lamb-Oseen
// angle, compact inside rho < reach, so the map is a bijection for any gain.
// Mirrored in thyllore-effect-core/src/flame/branch.rs.
const float FLAME_BRANCH_TAU = 6.283185307;
// Age-profile constants, mirrored from flame/constants.rs (BRANCH_BURNOUT_*).
const float FLAME_BRANCH_BURNOUT_START_FRACTION = 0.5;
const float FLAME_BRANCH_BURNOUT_RELEASE_FRACTION = 0.1;
const float FLAME_BRANCH_BURNOUT_MARGIN = 0.5;
const float FLAME_BRANCH_BURNOUT_TRUNK_INNER = 0.75;

bool flameBranchActive() {
    return flame.branchField.count > 0.5;
}

float flameBranchSmoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Ease-out winding (fastest at birth, decelerating to rest), then an unwind over
// envelopeTime so the map is the identity at death; the unwind is hidden outside
// the trunk by the burnout mask.
float flameBranchEnvelope(float age) {
    float life = flame.branchField.life;
    float envelopeTime = flame.branchField.envelopeTime;
    float t = clamp(age / max(life - envelopeTime, 1e-3), 0.0, 1.0);
    float easeOut = 1.0 - (1.0 - t) * (1.0 - t);
    return easeOut * (1.0 - flameBranchSmoothstep(life - envelopeTime, life, age));
}

// Burnout strength: rises from half the life to 1 when the unwind starts and
// releases in the last part of the unwind, when the remaining rotation is
// negligible, so the mask never jumps at death.
float flameBranchBurnout(float age) {
    float life = flame.branchField.life;
    float envelopeTime = flame.branchField.envelopeTime;
    float unwindStart = life - envelopeTime;
    float releaseStart = life - FLAME_BRANCH_BURNOUT_RELEASE_FRACTION * envelopeTime;
    return flameBranchSmoothstep(FLAME_BRANCH_BURNOUT_START_FRACTION * life, unwindStart, age)
        * (1.0 - flameBranchSmoothstep(releaseStart, life, age));
}

// (1 - exp(-rho^2 / rc^2)) / (2 pi rho^2) and its derivative in rho^2.
vec2 flameBranchLambOseen(float rhoSq, float coreRadius) {
    float coreSq = coreRadius * coreRadius;
    float x = rhoSq / coreSq;
    if (x < 1e-3) {
        return vec2(
            (1.0 - 0.5 * x + x * x / 6.0) / (FLAME_BRANCH_TAU * coreSq),
            (-0.5 + x / 3.0) / (FLAME_BRANCH_TAU * coreSq * coreSq));
    }
    float decay = exp(-x);
    return vec2(
        (1.0 - decay) / (FLAME_BRANCH_TAU * rhoSq),
        (decay * x - (1.0 - decay)) / (FLAME_BRANCH_TAU * rhoSq * rhoSq));
}

struct FlameVortexElement {
    vec3 center;
    vec2 inPlane;
    float reach;
    float coreRadius;
    float circulation;
};

bool flameVortexElementAt(int index, out FlameVortexElement element) {
    FlameBranchElement spawn = flame.branchField.elements[index];
    float age = flame.time - spawn.spawnTime;
    if (age < 0.0 || age >= flame.branchField.life) {
        return false;
    }
    element.inPlane = vec2(cos(spawn.azimuth), sin(spawn.azimuth));
    float lateral = spawn.side * spawn.trunkRadius
        * (flame.branchField.coreOffset + flame.branchField.driftRate * age);
    element.center = vec3(
        lateral * element.inPlane.x,
        spawn.spawnHeight + flame.branchField.riseRate * age,
        lateral * element.inPlane.y);
    float progress = age / flame.branchField.life;
    float reachRatio = flame.branchField.reachStart
        + (flame.branchField.reachEnd - flame.branchField.reachStart) * progress;
    element.reach = reachRatio * spawn.trunkRadius;
    element.coreRadius = flame.branchField.coreRadius * spawn.trunkRadius;
    element.circulation = spawn.side * flame.branchField.gain * FLAME_BRANCH_TAU
        * element.coreRadius * element.coreRadius * flameBranchEnvelope(age);
    return true;
}

// Each slice perpendicular to the horizontal vortex line rotates about the line
// by the windowed Lamb-Oseen angle; compact inside rho < reach, unit determinant.
vec3 flameVortexPullBackJvp(FlameVortexElement element, vec3 p, inout vec3 dir) {
    float aspect = flame.branchField.aspect;
    float ex = element.inPlane.x;
    float ez = element.inPlane.y;
    float qx = p.x - element.center.x;
    float qz = p.z - element.center.z;
    float u = qx * ex + qz * ez;
    float along = -qx * ez + qz * ex;
    float v = (p.y - element.center.y) * aspect;
    float reach = element.reach;
    float reachSq = reach * reach;
    float rhoSq = u * u + v * v;
    if (rhoSq >= reachSq) {
        return p;
    }
    float x = along / reach;
    if (abs(x) >= 1.0) {
        return p;
    }

    float window = (1.0 - x * x) * (1.0 - x * x);
    float s = rhoSq / reachSq;
    float gate = (1.0 - s) * (1.0 - s);
    vec2 profile = flameBranchLambOseen(rhoSq, element.coreRadius);
    float circulation = element.circulation;
    float psi = circulation * window * gate * profile.x;

    float du = dir.x * ex + dir.z * ez;
    float dAlong = -dir.x * ez + dir.z * ex;
    float dv = dir.y * aspect;
    float dRhoSq = 2.0 * (u * du + v * dv);
    float dWindow = -4.0 * x * (1.0 - x * x) * dAlong / reach;
    float dGate = -2.0 * (1.0 - s) * dRhoSq / reachSq;
    float dPsi = circulation
        * (dWindow * gate * profile.x + window * dGate * profile.x + window * gate * profile.y * dRhoSq);

    float sn = sin(psi);
    float cs = cos(psi);
    float u1 = u * cs - v * sn;
    float v1 = u * sn + v * cs;
    float du1 = du * cs - dv * sn - dPsi * v1;
    float dv1 = du * sn + dv * cs + dPsi * u1;
    dir = vec3(du1 * ex - dAlong * ez, dv1 / aspect, du1 * ez + dAlong * ex);
    return vec3(
        element.center.x + u1 * ex - along * ez,
        element.center.y + v1 / aspect,
        element.center.z + u1 * ez + along * ex);
}

// Density mask of one element at trunk-local p (before the pull-back): a plateau
// over the element's disc that only bites the medium outside the trunk, so the
// tongue dims away in place while the trunk keeps its material.
float flameVortexBurnoutMask(FlameVortexElement element, float burnout, float trunkRadius, vec3 p) {
    float ex = element.inPlane.x;
    float ez = element.inPlane.y;
    float qx = p.x - element.center.x;
    float qz = p.z - element.center.z;
    float u = qx * ex + qz * ez;
    float along = -qx * ez + qz * ex;
    float v = (p.y - element.center.y) * flame.branchField.aspect;
    float reach = max(element.reach, 1e-4);
    float outer = 1.0 + FLAME_BRANCH_BURNOUT_MARGIN;
    float rho = sqrt(u * u + v * v) / reach;
    float plateau = (1.0 - flameBranchSmoothstep(1.0, outer, rho))
        * (1.0 - flameBranchSmoothstep(1.0, outer, abs(along) / reach));

    float axisRadius = length(p.xz) / max(trunkRadius, 1e-4);
    float outsideTrunk = flameBranchSmoothstep(FLAME_BRANCH_BURNOUT_TRUNK_INNER, 1.0, axisRadius);
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
// inside the ring core; untouched samples show the smooth density in grey.
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
            vec2 q = ps.xz - element.center.xz;
            float radial = dot(q, element.inPlane);
            float axial = (ps.y - element.center.y) * flame.branchField.aspect;
            insideCore = radial * radial + axial * axial < element.coreRadius * element.coreRadius;
        }
    }
    if (bestDisplacement <= 1e-5) {
        return vec3(0.35 * clamp(density, 0.0, 1.0));
    }
    float strength = clamp(bestDisplacement / bestCoreRadius, 0.0, 1.0);
    vec3 color = flameBranchDebugHue(bestHash) * mix(0.3, 1.0, strength);
    return insideCore ? mix(color, vec3(1.0), 0.6) : color;
}

#endif
