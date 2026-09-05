#ifndef WIND_SHELL_INTEGRAL_GLSL
#define WIND_SHELL_INTEGRAL_GLSL

// Closed-form optical depth of the shell field along a ray: the ray is cut at the
// shell support boundaries and the envelope break (knots), and inside each piece
// the density is one polynomial in the piece-local variable sigma in [0, 1] whose
// integral is a sum of power-rule moments.
// Mirrored in thyllore-effect-core/src/wind/analytic/shell_integral.rs.
// Must be included after wind_shell_field.glsl.

const int WIND_MAX_KNOTS = 12;
const int WIND_POLY_TERMS = 12;
const float WIND_EMPTY_INTERVAL_EPSILON = 1e-6;

void windPushKnot(inout float knots[WIND_MAX_KNOTS], inout int count, float t, float lo, float hi) {
    if (t <= lo || t >= hi || count >= WIND_MAX_KNOTS) {
        return;
    }
    knots[count] = t;
    count += 1;
}

void windPushQuadraticRoots(
    float a, float b, float c, float lo, float hi,
    inout float knots[WIND_MAX_KNOTS], inout int count) {
    if (abs(a) < WIND_LINEAR_COEFFICIENT_EPSILON) {
        if (abs(b) >= WIND_LINEAR_COEFFICIENT_EPSILON) {
            windPushKnot(knots, count, -c / b, lo, hi);
        }
        return;
    }
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return;
    }
    float sqrtDiscriminant = sqrt(discriminant);
    windPushKnot(knots, count, (-b - sqrtDiscriminant) / (2.0 * a), lo, hi);
    windPushKnot(knots, count, (-b + sqrtDiscriminant) / (2.0 * a), lo, hi);
}

void windSortKnots(inout float knots[WIND_MAX_KNOTS], int count) {
    for (int i = 1; i < count; ++i) {
        float value = knots[i];
        int j = i;
        while (j > 0 && knots[j - 1] > value) {
            knots[j] = knots[j - 1];
            j -= 1;
        }
        knots[j] = value;
    }
}

int windRayKnots(vec3 o, vec3 d, float tNear, float tFar, out float knots[WIND_MAX_KNOTS]) {
    int count = 2;
    knots[0] = tNear;
    knots[1] = tFar;

    float qA = dot(d.xz, d.xz);
    float qB = 2.0 * dot(o.xz, d.xz);
    float qC = dot(o.xz, o.xz);

    float invHeight = 1.0 / windHeight();
    float radius0 = windWallRadius(o.y * invHeight);
    float radius1 = windWallRadiusSlope() * d.y * invHeight;
    float deltaA = qA - radius1 * radius1;
    float deltaB = qB - 2.0 * radius0 * radius1;
    float deltaC = qC - radius0 * radius0;
    windPushQuadraticRoots(deltaA, deltaB, deltaC - windWallWidthQ(), tNear, tFar, knots, count);
    windPushQuadraticRoots(deltaA, deltaB, deltaC + windWallWidthQ(), tNear, tFar, knots, count);

    if (windCoreActive()) {
        windPushQuadraticRoots(qA, qB, qC - windCoreRadiusSq(), tNear, tFar, knots, count);
    }

    if (abs(d.y) >= WIND_LINEAR_COEFFICIENT_EPSILON) {
        float fadeY = windFadeStart() * windHeight();
        windPushKnot(knots, count, (fadeY - o.y) / d.y, tNear, tFar);
    }

    windSortKnots(knots, count);
    return count;
}

void windPolyMul(float a[WIND_POLY_TERMS], float b[WIND_POLY_TERMS], out float product[WIND_POLY_TERMS]) {
    for (int k = 0; k < WIND_POLY_TERMS; ++k) {
        product[k] = 0.0;
    }
    for (int i = 0; i < WIND_POLY_TERMS; ++i) {
        if (a[i] == 0.0) {
            continue;
        }
        for (int j = 0; i + j < WIND_POLY_TERMS; ++j) {
            product[i + j] += a[i] * b[j];
        }
    }
}

void windPolyFromQuadratic(float c0, float c1, float c2, out float poly[WIND_POLY_TERMS]) {
    for (int k = 0; k < WIND_POLY_TERMS; ++k) {
        poly[k] = 0.0;
    }
    poly[0] = c0;
    poly[1] = c1;
    poly[2] = c2;
}

void windBiweightPoly(float u[WIND_POLY_TERMS], out float result[WIND_POLY_TERMS]) {
    float inside[WIND_POLY_TERMS];
    windPolyMul(u, u, inside);
    for (int k = 0; k < WIND_POLY_TERMS; ++k) {
        inside[k] = -inside[k];
    }
    inside[0] += 1.0;
    windPolyMul(inside, inside, result);
}

void windEnvelopePoly(float h0, float h1, float hMid, out float envelope[WIND_POLY_TERMS]) {
    for (int k = 0; k < WIND_POLY_TERMS; ++k) {
        envelope[k] = 0.0;
    }
    float fadeStart = windFadeStart();
    if (hMid <= fadeStart) {
        envelope[0] = 1.0;
        return;
    }
    float v0 = (h0 - fadeStart) / windTopFade();
    float v1 = h1 / windTopFade();
    envelope[0] = 1.0 - 3.0 * v0 * v0 + 2.0 * v0 * v0 * v0;
    envelope[1] = -6.0 * v0 * v1 + 6.0 * v0 * v0 * v1;
    envelope[2] = -3.0 * v1 * v1 + 6.0 * v0 * v1 * v1;
    envelope[3] = 2.0 * v1 * v1 * v1;
}

float windPieceOpticalDepth(vec3 o, vec3 d, float s0, float s1) {
    float pieceLength = s1 - s0;
    if (pieceLength <= WIND_EMPTY_INTERVAL_EPSILON) {
        return 0.0;
    }
    vec3 start = o + d * s0;
    float invHeight = 1.0 / windHeight();
    float h0 = start.y * invHeight;
    float h1 = pieceLength * d.y * invHeight;
    float hMid = h0 + 0.5 * h1;
    if (hMid < 0.0 || hMid > 1.0) {
        return 0.0;
    }

    float q0 = dot(start.xz, start.xz);
    float q1 = 2.0 * pieceLength * dot(start.xz, d.xz);
    float q2 = pieceLength * pieceLength * dot(d.xz, d.xz);

    float radius0 = windWallRadius(h0);
    float radius1 = windWallRadiusSlope() * h1;
    float invWidth = 1.0 / windWallWidthQ();
    float u[WIND_POLY_TERMS];
    windPolyFromQuadratic(
        (q0 - radius0 * radius0) * invWidth,
        (q1 - 2.0 * radius0 * radius1) * invWidth,
        (q2 - radius1 * radius1) * invWidth,
        u);
    float uMid = u[0] + 0.5 * u[1] + 0.25 * u[2];

    float shell[WIND_POLY_TERMS];
    for (int k = 0; k < WIND_POLY_TERMS; ++k) {
        shell[k] = 0.0;
    }
    if (abs(uMid) < 1.0) {
        float wall[WIND_POLY_TERMS];
        windBiweightPoly(u, wall);
        for (int k = 0; k < WIND_POLY_TERMS; ++k) {
            shell[k] += windWallStrength() * wall[k];
        }
    }
    if (windCoreActive()) {
        float invCore = 1.0 / windCoreRadiusSq();
        float uc[WIND_POLY_TERMS];
        windPolyFromQuadratic(q0 * invCore, q1 * invCore, q2 * invCore, uc);
        float ucMid = uc[0] + 0.5 * uc[1] + 0.25 * uc[2];
        if (ucMid < 1.0) {
            float core[WIND_POLY_TERMS];
            windBiweightPoly(uc, core);
            for (int k = 0; k < WIND_POLY_TERMS; ++k) {
                shell[k] += windCoreStrength() * core[k];
            }
        }
    }

    float envelope[WIND_POLY_TERMS];
    windEnvelopePoly(h0, h1, hMid, envelope);
    float density[WIND_POLY_TERMS];
    windPolyMul(envelope, shell, density);
    float momentSum = 0.0;
    for (int n = 0; n < WIND_POLY_TERMS; ++n) {
        momentSum += density[n] / float(n + 1);
    }
    return max(pieceLength * windSigmaT() * momentSum, 0.0);
}

float windOpticalDepth(vec3 o, vec3 d, float tNear, float tFar, out int knotCount) {
    knotCount = 0;
    if (tFar <= tNear) {
        return 0.0;
    }
    float knots[WIND_MAX_KNOTS];
    knotCount = windRayKnots(o, d, tNear, tFar, knots);
    float total = 0.0;
    for (int i = 1; i < knotCount; ++i) {
        total += windPieceOpticalDepth(o, d, knots[i - 1], knots[i]);
    }
    return total;
}

#endif
