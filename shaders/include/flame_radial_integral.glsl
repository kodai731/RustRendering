#ifndef FLAME_RADIAL_INTEGRAL_GLSL
#define FLAME_RADIAL_INTEGRAL_GLSL

// Closed-form emission integral of the compact-support radial density
//   rho(p) = F(h) * (1 - u^2)^2,  u = |p.xz| / (S * R(h)),  zero for u >= 1
// over a ray segment. Along the ray u^2(s) is a quadratic g(s); the height
// envelope F is a degree-7 polynomial (Chebyshev series) and capFade a piecewise
// cubic in raw h, so within one piece the whole integrand is a single polynomial
// in the piece-local variable and every band integral is exact power-rule
// moments — no within-band approximation, hence no band-resolution limit on the
// envelope (the upper-band stripe artifact of the former 3-point quadratic).
// Pieces and occupancy nodes are cut at field-fixed knots (the capFade bounds,
// the displaced tip and the CPU-computed envelope edge crossings), which enter
// and leave the support interval with zero contribution — continuous per ray,
// no per-ray integer switches.
//
// Must be included after FlameUBO, evaluateHeightFalloff, and flame_noise_field.glsl
// (flameBiweight / flameRadialSupportRadius live there).
// Mirrored in thyllore-render-core/src/flame_radial.rs; the accuracy tests live there.

const int FLAME_RADIAL_BAND_COUNT = 6;
const float FLAME_RADIAL_MIN_DIR_Y = 1e-4;
const float FLAME_RADIAL_MIN_HEIGHT_SPAN = 1e-5;
const int FLAME_ENVELOPE_KNOT_COUNT = 8;

// Interval where a*s^2 + b*s + c <= 1 clipped to [-halfWidth, halfWidth].
// Citardauq-form roots keep grazing rays (discriminant near zero) precise.
// Returns lo > hi when the support is empty.
vec2 flameSupportInterval(float a, float b, float c, float halfWidth) {
    if (a < 1e-12) {
        return c <= 1.0 ? vec2(-halfWidth, halfWidth) : vec2(1.0, -1.0);
    }
    float discriminant = b * b - 4.0 * a * (c - 1.0);
    if (discriminant <= 0.0) {
        return vec2(1.0, -1.0);
    }
    float root = sqrt(discriminant);
    float gauge = -0.5 * (b + (b >= 0.0 ? root : -root));
    float sFirst = gauge / a;
    float sSecond = (c - 1.0) / gauge;
    return vec2(
        max(min(sFirst, sSecond), -halfWidth),
        min(max(sFirst, sSecond), halfWidth));
}

// Integral (.x) and first moment (.y) of (f.x + f.y s + f.z s^2) * (1 - g(s))^2 over
// the part of [-halfWidth, halfWidth] inside the support g(s) = a s^2 + b s + c.
// Kept for the along-ray fallback, whose envelope is constant over the segment.
vec2 flameBiweightBandEmission(float a, float b, float c, vec3 f, float halfWidth) {
    vec2 interval = flameSupportInterval(a, b, c, halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }

    float m = 1.0 - c;
    float w0 = m * m;
    float w1 = -2.0 * b * m;
    float w2 = b * b - 2.0 * a * m;
    float w3 = 2.0 * a * b;
    float w4 = a * a;
    float e[7] = float[](
        f.x * w0,
        f.x * w1 + f.y * w0,
        f.x * w2 + f.y * w1 + f.z * w0,
        f.x * w3 + f.y * w2 + f.z * w1,
        f.x * w4 + f.y * w3 + f.z * w2,
        f.y * w4 + f.z * w3,
        f.z * w4);

    float powerLo = 1.0;
    float powerHi = 1.0;
    float moments[8];
    for (int n = 0; n < 8; ++n) {
        powerLo *= interval.x;
        powerHi *= interval.y;
        moments[n] = (powerHi - powerLo) / float(n + 1);
    }

    vec2 result = vec2(0.0);
    for (int n = 0; n < 7; ++n) {
        result += e[n] * vec2(moments[n], moments[n + 1]);
    }
    return result;
}

// Std of the unresolved erosion fluctuation inside one piece, in argument units.
// FLAME_EROSION_NOISE_SIGMA is the std of e(s) - e(node) for decorrelated
// fbm3 samples: value-noise std ~0.185, octave amplitudes (0.5, 0.25, 0.125) give
// fbm std ~0.106, and the difference of two decorrelated samples scales by sqrt(2).
// Chords shorter than one noise cell stay correlated, so sigma ramps in linearly —
// it vanishes as the nodes resolve the field.
const float FLAME_EROSION_NOISE_SIGMA = 0.15;

float flameErosionSigma(float height01, float chordLength) {
    float decorrelation = min(chordLength * flame.noiseFrequency, 1.0);
    return abs(flame.noiseAmplitude) * mix(0.2, 1.0, height01)
        * FLAME_EROSION_NOISE_SIGMA * decorrelation;
}

// ---- Exact envelope machinery (band-free integration) ----

// F monomial (in u = 2 hb - 1, from the UBO) affine-composed into the piece
// frame: u = uScale * zeta + uBias.
void flameEnvelopeMonomialLocal(float uScale, float uBias, out float target[8]) {
    for (int i = 0; i < 8; ++i) {
        target[i] = 0.0;
    }
    for (int k = 7; k >= 0; --k) {
        for (int i = 7; i >= 1; --i) {
            target[i] = target[i] * uBias + target[i - 1] * uScale;
        }
        target[0] = target[0] * uBias + flame.heightMonomial[k >> 2][k & 3];
    }
}

// Next field-fixed cut strictly above sCur along h(s) = h0 + hSlope * s: the
// envelope knots (in hb units, scaled by the displaced tip bx) and, with
// boundary displacement active, the capFade cubic bounds and the tip itself.
// Near-constant heights have no cuts inside any finite segment.
float flameNextEnvelopeCut(
    float sCur, float sEnd, float h0, float hSlope, float bx, bool useKnots) {
    if (abs(hSlope) < FLAME_RADIAL_MIN_DIR_Y) {
        return sEnd;
    }
    float sNext = sEnd;
    float invSlope = 1.0 / hSlope;
    if (useKnots) {
        for (int k = 0; k < FLAME_ENVELOPE_KNOT_COUNT; ++k) {
            float knot = flame.envelopeKnots[k >> 2][k & 3];
            if (knot > 1.5) {
                break;
            }
            float s = (knot * bx - h0) * invSlope;
            if (s > sCur + 1e-6 && s < sNext) {
                sNext = s;
            }
        }
    }
    if (flame.boundaryParams.x != 0.0) {
        float s94 = (0.94 - h0) * invSlope;
        if (s94 > sCur + 1e-6 && s94 < sNext) {
            sNext = s94;
        }
        float s100 = (1.0 - h0) * invSlope;
        if (s100 > sCur + 1e-6 && s100 < sNext) {
            sNext = s100;
        }
        float sTip = (bx - h0) * invSlope;
        if (sTip > sCur + 1e-6 && sTip < sNext) {
            sNext = sTip;
        }
    }
    return sNext;
}

// Exact integral (.x) and first moment about s = 0 (.y) of
//   F(h(s) / bx) * cap(h(s)) * (1 - g(s))^2
// over [s0, s1], which must lie inside the support and inside a single regime of
// cap and of the hb clamp. h(s) = h0 + hSlope * s. The polynomial is evaluated in
// the piece-centered variable so high powers stay conditioned.
vec2 flameExactPieceEmission(
    float a, float b, float c, float h0, float hSlope, float bx,
    bool tipRegime, bool capRegime, float s0, float s1) {
    float w = 0.5 * (s1 - s0);
    if (w <= 1e-7) {
        return vec2(0.0);
    }
    float sMid = 0.5 * (s0 + s1);
    float hMid = h0 + hSlope * sMid;

    float envelope[8];
    if (tipRegime) {
        // hb clamps to 1 above the displaced tip: F is constant there.
        envelope[0] = evaluateHeightFalloff(1.0);
        for (int i = 1; i < 8; ++i) {
            envelope[i] = 0.0;
        }
    } else {
        flameEnvelopeMonomialLocal(2.0 * hSlope / bx, 2.0 * hMid / bx - 1.0, envelope);
    }

    float integrand[11];
    if (capRegime) {
        // cap = 3 t^2 - 2 t^3 with t = (1 - h) / 0.06, composed into the piece frame.
        float tScale = -hSlope / 0.06;
        float tBias = (1.0 - hMid) / 0.06;
        float cap0 = tBias * tBias * (3.0 - 2.0 * tBias);
        float cap1 = tScale * tBias * (6.0 - 6.0 * tBias);
        float cap2 = tScale * tScale * (3.0 - 6.0 * tBias);
        float cap3 = -2.0 * tScale * tScale * tScale;
        for (int i = 0; i < 11; ++i) {
            integrand[i] = 0.0;
        }
        for (int i = 0; i < 8; ++i) {
            integrand[i] += envelope[i] * cap0;
            integrand[i + 1] += envelope[i] * cap1;
            integrand[i + 2] += envelope[i] * cap2;
            integrand[i + 3] += envelope[i] * cap3;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            integrand[i] = envelope[i];
        }
        for (int i = 8; i < 11; ++i) {
            integrand[i] = 0.0;
        }
    }

    float gb = 2.0 * a * sMid + b;
    float gc = (a * sMid + b) * sMid + c;
    float m = 1.0 - gc;
    float w0 = m * m;
    float w1 = -2.0 * gb * m;
    float w2 = gb * gb - 2.0 * a * m;
    float w3 = 2.0 * a * gb;
    float w4 = a * a;
    float e[15];
    for (int i = 0; i < 15; ++i) {
        e[i] = 0.0;
    }
    for (int i = 0; i < 11; ++i) {
        e[i] += integrand[i] * w0;
        e[i + 1] += integrand[i] * w1;
        e[i + 2] += integrand[i] * w2;
        e[i + 3] += integrand[i] * w3;
        e[i + 4] += integrand[i] * w4;
    }

    // Symmetric moments over [-w, w]: odd powers vanish, so even coefficients
    // feed the integral and odd coefficients feed the first moment.
    float integral = 0.0;
    float first = 0.0;
    float wPow = w;
    for (int n = 0; n < 15; ++n) {
        if ((n & 1) == 0) {
            integral += e[n] * 2.0 * wPow / float(n + 1);
        } else {
            first += e[n] * 2.0 * wPow * w / float(n + 2);
        }
        wPow *= w;
    }
    return vec2(integral, sMid * integral + first);
}

// Exact band emission: integral (.x) and first moment about the band center (.y)
// of F(h/bx) * cap(h) * (1 - g(s))^2 over the support part of
// [-halfWidth, halfWidth], split only at the cap / tip regime bounds (the
// envelope itself is one global polynomial — no other cuts are needed).
vec2 flameExactBandEmission(
    float a, float b, float c, float h0, float hSlope, float bx, float halfWidth) {
    vec2 interval = flameSupportInterval(a, b, c, halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }
    bool hasCap = flame.boundaryParams.x != 0.0;
    vec2 total = vec2(0.0);
    float sCur = interval.x;
    for (int piece = 0; piece < 4; ++piece) {
        float sNext = flameNextEnvelopeCut(sCur, interval.y, h0, hSlope, bx, false);
        float hPieceMid = h0 + hSlope * 0.5 * (sCur + sNext);
        bool beyondCap = hasCap && hPieceMid >= 1.0;
        if (!beyondCap) {
            total += flameExactPieceEmission(
                a, b, c, h0, hSlope, bx,
                hasCap && hPieceMid >= bx, hasCap && hPieceMid >= 0.94, sCur, sNext);
        }
        sCur = sNext;
        if (sCur >= interval.y - 1e-7) {
            break;
        }
    }
    return total;
}

// Radius R(h) of the radial density profile, in flame-local units. Texture fit
// bakes an arbitrary R(h) curve (profileParams.x flags it); the parametric taper
// stays the default.
float flameRadialRadiusScale(float height01) {
    if (flame.profileParams.x > 0.5) {
        return FLAME_SHELL_BASE_RADIUS
            * max(evaluateChebyshev8(flame.radiusCoefficients[0], flame.radiusCoefficients[1], height01), 0.05);
    }
    return FLAME_SHELL_BASE_RADIUS
        * mix(1.0, flame.styleParams1.x, pow(height01, flame.styleParams0.w));
}

// Squared reciprocal of the support radius S * R(h), which is what u^2 scales by.
float flameRadialSupportInvSq(float height01) {
    float scale = max(flameRadialSupportRadius() * flameRadialRadiusScale(height01), 1e-4);
    return 1.0 / (scale * scale);
}

float flameRadialDensityFactor(vec3 p, float height01) {
    float radiusSquared = dot(p.xz, p.xz);
    return flameBiweight(flameRadialSupportInvSq(height01) * radiusSquared);
}

// Contour wiggle: per-band field quantity from fbm at the actual 3D point.
// Returns 1.0 when contourParams.x == 0 (identity, matches old path).
float flameContourWiggle(vec3 p, float h) {
    if (flame.contourParams.x == 0.0) { return 1.0; }
    return 1.0 + flame.contourParams.x * (fbm3(vec3(p.x, h - flame.styleParams0.z * flame.time, p.z) * flame.noiseFrequency) * (2.0 / 0.875) - 1.0);
}

// Horizontal offset of the ray at a given height, with q = d.xz / d.y.
vec2 flameRayPointAtHeight(vec3 o, vec2 q, float height) {
    return o.xz + (height - o.y) * q;
}

// Fixed node spacing per band (node positions at fixed band fractions).
const int FLAME_OCCUPANCY_EDGE_SEGMENTS = 2;
const int FLAME_OCCUPANCY_MAX_NODES =
    FLAME_ENVELOPE_KNOT_COUNT + FLAME_OCCUPANCY_EDGE_SEGMENTS + 5;

// Occupancy integral (.x) and first moment about s = 0 (.y) of the smoothed
// threshold response over the support part of [-halfWidth, halfWidth] of the
// true ray p(s) = eP0 + s * ePd.
//
// The fbm quantities (erosion, boundary displacement, contour wiggle) are
// sampled at nodes whose positions are affine in the band — fixed band
// fractions plus the field-fixed cuts (envelope knots, capFade bounds,
// displaced tip) — never at support-adaptive positions: a support-interval end
// or the vertex of g moves with a kinked derivative as the ray sweeps
// (root/clip switches), and fbm sampled at kinked positions images fold lines
// onto the screen. A frozen single band sample is just as wrong the other way
// (whole-band binarization, screen-imaged swirls at the tip). Between nodes
// the fbm quantities are linear.
//
// The argument x(s) = dens(s) - eroded(s) is re-linearized on 3 fixed
// sub-pieces of each node piece clipped to the support. At the sub-nodes the
// density is EXACT in h — envelope (Clenshaw), capFade, wind bend, taper
// radius invSq(hb) — with only the fbm quantities interpolated: any band-frozen
// factor with a steep h dependence (the taper radius at the tip most of all)
// steps at the band boundaries and the threshold binarizes the step into
// stacked horizontal surfaces. The frozen quadratic survives only as a
// conservative support CLIP (widest node radius, extra margin): outside the
// true support the exact density vanishes, so an over-wide clip is harmless.
//
// Sigma models the sub-node-piece fluctuation of the erosion (chord = fbm node
// spacing, vanishing as nodes resolve the field) and fades toward the support
// boundary per sub-piece (mean of the sub-node envelope fades): the unresolved
// fluctuation is of the *faded* erosion, so it must vanish with the envelope
// like the argument does — a flat band sigma leaves phi_sigma with a positive
// floor at the clipped support surface.
// Mirrored in thyllore-render-core/src/flame_radial.rs (evaluate_occupancy_band /
// occupancy_band_pieces — the shared piece model; the exact-density closure is
// the GLSL caller's).
vec2 flameOccupancyBandExact(float nearFade, float halfWidth, vec3 eP0, vec3 ePd) {
    bool hasCap = flame.boundaryParams.x != 0.0;
    float bxCenter = hasCap ? flameBoundaryDisplacement(eP0.xz).x : 1.0;

    float nodes[FLAME_OCCUPANCY_MAX_NODES];
    int nodeCount = 0;
    nodes[nodeCount++] = -halfWidth;
    float bandWidth = 2.0 * halfWidth;
    int fraction = 1;
    float sCur = -halfWidth;
    for (int i = 0; i < FLAME_OCCUPANCY_MAX_NODES; ++i) {
        if (nodeCount >= FLAME_OCCUPANCY_MAX_NODES - 1) {
            break;
        }
        float sFraction = fraction < FLAME_OCCUPANCY_EDGE_SEGMENTS
            ? -halfWidth + bandWidth * float(fraction) / float(FLAME_OCCUPANCY_EDGE_SEGMENTS)
            : halfWidth;
        float sCut = flameNextEnvelopeCut(sCur, halfWidth, eP0.y, ePd.y, bxCenter, true);
        float sNext = min(sFraction, sCut);
        if (sNext >= halfWidth - 1e-7) {
            break;
        }
        nodes[nodeCount++] = sNext;
        if (sFraction <= sCut) {
            ++fraction;
        }
        sCur = sNext;
    }
    nodes[nodeCount++] = halfWidth;

    float radiusScaleNode[FLAME_OCCUPANCY_MAX_NODES];
    float clipScale = 1e30;
    for (int i = 0; i < nodeCount; ++i) {
        vec3 p = eP0 + nodes[i] * ePd;
        float h = clamp(p.y, 0.0, 1.0);
        vec2 boundaryNode = hasCap ? flameBoundaryDisplacement(p.xz) : vec2(1.0);
        radiusScaleNode[i] = max(flameContourWiggle(p, h), 1e-4);
        float nodeRadius = max(radiusScaleNode[i] * boundaryNode.y, 1e-4);
        float nodeScale = flameRadialSupportInvSq(clamp(h / boundaryNode.x, 0.0, 1.0))
            / (nodeRadius * nodeRadius);
        clipScale = min(clipScale, nodeScale);
    }

    // Conservative support clip: widest node radius with margin, wind bend
    // frozen at the band center. Outside the true support the exact density
    // vanishes, so the slack only costs a little integration range.
    clipScale *= 0.64;
    vec2 bendCenter = flameBendOffsetAt(clamp(eP0.y, 0.0, 1.0));
    vec2 exz = eP0.xz - bendCenter;
    vec2 interval = flameSupportInterval(
        clipScale * dot(ePd.xz, ePd.xz),
        2.0 * clipScale * dot(exz, ePd.xz),
        clipScale * dot(exz, exz),
        halfWidth);
    if (interval.y <= interval.x) {
        return vec2(0.0);
    }

    const int SUB = 3;
    float beta = flameCarveResidualStrength();
    float stepLength = length(ePd);
    vec2 total = vec2(0.0);
    for (int i = 1; i < nodeCount; ++i) {
        float pieceLo = max(nodes[i - 1], interval.x);
        float pieceHi = min(nodes[i], interval.y);
        float clipped = pieceHi - pieceLo;
        if (clipped < 1e-7) {
            continue;
        }
        float nodeSpan = max(nodes[i] - nodes[i - 1], 1e-7);
        float pieceMidH = clamp(eP0.y + (pieceLo + 0.5 * clipped) * ePd.y, 0.0, 1.0);
        float pieceSigma = flameErosionSigma(pieceMidH, clipped / float(SUB) * stepLength);
        float subArgument[SUB + 1];
        float subFade[SUB + 1];
        float subPlain[SUB + 1];
        for (int j = 0; j <= SUB; ++j) {
            float s = pieceLo + clipped * float(j) / float(SUB);
            float h = clamp(eP0.y + s * ePd.y, 0.0, 1.0);
            // The erosion and the boundary displacement are sampled exactly at
            // every sub-node: the tip occupancy IS the sublevel-set geometry of
            // the erosion over the raised displacement columns, and any
            // interpolant renders its expectation instead — a wide diffuse cap
            // separated from the body by a coherent transparent gap (the
            // historical top seam), where the true field bridges the gap with
            // individual columns and tongues. Only the contour wiggle (a
            // multiplicative low-frequency radius modulation) stays on the
            // node interpolant.
            vec3 pSub = eP0 + s * ePd;
            float fNode = clamp((s - nodes[i - 1]) / nodeSpan, 0.0, 1.0);
            float bx = 1.0;
            float radiusScale = mix(radiusScaleNode[i - 1], radiusScaleNode[i], fNode);
            if (hasCap) {
                vec2 boundarySub = flameBoundaryDisplacement(pSub.xz);
                bx = boundarySub.x;
                radiusScale *= boundarySub.y;
            }
            float erosion = flameNoiseErosionValue(pSub, h);
            float hb = clamp(h / bx, 0.0, 1.0);
            float capFade = hasCap ? smoothstep(1.0, 0.94, h) : 1.0;
            vec2 radial = eP0.xz + s * ePd.xz - flameBendOffsetAt(h);
            radiusScale = max(radiusScale, 1e-4);
            float uSquared = dot(radial, radial) * flameRadialSupportInvSq(hb)
                / (radiusScale * radiusScale);
            float dens = evaluateHeightFalloff(hb) * capFade
                * flameBiweight(uSquared) * nearFade;
            subArgument[j] = flameErodedArgument(dens, erosion);
            subFade[j] = flameEnvelopeFade(dens);
            subPlain[j] = dens;
        }
        float subSpan = clipped / float(SUB);
        for (int j = 1; j <= SUB; ++j) {
            FlameSmoothedResponse response = flameSmoothErosionResponse(
                pieceSigma * 0.5 * (subFade[j - 1] + subFade[j]));
            float sA = pieceLo + float(j - 1) * subSpan;
            float slope = (subArgument[j] - subArgument[j - 1]) / subSpan;
            vec2 carved = flameErosionResponseLinearIntegral(
                response, subArgument[j - 1] - slope * sA, slope, sA, sA + subSpan);
            // Residual (un-carveable) fraction: the same threshold response
            // applied to the un-eroded density, which is linear on the sub-piece
            // like the carved argument — one more closed-form evaluation, no
            // sampling and no fbm.
            if (beta > 0.0) {
                float plainSlope = (subPlain[j] - subPlain[j - 1]) / subSpan;
                vec2 plain = flameErosionResponseLinearIntegral(
                    flameSmoothErosionResponse(0.0),
                    subPlain[j - 1] - plainSlope * sA, plainSlope, sA, sA + subSpan);
                carved = mix(carved, plain, beta);
            }
            total += carved;
        }
    }
    return total;
}


// Pointwise field for the reference raymarch (mode 1): the same eroded threshold
// field the closed form approximates — true smoothstep, exact support membership.
float flamePointOccupancyDensity(vec3 p, float h, float wiggle) {
    vec2 boundary = flameBoundaryDisplacement(p.xz);
    float hb = clamp(h / boundary.x, 0.0, 1.0);
    float wb = wiggle * boundary.y;
    float capFade = flame.boundaryParams.x != 0.0 ? smoothstep(1.0, 0.94, h) : 1.0;
    float dSmooth = evaluateHeightFalloff(hb) * capFade
        * flameRadialDensityFactor(vec3(p.x / wb, p.y, p.z / wb), hb)
        * flameNearCameraFade(p);
    float erosion = flameNoiseErosionValue(p, h);
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
        dSmooth) * flameFieldSupportMask(dSmooth);
}

// Occupancy variant of the along-ray fallback: the band frame is the true ray
// around tCenter (pMidTrue), first moment about tCenter.
vec2 flameOccupancyAlongRay(vec3 d, float tNear, float tFar, vec3 pMidTrue) {
    float halfWidth = 0.5 * (tFar - tNear);
    float nearFade = flameNearCameraFade(pMidTrue);
    return flameOccupancyBandExact(nearFade, halfWidth, pMidTrue, d);
}

// ---- Emitter-generic occupancy (ring / SDF analytic path) ----
// Every non-trail emitter shares the eroded threshold field
//   smoothstep(edgeLow, edgeHigh, dSmooth - erosion) * [dSmooth > 0];
// only dSmooth differs per emitter. The closed form needs dSmooth only at a few
// nodes per band: the argument is piecewise-linearized between exact node values
// and the response integral stays closed form — the sharpness lives in the
// response, not in the nodes. The cylinder keeps its specialized band integral
// (exact support intervals); this generic path serves ring and SDF, unwarped
// like the cylinder pair so mode 0 and mode 1 stay a parity pair.
// Mirrored in thyllore-render-core/src/flame_radial.rs (evaluate_occupancy_node_band).

// Pointwise field for the reference raymarch (mode 1) on ring/SDF emitters:
// the same field the node-based closed form approximates.
float flamePointEmitterOccupancy(vec3 p, float h, float wiggle) {
    float dSmooth = flameEmitterSmoothDensityAt(p, h, wiggle) * flameNearCameraFade(p);
    float erosion = flame.noiseAmplitude != 0.0 ? flameNoiseErosionValue(p, h) : 0.0;
    return flameApplyCarveResidual(
        smoothstep(flame.styleParams1.y, flame.styleParams1.z, flameErodedArgument(dSmooth, erosion)),
        dSmooth) * flameFieldSupportMask(dSmooth);
}

const int FLAME_OCCUPANCY_NODE_SEGMENTS = 4;

// Occupancy integral (.x) and first moment in t (.y) of the smoothed threshold
// response over one t band, with dSmooth sampled at the segment nodes and the
// argument linear between them. Segments whose both node densities are zero lie
// outside the support and contribute nothing (exact membership at node resolution);
// densities are sampled before the erosion fbm so empty bands cost no noise at all.
// The fbm erosion is sampled per node like the density (a frozen band sample
// loses the sublevel-set geometry of the erosion — whole-band binarization and
// screen-imaged swirls at the tip); nodes not adjacent to any positive density
// skip the fbm. Sigma models only the sub-segment fluctuation (chord = segment
// length) and is faded per segment by the node envelope fades.
// The kernel-basis warp stays frozen at the density-weighted node position, not
// the band midpoint: the midpoint of a band that straddles empty space (ring
// seen from inside) lands away from the wall, so the frozen warp would be
// sampled where no density exists and sweep through world space as the camera
// moves — the view-dependent shimmer of the band freeze.
// Mirrored in thyllore-render-core/src/flame_radial.rs (density_weighted_node_t).
// Kernel basis: erosion exact per node, so residual sigma is skipped.
vec2 flameOccupancyNodeBand(vec3 o, vec3 d, float t0, float t1, float wiggleBand) {
    float dt = (t1 - t0) / float(FLAME_OCCUPANCY_NODE_SEGMENTS);
    float density[FLAME_OCCUPANCY_NODE_SEGMENTS + 1];
    float weightSum = 0.0;
    float tWeighted = 0.0;
    for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
        float t = t0 + float(node) * dt;
        vec3 p = o + t * d;
        vec2 boundaryNode = flameBoundaryDisplacement(p.xz);
        density[node] =
            flameEmitterSmoothDensityDisplacedAt(p, clamp(p.y, 0.0, 1.0), wiggleBand, boundaryNode);
        density[node] *= flameNearCameraFade(p);
        weightSum += density[node];
        tWeighted += density[node] * t;
    }
    if (weightSum <= 0.0) {
        return vec2(0.0);
    }

    bool kernelErosion = flameKernelModelActive() && flame.noiseAmplitude != 0.0;
    bool fbmErosion = !kernelErosion && flame.noiseAmplitude != 0.0;
    vec3 pSample = o + (tWeighted / weightSum) * d;
    float hSample = clamp(pSample.y, 0.0, 1.0);
    float blobSum[FLAME_OCCUPANCY_NODE_SEGMENTS + 1];
    if (kernelErosion) {
        // Warp frozen per band (like the legacy erosion freeze); only the blob
        // realization varies per node. Per blob u²(t) = at² + bt + c, evaluated
        // at the shared nodes.
        vec3 warpOffset = flameNoiseWarpedCoordinate(pSample, hSample) - pSample;
        for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
            blobSum[node] = 0.0;
        }
        vec3 ow = o + warpOffset;
        for (int i = 0; i < 96; ++i) {
            vec4 blob = flame.kernelBlobs[2 * i];
            float amp = flame.kernelBlobs[2 * i + 1].x;
            if (amp <= 0.0 || blob.w <= 0.0) {
                continue;
            }
            float invR2 = 1.0 / (blob.w * blob.w);
            vec3 rel = ow - blob.xyz;
            float a = dot(d, d) * invR2;
            float b = 2.0 * dot(d, rel) * invR2;
            float c = dot(rel, rel) * invR2;
            float tVertex = a > 1e-12 ? clamp(-0.5 * b / a, t0, t1) : t0;
            if ((a * tVertex + b) * tVertex + c >= 1.0) {
                continue;
            }
            for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
                float t = t0 + float(node) * dt;
                float inside = max(1.0 - ((a * t + b) * t + c), 0.0);
                blobSum[node] += amp * inside * inside;
            }
        }
    }
    float argument[FLAME_OCCUPANCY_NODE_SEGMENTS + 1];
    for (int node = 0; node <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++node) {
        float erosionNode = 0.0;
        float h = clamp(o.y + (t0 + float(node) * dt) * d.y, 0.0, 1.0);
        if (kernelErosion) {
            erosionNode = flameNoiseErosionFromValue(blobSum[node], h);
        } else if (fbmErosion) {
            bool adjacentSupport = density[node] > 0.0
                || (node > 0 && density[node - 1] > 0.0)
                || (node < FLAME_OCCUPANCY_NODE_SEGMENTS && density[node + 1] > 0.0);
            if (adjacentSupport) {
                erosionNode = flameNoiseErosionValue(o + (t0 + float(node) * dt) * d, h);
            }
        }
        argument[node] = flameErodedArgument(density[node], erosionNode);
    }
    float segmentChord = dt * length(d);
    vec2 total = vec2(0.0);
    for (int segment = 1; segment <= FLAME_OCCUPANCY_NODE_SEGMENTS; ++segment) {
        if (density[segment - 1] > 0.0 || density[segment] > 0.0) {
            // Per-segment sigma fade toward the support boundary (see
            // flameOccupancyBandExact): the fluctuation vanishes with the envelope.
            float tPrev = t0 + float(segment - 1) * dt;
            float hSegment = clamp(o.y + (tPrev + 0.5 * dt) * d.y, 0.0, 1.0);
            float sigma = fbmErosion ? flameErosionSigma(hSegment, segmentChord) : 0.0;
            FlameSmoothedResponse response = flameSmoothErosionResponse(sigma * 0.5
                * (flameEnvelopeFade(density[segment - 1]) + flameEnvelopeFade(density[segment])));
            float slope = (argument[segment] - argument[segment - 1]) / dt;
            total += flameErosionResponseLinearIntegral(
                response, argument[segment - 1] - slope * tPrev, slope, tPrev, tPrev + dt);
        }
    }
    return total;
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar);

// Rays with a near-constant height cannot be parameterized by h, so the moment is taken along t.
float integrateRadialEmissionAlongRay(vec3 o, vec3 d, float tNear, float tFar, float invSq) {
    float tCenter = 0.5 * (tNear + tFar);
    vec2 p = o.xz + tCenter * d.xz;
    float halfWidth = 0.5 * (tFar - tNear);
    vec2 boundary = flameBoundaryDisplacement(p);
    float hRaw = clamp(o.y + tCenter * d.y, 0.0, 1.0);
    float hMid = clamp(hRaw / boundary.x, 0.0, 1.0);
    float capFade = flame.boundaryParams.x != 0.0 ? smoothstep(1.0, 0.94, hRaw) : 1.0;
    float falloff = evaluateHeightFalloff(hMid) * capFade * flameNearCameraFade(o + tCenter * d);
    float invSqB = invSq / (boundary.y * boundary.y);
    return flameBiweightBandEmission(
        invSqB * dot(d.xz, d.xz), 2.0 * invSqB * dot(p, d.xz), invSqB * dot(p, p),
        vec3(falloff, 0.0, 0.0), halfWidth).x;
}

float integrateRadialEmission(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }

    float heightNear = clamp(o.y + tNear * d.y, 0.0, 1.0);
    float heightFar = clamp(o.y + tFar * d.y, 0.0, 1.0);
    float heightLo = min(heightNear, heightFar);
    float heightHi = max(heightNear, heightFar);
    if (abs(d.y) < FLAME_RADIAL_MIN_DIR_Y
        || heightHi - heightLo < FLAME_RADIAL_MIN_HEIGHT_SPAN) {
        float midHeight = clamp(o.y + 0.5 * (tNear + tFar) * d.y, 0.0, 1.0);
        vec3 pMid = o + 0.5 * (tNear + tFar) * d;
        float wMid = flameContourWiggle(pMid, midHeight);
        vec2 bendMid = flameBendOffsetAt(midHeight);
        if (flame.noiseAmplitude != 0.0) {
            return flameOccupancyAlongRay(d, tNear, tFar, pMid).x;
        }
        return integrateRadialEmissionAlongRay(
            vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, tNear, tFar,
            flameRadialSupportInvSq(midHeight) / (wMid * wMid));
    }

    // Only the slope is carried: monomial coefficients in h would grow as 1/d.y^2 and cancel away.
    vec2 q = d.xz / d.y;
    float quadratic = dot(q, q);

    // Trim to the widest support (over heights, wiggle and wind bend) so grazing rays
    // do not spend bands on empty range. Bands outside the true support integrate to
    // exactly zero, so the extra margin costs only band resolution, never correctness.
    float wTrim = (1.0 + max(flame.contourParams.x, 0.0))
        * (1.0 + 3.0 * abs(flame.boundaryParams.x) * max(flame.boundaryParams.w, 0.0));
    float widestRadius = flameRadialSupportRadius() * wTrim
        * (flame.profileParams.x > 0.5
            ? FLAME_SHELL_BASE_RADIUS * flame.profileParams.y
            : max(flameRadialRadiusScale(0.0), flameRadialRadiusScale(1.0)));
    float bendMax = length(flame.styleParams2.xy) * flame.styleParams2.z;
    if (quadratic > 1e-12) {
        float support = (widestRadius + bendMax) / sqrt(quadratic);
        float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
        heightLo = max(heightLo, closestApproachHeight - support);
        heightHi = min(heightHi, closestApproachHeight + support);
        if (heightHi <= heightLo) {
            return 0.0;
        }
    }

    float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
    float halfWidth = 0.5 * bandWidth;
    float total = 0.0;
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        float center = heightLo + (float(band) + 0.5) * bandWidth;
        vec2 pTrue = flameRayPointAtHeight(o, q, center);
        float nearFade = flameNearCameraFade(vec3(pTrue.x, center, pTrue.y));
        if (flame.noiseAmplitude != 0.0) {
            total += flameOccupancyBandExact(
                nearFade, halfWidth, vec3(pTrue.x, center, pTrue.y), vec3(q.x, 1.0, q.y)).x;
        } else {
            vec2 pxz = pTrue - flameBendOffsetAt(center);
            float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
            vec2 boundary = flame.boundaryParams.x != 0.0
                ? flameBoundaryDisplacement(pTrue) : vec2(1.0);
            float hbC = clamp(center / boundary.x, 0.0, 1.0);
            wBand *= boundary.y;
            float invSq = flameRadialSupportInvSq(hbC) / (wBand * wBand);
            float aq = invSq * quadratic;
            float bq = 2.0 * invSq * dot(pxz, q);
            float cq = invSq * dot(pxz, pxz);
            total += nearFade
                * flameExactBandEmission(aq, bq, cq, center, 1.0, boundary.x, halfWidth).x;
        }
    }
    return total / abs(d.y);
}

vec3 flameRampColor(float h) {
    if (flame.profileParams.z > 0.5) {
        float u = clamp(h, 0.0, 1.0) * 8.0 - 0.5;
        int i0 = int(clamp(floor(u), 0.0, 7.0));
        int i1 = min(i0 + 1, 7);
        float f = clamp(u - float(i0), 0.0, 1.0);
        return mix(flame.colorRamp[i0].rgb, flame.colorRamp[i1].rgb, f);
    }
    if (h < 0.5) {
        return mix(flame.colorBase.rgb, flame.colorMid.rgb, h * 2.0);
    }
    return mix(flame.colorMid.rgb, flame.colorTip.rgb, (h - 0.5) * 2.0);
}

// Beer-Lambert composite over per-band emissions — the one RTE assembly shared by
// the cylinder bands (camera order given by `reversed`) and the emitter-generic
// t bands (already camera-ordered, reversed = false).
vec4 flameCompositeRteBands(
    float bandEmission[FLAME_RADIAL_BAND_COUNT],
    float bandHeight[FLAME_RADIAL_BAND_COUNT],
    float bandEdge[FLAME_RADIAL_BAND_COUNT],
    bool reversed) {
    float total = 0.0;
    float heightMean = 0.0;
    for (int i = 0; i < FLAME_RADIAL_BAND_COUNT; ++i) {
        total += bandEmission[i];
        heightMean += bandEmission[i] * bandHeight[i];
    }
    heightMean = total > 1e-6 ? heightMean / total : 0.0;
    float tempNorm = clamp(total * 2.0, 0.0, 1.0) * (1.0 - 0.55 * heightMean);
    float boost = 1.0 + flame.styleParams1.w * tempNorm * tempNorm;

    vec3 radiance = vec3(0.0);
    vec3 sigmaRgb = flame.sigmaT * mix(vec3(1.0), vec3(1.0, 1.091, 1.333), clamp(flame.contourParams.w, 0.0, 1.0));
    vec3 transmittance = vec3(1.0);
    for (int i = 0; i < FLAME_RADIAL_BAND_COUNT; ++i) {
        int idx = reversed ? FLAME_RADIAL_BAND_COUNT - 1 - i : i;
        vec3 tau = sigmaRgb * bandEmission[idx];
        vec3 absorbed = vec3(1.0) - exp(-tau);
        radiance += transmittance * mix(flameRampColor(bandHeight[idx]), flame.colorTip.rgb, bandEdge[idx]) * flame.intensity * boost * absorbed;
        transmittance *= exp(-tau);
    }
    return vec4(radiance, 1.0 - dot(transmittance, vec3(1.0 / 3.0)));
}

vec4 integrateRadialRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }

    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    bool reversed = false;

    float heightNear = clamp(o.y + tNear * d.y, 0.0, 1.0);
    float heightFar = clamp(o.y + tFar * d.y, 0.0, 1.0);
    float heightLo = min(heightNear, heightFar);
    float heightHi = max(heightNear, heightFar);

    if (abs(d.y) < FLAME_RADIAL_MIN_DIR_Y || heightHi - heightLo < FLAME_RADIAL_MIN_HEIGHT_SPAN) {
        float dt = (tFar - tNear) / float(FLAME_RADIAL_BAND_COUNT);
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            float t0 = tNear + float(band) * dt;
            vec3 pMid = o + (t0 + 0.5 * dt) * d;
            float midHeight = clamp(pMid.y, 0.0, 1.0);
            float wMid = flameContourWiggle(pMid, midHeight);
            vec2 bendMid = flameBendOffsetAt(midHeight);
            float c;
            if (flame.noiseAmplitude != 0.0) {
                c = flameOccupancyAlongRay(d, t0, t0 + dt, pMid).x;
            } else {
                c = integrateRadialEmissionAlongRay(
                    vec3(o.x - bendMid.x, o.y, o.z - bendMid.y), d, t0, t0 + dt,
                    flameRadialSupportInvSq(midHeight) / (wMid * wMid));
            }
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = midHeight;
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(pMid.xz - bendMid) / max(flameRadialRadiusScale(midHeight), 1e-4)), 0.0, 1.0);
        }
    } else {
        vec2 q = d.xz / d.y;
        float quadratic = dot(q, q);

        float wTrim = (1.0 + max(flame.contourParams.x, 0.0))
        * (1.0 + 3.0 * abs(flame.boundaryParams.x) * max(flame.boundaryParams.w, 0.0));
        float widestRadius = flameRadialSupportRadius() * wTrim
            * (flame.profileParams.x > 0.5
                ? FLAME_SHELL_BASE_RADIUS * flame.profileParams.y
                : max(flameRadialRadiusScale(0.0), flameRadialRadiusScale(1.0)));
        float bendMax = length(flame.styleParams2.xy) * flame.styleParams2.z;
        if (quadratic > 1e-12) {
            float support = (widestRadius + bendMax) / sqrt(quadratic);
            float closestApproachHeight = o.y - dot(o.xz, q) / quadratic;
            heightLo = max(heightLo, closestApproachHeight - support);
            heightHi = min(heightHi, closestApproachHeight + support);
            if (heightHi <= heightLo) {
                return vec4(0.0);
            }
        }

        float bandWidth = (heightHi - heightLo) / float(FLAME_RADIAL_BAND_COUNT);
        float halfWidth = 0.5 * bandWidth;
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            float center = heightLo + (float(band) + 0.5) * bandWidth;
            vec2 pTrue = flameRayPointAtHeight(o, q, center);
            float nearFade = flameNearCameraFade(vec3(pTrue.x, center, pTrue.y));
            vec2 p = pTrue - flameBendOffsetAt(center);
            vec2 emission;
            if (flame.noiseAmplitude != 0.0) {
                emission = flameOccupancyBandExact(
                    nearFade, halfWidth, vec3(pTrue.x, center, pTrue.y), vec3(q.x, 1.0, q.y));
            } else {
                float wBand = flameContourWiggle(vec3(pTrue.x, center, pTrue.y), center);
                vec2 boundary = flame.boundaryParams.x != 0.0
                    ? flameBoundaryDisplacement(pTrue) : vec2(1.0);
                float hbC = clamp(center / boundary.x, 0.0, 1.0);
                wBand *= boundary.y;
                float invSq = flameRadialSupportInvSq(hbC) / (wBand * wBand);
                float aq = invSq * quadratic;
                float bq = 2.0 * invSq * dot(p, q);
                float cq = invSq * dot(p, p);
                emission = nearFade
                    * flameExactBandEmission(aq, bq, cq, center, 1.0, boundary.x, halfWidth);
            }
            float c = emission.x / abs(d.y);
            float meanOffset = emission.x > 1e-6 ? clamp(emission.y / emission.x, -halfWidth, halfWidth) : 0.0;
            bandEmission[band] = max(c, 0.0);
            bandHeight[band] = clamp(center + meanOffset, 0.0, 1.0);
            bandEdge[band] = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, length(p) / max(flameRadialRadiusScale(center), 1e-4)), 0.0, 1.0);
        }
        reversed = d.y < 0.0;
    }

    return flameCompositeRteBands(bandEmission, bandHeight, bandEdge, reversed);
}

// Narrow [tNear, tFar] to the ray's crossing of the ring's outer support
// cylinder, conservative over height (widest taper) and contour wiggle.
// Trimming keeps the band layout attached to the flame body: the raw proxy
// interval of an inside-the-ring ray extends past the support to wherever the
// proxy exit lands, so the band layout would jump wherever that exit switches
// between the top slab and the cone side — the visible horizontal seam where
// the frozen-turbulence texture changes character with the view angle.
// The outer cylinder is convex, so the trimmed span is a single interval whose
// endpoints vary continuously with the ray — no per-pixel topology switches
// (subtracting the inner hole would split the span in two and re-allocating
// integer band counts between the parts creates new arc-shaped seams; the hole
// only costs resolution, its bands stay empty by density gating). Returns
// false when the ray misses the support entirely.
// Mirrored in thyllore-render-core/src/flame_radial.rs (ring_support_span).
bool flameRingSupportSpan(vec3 o, vec3 d, inout float tNear, inout float tFar) {
    float rm = flame.emitterParams.y;
    float minorScale = max(1.0 - rm, 1e-3);
    float wTrim = (1.0 + max(flame.contourParams.x, 0.0))
        * (1.0 + 3.0 * abs(flame.boundaryParams.x) * max(flame.boundaryParams.w, 0.0));
    float taperMax = max(1.0, flame.styleParams1.x);
    float rOut = rm + minorScale * flameRadialSupportRadius() * taperMax * wTrim;

    float a = dot(d.xz, d.xz);
    float b = 2.0 * dot(o.xz, d.xz);
    float c = dot(o.xz, o.xz) - rOut * rOut;
    if (a < 1e-12) {
        return c <= 0.0;
    }
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant <= 0.0) {
        return false;
    }
    float root = sqrt(discriminant);
    tNear = max((-b - root) / (2.0 * a), tNear);
    tFar = min((-b + root) / (2.0 * a), tFar);
    return tFar > tNear;
}

// Emitter-generic occupancy bands over t — the one band fill shared by the scalar
// emission integral and the RTE composite. The t bands are camera-ordered; for
// the ring they cover the (convex) outer-support crossing instead of the raw
// proxy interval.
void flameEmitterOccupancyBands(
    vec3 o, vec3 d, float tNear, float tFar,
    out float bandEmission[FLAME_RADIAL_BAND_COUNT],
    out float bandHeight[FLAME_RADIAL_BAND_COUNT],
    out float bandEdge[FLAME_RADIAL_BAND_COUNT]) {
    bool ringEmitter = flame.emitterParams.x >= 0.5 && flame.emitterParams.x < 1.5;
    if (ringEmitter && !flameRingSupportSpan(o, d, tNear, tFar)) {
        float hFallback = clamp(o.y + 0.5 * (tNear + tFar) * d.y, 0.0, 1.0);
        for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
            bandEmission[band] = 0.0;
            bandHeight[band] = hFallback;
            bandEdge[band] = 0.0;
        }
        return;
    }
    float dt = (tFar - tNear) / float(FLAME_RADIAL_BAND_COUNT);
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        float t0 = tNear + float(band) * dt;
        vec3 pMid = o + (t0 + 0.5 * dt) * d;
        float hMid = clamp(pMid.y, 0.0, 1.0);
        float wBand = flameContourWiggle(pMid, hMid);
        vec2 occupancy = flameOccupancyNodeBand(o, d, t0, t0 + dt, wBand);
        float tMean = occupancy.x > 1e-6 ? clamp(occupancy.y / occupancy.x, t0, t0 + dt) : t0 + 0.5 * dt;
        vec3 pMean = o + tMean * d;
        float hMean = clamp(pMean.y, 0.0, 1.0);
        bandEmission[band] = max(occupancy.x, 0.0);
        bandHeight[band] = hMean;
        float edge = 0.0;
        if (flame.emitterParams.x < 1.5) {
            float rm = flame.emitterParams.x >= 0.5 ? flame.emitterParams.y : 0.0;
            float minorScale = flame.emitterParams.x >= 0.5 ? max(1.0 - rm, 1e-3) : 1.0;
            float taperR = mix(1.0, flame.styleParams1.x, pow(hMean, flame.styleParams0.w));
            float rhoNorm = abs((length(pMean.xz) - rm) / minorScale) / max(taperR, 1e-4);
            edge = clamp(flame.colorTip.w * smoothstep(0.6, 1.2, rhoNorm), 0.0, 1.0);
        }
        bandEdge[band] = edge;
    }
}

vec4 integrateEmitterOccupancyRTE(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return vec4(0.0);
    }
    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    flameEmitterOccupancyBands(o, d, tNear, tFar, bandEmission, bandHeight, bandEdge);
    return flameCompositeRteBands(bandEmission, bandHeight, bandEdge, false);
}

float integrateEmitterOccupancy(vec3 o, vec3 d, float tNear, float tFar) {
    if (tFar <= tNear) {
        return 0.0;
    }
    float bandEmission[FLAME_RADIAL_BAND_COUNT];
    float bandHeight[FLAME_RADIAL_BAND_COUNT];
    float bandEdge[FLAME_RADIAL_BAND_COUNT];
    flameEmitterOccupancyBands(o, d, tNear, tFar, bandEmission, bandHeight, bandEdge);
    float total = 0.0;
    for (int band = 0; band < FLAME_RADIAL_BAND_COUNT; ++band) {
        total += bandEmission[band];
    }
    return total;
}

#endif
