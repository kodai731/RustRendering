use super::*;
use thyllore_math_core::evaluate_chebyshev;

pub fn integrate_emission_segment(source: f32, sigma_t: f32, dt: f32) -> f32 {
    let x = sigma_t * dt;
    if x < 1e-3 {
        source * dt * (1.0 - 0.5 * x + x * x / 6.0)
    } else {
        source * (1.0 - (-x).exp()) / sigma_t
    }
}

/// Evaluate self-shadow optical depth for a point in flame-local space.
/// Uses layered concentric cylinders (3 layers) with Chebyshev-evaluated density.
pub fn evaluate_self_shadow_optical_depth(
    p_local: [f32; 3],
    light_dir_local: [f32; 3],
    coefficients: &FlameCoefficients,
    sigma_t: f32,
) -> f32 {
    // Layer radii S = [1/3, 2/3, 1], midpoints m = [1/6, 0.5, 5/6]
    let s: [f32; 3] = [1.0 / 3.0, 2.0 / 3.0, 1.0];
    let m: [f32; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

    // Evaluate density at each layer midpoint using Chebyshev coefficients
    let radial_series = thyllore_math_core::ChebyshevSeries::new(
        coefficients.radial.iter().flatten().copied().collect(),
        (0.0, 1.0),
    );
    let mut dens = [0.0f32; 4];
    for k in 0..3 {
        dens[k] = evaluate_chebyshev(&radial_series, m[k]);
    }
    dens[3] = 0.0;

    // Compute weights w_k = dens_k - dens_{k+1}
    let w: [f32; 3] = [dens[0] - dens[1], dens[1] - dens[2], dens[2] - dens[3]];

    let px = p_local[0];
    let py = p_local[1];
    let pz = p_local[2];
    let lx = light_dir_local[0];
    let ly = light_dir_local[1];
    let lz = light_dir_local[2];

    // For each layer, compute the integral I_k
    let mut total: f32 = 0.0;

    for k in 0..3 {
        let sk = s[k];
        let a = lx * lx + lz * lz;

        // Find intersection of cylinder (x^2 + z^2 = S_k^2) and ray p + s*L
        let (s0, s1) = if a < 1e-6 {
            // Ray is parallel to cylinder axis
            if px * px + pz * pz <= sk * sk {
                (0.0, 1e4)
            } else {
                continue;
            }
        } else {
            // Solve quadratic: a*s^2 + 2*(px*lx + pz*lz)*s + (px^2 + pz^2 - sk^2) = 0
            let b = 2.0 * (px * lx + pz * lz);
            let c = px * px + pz * pz - sk * sk;
            let disc = b * b - 4.0 * a * c;

            if disc <= 0.0 {
                continue;
            }

            let sqrt_disc = disc.sqrt();
            let mut s_start = (-b - sqrt_disc) / (2.0 * a);
            let s_end = (-b + sqrt_disc) / (2.0 * a);

            // Clip to s >= 0
            if s_end < 0.0 {
                continue;
            }
            if s_start < 0.0 {
                s_start = 0.0;
            }

            (s_start, s_end)
        };

        // Clip interval by height h(s) = p.y + s*L.y in [0, 1]
        let mut lo = s0;
        let mut hi = s1;

        if ly.abs() < 1e-4 {
            // h is approximately constant
            if py < 0.0 || py > 1.0 {
                continue;
            }
            // F is coefficients.height evaluated at p.y
            let height_series = thyllore_math_core::ChebyshevSeries::new(
                coefficients.height.iter().flatten().copied().collect(),
                (0.0, 1.0),
            );
            let f_val = evaluate_chebyshev(&height_series, py);
            total += w[k] * f_val * (hi - lo);
        } else {
            // h(s) = py + s*ly, find where h in [0, 1]
            // s_lo = (0 - py) / ly, s_hi = (1 - py) / ly
            let mut s_lo = (0.0 - py) / ly;
            let mut s_hi = (1.0 - py) / ly;

            if s_lo > s_hi {
                std::mem::swap(&mut s_lo, &mut s_hi);
            }

            // Clip [lo, hi] by [s_lo, s_hi]
            lo = lo.max(s_lo);
            hi = hi.min(s_hi);

            if lo >= hi {
                continue;
            }

            // I_k = (H1(h(s1)) - H1(h(s0))) / L.y
            let h_s0 = py + lo * ly;
            let h_s1 = py + hi * ly;

            let height_prim_series = thyllore_math_core::ChebyshevSeries::new(
                coefficients
                    .height_primitive
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
                (0.0, 1.0),
            );
            let h1_s0 = evaluate_chebyshev(&height_prim_series, h_s0);
            let h1_s1 = evaluate_chebyshev(&height_prim_series, h_s1);

            total += w[k] * (h1_s1 - h1_s0) / ly;
        }
    }

    (sigma_t * total).max(0.0)
}
