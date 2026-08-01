//! Kernel-primitive turbulence: deterministic biweight blobs replacing the fbm
//! erosion when `FlameEffect::turbulence_model == 1`.
//!
//! The field is purely additive — `density = core + sum_i blob_i` with no
//! threshold — so the ray integral of a realization is exact closed form:
//! each blob is a compact-support biweight whose squared normalized distance
//! along a ray is a quadratic, reducing every band integral to power moments
//! (see `thyllore_math_core::compact_support`). Blob placement is a stateless
//! function of (effect parameters, time): the same inputs always produce the
//! same blob list, keeping batch capture deterministic and mode 0/1 parity
//! constructive (both consume the identical list).
//!
//! GLSL consumes the list through the `kernelBlobs` UBO array filled by
//! `build_flame_ubo`; the closed-form band integral is mirrored in
//! `shaders/include/flame_radial_integral.glsl` (flameKernelBlobBandIntegral).

use thyllore_math_core::{integrate_powers, solve_support_interval};

pub const KERNEL_BLOB_COUNT: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KernelBlob {
    pub center: [f32; 3],
    pub radius: f32,
    pub amplitude: f32,
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash01(slot: u32, stream: u32) -> f32 {
    (hash_u32(slot.wrapping_mul(0x9e37_79b9) ^ stream.wrapping_mul(0x85eb_ca6b)) >> 8) as f32
        / (1u32 << 24) as f32
}

#[derive(Clone, Copy, Debug)]
pub struct KernelBlobParams {
    /// 0 = cylinder column, 1 = ring circle (matches emitter_kind 0/1).
    pub emitter_kind: u32,
    /// Ring major radius in flame-local units (emitter_params.y of the UBO).
    pub ring_major_norm: f32,
    /// Base blob radius in flame-local units.
    pub blob_size: f32,
    /// Peak blob density amplitude.
    pub blob_amp: f32,
    /// Upward advection speed; one life spans the unit height.
    pub rise_speed: f32,
    pub time: f32,
}

/// Life-cycle amplitude envelope: zero at both ends of the period so the
/// wrap of `fract` never pops, peaking near the lower third of the rise.
fn life_envelope(u: f32) -> f32 {
    3.0 * u.sqrt() * (1.0 - u) * (1.0 - u).sqrt()
}

pub fn generate_kernel_blobs(params: &KernelBlobParams) -> [KernelBlob; KERNEL_BLOB_COUNT] {
    let mut blobs = [KernelBlob::default(); KERNEL_BLOB_COUNT];
    let rise = params.rise_speed.max(0.1);
    for (slot, blob) in blobs.iter_mut().enumerate() {
        let k = slot as u32;
        let lifetime = (0.8 + 0.5 * hash01(k, 0)) / rise;
        let phase = hash01(k, 1);
        let u = (params.time / lifetime + phase).fract();

        let angle = std::f32::consts::TAU
            * (slot as f32 / KERNEL_BLOB_COUNT as f32 + 0.37 * hash01(k, 2));
        let (sin_a, cos_a) = angle.sin_cos();
        let spawn_radius = if params.emitter_kind == 1 {
            params.ring_major_norm * (1.0 - 0.15 * u)
        } else {
            0.35 * hash01(k, 3).sqrt() * (1.0 - 0.6 * u)
        };

        let drift_phase = std::f32::consts::TAU * hash01(k, 4);
        let drift = 0.05 * (std::f32::consts::TAU * (2.0 * u) + drift_phase).sin();

        blob.center = [
            spawn_radius * cos_a - drift * sin_a,
            u,
            spawn_radius * sin_a + drift * cos_a,
        ];
        blob.radius = params.blob_size * (0.7 + 0.6 * hash01(k, 5)) * (0.6 + 0.8 * u);
        blob.amplitude = params.blob_amp * (0.7 + 0.6 * hash01(k, 6)) * life_envelope(u);
    }
    blobs
}

/// Pointwise blob field density (the mode 1 reference of the closed form).
pub fn evaluate_kernel_blob_density(blobs: &[KernelBlob], p: [f32; 3]) -> f32 {
    let mut total = 0.0;
    for blob in blobs {
        if blob.amplitude <= 0.0 || blob.radius <= 0.0 {
            continue;
        }
        let dx = p[0] - blob.center[0];
        let dy = p[1] - blob.center[1];
        let dz = p[2] - blob.center[2];
        let u2 = (dx * dx + dy * dy + dz * dz) / (blob.radius * blob.radius);
        let inside = (1.0 - u2).max(0.0);
        total += blob.amplitude * inside * inside;
    }
    total
}

/// Closed-form integral (.0) and first moment in t (.1) of one blob over the
/// ray segment [t0, t1]. Mirrored in GLSL as `flameKernelBlobBandIntegral`.
pub fn evaluate_kernel_blob_band(
    o: [f32; 3],
    d: [f32; 3],
    t0: f32,
    t1: f32,
    blob: &KernelBlob,
) -> (f32, f32) {
    if blob.amplitude <= 0.0 || blob.radius <= 0.0 {
        return (0.0, 0.0);
    }
    let inv_r2 = 1.0 / (blob.radius * blob.radius);
    let ox = [
        o[0] - blob.center[0],
        o[1] - blob.center[1],
        o[2] - blob.center[2],
    ];
    let a = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * inv_r2;
    let b = 2.0 * (d[0] * ox[0] + d[1] * ox[1] + d[2] * ox[2]) * inv_r2;
    let c = (ox[0] * ox[0] + ox[1] * ox[1] + ox[2] * ox[2]) * inv_r2;

    let Some((s_lo, s_hi)) = solve_support_interval(a, b, c, t0, t1) else {
        return (0.0, 0.0);
    };

    // Recenter the quadratic on the clipped support midpoint: the raw (b, c) grow
    // with the blob distance and cancel catastrophically in f32, while the centered
    // coefficients stay O(1) over the support.
    let t_center = 0.5 * (s_lo + s_hi);
    let b_centered = 2.0 * a * t_center + b;
    let c_centered = (a * t_center + b) * t_center + c;

    let m = 1.0 - c_centered;
    let e = [
        m * m,
        -2.0 * b_centered * m,
        b_centered * b_centered - 2.0 * a * m,
        2.0 * a * b_centered,
        a * a,
    ];
    let powers = integrate_powers(s_lo - t_center, s_hi - t_center);
    let mut integral = 0.0;
    let mut first_moment = 0.0;
    for (n, coefficient) in e.iter().enumerate() {
        integral += coefficient * powers[n];
        first_moment += coefficient * powers[n + 1];
    }
    first_moment += t_center * integral;
    (blob.amplitude * integral, blob.amplitude * first_moment)
}

pub const KERNEL_NODE_SEGMENTS: usize = 4;

/// Mirror of `flameKernelNodeBand`: the smooth core on shared nodes (trapezoid
/// in t with exact first moment) plus the closed-form blob sum.
pub fn evaluate_kernel_node_band(
    o: [f32; 3],
    d: [f32; 3],
    t0: f32,
    t1: f32,
    core_weight: f32,
    core_density_at: impl Fn([f32; 3]) -> f32,
    blobs: &[KernelBlob],
) -> (f32, f32) {
    let mut total = (0.0f32, 0.0f32);
    for blob in blobs {
        let (integral, moment) = evaluate_kernel_blob_band(o, d, t0, t1, blob);
        total.0 += integral;
        total.1 += moment;
    }

    let dt = (t1 - t0) / KERNEL_NODE_SEGMENTS as f32;
    let point_at = |t: f32| [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    let mut prev = core_density_at(point_at(t0));
    for segment in 1..=KERNEL_NODE_SEGMENTS {
        let t = t0 + segment as f32 * dt;
        let cur = core_density_at(point_at(t));
        let t_prev = t - dt;
        total.0 += core_weight * 0.5 * dt * (prev + cur);
        total.1 += core_weight * dt * (prev * (2.0 * t_prev + t) + cur * (t_prev + 2.0 * t)) / 6.0;
        prev = cur;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params(emitter_kind: u32, time: f32) -> KernelBlobParams {
        KernelBlobParams {
            emitter_kind,
            ring_major_norm: 0.6,
            blob_size: 0.15,
            blob_amp: 1.0,
            rise_speed: 1.5,
            time,
        }
    }

    #[test]
    fn test_generation_is_deterministic() {
        let params = test_params(1, 3.7);
        assert_eq!(generate_kernel_blobs(&params), generate_kernel_blobs(&params));
    }

    #[test]
    fn test_blobs_stay_in_local_bounds_and_wrap_without_pop() {
        for kind in [0u32, 1] {
            for step in 0..200 {
                let params = test_params(kind, step as f32 * 0.173);
                for blob in generate_kernel_blobs(&params) {
                    assert!((0.0..=1.0).contains(&blob.center[1]));
                    let r_xz = (blob.center[0] * blob.center[0]
                        + blob.center[2] * blob.center[2])
                        .sqrt();
                    assert!(r_xz <= 0.9, "xz radius {r_xz} out of local bounds");
                    assert!(blob.radius > 0.0 && blob.amplitude >= 0.0);
                }
            }
        }
        assert_eq!(life_envelope(0.0), 0.0);
        assert_eq!(life_envelope(1.0), 0.0);
    }

    #[test]
    fn test_band_integral_matches_quadrature() {
        let blob = KernelBlob {
            center: [0.2, 0.5, -0.1],
            radius: 0.18,
            amplitude: 0.8,
        };
        let rays = [
            ([0.0, 0.4, -2.0], [0.1, 0.05, 1.0]),
            ([0.2, 0.5, -0.5], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.8, 0.35, -0.05]),
            ([0.2, 0.5, -0.1], [0.0, 1.0, 0.0]),
        ];
        for (o, d) in rays {
            for (t0, t1) in [(0.0f32, 3.0f32), (0.3, 0.6), (1.5, 4.0)] {
                let (integral, first_moment) = evaluate_kernel_blob_band(o, d, t0, t1, &blob);
                let steps = 40000;
                let dt = (t1 - t0) as f64 / steps as f64;
                let mut ref_integral = 0.0f64;
                let mut ref_moment = 0.0f64;
                for i in 0..steps {
                    let t = t0 as f64 + (i as f64 + 0.5) * dt;
                    let p = [
                        o[0] as f64 + t * d[0] as f64,
                        o[1] as f64 + t * d[1] as f64,
                        o[2] as f64 + t * d[2] as f64,
                    ];
                    let value = evaluate_kernel_blob_density(
                        &[blob],
                        [p[0] as f32, p[1] as f32, p[2] as f32],
                    ) as f64;
                    ref_integral += value * dt;
                    ref_moment += value * t * dt;
                }
                assert!(
                    (integral as f64 - ref_integral).abs() < 1e-4,
                    "integral {integral} vs {ref_integral} for o={o:?} d={d:?} [{t0},{t1}]"
                );
                assert!(
                    (first_moment as f64 - ref_moment).abs() < 3e-4,
                    "moment {first_moment} vs {ref_moment} for o={o:?} d={d:?} [{t0},{t1}]"
                );
            }
        }
    }

    #[test]
    fn test_node_band_matches_quadrature() {
        let blobs = [
            KernelBlob {
                center: [0.1, 0.4, 0.0],
                radius: 0.2,
                amplitude: 0.9,
            },
            KernelBlob {
                center: [-0.2, 0.7, 0.15],
                radius: 0.12,
                amplitude: 1.4,
            },
        ];
        let core = |p: [f32; 3]| {
            let r2 = p[0] * p[0] + p[2] * p[2];
            ((1.0 - p[1]).max(0.0)) * (-3.0 * r2).exp()
        };
        let o = [-0.8, 0.2, -0.3];
        let d = [0.9, 0.45, 0.3];
        let (t0, t1) = (0.0f32, 1.4f32);
        let core_weight = 0.8;

        let (integral, moment) =
            evaluate_kernel_node_band(o, d, t0, t1, core_weight, core, &blobs);

        // Reference: quadrature of the *piecewise-linear* core (the linearization
        // is the model's contract, not an error source) plus the exact blob field.
        let node_dt = (t1 - t0) / KERNEL_NODE_SEGMENTS as f32;
        let node_values: Vec<f32> = (0..=KERNEL_NODE_SEGMENTS)
            .map(|n| {
                let t = t0 + n as f32 * node_dt;
                core([o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]])
            })
            .collect();
        let linear_core = |t: f64| {
            let s = ((t - t0 as f64) / node_dt as f64)
                .clamp(0.0, KERNEL_NODE_SEGMENTS as f64 - 1e-9);
            let idx = s.floor() as usize;
            let frac = s - idx as f64;
            node_values[idx] as f64 * (1.0 - frac) + node_values[idx + 1] as f64 * frac
        };
        let steps = 80000;
        let dt = (t1 - t0) as f64 / steps as f64;
        let mut ref_integral = 0.0f64;
        let mut ref_moment = 0.0f64;
        for i in 0..steps {
            let t = t0 as f64 + (i as f64 + 0.5) * dt;
            let p = [
                (o[0] as f64 + t * d[0] as f64) as f32,
                (o[1] as f64 + t * d[1] as f64) as f32,
                (o[2] as f64 + t * d[2] as f64) as f32,
            ];
            let value = core_weight as f64 * linear_core(t)
                + evaluate_kernel_blob_density(&blobs, p) as f64;
            ref_integral += value * dt;
            ref_moment += value * t * dt;
        }
        assert!(
            (integral as f64 - ref_integral).abs() < 2e-4,
            "integral {integral} vs {ref_integral}"
        );
        assert!(
            (moment as f64 - ref_moment).abs() < 2e-4,
            "moment {moment} vs {ref_moment}"
        );
    }

    #[test]
    fn test_band_split_additivity() {
        let blob = KernelBlob {
            center: [0.0, 0.5, 0.0],
            radius: 0.25,
            amplitude: 1.3,
        };
        let o = [-1.0, 0.45, 0.05];
        let d = [1.0, 0.02, -0.03];
        let (whole, whole_m) = evaluate_kernel_blob_band(o, d, 0.0, 3.0, &blob);
        let mut split = 0.0;
        let mut split_m = 0.0;
        for band in 0..6 {
            let t0 = band as f32 * 0.5;
            let (part, part_m) = evaluate_kernel_blob_band(o, d, t0, t0 + 0.5, &blob);
            split += part;
            split_m += part_m;
        }
        assert!((whole - split).abs() < 1e-5);
        assert!((whole_m - split_m).abs() < 1e-5);
    }
}
