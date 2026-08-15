//! Full numerical replay of the analytic flame path, driven by the packed
//! [`FlameUBO`] — the exact struct the GPU receives — so no parameter mapping
//! can diverge between this trace and the shader. Every intermediate of the
//! per-node / per-segment computation (warp chain, mode sum, density chain,
//! erf response, RTE composite, self-shadow) is recorded into a JSON document
//! for offline numerical root-cause analysis of rendering artifacts.
//!
//! Not replayed (recorded as `not_replayed` in the header):
//!   * temporal history blend (`temporalData.x`) — cross-frame state
//!   * auto exposure / scene composite — outside the flame shader
//!   * proxy-raster ray interval — approximated by slab [0,1] + outer cylinder
//!   * mode 3 IGN jitter and the SDF billboard emitter (texture-bound)

use crate::flame_fbm_mirror::fbm3;
use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
use serde_json::{json, Value};
use thyllore_effect_core::flame_wave::{WAVE_JITTER_K, WAVE_JITTER_PHASE, WAVE_JITTER_RANK};
use thyllore_effect_core::WallProbeView;
use thyllore_effect_core::{
    branch_pull_back, branch_pull_back_jvp, build_flame_ubo, FlameBaked, FlameEffect,
    FlameTemporalAccum, FlameUBO,
};
use thyllore_math_core::{integrate_erf_response_linear, smooth_erf_response, ErfResponseModel};

const SEGMENTS: usize = 64;
const EROSION_SLOTS: usize = thyllore_effect_core::flame_wave::WAVE_EROSION_SLOTS;
const WARP_BASE: usize = thyllore_effect_core::flame_wave::WAVE_WARP_BASE;
const WARP_COUNT: usize = 16;
const MEDIUM_SWIRL_BASE: usize = thyllore_effect_core::flame_wave::WAVE_MEDIUM_SWIRL_BASE;
const MEDIUM_SWIRL_COUNT: usize = thyllore_effect_core::flame_wave::WAVE_MEDIUM_SWIRL_MODE_COUNT;
const DETAIL_BASE: usize = thyllore_effect_core::flame_wave::WAVE_DETAIL_BASE;
const DETAIL_COUNT: usize = 64;
const SHELL_BASE_RADIUS: f32 = 0.5;
const SUPPORT_HEADROOM: f32 = 1.5;
const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Segment estimator selection. Legacy mirrors the production GPU
/// value-sampling path bit-for-bit and is the default. Faddeeva is the
/// continuous-functional estimator — **CPU-mirror / fringe-diagnosis only,
/// never in production** (reference copy:
/// reference_glsl/flame_wave_continuous.glsl). Opt in with
/// THYLLORE_FLAME_TRACE_INTEGRATOR=faddeeva.
#[derive(Clone, Copy, PartialEq)]
enum SegmentIntegrator {
    Legacy,
    Faddeeva,
}

impl SegmentIntegrator {
    fn from_env() -> Self {
        match std::env::var("THYLLORE_FLAME_TRACE_INTEGRATOR").as_deref() {
            Ok("faddeeva") => SegmentIntegrator::Faddeeva,
            _ => SegmentIntegrator::Legacy,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            SegmentIntegrator::Legacy => "legacy",
            SegmentIntegrator::Faddeeva => "faddeeva",
        }
    }
}

fn glsl_fract(x: f32) -> f32 {
    x - x.floor()
}

/// Mirrors GLSL `interleavedGradientNoise` from flame_noise.glsl.
fn interleaved_gradient_noise(coord: [f32; 2]) -> f32 {
    let dot = coord[0] * 0.06711056 + coord[1] * 0.00583715;
    glsl_fract(52.9829189 * glsl_fract(dot))
}

fn transform_point(matrix: &Matrix4<f32>, point: [f32; 3]) -> [f32; 3] {
    let v = matrix * Vector4::new(point[0], point[1], point[2], 1.0);
    [v.x, v.y, v.z]
}

fn transform_vector(matrix: &Matrix4<f32>, vector: [f32; 3]) -> [f32; 3] {
    let v = matrix * Vector4::new(vector[0], vector[1], vector[2], 0.0);
    [v.x, v.y, v.z]
}

const EROSION_MEAN_SHRINK: f32 = 0.0875;
const PLATEAU_CARVE_BOOST: f32 = 1.0;
/// Mirror of FLAME_SUPPORT_BISECTION_STEPS in flame_radial_integral.glsl.
const SUPPORT_BISECTION_STEPS: usize = 8;
const EROSION_SHELL_REF: f32 = 0.30;
/// Fixed scan grid for mean-line shell crossings, independent of the segment
/// count so the reference cutoff cannot change with the trace lattice.
const CROSSING_SCAN_INTERVALS: usize = 256;
const CROSSING_BISECTION_STEPS: usize = 20;
/// Sigma-side capture bands over log2(omega): band powers reproduce the
/// folded sigma to ~2% (verified vs the exact per-mode fold), and sigma is
/// the only per-segment-adaptive quantity — the resolved value is exact.
const CAPTURE_BANDS: usize = 8;
/// Positive nodes / doubled weights of the 8-point Gauss-Hermite rule,
/// prescaled so E[f(z)] = sum w * f(sqrt(2) sigma x) for even f, z ~ N(0, sigma^2).
const GAUSS_HERMITE_8: [(f32, f32); 4] = [
    (0.381_187_0, 0.746_024_5),
    (1.157_193_7, 0.234_481_0),
    (1.981_656_8, 0.019_271_2),
    (2.930_637_4, 0.000_224_6),
];

fn mixf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        mixf(a[0], b[0], t),
        mixf(a[1], b[1], t),
        mixf(a[2], b[2], t),
    ]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cheb8(c0: [f32; 4], c1: [f32; 4], x01: f32) -> f32 {
    let u = 2.0 * x01 - 1.0;
    let t = 2.0 * u;
    let b7 = c1[3];
    let b6 = t * b7 + c1[2];
    let b5 = t * b6 - b7 + c1[1];
    let b4 = t * b5 - b6 + c1[0];
    let b3 = t * b4 - b5 + c0[3];
    let b2 = t * b3 - b4 + c0[2];
    let b1 = t * b2 - b3 + c0[1];
    u * b1 - b2 + c0[0]
}

fn cheb12(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], x01: f32) -> f32 {
    let u = 2.0 * x01 - 1.0;
    let t = 2.0 * u;
    let b11 = c2[3];
    let b10 = t * b11 + c2[2];
    let b9 = t * b10 - b11 + c2[1];
    let b8 = t * b9 - b10 + c2[0];
    let b7 = t * b8 - b9 + c1[3];
    let b6 = t * b7 - b8 + c1[2];
    let b5 = t * b6 - b7 + c1[1];
    let b4 = t * b5 - b6 + c1[0];
    let b3 = t * b4 - b5 + c0[3];
    let b2 = t * b3 - b4 + c0[2];
    let b1 = t * b2 - b3 + c0[1];
    u * b1 - b2 + c0[0]
}

struct UboCtx<'a> {
    u: &'a FlameUBO,
    camera_world: [f32; 3],
    advect: [f32; 3],
    aniso_axis: [f32; 3],
    jitter_scale: f32,
    erf: ErfResponseModel,
}

impl<'a> UboCtx<'a> {
    fn new(u: &'a FlameUBO, camera_world: [f32; 3]) -> Self {
        let sp0 = u.warp_style;
        let sp2 = u.wind_bend;
        let adv_dir = [sp2.wind_direction[0], sp0.rise_speed, sp2.wind_direction[1]];
        let advect = [
            adv_dir[0] * u.time,
            adv_dir[1] * u.time,
            adv_dir[2] * u.time,
        ];
        let mut axis = [0.0, 1.0, 0.0];
        let adv_sq = dot3(adv_dir, adv_dir);
        if u.contour_params.aniso_axis_advect > 0.0 && adv_sq > 1e-8 {
            let inv = 1.0 / adv_sq.sqrt();
            let n = [adv_dir[0] * inv, adv_dir[1] * inv, adv_dir[2] * inv];
            let t = u.contour_params.aniso_axis_advect.clamp(0.0, 1.0);
            let m = [
                mixf(axis[0], n[0], t),
                mixf(axis[1], n[1], t),
                mixf(axis[2], n[2], t),
            ];
            let len = dot3(m, m).sqrt();
            axis = [m[0] / len, m[1] / len, m[2] / len];
        }
        let jitter_scale = if u.wave_jitter[0][3] > 0.0 {
            u.wave_jitter[0][3]
        } else {
            1.0
        };
        let erf = ErfResponseModel {
            center: u.erosion_response.center,
            kappa: u.erosion_response.kappa,
            gaussian_weights: [u.erosion_response.weight1, u.erosion_response.weight2],
        };
        Self {
            u,
            camera_world,
            advect,
            aniso_axis: axis,
            jitter_scale,
            erf,
        }
    }

    fn aniso_compress(&self, v: [f32; 3], axial_scale: f32) -> [f32; 3] {
        let d = dot3(v, self.aniso_axis) * (1.0 - axial_scale);
        [
            v[0] - d * self.aniso_axis[0],
            v[1] - d * self.aniso_axis[1],
            v[2] - d * self.aniso_axis[2],
        ]
    }

    fn aniso_expand(&self, v: [f32; 3], axial_scale: f32) -> [f32; 3] {
        let d = dot3(v, self.aniso_axis) * (1.0 / axial_scale - 1.0);
        [
            v[0] + d * self.aniso_axis[0],
            v[1] + d * self.aniso_axis[1],
            v[2] + d * self.aniso_axis[2],
        ]
    }

    fn bend_offset(&self, h: f32) -> [f32; 2] {
        let sp2 = self.u.wind_bend;
        let s = sp2.bend_amount * h.powf(sp2.bend_power);
        [sp2.wind_direction[0] * s, sp2.wind_direction[1] * s]
    }

    /// flameMeanderOffsetAt mirror: animated meander displacement of the centerline.
    fn meander_offset(&self, h: f32) -> [f32; 2] {
        let t = self.u.time;
        let mut offset = [0.0f32; 2];
        for mode in &self.u.meander_modes {
            let wave = (mode.kappa * h - mode.omega * t + mode.phase).sin();
            offset[0] += wave * mode.direction[0];
            offset[1] += wave * mode.direction[1];
        }
        let amp = self.u.support_motion.meander_amp * h;
        [amp * offset[0], amp * offset[1]]
    }

    /// flameMeanderShifted mirror: p shifted by meander offset at height h.
    fn meander_shifted(&self, p: [f32; 3], h: f32) -> [f32; 3] {
        let off = self.meander_offset(h);
        [p[0] - off[0], p[1], p[2] - off[1]]
    }

    /// flameSupportPosition mirror: meander removed, then pulled back through
    /// the branch elements; returns the coordinate and its height.
    fn support_position(&self, p: [f32; 3], h: f32) -> ([f32; 3], f32) {
        let ps = self.meander_shifted(p, h);
        if self.u.branch_field.count > 0.5 {
            let pulled = branch_pull_back(&self.u.branch_field, ps, self.u.time);
            return (pulled, pulled[1].clamp(0.0, 1.0));
        }
        (ps, h)
    }

    fn wave_mode(&self, slot: usize) -> ([f32; 4], [f32; 4]) {
        (self.u.wave_modes[2 * slot], self.u.wave_modes[2 * slot + 1])
    }

    /// flameMediumSwirlShear mirror: azimuthal transport of the RTE medium
    fn medium_swirl_shear(
        &self,
        z: [f32; 3],
        v: [f32; 3],
        strength: f32,
        displacement: &mut [f32; 3],
        rate: &mut [f32; 3],
    ) {
        let drift_time = self.u.noise_scroll_speed * self.u.time;
        for m in 0..MEDIUM_SWIRL_COUNT {
            let (kv, dv) = self.wave_mode(MEDIUM_SWIRL_BASE + m);
            let k = [kv[0], kv[1], kv[2]];
            let curl = [dv[1], 0.0, dv[3]];
            let angle = dot3(k, z) + dv[0] + dv[2] * drift_time;
            let shear = strength * kv[3] * angle.cos();
            let fp = -strength * kv[3] * angle.sin();
            let kdv = dot3(k, v);
            for axis in 0..3 {
                displacement[axis] += curl[axis] * shear;
                rate[axis] += curl[axis] * fp * kdv;
            }
        }
    }

    /// S1 mirror (flameWarpMapJvp): displacement sum (warp_form_params.x > 0.5)
    /// or the legacy sequential shear composition, with the Jacobian-vector
    /// product on v.
    fn warp_map_jvp(&self, z: &mut [f32; 3], v: &mut [f32; 3], strength: f32) {
        if self.u.warp_form_params.displacement_form > 0.5 {
            let z0 = *z;
            let v0 = *v;
            let mut displacement = [0.0f32; 3];
            let mut rate = [0.0f32; 3];
            for m in 0..WARP_COUNT {
                let (kv, dv) = self.wave_mode(WARP_BASE + m);
                let k = [kv[0], kv[1], kv[2]];
                let curl = [dv[1], dv[2], dv[3]];
                let angle = dot3(k, z0) + dv[0];
                let shear = strength * kv[3] * angle.cos();
                let fp = -strength * kv[3] * angle.sin();
                let kdv = dot3(k, v0);
                for axis in 0..3 {
                    displacement[axis] += curl[axis] * shear;
                    rate[axis] += curl[axis] * fp * kdv;
                }
            }
            self.medium_swirl_shear(z0, v0, strength, &mut displacement, &mut rate);
            for axis in 0..3 {
                z[axis] = z0[axis] + displacement[axis];
                v[axis] = v0[axis] + rate[axis];
            }
            return;
        }
        for m in 0..WARP_COUNT {
            let (kv, dv) = self.wave_mode(WARP_BASE + m);
            let k = [kv[0], kv[1], kv[2]];
            let curl = [dv[1], dv[2], dv[3]];
            let angle = dot3(k, *z) + dv[0];
            let shear = strength * kv[3] * angle.cos();
            let fp = -strength * kv[3] * angle.sin();
            let kdv = dot3(k, *v);
            for axis in 0..3 {
                z[axis] += curl[axis] * shear;
                v[axis] += curl[axis] * fp * kdv;
            }
        }
        let drift_time = self.u.noise_scroll_speed * self.u.time;
        for m in 0..MEDIUM_SWIRL_COUNT {
            let (kv, dv) = self.wave_mode(MEDIUM_SWIRL_BASE + m);
            let k = [kv[0], kv[1], kv[2]];
            let curl = [dv[1], 0.0, dv[3]];
            let angle = dot3(k, *z) + dv[0] + dv[2] * drift_time;
            let shear = strength * kv[3] * angle.cos();
            let fp = -strength * kv[3] * angle.sin();
            let kdv = dot3(k, *v);
            for axis in 0..3 {
                z[axis] += curl[axis] * shear;
                v[axis] += curl[axis] * fp * kdv;
            }
        }
    }

    /// flameWaveFlowWarpRate: returns (warped point q, rate dq/dt along dir).
    fn flow_warp_rate(&self, pb: [f32; 3], dir: [f32; 3], h: f32) -> ([f32; 3], [f32; 3]) {
        let sp0 = self.u.warp_style;
        let strength = self.warp_strength(h);
        if strength == 0.0 {
            return (pb, dir);
        }
        let c = self.aniso_compress(pb, 0.35);
        let mut z = [
            c[0] * sp0.warp_freq - self.advect[0],
            c[1] * sp0.warp_freq - self.advect[1],
            c[2] * sp0.warp_freq - self.advect[2],
        ];
        let cv = self.aniso_compress(dir, 0.35);
        let mut v = [
            cv[0] * sp0.warp_freq,
            cv[1] * sp0.warp_freq,
            cv[2] * sp0.warp_freq,
        ];
        self.warp_map_jvp(&mut z, &mut v, strength);
        let q_pre = [
            z[0] / sp0.warp_freq + self.advect[0] / sp0.warp_freq,
            z[1] / sp0.warp_freq + self.advect[1] / sp0.warp_freq,
            z[2] / sp0.warp_freq + self.advect[2] / sp0.warp_freq,
        ];
        let rate_pre = [
            v[0] / sp0.warp_freq,
            v[1] / sp0.warp_freq,
            v[2] / sp0.warp_freq,
        ];
        (
            self.aniso_expand(q_pre, 0.35),
            self.aniso_expand(rate_pre, 0.35),
        )
    }

    fn jitter_state(&self, w: [f32; 3], rate: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let mut psi = [0.0f32; 3];
        let mut psi_rate = [0.0f32; 3];
        for m in 0..WAVE_JITTER_RANK {
            let k = WAVE_JITTER_K[m];
            let angle = self.jitter_scale * dot3(k, w) + WAVE_JITTER_PHASE[m];
            psi[m] = angle.sin();
            psi_rate[m] = angle.cos() * self.jitter_scale * dot3(k, rate);
        }
        (psi, psi_rate)
    }

    fn detail_noise(&self, p: [f32; 3]) -> f32 {
        let mut z = 0.0f32;
        for n in 0..DETAIL_COUNT {
            let (kv, ph) = self.wave_mode(DETAIL_BASE + n);
            z += kv[3] * (dot3([kv[0], kv[1], kv[2]], p) + ph[0]).sin();
        }
        z * (2.0 / 0.875)
    }

    fn contour_wiggle(&self, p: [f32; 3], h: f32) -> f32 {
        if self.u.contour_params.wiggle_amp == 0.0 || self.unified() {
            return 1.0;
        }
        let q = [
            p[0] * self.u.noise_frequency,
            (h - self.u.warp_style.rise_speed * self.u.time) * self.u.noise_frequency,
            p[2] * self.u.noise_frequency,
        ];
        1.0 + self.u.contour_params.wiggle_amp * self.detail_noise(q)
    }

    fn boundary_displacement(&self, x: f32, z: f32) -> [f32; 2] {
        let bp = self.u.boundary_params;
        if bp.amp == 0.0 || self.unified() {
            return [1.0, 1.0];
        }
        let q = [x * bp.freq, -bp.speed * self.u.time, z * bp.freq];
        let height_noise = ((fbm3(q) * (2.0 / 0.875) - 1.0) * 3.0).min(1.0);
        let radius_noise =
            (fbm3([q[0] + 13.7, q[1] + 41.3, q[2] + 7.9]) * (2.0 / 0.875) - 1.0) * 3.0;
        [
            (1.0 + bp.amp * height_noise).max(0.2),
            (1.0 + bp.amp * bp.radius_ratio * radius_noise).max(0.2),
        ]
    }

    fn cap_fade(&self, h: f32, bx: f32) -> f32 {
        if self.u.boundary_params.amp == 0.0 || bx <= 1.0 {
            return 1.0;
        }
        smoothstep(1.0, 2.0 - bx, h)
    }

    fn height_falloff(&self, hb: f32) -> f32 {
        cheb8(
            self.u.height_coefficients[0],
            self.u.height_coefficients[1],
            hb,
        )
    }

    fn radial_support_radius(&self) -> f32 {
        self.u.support_motion.support_margin
            * (2.0 / self.u.radial_sharpness.max(1e-3f32))
                .sqrt()
                .min(SUPPORT_HEADROOM)
    }

    fn radial_radius_scale(&self, hb: f32) -> f32 {
        if self.u.profile_params.radius_active > 0.5 {
            SHELL_BASE_RADIUS
                * cheb8(
                    self.u.radius_coefficients[0],
                    self.u.radius_coefficients[1],
                    hb,
                )
                .max(0.05)
        } else {
            SHELL_BASE_RADIUS
                * mixf(
                    1.0,
                    self.u.edge_style.radius_tip_ratio,
                    hb.powf(self.u.warp_style.taper_power),
                )
        }
    }

    fn radial_factor(&self, px: f32, pz: f32, hb: f32) -> f32 {
        let scale = (self.radial_support_radius() * self.radial_radius_scale(hb)).max(1e-4);
        let u2 = (px * px + pz * pz) / (scale * scale);
        let inside = (1.0 - u2).max(0.0);
        inside * inside
    }

    fn near_camera_fade(&self, p_local: [f32; 3]) -> f32 {
        let radius = self.u.near_fade_params.radius;
        if radius <= 0.0 {
            return 1.0;
        }
        let pw = transform_point(&self.u.model, p_local);
        let d = [
            pw[0] - self.camera_world[0],
            pw[1] - self.camera_world[1],
            pw[2] - self.camera_world[2],
        ];
        smoothstep(0.0, radius, dot3(d, d).sqrt())
    }

    fn envelope_fade(&self, d_smooth: f32) -> f32 {
        (d_smooth / self.u.edge_style.edge_high.max(1e-3)).min(1.0)
    }

    /// flameEnvelopeRemainingMu: the flame's own height scale shared by the
    /// tip-carve lambda and the warp strain profile.
    fn envelope_remaining_mu(&self, h: f32) -> f32 {
        let primitive = cheb12(
            self.u.height_primitive_coefficients[0],
            self.u.height_primitive_coefficients[1],
            self.u.height_primitive_coefficients[2],
            h,
        );
        let tc = self.u.tip_carve_params;
        ((tc.primitive_top - primitive) * tc.inv_primitive_range).clamp(0.0, 1.0)
    }

    fn unified(&self) -> bool {
        self.u.unified_params.enabled > 0.5
    }

    fn unified_sigma_floor(&self, h: f32, density: f32, u_squared: f32) -> f32 {
        if !self.unified() {
            return 0.0;
        }
        let mut sigma_floor =
            self.u.unified_params.sigma_floor * self.tip_carve_lambda(h) * density
                / EROSION_SHELL_REF;
        sigma_floor *= mixf(
            1.0,
            1.0 - self.u.spread_params.edge_outer_sharpen,
            self.flame_carve_residual_outer_gate(u_squared),
        );
        sigma_floor
    }

    fn tip_carve_lambda(&self, h: f32) -> f32 {
        let tc = self.u.tip_carve_params;
        1.0 + tc.depth * (-self.envelope_remaining_mu(h) * tc.inv_reach).exp()
    }

    /// Mirror of flameBurnoutBoost (D design): remaining-luminous-fraction
    /// deepening factor of the deterministic erosion mean shrink.
    /// warp_form_params.y = burnout gain, tip_carve_params.y = 1/reach.
    fn burnout_boost(&self, h: f32) -> f32 {
        self.u.warp_form_params.burnout_gain
            * (-self.envelope_remaining_mu(h) * self.u.tip_carve_params.inv_reach).exp()
    }

    /// Mirror of flameCarveResidualOuterGate in flame_noise_field.glsl:
    /// Hermite smoothstep from inner (pre-expanded support edge in u^2 units) to 1.0.
    /// margin <= 1.0 returns 0.0 (no boost).
    fn flame_carve_residual_outer_gate(&self, u_squared: f32) -> f32 {
        let margin = self.u.support_motion.support_margin;
        if margin <= 1.0 {
            return 0.0;
        }
        let w = u_squared * margin * margin;
        let t = ((w - 1.0) / 0.5).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// flamePlateauCarveReach: erosion boost reach — absolute core-radius reach function.
    /// w = u_squared * margin * margin is core-radius squared; zero inside core, linear increase outside.
    fn flame_plateau_carve_reach(&self, u_squared: f32) -> f32 {
        let margin = self.u.support_motion.support_margin;
        let w = u_squared * margin * margin;
        (w - 1.0).max(0.0)
    }

    /// flameWarpStrain: asymptotic dimensionless strain of the flow warp.
    fn warp_strain(&self, h: f32) -> f32 {
        let ws = self.u.warp_strain_params;
        ws.strain_base
            + (ws.strain_tip - ws.strain_base)
                * (-self.envelope_remaining_mu(h) * ws.inv_reach).exp()
    }

    /// flameWarpStrength: strain(h) / K.
    fn warp_strength(&self, h: f32) -> f32 {
        self.warp_strain(h) * self.u.warp_strain_params.inv_strain_norm
    }

    /// flameMediumSpreadScale mirror: age-coordinate radial contraction of the
    /// noise sampling toward the luminous tip (reach shares the tip carve).
    fn medium_spread_scale(&self, h: f32) -> f32 {
        let sigma = self.u.spread_params.gain
            * (-self.envelope_remaining_mu(h) * self.u.tip_carve_params.inv_reach).exp();
        (-sigma * h).exp()
    }

    fn eroded_argument(&self, d_smooth: f32, erosion: f32) -> f32 {
        d_smooth - (erosion.max(0.0) + erosion.min(0.0) * self.envelope_fade(d_smooth))
    }

    /// d(shaped)/dz of the tanh noise shaping, expressed through the shaped
    /// value itself (the chain-rule scale from z units to argument units).
    fn shaping_deriv(&self, shaped: f32) -> f32 {
        let inv_scale = self.u.wave_params.inverse_scale;
        let amp = self.u.wave_params.amplitude;
        if inv_scale > 0.0 {
            let tval = (shaped - 0.4375) / amp;
            amp * inv_scale * (1.0 - tval * tval)
        } else {
            1.0
        }
    }

    /// Statistical linearization of the tanh shaping for Gaussian carrier z
    /// of std `sigma_z`: least-squares gain (= E[d shaped/dz] by Stein's
    /// lemma) and the distortion std left outside the linear part. The gain
    /// saturates for deep tanh drive, which bounds the modulation depth to
    /// the true +-amp swing — the raw origin slope amp*inv_scale does not and
    /// inflates sigma by orders of magnitude (P1b diagnosis, 2026-08-09).
    fn shaping_statistical_gain(&self, sigma_z: f32) -> (f32, f32) {
        let inv_scale = self.u.wave_params.inverse_scale;
        let amp = self.u.wave_params.amplitude;
        if inv_scale <= 0.0 {
            return (1.0, 0.0);
        }
        let mut e_sech2 = 0.0f32;
        let mut e_tanh2 = 0.0f32;
        for (node, weight) in GAUSS_HERMITE_8 {
            let t = (inv_scale * std::f32::consts::SQRT_2 * sigma_z * node).tanh();
            e_sech2 += weight * (1.0 - t * t);
            e_tanh2 += weight * t * t;
        }
        let gain = amp * inv_scale * e_sech2;
        let variance = amp * amp * e_tanh2;
        let distortion = (variance - gain * gain * sigma_z * sigma_z).max(0.0).sqrt();
        (gain, distortion)
    }

    /// flameWaveNodeDensity (cylinder / ring generic branch): returns the full
    /// factor decomposition alongside the product.
    fn node_density(&self, p: [f32; 3], h: f32) -> NodeDensity {
        let wiggle = self.contour_wiggle(p, h);
        let boundary = self.boundary_displacement(p[0], p[2]);
        let emitter = self.u.emitter_params.kind;
        let near_fade = self.near_camera_fade(p);
        if emitter < 0.5 {
            let hb = (h / boundary[0]).clamp(0.0, 1.0);
            let wb = (wiggle * boundary[1]).max(1e-4);
            let height_falloff = self.height_falloff(hb);
            let cap_fade = self.cap_fade(h, boundary[0]);
            let radial = self.radial_factor(p[0] / wb, p[2] / wb, hb);
            NodeDensity {
                wiggle,
                boundary,
                height_falloff,
                cap_fade,
                radial,
                near_fade,
                density: height_falloff * cap_fade * radial * near_fade,
            }
        } else {
            // Ring: flameEmitterSmoothDensityDisplacedAt (billboard SDF not replayed).
            let hb = (h / boundary[0]).clamp(0.0, 1.0);
            let taper_r = mixf(
                1.0,
                self.u.edge_style.radius_tip_ratio,
                hb.powf(self.u.warp_style.taper_power),
            );
            let rm = self.u.emitter_params.ring_major_ratio;
            let minor = (1.0 - rm).max(1e-3);
            let rho = ((p[0] * p[0] + p[2] * p[2]).sqrt() - rm) / minor;
            let rn = rho.abs() / (taper_r * wiggle * boundary[1]).max(1e-4);
            let uu = rn / self.radial_support_radius();
            let inside = (1.0 - uu * uu).max(0.0);
            let height_falloff = self.height_falloff(hb);
            let cap_fade = self.cap_fade(h, boundary[0]);
            let radial = inside * inside;
            NodeDensity {
                wiggle,
                boundary,
                height_falloff,
                cap_fade,
                radial,
                near_fade,
                density: height_falloff * radial * cap_fade * near_fade,
            }
        }
    }

    /// flameMediumTwistAngle mirror: node-frozen azimuthal twist of the noise
    /// coordinate (Lamb-Oseen radial profile x two counter-rotating axial modes).
    fn twist_angle(&self, r_squared: f32, h: f32) -> f32 {
        let twist = &self.u.twist_field;
        let radial = twist.core_radius_sq / (r_squared + twist.core_radius_sq);
        let t = self.u.time;
        let wave: f32 = twist
            .modes
            .iter()
            .map(|mode| mode.amp * (mode.kappa * h + mode.omega * t + mode.phase).cos())
            .sum();
        self.u.spread_params.twist_gain * radial * h * wave
    }

    /// Warped noise frame shared by the node argument and the segment mode
    /// frame: bent point, warp image, noise coordinate, its rate along the
    /// ray, and the jitter state.
    fn wave_frame(&self, p: [f32; 3], d: [f32; 3], h: f32) -> WaveFrame {
        let bend = self.bend_offset(h);
        let mut pbu = [p[0] - bend[0], p[1], p[2] - bend[1]];
        let mut du = d;
        let mut h = h;
        if self.u.branch_field.count > 0.5 {
            let (pulled, pulled_dir) =
                branch_pull_back_jvp(&self.u.branch_field, pbu, du, self.u.time);
            pbu = pulled;
            du = pulled_dir;
            h = pulled[1].clamp(0.0, 1.0);
        }
        if self.u.spread_params.twist_gain != 0.0 {
            let r_squared = pbu[0] * pbu[0] + pbu[2] * pbu[2];
            let phi = self.twist_angle(r_squared, h);
            let (sp, cp) = phi.sin_cos();
            pbu = [cp * pbu[0] - sp * pbu[2], pbu[1], sp * pbu[0] + cp * pbu[2]];
            du = [cp * d[0] - sp * d[2], d[1], sp * d[0] + cp * d[2]];
        }
        let spread = self.medium_spread_scale(h);
        let spread_y = 1.0 / (spread * spread);
        let pb = [pbu[0] * spread, pbu[1] * spread_y, pbu[2] * spread];
        let ds = [du[0] * spread, du[1] * spread_y, du[2] * spread];
        let (q, rate_raw) = self.flow_warp_rate(pb, ds, h);
        let cw = self.aniso_compress(q, self.u.temporal_data.noise_aniso_y);
        let w = [
            cw[0] * self.u.noise_frequency - self.advect[0],
            cw[1] * self.u.noise_frequency - self.advect[1],
            cw[2] * self.u.noise_frequency - self.advect[2],
        ];
        let cr = self.aniso_compress(rate_raw, self.u.temporal_data.noise_aniso_y);
        let rate = [
            cr[0] * self.u.noise_frequency,
            cr[1] * self.u.noise_frequency,
            cr[2] * self.u.noise_frequency,
        ];
        let (jitter_psi, jitter_psi_rate) = self.jitter_state(w, rate);
        WaveFrame {
            pb,
            q,
            w,
            rate,
            jitter_psi,
            jitter_psi_rate,
            h,
        }
    }

    /// flameWaveNodeArgumentLocal with every intermediate recorded.
    fn node_argument(
        &self,
        p: [f32; 3],
        d: [f32; 3],
        h: f32,
        density: f32,
        dt: f32,
    ) -> NodeArgument {
        let WaveFrame {
            pb,
            q,
            w,
            rate,
            jitter_psi,
            jitter_psi_rate,
            h: hs,
        } = self.wave_frame(p, d, h);

        // Compute u_squared (normalized radius squared) for plateau boost.
        // Use the support position for density-side coordinate, matching the shader.
        let (ps, h_support) = self.support_position(p, h);
        let boundary = self.boundary_displacement(ps[0], ps[2]);
        let hb = (h_support / boundary[0]).clamp(0.0, 1.0);
        let emitter = self.u.emitter_params.kind;
        let u_squared = if emitter < 0.5 {
            // Cylinder: normalized radius squared from radial_factor scale
            let scale = (self.radial_support_radius() * self.radial_radius_scale(hb)).max(1e-4);
            (ps[0] * ps[0] + ps[2] * ps[2]) / (scale * scale)
        } else {
            // Ring: from radial distance normalized by support radius
            let rm = self.u.emitter_params.ring_major_ratio;
            let minor = (1.0 - rm).max(1e-3);
            let rho = ((ps[0] * ps[0] + ps[2] * ps[2]).sqrt() - rm) / minor;
            let rn = rho.abs() / (boundary[1]).max(1e-4);
            let uu = rn / self.radial_support_radius();
            uu * uu
        };

        let eddy_time = self.u.noise_scroll_speed * self.u.time;
        let mut z_low = 0.0f32;
        let mut unresolved = 0.0f32;
        let mut mode_values = Vec::new();
        let mut mode_weights = Vec::new();
        let record_modes = dt > 0.0;
        let count = (self.u.wave_params.tracked_count as usize).min(EROSION_SLOTS);
        for pass in 0..2 {
            let mut z_acc = 0.0f32;
            for n in 0..count {
                let (kv, ph) = self.wave_mode(n);
                let is_high = ph[2] != 0.0;
                if (pass == 0) == is_high {
                    continue;
                }
                let k = [kv[0], kv[1], kv[2]];
                let jn = n.min(self.u.wave_jitter.len() - 1);
                let jit = [
                    self.u.wave_jitter[jn][0],
                    self.u.wave_jitter[jn][1],
                    self.u.wave_jitter[jn][2],
                ];
                let angle = dot3(k, w) + ph[0] + ph[1] * eddy_time + dot3(jit, jitter_psi);
                let beta_phase = dot3(k, rate) + dot3(jit, jitter_psi_rate);
                let beta = beta_phase * dt / std::f32::consts::PI;
                let b2 = beta * beta;
                let weight = (-b2 * b2).exp();
                let carrier = angle.sin();
                if pass == 0 {
                    z_acc += weight * kv[3] * carrier;
                    unresolved += 0.5 * kv[3] * kv[3] * (1.0 - weight * weight);
                } else {
                    let envelope = 1.0 + ph[2] * z_low;
                    z_acc += envelope * weight * kv[3] * carrier;
                    unresolved +=
                        envelope * envelope * 0.5 * kv[3] * kv[3] * (1.0 - weight * weight);
                }
                if record_modes {
                    mode_values.push(kv[3] * carrier);
                    mode_weights.push(weight);
                }
            }
            if pass == 0 {
                z_low = z_acc;
            } else {
                z_low += z_acc; // z = z_low + high sum; reuse variable below
            }
        }
        let z = z_low; // after both passes z_low holds the full sum
                       // Recompute the true z_low (first pass only) for the record.
        let mut z_low_only = 0.0f32;
        for n in 0..count {
            let (kv, ph) = self.wave_mode(n);
            if ph[2] != 0.0 {
                continue;
            }
            let k = [kv[0], kv[1], kv[2]];
            let jit = [
                self.u.wave_jitter[n][0],
                self.u.wave_jitter[n][1],
                self.u.wave_jitter[n][2],
            ];
            let angle = dot3(k, w) + ph[0] + ph[1] * eddy_time + dot3(jit, jitter_psi);
            let beta_phase = dot3(k, rate) + dot3(jit, jitter_psi_rate);
            let beta = beta_phase * dt / std::f32::consts::PI;
            let b2 = beta * beta;
            let weight = (-b2 * b2).exp();
            z_low_only += weight * kv[3] * angle.sin();
        }

        let mut unresolved_total = unresolved + self.u.wave_cf_params.skipped_power_plain;
        let env_skip = 1.0 + self.u.wave_params.env_coeff * z_low_only;
        unresolved_total += self.u.wave_cf_params.skipped_power_env * env_skip * env_skip;

        let sigma_noise = unresolved_total.sqrt();
        let inv_scale = self.u.wave_params.inverse_scale;
        let amp = self.u.wave_params.amplitude;
        let shaped = if inv_scale > 0.0 {
            0.4375 + amp * (z * inv_scale).tanh()
        } else {
            0.4375 + z
        };
        let lambda = self.tip_carve_lambda(hs);
        let mu = self.envelope_remaining_mu(hs);
        let strain = self.warp_strain(hs);
        let erosion = self.u.noise_amplitude
            * (mixf(0.2, 1.0, hs) * EROSION_MEAN_SHRINK * (1.0 + self.burnout_boost(hs))
                + PLATEAU_CARVE_BOOST * self.flame_plateau_carve_reach(u_squared)
                + self.u.spread_params.erosion_noise_gain
                    * lambda
                    * (density / EROSION_SHELL_REF)
                    * (shaped - 0.4375));
        let argument = self.eroded_argument(density, erosion);
        NodeArgument {
            pb,
            q,
            w,
            rate,
            jitter_psi,
            jitter_psi_rate,
            z_low: z_low_only,
            z,
            shaped,
            sigma_noise,
            lambda,
            mu,
            strain,
            erosion,
            envelope_fade: self.envelope_fade(density),
            argument,
            mode_values,
            mode_weights,
        }
    }

    /// Carrier constants shared by every point of a ray: per-mode amplitude
    /// with the high-mode envelope folded as statistical power (sampling its
    /// realized value would re-imprint the sampling lattice), the non-modal
    /// unresolved std and the total carrier std (both z units).
    fn carrier_amplitudes(&self) -> CarrierAmplitudes {
        let count = (self.u.wave_params.tracked_count as usize).min(EROSION_SLOTS);

        let mut low_power = 0.0f32;
        for n in 0..count {
            let (kv, ph) = self.wave_mode(n);
            if ph[2] == 0.0 {
                low_power += 0.5 * kv[3] * kv[3];
            }
        }

        let mut amplitudes = Vec::with_capacity(count);
        let mut modal_power = 0.0f32;
        let mut k_min = f32::INFINITY;
        for n in 0..count {
            let (kv, ph) = self.wave_mode(n);
            let envelope_rms = if ph[2] != 0.0 {
                (1.0 + ph[2] * ph[2] * low_power).sqrt()
            } else {
                1.0
            };
            let amplitude = envelope_rms * kv[3];
            modal_power += 0.5 * amplitude * amplitude;
            k_min = k_min.min(dot3([kv[0], kv[1], kv[2]], [kv[0], kv[1], kv[2]]).sqrt());
            amplitudes.push(amplitude);
        }
        if !k_min.is_finite() {
            k_min = 1.0;
        }

        let env_skip_power =
            1.0 + self.u.wave_params.env_coeff * self.u.wave_params.env_coeff * low_power;
        let sigma_base = (self.u.wave_cf_params.skipped_power_plain
            + self.u.wave_cf_params.skipped_power_env * env_skip_power)
            .max(0.0)
            .sqrt();
        let sigma_z = (modal_power + sigma_base * sigma_base).sqrt();
        CarrierAmplitudes {
            amplitudes,
            sigma_base,
            sigma_z,
            modal_power,
            k_min,
        }
    }

    /// Carrier state at one ray point: the slow value is the exact per-mode
    /// capture sum at the ray-fixed reference cutoff (no compression — any
    /// lossy per-node reconstruction turns into argument noise at the shell),
    /// while the modal power is hat-distributed over log2(|omega|) bands for
    /// the segment-local sigma fold.
    fn carrier_slow_state(
        &self,
        p: [f32; 3],
        d: [f32; 3],
        h: f32,
        carrier: &CarrierAmplitudes,
        alpha_ref: f32,
    ) -> CarrierSlowState {
        let frame = self.wave_frame(p, d, h);
        let eddy_time = self.u.noise_scroll_speed * self.u.time;
        let count = (self.u.wave_params.tracked_count as usize).min(EROSION_SLOTS);

        let rate_len = dot3(frame.rate, frame.rate).sqrt();
        let omega0 = (carrier.k_min * rate_len).max(1e-3) * 0.5;
        let inv_ln2 = std::f32::consts::LOG2_E;

        let mut z_slow = 0.0f32;
        let mut power_band = [0.0f32; CAPTURE_BANDS];
        for n in 0..count {
            let (kv, ph) = self.wave_mode(n);
            let k = [kv[0], kv[1], kv[2]];
            let jn = n.min(self.u.wave_jitter.len() - 1);
            let jit = [
                self.u.wave_jitter[jn][0],
                self.u.wave_jitter[jn][1],
                self.u.wave_jitter[jn][2],
            ];
            let angle = dot3(k, frame.w) + ph[0] + ph[1] * eddy_time + dot3(jit, frame.jitter_psi);
            let omega = dot3(k, frame.rate) + dot3(jit, frame.jitter_psi_rate);
            let amplitude = carrier.amplitudes[n];
            let power = 0.5 * amplitude * amplitude;

            let kappa = omega * alpha_ref * std::f32::consts::FRAC_1_SQRT_2;
            let g_ref = if kappa.is_finite() {
                (-kappa * kappa).exp()
            } else {
                0.0
            };
            z_slow += g_ref * amplitude * angle.sin();

            let u = ((omega.abs() / omega0).max(1e-6).ln() * inv_ln2)
                .clamp(0.0, (CAPTURE_BANDS - 1) as f32);
            for band in 0..CAPTURE_BANDS {
                let hat = (1.0 - (u - band as f32).abs()).max(0.0);
                power_band[band] += hat * power;
            }
        }
        CarrierSlowState {
            z_slow,
            power_band,
            omega0,
        }
    }

    /// Conditional mean of `shaped - 0.4375` given the resolved slow carrier
    /// `u`, averaging the tanh over the unresolved Gaussian residual of std
    /// `sigma_fast` (both z units). The saturation of the slow swing — the
    /// dominant nonlinearity at production modulation depth — is kept exact.
    fn shaped_delta_mean(&self, u: f32, sigma_fast: f32) -> f32 {
        let inv_scale = self.u.wave_params.inverse_scale;
        let amp = self.u.wave_params.amplitude;
        if inv_scale <= 0.0 {
            return u;
        }
        let mut mean = 0.0f32;
        for (node, weight) in GAUSS_HERMITE_8 {
            let offset = std::f32::consts::SQRT_2 * sigma_fast * node;
            mean += weight
                * 0.5
                * ((inv_scale * (u + offset)).tanh() + (inv_scale * (u - offset)).tanh());
        }
        amp * mean
    }

    fn ramp_color(&self, h: f32) -> [f32; 3] {
        if self.u.profile_params.color_active > 0.5 {
            let u = h.clamp(0.0, 1.0) * 8.0 - 0.5;
            let i0 = (u.floor().clamp(0.0, 7.0)) as usize;
            let i1 = (i0 + 1).min(7);
            let f = (u - i0 as f32).clamp(0.0, 1.0);
            let a = self.u.color_ramp[i0];
            let b = self.u.color_ramp[i1];
            mix3([a[0], a[1], a[2]], [b[0], b[1], b[2]], f)
        } else if h < 0.5 {
            mix3(
                [
                    self.u.color_base.rgb[0],
                    self.u.color_base.rgb[1],
                    self.u.color_base.rgb[2],
                ],
                [
                    self.u.color_mid.rgb[0],
                    self.u.color_mid.rgb[1],
                    self.u.color_mid.rgb[2],
                ],
                h * 2.0,
            )
        } else {
            mix3(
                [
                    self.u.color_mid.rgb[0],
                    self.u.color_mid.rgb[1],
                    self.u.color_mid.rgb[2],
                ],
                [
                    self.u.color_tip.rgb[0],
                    self.u.color_tip.rgb[1],
                    self.u.color_tip.rgb[2],
                ],
                (h - 0.5) * 2.0,
            )
        }
    }

    /// computeSelfShadowTau (layered concentric cylinders).
    fn self_shadow_tau(&self, p: [f32; 3], l: [f32; 3]) -> f32 {
        let s = [1.0f32 / 3.0, 2.0 / 3.0, 1.0];
        let m = [1.0f32 / 6.0, 0.5, 5.0 / 6.0];
        let mut dens = [0.0f32; 4];
        for k in 0..3 {
            dens[k] = cheb8(
                self.u.radial_coefficients[0],
                self.u.radial_coefficients[1],
                m[k],
            );
        }
        let w = [dens[0] - dens[1], dens[1] - dens[2], dens[2] - dens[3]];
        let (px, py, pz) = (p[0], p[1], p[2]);
        let (lx, ly, lz) = (l[0], l[1], l[2]);
        let mut total = 0.0f32;
        for k in 0..3 {
            let sk = s[k];
            let a = lx * lx + lz * lz;
            let (s0, s1);
            if a < 1e-6 {
                if px * px + pz * pz <= sk * sk {
                    s0 = 0.0;
                    s1 = 1e4;
                } else {
                    continue;
                }
            } else {
                let b = 2.0 * (px * lx + pz * lz);
                let c = px * px + pz * pz - sk * sk;
                let disc = b * b - 4.0 * a * c;
                if disc <= 0.0 {
                    continue;
                }
                let root = disc.sqrt();
                let mut lo = (-b - root) / (2.0 * a);
                let hi = (-b + root) / (2.0 * a);
                if hi < 0.0 {
                    continue;
                }
                if lo < 0.0 {
                    lo = 0.0;
                }
                s0 = lo;
                s1 = hi;
            }
            let (mut lo, mut hi) = (s0, s1);
            if ly.abs() < 1e-4 {
                if !(0.0..=1.0).contains(&py) {
                    continue;
                }
                let f_val = cheb8(
                    self.u.height_coefficients[0],
                    self.u.height_coefficients[1],
                    py,
                );
                total += w[k] * f_val * (hi - lo);
            } else {
                let mut s_lo = (0.0 - py) / ly;
                let mut s_hi = (1.0 - py) / ly;
                if s_lo > s_hi {
                    std::mem::swap(&mut s_lo, &mut s_hi);
                }
                lo = lo.max(s_lo);
                hi = hi.min(s_hi);
                if lo >= hi {
                    continue;
                }
                let h_s0 = py + lo * ly;
                let h_s1 = py + hi * ly;
                let h1_s0 = cheb12(
                    self.u.height_primitive_coefficients[0],
                    self.u.height_primitive_coefficients[1],
                    self.u.height_primitive_coefficients[2],
                    h_s0,
                );
                let h1_s1 = cheb12(
                    self.u.height_primitive_coefficients[0],
                    self.u.height_primitive_coefficients[1],
                    self.u.height_primitive_coefficients[2],
                    h_s1,
                );
                total += w[k] * (h1_s1 - h1_s0) / ly;
            }
        }
        total * self.u.sigma_t
    }
}

struct NodeDensity {
    wiggle: f32,
    boundary: [f32; 2],
    height_falloff: f32,
    cap_fade: f32,
    radial: f32,
    near_fade: f32,
    density: f32,
}

struct WaveFrame {
    pb: [f32; 3],
    q: [f32; 3],
    w: [f32; 3],
    rate: [f32; 3],
    jitter_psi: [f32; 3],
    jitter_psi_rate: [f32; 3],
    h: f32,
}

/// Ray-constant carrier data (z units): per-mode amplitudes with the
/// high-mode envelope folded as power, non-modal unresolved std, total std,
/// total modal power and the largest wave number (capture grid scale).
struct CarrierAmplitudes {
    amplitudes: Vec<f32>,
    sigma_base: f32,
    sigma_z: f32,
    modal_power: f32,
    k_min: f32,
}

/// Per-node carrier state: the resolved slow value is exact at the
/// ray-fixed reference cutoff; the band powers let a segment evaluate the
/// sigma-side folded power at any local cutoff without per-mode state.
#[derive(Clone, Copy)]
struct CarrierSlowState {
    z_slow: f32,
    power_band: [f32; CAPTURE_BANDS],
    omega0: f32,
}

impl CarrierSlowState {
    fn captured_power(&self, alpha: f32) -> f32 {
        if !alpha.is_finite() {
            return 0.0;
        }
        let mut sum = 0.0;
        for band in 0..CAPTURE_BANDS {
            let omega_band = self.omega0 * (1u32 << band) as f32;
            let k = omega_band * alpha * std::f32::consts::FRAC_1_SQRT_2;
            let g = if k.is_finite() { (-k * k).exp() } else { 0.0 };
            sum += self.power_band[band] * g * g;
        }
        sum
    }
}

struct SegmentEstimate {
    integral: f32,
    first_moment: f32,
    sigma: f32,
    shaping_deriv: f32,
    linear_correction: f32,
    sigma_eff_raw: f32,
    sigma_floor: f32,
}

/// Faddeeva estimator v3 (continuous ray integrator P1b, deep-modulation
/// revision): captured (slow) modes are tracked as values inside the
/// argument itself — their physical cutoff sits far below any segment
/// Nyquist, so value sampling cannot alias — while uncaptured modes fold
/// into the smoothing sigma through the tanh conditional statistics.
fn mean_argument_at(ctx: &UboCtx, o: [f32; 3], d: [f32; 3], t: f32) -> f32 {
    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    let (ps, hs) = ctx.support_position(p, p[1].clamp(0.0, 1.0));
    let density = ctx.node_density(ps, hs).density;
    // Compute u_squared (normalized radius squared) for plateau boost.
    let boundary = ctx.boundary_displacement(ps[0], ps[2]);
    let hb = (hs / boundary[0]).clamp(0.0, 1.0);
    let emitter = ctx.u.emitter_params.kind;
    let u_squared = if emitter < 0.5 {
        let scale = (ctx.radial_support_radius() * ctx.radial_radius_scale(hb)).max(1e-4);
        (ps[0] * ps[0] + ps[2] * ps[2]) / (scale * scale)
    } else if emitter < 1.5 {
        let taper_r = mixf(
            1.0,
            ctx.u.edge_style.radius_tip_ratio,
            hb.powf(ctx.u.warp_style.taper_power),
        );
        let rm = ctx.u.emitter_params.ring_major_ratio;
        let minor = (1.0 - rm).max(1e-3);
        let rho = ((ps[0] * ps[0] + ps[2] * ps[2]).sqrt() - rm) / minor;
        let rn = rho.abs() / (taper_r * boundary[1]).max(1e-4);
        let uu = rn / ctx.radial_support_radius();
        uu * uu
    } else {
        0.0
    };
    let mean_erosion = ctx.u.noise_amplitude
        * (mixf(0.2, 1.0, hs) * EROSION_MEAN_SHRINK * (1.0 + ctx.burnout_boost(hs))
            + PLATEAU_CARVE_BOOST * ctx.flame_plateau_carve_reach(u_squared));
    ctx.eroded_argument(density, mean_erosion)
}

/// Argument-unit sigma of the folded carrier share at the given geometry:
/// gain-passed base + folded modal power, plus the tanh distortion residual.
fn folded_sigma_argument(
    geometry: f32,
    gain: f32,
    distortion: f32,
    carrier: &CarrierAmplitudes,
    folded_power: f32,
) -> f32 {
    let base_z = carrier.sigma_base * carrier.sigma_base + folded_power;
    (geometry * geometry * (gain * gain * base_z + distortion * distortion)).sqrt()
}

/// Full argument at one point: density, mean shrink, and the conditional tanh
/// mean of the resolved slow carrier over the folded residual.
#[allow(clippy::too_many_arguments)]
fn argument_with_slow(
    ctx: &UboCtx,
    o: [f32; 3],
    d: [f32; 3],
    t: f32,
    density: f32,
    z_slow: f32,
    sigma_fast: f32,
) -> f32 {
    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    let h = p[1].clamp(0.0, 1.0);
    let boundary = ctx.boundary_displacement(p[0], p[2]);
    let hb = (h / boundary[0]).clamp(0.0, 1.0);
    let emitter = ctx.u.emitter_params.kind;
    let u_squared = if emitter < 0.5 {
        let scale = (ctx.radial_support_radius() * ctx.radial_radius_scale(hb)).max(1e-4);
        (p[0] * p[0] + p[2] * p[2]) / (scale * scale)
    } else {
        let rm = ctx.u.emitter_params.ring_major_ratio;
        let minor = (1.0 - rm).max(1e-3);
        let rho = ((p[0] * p[0] + p[2] * p[2]).sqrt() - rm) / minor;
        let rn = rho.abs() / (boundary[1]).max(1e-4);
        let uu = rn / ctx.radial_support_radius();
        uu * uu
    };
    let shaped_delta = ctx.shaped_delta_mean(z_slow, sigma_fast);
    let erosion = ctx.u.noise_amplitude
        * (mixf(0.2, 1.0, h) * EROSION_MEAN_SHRINK * (1.0 + ctx.burnout_boost(h))
            + PLATEAU_CARVE_BOOST * ctx.flame_plateau_carve_reach(u_squared)
            + ctx.u.spread_params.erosion_noise_gain
                * ctx.tip_carve_lambda(h)
                * (density / EROSION_SHELL_REF)
                * shaped_delta);
    ctx.eroded_argument(density, erosion)
}

/// Ray-fixed reference capture cutoff: the sharpest (smallest-alpha) mean-line
/// shell crossing decides which modes the whole ray resolves as values. The
/// scan is density-only (no mode loops) on a lattice-independent grid; the
/// shell width uses the conservative full-fold sigma at the crossing.
fn solve_reference_cutoff(
    ctx: &UboCtx,
    o: [f32; 3],
    d: [f32; 3],
    t0: f32,
    span_total: f32,
    carrier: &CarrierAmplitudes,
    gain: f32,
    distortion: f32,
) -> f32 {
    let center = ctx.erf.center;
    let f = |t: f32| mean_argument_at(ctx, o, d, t) - center;

    let scan_dt = span_total / CROSSING_SCAN_INTERVALS as f32;
    let slope_eps = 1e-3 * span_total;
    let mut alpha_ref = f32::INFINITY;
    let mut f_a = f(t0);
    for interval in 0..CROSSING_SCAN_INTERVALS {
        let t_a = t0 + interval as f32 * scan_dt;
        let t_b = t_a + scan_dt;
        let f_b = f(t_b);
        if f_a * f_b < 0.0 {
            let (mut lo, mut hi) = (t_a, t_b);
            let lo_negative = f_a < 0.0;
            for _ in 0..CROSSING_BISECTION_STEPS {
                let mid = 0.5 * (lo + hi);
                if (f(mid) < 0.0) == lo_negative {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t_star = 0.5 * (lo + hi);
            let p_star = [
                o[0] + t_star * d[0],
                o[1] + t_star * d[1],
                o[2] + t_star * d[2],
            ];
            let (ps_star, h_star) = ctx.support_position(p_star, p_star[1].clamp(0.0, 1.0));
            let density = ctx.node_density(ps_star, h_star).density;
            if density > 0.0 {
                let fade = ctx.envelope_fade(density);
                let geometry = ctx.u.noise_amplitude
                    * ctx.tip_carve_lambda(h_star)
                    * (density / EROSION_SHELL_REF)
                    * fade;
                let u_squared = (h_star - 1.0).powi(2);
                let sigma_floor = ctx.unified_sigma_floor(h_star, density, u_squared) * fade;
                let sigma_full =
                    folded_sigma_argument(geometry, gain, distortion, carrier, carrier.modal_power)
                        .max(sigma_floor);
                let kappa_eff = smooth_erf_response(&ctx.erf, sigma_full).kappa_eff;
                let shell_width = 1.0 / (std::f32::consts::SQRT_2 * kappa_eff);
                let slope = (f(t_star + slope_eps) - f(t_star - slope_eps)) / (2.0 * slope_eps);
                if slope.abs() > 1e-6 {
                    alpha_ref = alpha_ref.min(shell_width / slope.abs());
                }
            }
        }
        f_a = f_b;
    }
    alpha_ref
}

/// Faddeeva estimator (continuous ray integrator, deep-modulation form):
/// modes below the ray's reference cutoff are tracked as exact values inside
/// the argument (their cutoff sits far below any segment Nyquist, so value
/// sampling cannot alias); the sigma fold adapts per segment to the realized
/// slope through the band powers, folding additionally wherever the shell is
/// crossed slower than at the reference crossing.
#[allow(clippy::too_many_arguments)]
fn faddeeva_segment_estimate(
    ctx: &UboCtx,
    o: [f32; 3],
    d: [f32; 3],
    seg_start: f32,
    seg_end: f32,
    density_start: f32,
    density_end: f32,
    h_mid: f32,
    fade_avg: f32,
    carrier: &CarrierAmplitudes,
    gain: f32,
    distortion: f32,
    alpha_ref: f32,
    state_start: &CarrierSlowState,
    state_end: &CarrierSlowState,
) -> SegmentEstimate {
    let span = seg_end - seg_start;
    let density_avg = 0.5 * (density_start + density_end);
    let geometry = ctx.u.noise_amplitude
        * ctx.tip_carve_lambda(h_mid)
        * (density_avg / EROSION_SHELL_REF)
        * fade_avg;
    let u_squared = (h_mid - 1.0).powi(2);
    let sigma_floor = ctx.unified_sigma_floor(h_mid, density_avg, u_squared) * fade_avg;

    let captured_ref =
        0.5 * (state_start.captured_power(alpha_ref) + state_end.captured_power(alpha_ref));
    let sigma_fast = (carrier.sigma_base * carrier.sigma_base
        + (carrier.modal_power - captured_ref).max(0.0))
    .sqrt();
    let arg_start = argument_with_slow(
        ctx,
        o,
        d,
        seg_start,
        density_start,
        state_start.z_slow,
        sigma_fast,
    );
    let arg_end = argument_with_slow(
        ctx,
        o,
        d,
        seg_end,
        density_end,
        state_end.z_slow,
        sigma_fast,
    );
    let slope = (arg_end - arg_start) / span;
    // Segment-local sigma: fold everything the local realized slope cannot
    // resolve. Two fixed-point iterations settle sigma <-> alpha.
    let mut folded = (carrier.modal_power - captured_ref).max(0.0);
    let mut sigma_smooth = 0.0;
    for _ in 0..2 {
        let sigma_raw = folded_sigma_argument(geometry, gain, distortion, carrier, folded);
        sigma_smooth = sigma_raw.max(sigma_floor);
        let kappa_eff = smooth_erf_response(&ctx.erf, sigma_smooth).kappa_eff;
        let shell_width = 1.0 / (std::f32::consts::SQRT_2 * kappa_eff);
        let alpha_local = if slope.abs() > 1e-6 {
            (shell_width / slope.abs()).max(alpha_ref)
        } else {
            f32::INFINITY
        };
        let captured_local =
            0.5 * (state_start.captured_power(alpha_local) + state_end.captured_power(alpha_local));
        folded = (carrier.modal_power - captured_local.min(captured_ref)).max(0.0);
    }

    let (integral, first_moment) = integrate_erf_response_linear(
        &ctx.erf,
        sigma_smooth,
        arg_start - slope * seg_start,
        slope,
        seg_start,
        seg_end,
    );
    SegmentEstimate {
        integral,
        first_moment,
        sigma: sigma_smooth,
        shaping_deriv: gain,
        linear_correction: 0.0,
        sigma_eff_raw: folded_sigma_argument(geometry, gain, distortion, carrier, folded),
        sigma_floor,
    }
}

struct NodeArgument {
    pb: [f32; 3],
    q: [f32; 3],
    w: [f32; 3],
    rate: [f32; 3],
    jitter_psi: [f32; 3],
    jitter_psi_rate: [f32; 3],
    z_low: f32,
    z: f32,
    shaped: f32,
    sigma_noise: f32,
    lambda: f32,
    mu: f32,
    strain: f32,
    erosion: f32,
    envelope_fade: f32,
    argument: f32,
    mode_values: Vec<f32>,
    mode_weights: Vec<f32>,
}

fn slab_interval(origin_y: f32, dir_y: f32, y_top: f32, t_max: f32) -> Option<(f32, f32)> {
    if dir_y.abs() < 1e-6 {
        return (0.0..=y_top).contains(&origin_y).then_some((0.0, t_max));
    }
    let a = (0.0 - origin_y) / dir_y;
    let b = (y_top - origin_y) / dir_y;
    let lo = a.min(b).max(0.0);
    let hi = a.max(b).min(t_max);
    (hi > lo).then_some((lo, hi))
}

fn outer_cylinder_interval(
    o: [f32; 3],
    d: [f32; 3],
    r_out: f32,
    t_near: f32,
    t_far: f32,
) -> Option<(f32, f32)> {
    let a = d[0] * d[0] + d[2] * d[2];
    let b = 2.0 * (o[0] * d[0] + o[2] * d[2]);
    let c = o[0] * o[0] + o[2] * o[2] - r_out * r_out;
    if a < 1e-12 {
        return (c <= 0.0).then_some((t_near, t_far));
    }
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let lo = ((-b - root) / (2.0 * a)).max(t_near);
    let hi = ((-b + root) / (2.0 * a)).min(t_far);
    (hi > lo).then_some((lo, hi))
}

fn branch_field_json(field: &thyllore_effect_core::FlameBranchField) -> Value {
    let count = (field.count as usize).min(thyllore_effect_core::BRANCH_MAX_ELEMENTS);
    json!({
        "count": field.count,
        "period": round5(field.period),
        "life": round5(field.life),
        "gain": round5(field.gain),
        "rise_rate": round5(field.rise_rate),
        "drift_rate": round5(field.drift_rate),
        "aspect": round5(field.aspect),
        "core_radius": round5(field.core_radius),
        "ring_radius": vec_json(&[field.ring_radius_start, field.ring_radius_end]),
        "envelope_time": round5(field.envelope_time),
        "arc_half_width": round5(field.arc_half_width),
        "bounding_pad": vec_json(&[field.bounding_pad, field.bounding_pad_y]),
        "elements": field.elements[..count]
            .iter()
            .map(|e| vec_json(&[e.spawn_time, e.side, e.azimuth, e.spawn_height, e.kind, e.hash01]))
            .collect::<Vec<_>>(),
    })
}

fn round5(x: f32) -> Value {
    if !x.is_finite() {
        return json!(null);
    }
    // 6 significant digits keeps the file readable and diffable.
    let s = format!("{:.6e}", x);
    let v: f64 = s.parse().unwrap_or(0.0);
    json!(v)
}

fn vec_json(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|v| round5(*v)).collect())
}

/// Trace every intermediate of the analytic flame path for a grid of view
/// rays, taking the flame's ECS components as the unit of input. The UBO the
/// shader sees is derived through the same builder the renderer uses.
pub fn trace_flame_field(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
    view: &WallProbeView,
) -> Value {
    trace_flame_field_ubo(&build_flame_ubo(effect, baked, temporal), view)
}

/// UBO-level entry for replay tooling that reconstructs a dumped UBO directly.
/// `mode_trace_ray` (col, row) selects one ray that additionally records
/// the per-mode carrier values and low-pass weights at every node.
pub fn trace_flame_field_ubo(ubo: &FlameUBO, view: &WallProbeView) -> Value {
    let cols = env_usize("THYLLORE_FLAME_TRACE_COLS", 13);
    let rows = env_usize("THYLLORE_FLAME_TRACE_ROWS", 49);
    let segments = env_usize("THYLLORE_FLAME_TRACE_SEGMENTS", SEGMENTS);
    let integrator = SegmentIntegrator::from_env();
    let apply_jitter = env_usize("THYLLORE_FLAME_TRACE_JITTER", 1) != 0;
    let inverse_model = ubo.inverse_model;
    let ctx = UboCtx::new(ubo, view.position);

    let tan_half = (view.fov_y_radians * 0.5).tan();
    let aspect = (view.viewport_size_px[0] / view.viewport_size_px[1].max(1.0)).max(1e-3);
    let forward = Vector3::from(view.forward);
    let right = Vector3::from(view.right);
    let up = Vector3::from(view.up);
    let camera_local = transform_point(&inverse_model, view.position);
    let t_max = dot3(camera_local, camera_local).sqrt() + 4.0;

    let wiggle_trim = 1.0 + ctx.u.contour_params.wiggle_amp.max(0.0);
    let boundary_trim =
        1.0 + 3.0 * ctx.u.boundary_params.amp.abs() * ctx.u.boundary_params.radius_ratio.max(0.0);
    let taper_max = ctx.u.edge_style.radius_tip_ratio.max(1.0);
    let r_out = SHELL_BASE_RADIUS * SUPPORT_HEADROOM * taper_max * wiggle_trim * boundary_trim
        + ctx.u.branch_field.bounding_pad;
    let y_top = 1.0 + ctx.u.branch_field.bounding_pad_y;

    let sigma_rgb = {
        let t = ctx.u.contour_params.sigma_dispersion.clamp(0.0, 1.0);
        [
            ctx.u.sigma_t * mixf(1.0, 1.0, t),
            ctx.u.sigma_t * mixf(1.0, 1.091, t),
            ctx.u.sigma_t * mixf(1.0, 1.333, t),
        ]
    };

    let mode_trace_ray = (cols / 2, rows / 2);
    let mut rays_json = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let ndc = [
                (col as f32 + 0.5) / cols as f32 * 2.0 - 1.0,
                (row as f32 + 0.5) / rows as f32 * 2.0 - 1.0,
            ];
            let dir_world =
                (forward + right * ndc[0] * tan_half * aspect + up * ndc[1] * tan_half).normalize();
            let dl = transform_vector(&inverse_model, [dir_world.x, dir_world.y, dir_world.z]);
            let dl_len = dot3(dl, dl).sqrt().max(1e-8);
            let o = camera_local;
            let d = [dl[0] / dl_len, dl[1] / dl_len, dl[2] / dl_len];

            let span = slab_interval(o[1], d[1], y_top, t_max)
                .and_then(|(lo, hi)| outer_cylinder_interval(o, d, r_out, lo, hi));
            let (t0, t1) = match span {
                Some(pair) => pair,
                None => {
                    rays_json.push(json!({"ndc": [round5(ndc[0]), round5(ndc[1])], "hit": false}));
                    continue;
                }
            };

            let record_modes = (col, row) == mode_trace_ray;
            let ray = trace_ray(
                &ctx,
                o,
                d,
                t0,
                t1,
                sigma_rgb,
                record_modes,
                col as f32,
                row as f32,
                apply_jitter,
                segments,
                integrator,
            );
            let mut obj = ray;
            obj["ndc"] = json!([round5(ndc[0]), round5(ndc[1])]);
            obj["hit"] = json!(true);
            rays_json.push(obj);
        }
    }

    json!({
        "schema": "flame-field-trace-v1",
        "segments": segments,
        "integrator": integrator.name(),
        "grid": {"cols": cols, "rows": rows},
        "mode_trace_ray": {"col": mode_trace_ray.0, "row": mode_trace_ray.1},
        "view": {
            "position": vec_json(&view.position),
            "forward": vec_json(&view.forward),
            "right": vec_json(&view.right),
            "up": vec_json(&view.up),
            "fov_y_radians": round5(view.fov_y_radians),
            "viewport_size_px": vec_json(&view.viewport_size_px),
        },
        "ubo": {
            "time": round5(ubo.time),
            "sigma_t": round5(ubo.sigma_t),
            "intensity": round5(ubo.intensity),
            "height_axis_scale": round5(ubo.height_axis_scale),
            "noise_amplitude": round5(ubo.noise_amplitude),
            "tip_carve_params": vec_json(&[ubo.tip_carve_params.depth, ubo.tip_carve_params.inv_reach, ubo.tip_carve_params.primitive_top, ubo.tip_carve_params.inv_primitive_range]),
            "warp_strain_params": vec_json(&[ubo.warp_strain_params.strain_base, ubo.warp_strain_params.strain_tip, ubo.warp_strain_params.inv_reach, ubo.warp_strain_params.inv_strain_norm]),
            "warp_form_params": vec_json(&[ubo.warp_form_params.displacement_form, ubo.warp_form_params.burnout_gain]),
            "height_primitive_coefficients": Value::Array(
                ubo.height_primitive_coefficients.iter().map(|s| vec_json(s)).collect()),
            "noise_frequency": round5(ubo.noise_frequency),
            "noise_scroll_speed": round5(ubo.noise_scroll_speed),
            "radial_sharpness": round5(ubo.radial_sharpness),
            "style_params0": vec_json(&[ubo.warp_style.warp_amp, ubo.warp_style.warp_freq, ubo.warp_style.rise_speed, ubo.warp_style.taper_power]),
            "style_params1": vec_json(&[ubo.edge_style.radius_tip_ratio, ubo.edge_style.edge_low, ubo.edge_style.edge_high, ubo.edge_style.white_boost]),
            "style_params2": vec_json(&[ubo.wind_bend.wind_direction[0], ubo.wind_bend.wind_direction[1], ubo.wind_bend.bend_amount, ubo.wind_bend.bend_power]),
            "temporal_data": vec_json(&[ubo.temporal_data.accum_weight, ubo.temporal_data.frame_index, ubo.temporal_data.noise_aniso_y, ubo.temporal_data.warp_y_scale]),
            "light_data": vec_json(&[ubo.light_data.direction[0], ubo.light_data.direction[1], ubo.light_data.direction[2], ubo.light_data.self_shadow_strength]),
            "emitter_params": vec_json(&[ubo.emitter_params.kind, ubo.emitter_params.ring_major_ratio, ubo.emitter_params.ring_angular_speed, ubo.emitter_params.sdf_slab_depth]),
            "contour_params": vec_json(&[ubo.contour_params.wiggle_amp, ubo.contour_params.aniso_axis_advect, ubo.contour_params.rte_bands, ubo.contour_params.sigma_dispersion]),
            "erosion_response": vec_json(&[ubo.erosion_response.center, ubo.erosion_response.kappa, ubo.erosion_response.weight1, ubo.erosion_response.weight2]),
            "wave_cf_params": vec_json(&[ubo.wave_cf_params.enabled, ubo.wave_cf_params.shear_layer_count, ubo.wave_cf_params.skipped_power_plain, ubo.wave_cf_params.skipped_power_env]),
            "unified_params": vec_json(&[ubo.unified_params.enabled, ubo.unified_params.sigma_floor]),
            "spread_params": vec_json(&[ubo.spread_params.gain, ubo.spread_params.edge_outer_sharpen, ubo.spread_params.twist_gain, ubo.spread_params.erosion_noise_gain]),
            "support_params": vec_json(&[
                ubo.support_motion.support_margin,
                ubo.support_motion.meander_amp,
                ubo.support_motion.swirl_speed,
                ubo.support_motion.twist_speed,
            ]),
            "boundary_params": vec_json(&[ubo.boundary_params.amp, ubo.boundary_params.freq, ubo.boundary_params.speed, ubo.boundary_params.radius_ratio]),
            "near_fade_params": vec_json(&[ubo.near_fade_params.radius, ubo.near_fade_params.carve_residual, ubo.near_fade_params.edge_low, ubo.near_fade_params.edge_high]),
            "profile_params": vec_json(&[ubo.profile_params.radius_active, ubo.profile_params.radius_max, ubo.profile_params.color_active]),
            "wave_params": vec_json(&[ubo.wave_params.tracked_count, ubo.wave_params.env_coeff, ubo.wave_params.inverse_scale, ubo.wave_params.amplitude]),
            "branch_field": branch_field_json(&ubo.branch_field),
            "jitter_kappa_scale": round5(ctx.jitter_scale),
            "advect": vec_json(&ctx.advect),
            "aniso_axis": vec_json(&ctx.aniso_axis),
        },
        "not_replayed": [
            "temporal history blend (temporal_data.accum_weight)",
            "auto exposure / scene composite",
            "proxy raster interval (slab + outer cylinder approximation)",
            "mode 3 IGN jitter",
            "SDF billboard emitter",
        ],
        "rays": Value::Array(rays_json),
    })
}

#[allow(clippy::too_many_arguments)]
fn trace_ray(
    ctx: &UboCtx,
    o: [f32; 3],
    d: [f32; 3],
    t0: f32,
    t1: f32,
    sigma_rgb: [f32; 3],
    record_modes: bool,
    col: f32,
    row: f32,
    apply_jitter: bool,
    segments: usize,
    integrator: SegmentIntegrator,
) -> Value {
    let mut t0 = t0;
    let dt = (t1 - t0) / segments as f32;

    if apply_jitter {
        let jitter = interleaved_gradient_noise([col, row]);
        t0 += (jitter - 0.5) * dt;
    }

    // Per-node chain.
    let mut nodes: Vec<(f32, NodeDensity, Option<NodeArgument>)> = Vec::with_capacity(segments + 1);
    for node in 0..=segments {
        let t = t0 + node as f32 * dt;
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        let h = p[1].clamp(0.0, 1.0);
        // Use meander-shifted p for density-side coordinate, matching the shader.
        let (ps, hs) = ctx.support_position(p, h);
        let nd = ctx.node_density(ps, hs);
        let arg = if nd.density > 0.0 {
            Some(ctx.node_argument(p, d, h, nd.density, dt))
        } else {
            None
        };
        nodes.push((t, nd, arg));
    }

    let density_at = |t: f32| -> f32 {
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        let h = p[1].clamp(0.0, 1.0);
        // Use meander-shifted p for density-side coordinate, matching the shader.
        let (ps, hs) = ctx.support_position(p, h);
        ctx.node_density(ps, hs).density
    };
    let support_crossing = |t_dead: f32, t_live: f32| -> f32 {
        let (mut t_dead, mut t_live) = (t_dead, t_live);
        for _ in 0..SUPPORT_BISECTION_STEPS {
            let t_mid = 0.5 * (t_dead + t_live);
            if density_at(t_mid) > 0.0 {
                t_live = t_mid;
            } else {
                t_dead = t_mid;
            }
        }
        0.5 * (t_dead + t_live)
    };
    let argument_at = |t: f32, density: f32| -> NodeArgument {
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        ctx.node_argument(p, d, p[1].clamp(0.0, 1.0), density, dt)
    };

    // Segment walk (mirror of flameWaveOccupancySegments, rte = true).
    let carrier = match integrator {
        SegmentIntegrator::Faddeeva => Some(ctx.carrier_amplitudes()),
        SegmentIntegrator::Legacy => None,
    };
    let carrier_chain = carrier
        .as_ref()
        .map(|c| ctx.shaping_statistical_gain(c.sigma_z))
        .unwrap_or((1.0, 0.0));
    let alpha_ref = match (&integrator, &carrier) {
        (SegmentIntegrator::Faddeeva, Some(c)) => solve_reference_cutoff(
            ctx,
            o,
            d,
            t0,
            dt * segments as f32,
            c,
            carrier_chain.0,
            carrier_chain.1,
        ),
        _ => f32::INFINITY,
    };
    let state_at = |t: f32| -> CarrierSlowState {
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        ctx.carrier_slow_state(
            p,
            d,
            p[1].clamp(0.0, 1.0),
            carrier.as_ref().expect("carrier"),
            alpha_ref,
        )
    };
    let node_states: Vec<Option<CarrierSlowState>> = match integrator {
        SegmentIntegrator::Faddeeva => nodes
            .iter()
            .map(|(t, nd, _)| (nd.density > 0.0).then(|| state_at(*t)))
            .collect(),
        SegmentIntegrator::Legacy => Vec::new(),
    };
    let residual = ctx.u.near_fade_params.carve_residual.clamp(0.0, 1.0);
    let mut total = 0.0f32;
    let mut height_mean_num = 0.0f32;
    let mut radiance_pre = [0.0f32; 3];
    let mut transmittance = [1.0f32; 3];
    let mut trans_track: Vec<(f32, f32)> = Vec::with_capacity(segments);
    let mut segments_json = Vec::with_capacity(segments);
    for segment in 0..segments {
        let t_prev = t0 + segment as f32 * dt;
        let t = t_prev + dt;
        let prev_density = nodes[segment].1.density;
        let density = nodes[segment + 1].1.density;
        if prev_density <= 0.0 && density <= 0.0 {
            segments_json.push(json!(null));
            continue;
        }
        let entering = prev_density <= 0.0;
        let exiting = density <= 0.0;
        let seg_start = if entering {
            support_crossing(t_prev, t)
        } else {
            t_prev
        };
        let seg_end = if exiting {
            support_crossing(t, t_prev)
        } else {
            t
        };
        let span = seg_end - seg_start;
        if span < 1e-4 * dt {
            segments_json.push(json!(null));
            continue;
        }
        let density_start = if entering { 0.0 } else { prev_density };
        let density_end = if exiting { 0.0 } else { density };
        let start_arg = if entering {
            argument_at(seg_start, 0.0)
        } else {
            match &nodes[segment].2 {
                Some(a) => clone_argument(a),
                None => argument_at(t_prev, prev_density),
            }
        };
        let end_arg = if exiting {
            argument_at(seg_end, 0.0)
        } else {
            match &nodes[segment + 1].2 {
                Some(a) => clone_argument(a),
                None => argument_at(t, density),
            }
        };

        let h_mid = (o[1] + (seg_start + 0.5 * span) * d[1]).clamp(0.0, 1.0);
        let fade_avg = 0.5 * (ctx.envelope_fade(density_start) + ctx.envelope_fade(density_end));
        let estimate = match integrator {
            SegmentIntegrator::Legacy => {
                let shaping_deriv_avg =
                    0.5 * (ctx.shaping_deriv(start_arg.shaped) + ctx.shaping_deriv(end_arg.shaped));
                let sigma_eff_raw = ctx.u.spread_params.erosion_noise_gain
                    * 0.5
                    * (start_arg.sigma_noise + end_arg.sigma_noise)
                    * shaping_deriv_avg
                    * ctx.u.noise_amplitude.abs()
                    * ctx.tip_carve_lambda(h_mid)
                    * (0.5 * (density_start + density_end) / EROSION_SHELL_REF)
                    * 0.5
                    * (ctx.envelope_fade(density_start) + ctx.envelope_fade(density_end));
                let slope = (end_arg.argument - start_arg.argument) / span;
                let u_squared = (h_mid - 1.0).powi(2);
                let sigma_floor =
                    ctx.unified_sigma_floor(h_mid, 0.5 * (density_start + density_end), u_squared)
                        * fade_avg;
                let sigma_eff = sigma_eff_raw.max(sigma_floor);
                let (integral, first_moment) = integrate_erf_response_linear(
                    &ctx.erf,
                    sigma_eff,
                    start_arg.argument - slope * seg_start,
                    slope,
                    seg_start,
                    seg_end,
                );
                SegmentEstimate {
                    integral,
                    first_moment,
                    sigma: sigma_eff,
                    shaping_deriv: shaping_deriv_avg,
                    linear_correction: 0.0,
                    sigma_eff_raw,
                    sigma_floor,
                }
            }
            SegmentIntegrator::Faddeeva => {
                let start_state = if entering {
                    state_at(seg_start)
                } else {
                    match &node_states[segment] {
                        Some(state) => *state,
                        None => state_at(seg_start),
                    }
                };
                let end_state = if exiting {
                    state_at(seg_end)
                } else {
                    match &node_states[segment + 1] {
                        Some(state) => *state,
                        None => state_at(seg_end),
                    }
                };
                faddeeva_segment_estimate(
                    ctx,
                    o,
                    d,
                    seg_start,
                    seg_end,
                    density_start,
                    density_end,
                    h_mid,
                    fade_avg,
                    carrier.as_ref().expect("carrier amplitudes"),
                    carrier_chain.0,
                    carrier_chain.1,
                    alpha_ref,
                    &start_state,
                    &end_state,
                )
            }
        };
        let (mut integral, mut first_moment) = (estimate.integral, estimate.first_moment);
        if residual > 0.0 {
            let plain_slope = (density_end - density_start) / span;
            let (plain, plain_moment) = integrate_erf_response_linear(
                &ctx.erf,
                0.0,
                density_start - plain_slope * seg_start,
                plain_slope,
                seg_start,
                seg_end,
            );
            integral += residual * (plain - integral);
            first_moment += residual * (plain_moment - first_moment);
        }
        let emission = integral.max(0.0);
        let t_mean = if integral > 1e-6 {
            (first_moment / integral).clamp(seg_start, seg_end)
        } else {
            t0 + (segment as f32 + 0.5) * dt
        };
        total += emission;

        let p_mean = [
            o[0] + t_mean * d[0],
            o[1] + t_mean * d[1],
            o[2] + t_mean * d[2],
        ];
        let h_mean = p_mean[1].clamp(0.0, 1.0);
        height_mean_num += emission * h_mean;
        let mut edge = 0.0;
        if ctx.u.emitter_params.kind < 1.5 {
            let rm = if ctx.u.emitter_params.kind >= 0.5 {
                ctx.u.emitter_params.ring_major_ratio
            } else {
                0.0
            };
            let minor = if ctx.u.emitter_params.kind >= 0.5 {
                (1.0 - rm).max(1e-3)
            } else {
                1.0
            };
            let taper_r = mixf(
                1.0,
                ctx.u.edge_style.radius_tip_ratio,
                h_mean.powf(ctx.u.warp_style.taper_power),
            );
            let rho_norm = (((p_mean[0] * p_mean[0] + p_mean[2] * p_mean[2]).sqrt() - rm) / minor)
                .abs()
                / taper_r.max(1e-4);
            edge = (ctx.u.color_tip.edge_temperature_blend * smoothstep(0.6, 1.2, rho_norm))
                .clamp(0.0, 1.0);
        }
        let ramp = ctx.ramp_color(h_mean);
        let color = mix3(
            ramp,
            [
                ctx.u.color_tip.rgb[0],
                ctx.u.color_tip.rgb[1],
                ctx.u.color_tip.rgb[2],
            ],
            edge,
        );
        let tau = [
            sigma_rgb[0] * emission,
            sigma_rgb[1] * emission,
            sigma_rgb[2] * emission,
        ];
        for c in 0..3 {
            let absorbed = 1.0 - (-tau[c]).exp();
            radiance_pre[c] += transmittance[c] * color[c] * absorbed;
            transmittance[c] *= (-tau[c]).exp();
        }
        let mean_trans = (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;
        trans_track.push((t_mean, mean_trans));

        segments_json.push(json!({
            "seg_start": round5(seg_start),
            "seg_end": round5(seg_end),
            "entering": entering,
            "exiting": exiting,
            "arg_start": round5(start_arg.argument),
            "arg_end": round5(end_arg.argument),
            "sigma_eff_raw": round5(estimate.sigma_eff_raw),
            "sigma_floor": round5(estimate.sigma_floor),
            "sigma_eff": round5(estimate.sigma),
            "shaping_deriv_avg": round5(estimate.shaping_deriv),
            "linear_correction": round5(estimate.linear_correction),
            "emission": round5(emission),
            "t_mean": round5(t_mean),
            "h_mean": round5(h_mean),
            "edge": round5(edge),
            "tau": vec_json(&tau),
            "transmittance_after": vec_json(&transmittance),
        }));
    }

    let height_mean = if total > 1e-6 {
        height_mean_num / total
    } else {
        0.0
    };
    let temp_norm = (total * 2.0).clamp(0.0, 1.0) * (1.0 - 0.55 * height_mean);
    let boost = 1.0 + ctx.u.edge_style.white_boost * temp_norm * temp_norm;
    let mut radiance = [
        radiance_pre[0] * ctx.u.intensity * boost,
        radiance_pre[1] * ctx.u.intensity * boost,
        radiance_pre[2] * ctx.u.intensity * boost,
    ];
    let alpha_rte = 1.0 - (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;

    // Self shadow at the interval midpoint (lightData.w gate).
    let mut self_shadow_tau = 0.0f32;
    let mut self_shadow_factor = 1.0f32;
    if ctx.u.light_data.self_shadow_strength > 0.0 {
        let t_mid = 0.5 * (t0 + t1);
        let p_mid = [
            o[0] + t_mid * d[0],
            o[1] + t_mid * d[1],
            o[2] + t_mid * d[2],
        ];
        let l = Vector3::new(
            ctx.u.light_data.direction[0],
            ctx.u.light_data.direction[1],
            ctx.u.light_data.direction[2],
        )
        .normalize();
        self_shadow_tau = ctx.self_shadow_tau(p_mid, [l.x, l.y, l.z]);
        self_shadow_factor = mixf(
            1.0,
            (-self_shadow_tau).exp(),
            ctx.u.light_data.self_shadow_strength,
        );
        for c in &mut radiance {
            *c *= self_shadow_factor;
        }
    }

    let luma = radiance[0] * LUMA_WEIGHTS[0]
        + radiance[1] * LUMA_WEIGHTS[1]
        + radiance[2] * LUMA_WEIGHTS[2];
    let alpha_final = alpha_rte * smoothstep(0.0, ctx.u.color_base.occlusion_lum_ref, luma);

    // Optical front: the t where the accumulated opacity first reaches half
    // of its final value — the slice the RTE composite weights most, i.e.
    // where visible banding lives (works for optically thin rays too).
    let final_mean_trans = (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;
    let target = 0.5 * (1.0 + final_mean_trans);
    let t_front = trans_track
        .iter()
        .find(|(_, trans)| *trans <= target)
        .map(|(t, _)| *t);
    let front_json = match t_front {
        Some(tf) => {
            let p = [o[0] + tf * d[0], o[1] + tf * d[1], o[2] + tf * d[2]];
            let h = p[1].clamp(0.0, 1.0);
            // Use meander-shifted p for density-side coordinate, matching the shader.
            let (ps, hs) = ctx.support_position(p, h);
            let nd = ctx.node_density(ps, hs);
            let a = ctx.node_argument(p, d, h, nd.density, dt);
            json!({
                "t": round5(tf),
                "h": round5(h),
                "density": round5(nd.density),
                "wiggle": round5(nd.wiggle),
                "boundary": vec_json(&nd.boundary),
                "warp_d": vec_json(&[a.q[0] - a.pb[0], a.q[1] - a.pb[1], a.q[2] - a.pb[2]]),
                "w": vec_json(&a.w),
                "jitter_psi": vec_json(&a.jitter_psi),
                "z_low": round5(a.z_low),
                "z": round5(a.z),
                "shaped_noise": round5(a.shaped),
                "sigma_noise": round5(a.sigma_noise),
                "lambda": round5(a.lambda),
                "mu": round5(a.mu),
                "strain": round5(a.strain),
                "erosion": round5(a.erosion),
                "argument": round5(a.argument),
            })
        }
        None => json!(null),
    };

    // Per-node arrays (structure of arrays).
    macro_rules! node_arr {
        ($f:expr) => {
            Value::Array(nodes.iter().map($f).collect::<Vec<_>>())
        };
    }
    let nodes_json = json!({
        "t": node_arr!(|(t, _, _)| round5(*t)),
        "density": node_arr!(|(_, nd, _)| round5(nd.density)),
        "wiggle": node_arr!(|(_, nd, _)| round5(nd.wiggle)),
        "boundary_x": node_arr!(|(_, nd, _)| round5(nd.boundary[0])),
        "boundary_y": node_arr!(|(_, nd, _)| round5(nd.boundary[1])),
        "height_falloff": node_arr!(|(_, nd, _)| round5(nd.height_falloff)),
        "cap_fade": node_arr!(|(_, nd, _)| round5(nd.cap_fade)),
        "radial_factor": node_arr!(|(_, nd, _)| round5(nd.radial)),
        "near_fade": node_arr!(|(_, nd, _)| round5(nd.near_fade)),
        "w_x": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.w[0]))),
        "w_y": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.w[1]))),
        "w_z": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.w[2]))),
        "rate_x": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.rate[0]))),
        "rate_y": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.rate[1]))),
        "rate_z": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.rate[2]))),
        "warp_dx": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.q[0] - a.pb[0]))),
        "warp_dy": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.q[1] - a.pb[1]))),
        "warp_dz": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.q[2] - a.pb[2]))),
        "jitter_psi": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| vec_json(&a.jitter_psi))),
        "jitter_psi_rate": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| vec_json(&a.jitter_psi_rate))),
        "z_low": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.z_low))),
        "z": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.z))),
        "shaped_noise": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.shaped))),
        "sigma_noise": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.sigma_noise))),
        "lambda": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.lambda))),
        "mu": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.mu))),
        "strain": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.strain))),
        "erosion": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.erosion))),
        "envelope_fade": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.envelope_fade))),
        "argument": node_arr!(|(_, _, a)| a.as_ref().map_or(json!(null), |a| round5(a.argument))),
    });

    let mode_trace = if record_modes {
        json!({
            "per_node_mode_values": Value::Array(
                nodes
                    .iter()
                    .map(|(_, _, a)| a.as_ref().map_or(json!(null), |a| vec_json(&a.mode_values)))
                    .collect(),
            ),
            "per_node_mode_weights": Value::Array(
                nodes
                    .iter()
                    .map(|(_, _, a)| a.as_ref().map_or(json!(null), |a| vec_json(&a.mode_weights)))
                    .collect(),
            ),
        })
    } else {
        json!(null)
    };

    json!({
        "origin": vec_json(&o),
        "dir": vec_json(&d),
        "t_near": round5(t0),
        "t_far": round5(t1),
        "nodes": nodes_json,
        "segments": Value::Array(segments_json),
        "front": front_json,
        "mode_trace": mode_trace,
        "final": {
            "emission_total": round5(total),
            "height_mean": round5(height_mean),
            "temp_norm": round5(temp_norm),
            "boost": round5(boost),
            "radiance": vec_json(&radiance),
            "alpha_rte": round5(alpha_rte),
            "alpha_final": round5(alpha_final),
            "self_shadow_tau": round5(self_shadow_tau),
            "self_shadow_factor": round5(self_shadow_factor),
        },
    })
}

fn clone_argument(a: &NodeArgument) -> NodeArgument {
    NodeArgument {
        pb: a.pb,
        q: a.q,
        w: a.w,
        rate: a.rate,
        jitter_psi: a.jitter_psi,
        jitter_psi_rate: a.jitter_psi_rate,
        z_low: a.z_low,
        z: a.z,
        shaped: a.shaped,
        sigma_noise: a.sigma_noise,
        lambda: a.lambda,
        mu: a.mu,
        strain: a.strain,
        erosion: a.erosion,
        envelope_fade: a.envelope_fade,
        argument: a.argument,
        mode_values: a.mode_values.clone(),
        mode_weights: a.mode_weights.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::FlameEffect;

    #[test]
    fn test_trace_default_effect_produces_finite_emission() {
        std::env::set_var("THYLLORE_FLAME_TRACE_COLS", "5");
        std::env::set_var("THYLLORE_FLAME_TRACE_ROWS", "5");
        let effect = FlameEffect::default();
        let view = WallProbeView {
            position: [0.0, 0.5, 3.0],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_radians: 45f32.to_radians(),
            viewport_size_px: [1280.0, 720.0],
        };
        let trace = trace_flame_field(&effect, &Default::default(), &Default::default(), &view);
        let rays = trace["rays"].as_array().unwrap();
        assert_eq!(rays.len(), 25);
        let any_emission = rays.iter().any(|r| {
            r["final"]["emission_total"]
                .as_f64()
                .map(|v| v > 0.0)
                .unwrap_or(false)
        });
        assert!(any_emission, "no ray produced emission");
        for ray in rays {
            if let Some(nodes) = ray["nodes"].as_object() {
                for (_, arr) in nodes {
                    for v in arr.as_array().unwrap() {
                        if let Some(x) = v.as_f64() {
                            assert!(x.is_finite());
                        }
                    }
                }
            }
        }
    }

    /// Branch elements: an inactive spawner leaves the support position and
    /// warp frame untouched; a live element moves the pulled-back point, keeps
    /// the frame finite, and its support height follows the pulled-back y.
    #[test]
    fn test_branch_elements_transport_support_and_frame() {
        let baked = Default::default();
        let trail = Default::default();
        let mut effect = FlameEffect::default();
        effect.time = 2.3;
        let ubo_off = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let off = UboCtx::new(&ubo_off, [0.0, 0.5, 3.0]);
        let p = [0.45, 0.42, 0.1];
        let (ps_off, h_off) = off.support_position(p, 0.42);
        assert_eq!(ps_off, off.meander_shifted(p, 0.42));
        assert_eq!(h_off, 0.42);

        effect.branch.period = 0.4;
        effect.branch.gain = 3.0;
        effect.branch.spread = 0.0;
        let ubo_on = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        assert!(ubo_on.branch_field.count > 0.0);
        let on = UboCtx::new(&ubo_on, [0.0, 0.5, 3.0]);
        let moved = (0..12)
            .map(|i| [0.2 + 0.1 * i as f32, 0.3 + 0.03 * i as f32, 0.0])
            .any(|q| {
                let (ps, hs) = on.support_position(q, q[1]);
                assert_eq!(hs, ps[1].clamp(0.0, 1.0));
                ps != on.meander_shifted(q, q[1])
            });
        assert!(moved, "a live element must move at least one sample");
        let frame = on.wave_frame(p, [0.0, 0.0, -1.0], 0.42);
        assert!(frame
            .w
            .iter()
            .chain(frame.rate.iter())
            .all(|v| v.is_finite()));
        assert!((0.0..=1.0).contains(&frame.h));
    }

    /// twist_speed = 0 delegates the twist rate to swirl_speed; > 0 owns it.
    #[test]
    fn test_twist_speed_delegates_to_swirl_speed_at_zero() {
        let baked = Default::default();
        let trail = Default::default();
        let mut effect = FlameEffect::default();
        effect.twist.gain = 2.0;
        effect.time = 1.7;

        effect.swirl.speed = 1.3;
        effect.twist.speed = 0.0;
        let ubo_delegate = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let delegate = UboCtx::new(&ubo_delegate, [0.0, 0.5, 3.0]).twist_angle(0.09, 0.6);

        effect.swirl.speed = 0.4;
        effect.twist.speed = 1.3;
        let ubo_own = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let own = UboCtx::new(&ubo_own, [0.0, 0.5, 3.0]).twist_angle(0.09, 0.6);
        assert_eq!(delegate, own, "twist_speed must override the rate exactly");

        effect.twist.speed = 2.6;
        let ubo_fast = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let fast = UboCtx::new(&ubo_fast, [0.0, 0.5, 3.0]).twist_angle(0.09, 0.6);
        assert_ne!(
            delegate, fast,
            "a different twist_speed must change the angle"
        );
    }

    /// Burnout (D design): gain 0 leaves the boost at exactly 0, a positive
    /// gain deepens the mean shrink toward the luminous top (mu asymptote,
    /// near-zero at the base), and a larger tip_carve_reach descends deeper.
    #[test]
    fn test_burnout_boost_zero_off_and_mu_monotone() {
        let baked = Default::default();
        let trail = Default::default();
        let mut effect = FlameEffect::default();
        let ubo_off = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let off = UboCtx::new(&ubo_off, [0.0, 0.5, 3.0]);
        for h in [0.0, 0.5, 1.0] {
            assert_eq!(off.burnout_boost(h), 0.0);
        }

        effect.burnout_gain = 2.0;
        let ubo_on = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let on = UboCtx::new(&ubo_on, [0.0, 0.5, 3.0]);
        let base = on.burnout_boost(0.0);
        let mid = on.burnout_boost(0.5);
        let top = on.burnout_boost(1.0);
        assert!(base < 0.05, "the base must stay essentially untouched");
        assert!(
            top > mid && mid > base,
            "boost must deepen toward the luminous top"
        );
        assert!(top <= effect.burnout_gain);

        effect.tip_carve.reach *= 4.0;
        let ubo_deep = thyllore_effect_core::build_flame_ubo(&effect, &baked, &trail);
        let deep = UboCtx::new(&ubo_deep, [0.0, 0.5, 3.0]);
        assert!(
            deep.burnout_boost(0.5) > mid,
            "a larger reach must burn deeper down the column"
        );
    }

    /// The Faddeeva estimator must produce finite, non-trivial emission on
    /// the default effect (direct trace_ray call — no env, race-free).
    #[test]
    fn test_faddeeva_ray_produces_finite_emission() {
        let effect = FlameEffect::default();
        let ubo = thyllore_effect_core::build_flame_ubo(
            &effect,
            &Default::default(),
            &Default::default(),
        );
        let ctx = UboCtx::new(&ubo, [0.0, 0.5, 3.0]);
        let o = [0.0, 0.5, 3.0];
        let d = [0.0, 0.0, -1.0];
        let sigma_rgb = [ubo.sigma_t; 3];
        for integrator in [SegmentIntegrator::Legacy, SegmentIntegrator::Faddeeva] {
            let ray = trace_ray(
                &ctx, o, d, 2.0, 4.0, sigma_rgb, false, 0.0, 0.0, false, 64, integrator,
            );
            let total = ray["final"]["emission_total"].as_f64().unwrap();
            assert!(total.is_finite(), "emission not finite");
            assert!(total > 0.0, "no emission through the flame center");
            for value in ray["final"]["radiance"].as_array().unwrap() {
                assert!(value.as_f64().unwrap().is_finite());
            }
        }
    }
}
