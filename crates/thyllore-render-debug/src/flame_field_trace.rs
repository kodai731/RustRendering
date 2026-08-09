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

use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
use serde_json::{json, Value};
use thyllore_math_core::{integrate_erf_response_linear, ErfResponseModel};
use thyllore_render_core::WallProbeView;
use thyllore_render_core::flame_wave::{
    WAVE_JITTER_K, WAVE_JITTER_PHASE, WAVE_JITTER_RANK,
};
use thyllore_render_core::FlameUBO;

const SEGMENTS: usize = 64;
const EROSION_SLOTS: usize = 96;
const WARP_BASE: usize = 96;
const WARP_COUNT: usize = 16;
const DETAIL_BASE: usize = 112;
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

fn glsl_fract(x: f32) -> f32 {
    x - x.floor()
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
const EROSION_SHELL_REF: f32 = 0.30;

fn mixf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [mixf(a[0], b[0], t), mixf(a[1], b[1], t), mixf(a[2], b[2], t)]
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

fn hash_cell(cell: [f32; 3]) -> f32 {
    let mut p = [
        glsl_fract(cell[0] * 0.3183099 + 0.1),
        glsl_fract(cell[1] * 0.3183099 + 0.2),
        glsl_fract(cell[2] * 0.3183099 + 0.3),
    ];
    for v in &mut p {
        *v *= 17.0;
    }
    glsl_fract(p[0] * p[1] * p[2] * (p[0] + p[1] + p[2]))
}

fn value_noise3(p: [f32; 3]) -> f32 {
    let cell = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [glsl_fract(p[0]), glsl_fract(p[1]), glsl_fract(p[2])];
    let u = [
        f[0] * f[0] * (3.0 - 2.0 * f[0]),
        f[1] * f[1] * (3.0 - 2.0 * f[1]),
        f[2] * f[2] * (3.0 - 2.0 * f[2]),
    ];
    let h = |dx: f32, dy: f32, dz: f32| hash_cell([cell[0] + dx, cell[1] + dy, cell[2] + dz]);
    let nx00 = mixf(h(0.0, 0.0, 0.0), h(1.0, 0.0, 0.0), u[0]);
    let nx10 = mixf(h(0.0, 1.0, 0.0), h(1.0, 1.0, 0.0), u[0]);
    let nx01 = mixf(h(0.0, 0.0, 1.0), h(1.0, 0.0, 1.0), u[0]);
    let nx11 = mixf(h(0.0, 1.0, 1.0), h(1.0, 1.0, 1.0), u[0]);
    let nxy0 = mixf(nx00, nx10, u[1]);
    let nxy1 = mixf(nx01, nx11, u[1]);
    mixf(nxy0, nxy1, u[2])
}

fn fbm3(p: [f32; 3]) -> f32 {
    let mut sum = 0.0f32;
    let mut amplitude = 0.5f32;
    let mut q = p;
    for _ in 0..3 {
        sum += amplitude * value_noise3(q);
        for axis in 0..3 {
            q[axis] = q[axis] * 2.02 + 13.7;
        }
        amplitude *= 0.5;
    }
    sum
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
        let sp0 = u.style_params0;
        let sp2 = u.style_params2;
        let adv_dir = [sp2[0], sp0[2], sp2[1]];
        let advect = [
            adv_dir[0] * u.time,
            adv_dir[1] * u.time,
            adv_dir[2] * u.time,
        ];
        let mut axis = [0.0, 1.0, 0.0];
        let adv_sq = dot3(adv_dir, adv_dir);
        if u.contour_params[1] > 0.0 && adv_sq > 1e-8 {
            let inv = 1.0 / adv_sq.sqrt();
            let n = [adv_dir[0] * inv, adv_dir[1] * inv, adv_dir[2] * inv];
            let t = u.contour_params[1].clamp(0.0, 1.0);
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
            center: u.erosion_response[0],
            kappa: u.erosion_response[1],
            gaussian_weights: [u.erosion_response[2], u.erosion_response[3]],
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
        let sp2 = self.u.style_params2;
        let s = sp2[2] * h.powf(sp2[3]);
        [sp2[0] * s, sp2[1] * s]
    }

    fn wave_mode(&self, slot: usize) -> ([f32; 4], [f32; 4]) {
        (
            self.u.wave_modes[2 * slot],
            self.u.wave_modes[2 * slot + 1],
        )
    }

    /// flameWaveFlowWarpRate: returns (warped point q, rate dq/dt along dir).
    fn flow_warp_rate(&self, pb: [f32; 3], dir: [f32; 3], h: f32) -> ([f32; 3], [f32; 3]) {
        let sp0 = self.u.style_params0;
        let amp = sp0[0] * mixf(0.15, 1.0, h);
        if amp == 0.0 {
            return (pb, dir);
        }
        let c = self.aniso_compress(pb, 0.35);
        let mut z = [
            c[0] * sp0[1] - self.advect[0],
            c[1] * sp0[1] - self.advect[1],
            c[2] * sp0[1] - self.advect[2],
        ];
        let cv = self.aniso_compress(dir, 0.35);
        let mut v = [cv[0] * sp0[1], cv[1] * sp0[1], cv[2] * sp0[1]];
        let strength = amp * 0.96;
        for m in 0..WARP_COUNT {
            let (kv, dv) = self.wave_mode(WARP_BASE + m);
            let k = [kv[0], kv[1], kv[2]];
            let curl = [dv[1], dv[2], dv[3]];
            let angle = dot3(k, z) + dv[0];
            let shear = strength * kv[3] * angle.cos();
            let fp = -strength * kv[3] * angle.sin();
            let kdv = dot3(k, v);
            for axis in 0..3 {
                z[axis] += curl[axis] * shear;
                v[axis] += curl[axis] * fp * kdv;
            }
        }
        let q_pre = [
            z[0] / sp0[1] + self.advect[0] / sp0[1],
            z[1] / sp0[1] + self.advect[1] / sp0[1],
            z[2] / sp0[1] + self.advect[2] / sp0[1],
        ];
        let rate_pre = [v[0] / sp0[1], v[1] / sp0[1], v[2] / sp0[1]];
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
        if self.u.contour_params[0] == 0.0 {
            return 1.0;
        }
        let q = [
            p[0] * self.u.noise_frequency,
            (h - self.u.style_params0[2] * self.u.time) * self.u.noise_frequency,
            p[2] * self.u.noise_frequency,
        ];
        1.0 + self.u.contour_params[0] * self.detail_noise(q)
    }

    fn boundary_displacement(&self, x: f32, z: f32) -> [f32; 2] {
        let bp = self.u.boundary_params;
        if bp[0] == 0.0 {
            return [1.0, 1.0];
        }
        let q = [x * bp[1], -bp[2] * self.u.time, z * bp[1]];
        let height_noise = ((fbm3(q) * (2.0 / 0.875) - 1.0) * 3.0).min(1.0);
        let radius_noise = (fbm3([q[0] + 13.7, q[1] + 41.3, q[2] + 7.9]) * (2.0 / 0.875) - 1.0) * 3.0;
        [
            (1.0 + bp[0] * height_noise).max(0.2),
            (1.0 + bp[0] * bp[3] * radius_noise).max(0.2),
        ]
    }

    fn cap_fade(&self, h: f32, bx: f32) -> f32 {
        if self.u.boundary_params[0] == 0.0 || bx <= 1.0 {
            return 1.0;
        }
        smoothstep(1.0, 2.0 - bx, h)
    }

    fn height_falloff(&self, hb: f32) -> f32 {
        cheb8(self.u.height_coefficients[0], self.u.height_coefficients[1], hb)
    }

    fn radial_support_radius(&self) -> f32 {
        (2.0 / self.u.radialSharpness.max(1e-3f32)).sqrt().min(SUPPORT_HEADROOM)
    }

    fn radial_radius_scale(&self, hb: f32) -> f32 {
        if self.u.profile_params[0] > 0.5 {
            SHELL_BASE_RADIUS
                * cheb8(self.u.radius_coefficients[0], self.u.radius_coefficients[1], hb).max(0.05)
        } else {
            SHELL_BASE_RADIUS
                * mixf(1.0, self.u.style_params1[0], hb.powf(self.u.style_params0[3]))
        }
    }

    fn radial_factor(&self, px: f32, pz: f32, hb: f32) -> f32 {
        let scale = (self.radial_support_radius() * self.radial_radius_scale(hb)).max(1e-4);
        let u2 = (px * px + pz * pz) / (scale * scale);
        let inside = (1.0 - u2).max(0.0);
        inside * inside
    }

    fn near_camera_fade(&self, p_local: [f32; 3]) -> f32 {
        let radius = self.u.near_fade_params[0];
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
        (d_smooth / self.u.style_params1[2].max(1e-3)).min(1.0)
    }

    fn tip_carve_lambda(&self, h: f32) -> f32 {
        let primitive = cheb12(
            self.u.height_primitive_coefficients[0],
            self.u.height_primitive_coefficients[1],
            self.u.height_primitive_coefficients[2],
            h,
        );
        let tc = self.u.tip_carve_params;
        let mu = ((tc[2] - primitive) * tc[3]).clamp(0.0, 1.0);
        1.0 + tc[0] * (-mu * tc[1]).exp()
    }

    fn eroded_argument(&self, d_smooth: f32, erosion: f32) -> f32 {
        d_smooth - (erosion.max(0.0) + erosion.min(0.0) * self.envelope_fade(d_smooth))
    }

    /// flameWaveNodeDensity (cylinder / ring generic branch): returns the full
    /// factor decomposition alongside the product.
    fn node_density(&self, p: [f32; 3], h: f32) -> NodeDensity {
        let wiggle = self.contour_wiggle(p, h);
        let boundary = self.boundary_displacement(p[0], p[2]);
        let emitter = self.u.emitter_params.x;
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
            let taper_r = mixf(1.0, self.u.style_params1[0], hb.powf(self.u.style_params0[3]));
            let rm = self.u.emitter_params.y;
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

    /// flameWaveNodeArgumentLocal with every intermediate recorded.
    fn node_argument(&self, p: [f32; 3], d: [f32; 3], h: f32, density: f32, dt: f32) -> NodeArgument {
        let bend = self.bend_offset(h);
        let pb = [p[0] - bend[0], p[1], p[2] - bend[1]];
        let (q, rate_raw) = self.flow_warp_rate(pb, d, h);
        let cw = self.aniso_compress(q, self.u.temporal_data.z);
        let w = [
            cw[0] * self.u.noise_frequency - self.advect[0],
            cw[1] * self.u.noise_frequency - self.advect[1],
            cw[2] * self.u.noise_frequency - self.advect[2],
        ];
        let cr = self.aniso_compress(rate_raw, self.u.temporal_data.z);
        let rate = [
            cr[0] * self.u.noise_frequency,
            cr[1] * self.u.noise_frequency,
            cr[2] * self.u.noise_frequency,
        ];
        let (jitter_psi, jitter_psi_rate) = self.jitter_state(w, rate);

        let eddy_time = self.u.noise_scroll_speed * self.u.time;
        let count = (self.u.wave_params[0] as usize).min(EROSION_SLOTS);
        let mut z_low = 0.0f32;
        let mut unresolved = 0.0f32;
        let mut mode_values = Vec::new();
        let mut mode_weights = Vec::new();
        let record_modes = dt > 0.0;
        for pass in 0..2 {
            let mut z_acc = 0.0f32;
            for n in 0..count {
                let (kv, ph) = self.wave_mode(n);
                let is_high = ph[2] != 0.0;
                if (pass == 0) == is_high {
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
                let carrier = angle.sin();
                if pass == 0 {
                    z_acc += weight * kv[3] * carrier;
                    unresolved += 0.5 * kv[3] * kv[3] * (1.0 - weight * weight);
                } else {
                    let envelope = 1.0 + ph[2] * z_low;
                    z_acc += envelope * weight * kv[3] * carrier;
                    unresolved += envelope * envelope * 0.5 * kv[3] * kv[3] * (1.0 - weight * weight);
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

        let mut unresolved_total = unresolved + self.u.wave_cf_params[2];
        let env_skip = 1.0 + self.u.wave_params[1] * z_low_only;
        unresolved_total += self.u.wave_cf_params[3] * env_skip * env_skip;

        let sigma_noise = unresolved_total.sqrt();
        let inv_scale = self.u.wave_params[2];
        let amp = self.u.wave_params[3];
        let shaped = if inv_scale > 0.0 {
            0.4375 + amp * (z * inv_scale).tanh()
        } else {
            0.4375 + z
        };
        let lambda = self.tip_carve_lambda(h);
        let erosion = self.u.noise_amplitude
            * (mixf(0.2, 1.0, h) * EROSION_MEAN_SHRINK
                + lambda * (density / EROSION_SHELL_REF) * (shaped - 0.4375));
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
            erosion,
            envelope_fade: self.envelope_fade(density),
            argument,
            mode_values,
            mode_weights,
        }
    }

    fn ramp_color(&self, h: f32) -> [f32; 3] {
        if self.u.profile_params[2] > 0.5 {
            let u = h.clamp(0.0, 1.0) * 8.0 - 0.5;
            let i0 = (u.floor().clamp(0.0, 7.0)) as usize;
            let i1 = (i0 + 1).min(7);
            let f = (u - i0 as f32).clamp(0.0, 1.0);
            let a = self.u.color_ramp[i0];
            let b = self.u.color_ramp[i1];
            mix3([a[0], a[1], a[2]], [b[0], b[1], b[2]], f)
        } else if h < 0.5 {
            mix3(
                [self.u.color_base.x, self.u.color_base.y, self.u.color_base.z],
                [self.u.color_mid.x, self.u.color_mid.y, self.u.color_mid.z],
                h * 2.0,
            )
        } else {
            mix3(
                [self.u.color_mid.x, self.u.color_mid.y, self.u.color_mid.z],
                [self.u.color_tip.x, self.u.color_tip.y, self.u.color_tip.z],
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
            dens[k] = cheb8(self.u.radial_coefficients[0], self.u.radial_coefficients[1], m[k]);
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
                let f_val = cheb8(self.u.height_coefficients[0], self.u.height_coefficients[1], py);
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
    erosion: f32,
    envelope_fade: f32,
    argument: f32,
    mode_values: Vec<f32>,
    mode_weights: Vec<f32>,
}

fn slab_interval(origin_y: f32, dir_y: f32, t_max: f32) -> Option<(f32, f32)> {
    if dir_y.abs() < 1e-6 {
        return (0.0..=1.0).contains(&origin_y).then_some((0.0, t_max));
    }
    let a = (0.0 - origin_y) / dir_y;
    let b = (1.0 - origin_y) / dir_y;
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
/// rays. `mode_trace_ray` (col, row) selects one ray that additionally records
/// the per-mode carrier values and low-pass weights at every node.
pub fn trace_flame_field(ubo: &FlameUBO, view: &WallProbeView) -> Value {
    let cols = env_usize("THYLLORE_FLAME_TRACE_COLS", 13);
    let rows = env_usize("THYLLORE_FLAME_TRACE_ROWS", 49);
    let inverse_model = ubo.inverse_model;
    let ctx = UboCtx::new(ubo, view.position);

    let tan_half = (view.fov_y_radians * 0.5).tan();
    let aspect = (view.viewport_size_px[0] / view.viewport_size_px[1].max(1.0)).max(1e-3);
    let forward = Vector3::from(view.forward);
    let right = Vector3::from(view.right);
    let up = Vector3::from(view.up);
    let camera_local = transform_point(&inverse_model, view.position);
    let t_max = dot3(camera_local, camera_local).sqrt() + 4.0;

    let wiggle_trim = 1.0 + ctx.u.contour_params[0].max(0.0);
    let boundary_trim = 1.0
        + 3.0 * ctx.u.boundary_params[0].abs() * ctx.u.boundary_params[3].max(0.0);
    let taper_max = ctx.u.style_params1[0].max(1.0);
    let r_out = SHELL_BASE_RADIUS * SUPPORT_HEADROOM * taper_max * wiggle_trim * boundary_trim;

    let sigma_rgb = {
        let t = ctx.u.contour_params[3].clamp(0.0, 1.0);
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

            let span = slab_interval(o[1], d[1], t_max)
                .and_then(|(lo, hi)| outer_cylinder_interval(o, d, r_out, lo, hi));
            let (t0, t1) = match span {
                Some(pair) => pair,
                None => {
                    rays_json.push(json!({"ndc": [round5(ndc[0]), round5(ndc[1])], "hit": false}));
                    continue;
                }
            };

            let record_modes = (col, row) == mode_trace_ray;
            let ray = trace_ray(&ctx, o, d, t0, t1, sigma_rgb, record_modes);
            let mut obj = ray;
            obj["ndc"] = json!([round5(ndc[0]), round5(ndc[1])]);
            obj["hit"] = json!(true);
            rays_json.push(obj);
        }
    }

    json!({
        "schema": "flame-field-trace-v1",
        "segments": SEGMENTS,
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
            "tip_carve_params": vec_json(&ubo.tip_carve_params),
            "height_primitive_coefficients": Value::Array(
                ubo.height_primitive_coefficients.iter().map(|s| vec_json(s)).collect()),
            "noise_frequency": round5(ubo.noise_frequency),
            "noise_scroll_speed": round5(ubo.noise_scroll_speed),
            "radial_sharpness": round5(ubo.radialSharpness),
            "style_params0": vec_json(&ubo.style_params0),
            "style_params1": vec_json(&ubo.style_params1),
            "style_params2": vec_json(&ubo.style_params2),
            "temporal_data": vec_json(&[ubo.temporal_data.x, ubo.temporal_data.y, ubo.temporal_data.z, ubo.temporal_data.w]),
            "light_data": vec_json(&[ubo.light_data.x, ubo.light_data.y, ubo.light_data.z, ubo.light_data.w]),
            "emitter_params": vec_json(&[ubo.emitter_params.x, ubo.emitter_params.y, ubo.emitter_params.z, ubo.emitter_params.w]),
            "contour_params": vec_json(&ubo.contour_params),
            "erosion_response": vec_json(&ubo.erosion_response),
            "wave_cf_params": vec_json(&ubo.wave_cf_params),
            "boundary_params": vec_json(&ubo.boundary_params),
            "near_fade_params": vec_json(&ubo.near_fade_params),
            "profile_params": vec_json(&ubo.profile_params),
            "wave_params": vec_json(&ubo.wave_params),
            "jitter_kappa_scale": round5(ctx.jitter_scale),
            "advect": vec_json(&ctx.advect),
            "aniso_axis": vec_json(&ctx.aniso_axis),
        },
        "not_replayed": [
            "temporal history blend (temporal_data.x)",
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
) -> Value {
    let dt = (t1 - t0) / SEGMENTS as f32;

    // Per-node chain.
    let mut nodes: Vec<(f32, NodeDensity, Option<NodeArgument>)> = Vec::with_capacity(SEGMENTS + 1);
    for node in 0..=SEGMENTS {
        let t = t0 + node as f32 * dt;
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        let h = p[1].clamp(0.0, 1.0);
        let nd = ctx.node_density(p, h);
        let arg = if nd.density > 0.0 {
            Some(ctx.node_argument(p, d, h, nd.density, dt))
        } else {
            None
        };
        nodes.push((t, nd, arg));
    }

    let density_at = |t: f32| -> f32 {
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        ctx.node_density(p, p[1].clamp(0.0, 1.0)).density
    };
    let support_crossing = |t_dead: f32, t_live: f32| -> f32 {
        let (mut t_dead, mut t_live) = (t_dead, t_live);
        for _ in 0..8 {
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
    let residual = ctx.u.near_fade_params[1].clamp(0.0, 1.0);
    let inv_scale = ctx.u.wave_params[2];
    let amp = ctx.u.wave_params[3];
    let mut total = 0.0f32;
    let mut height_mean_num = 0.0f32;
    let mut radiance_pre = [0.0f32; 3];
    let mut transmittance = [1.0f32; 3];
    let mut trans_track: Vec<(f32, f32)> = Vec::with_capacity(SEGMENTS);
    let mut segments_json = Vec::with_capacity(SEGMENTS);
    for segment in 0..SEGMENTS {
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
        let seg_end = if exiting { support_crossing(t, t_prev) } else { t };
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

        let shaping_deriv = |shaped: f32| -> f32 {
            if inv_scale > 0.0 {
                let tval = (shaped - 0.4375) / amp;
                amp * inv_scale * (1.0 - tval * tval)
            } else {
                1.0
            }
        };
        let shaping_deriv_avg = 0.5 * (shaping_deriv(start_arg.shaped) + shaping_deriv(end_arg.shaped));
        let h_mid = (o[1] + (seg_start + 0.5 * span) * d[1]).clamp(0.0, 1.0);
        let sigma_eff = 0.5 * (start_arg.sigma_noise + end_arg.sigma_noise)
            * shaping_deriv_avg
            * ctx.u.noise_amplitude.abs()
            * ctx.tip_carve_lambda(h_mid)
            * (0.5 * (density_start + density_end) / EROSION_SHELL_REF)
            * 0.5
            * (ctx.envelope_fade(density_start) + ctx.envelope_fade(density_end));
        let slope = (end_arg.argument - start_arg.argument) / span;
        let (mut integral, mut first_moment) = integrate_erf_response_linear(
            &ctx.erf,
            sigma_eff,
            start_arg.argument - slope * seg_start,
            slope,
            seg_start,
            seg_end,
        );
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

        let p_mean = [o[0] + t_mean * d[0], o[1] + t_mean * d[1], o[2] + t_mean * d[2]];
        let h_mean = p_mean[1].clamp(0.0, 1.0);
        height_mean_num += emission * h_mean;
        let mut edge = 0.0;
        if ctx.u.emitter_params.x < 1.5 {
            let rm = if ctx.u.emitter_params.x >= 0.5 {
                ctx.u.emitter_params.y
            } else {
                0.0
            };
            let minor = if ctx.u.emitter_params.x >= 0.5 {
                (1.0 - rm).max(1e-3)
            } else {
                1.0
            };
            let taper_r = mixf(1.0, ctx.u.style_params1[0], h_mean.powf(ctx.u.style_params0[3]));
            let rho_norm = (((p_mean[0] * p_mean[0] + p_mean[2] * p_mean[2]).sqrt() - rm) / minor)
                .abs()
                / taper_r.max(1e-4);
            edge = (ctx.u.color_tip.w * smoothstep(0.6, 1.2, rho_norm)).clamp(0.0, 1.0);
        }
        let ramp = ctx.ramp_color(h_mean);
        let color = mix3(
            ramp,
            [ctx.u.color_tip.x, ctx.u.color_tip.y, ctx.u.color_tip.z],
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
            "sigma_eff": round5(sigma_eff),
            "shaping_deriv_avg": round5(shaping_deriv_avg),
            "emission": round5(emission),
            "t_mean": round5(t_mean),
            "h_mean": round5(h_mean),
            "edge": round5(edge),
            "tau": vec_json(&tau),
            "transmittance_after": vec_json(&transmittance),
        }));
    }

    let height_mean = if total > 1e-6 { height_mean_num / total } else { 0.0 };
    let temp_norm = (total * 2.0).clamp(0.0, 1.0) * (1.0 - 0.55 * height_mean);
    let boost = 1.0 + ctx.u.style_params1[3] * temp_norm * temp_norm;
    let mut radiance = [
        radiance_pre[0] * ctx.u.intensity * boost,
        radiance_pre[1] * ctx.u.intensity * boost,
        radiance_pre[2] * ctx.u.intensity * boost,
    ];
    let alpha_rte = 1.0 - (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;

    // Self shadow at the interval midpoint (lightData.w gate).
    let mut self_shadow_tau = 0.0f32;
    let mut self_shadow_factor = 1.0f32;
    if ctx.u.light_data.w > 0.0 {
        let t_mid = 0.5 * (t0 + t1);
        let p_mid = [o[0] + t_mid * d[0], o[1] + t_mid * d[1], o[2] + t_mid * d[2]];
        let l = Vector3::new(ctx.u.light_data.x, ctx.u.light_data.y, ctx.u.light_data.z)
            .normalize();
        self_shadow_tau = ctx.self_shadow_tau(p_mid, [l.x, l.y, l.z]);
        self_shadow_factor = mixf(1.0, (-self_shadow_tau).exp(), ctx.u.light_data.w);
        for c in &mut radiance {
            *c *= self_shadow_factor;
        }
    }

    let luma = radiance[0] * LUMA_WEIGHTS[0] + radiance[1] * LUMA_WEIGHTS[1] + radiance[2] * LUMA_WEIGHTS[2];
    let alpha_final = alpha_rte * smoothstep(0.0, ctx.u.color_base.w, luma);

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
            let nd = ctx.node_density(p, h);
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
    use thyllore_render_core::FlameEffect;

    #[test]
    fn test_trace_default_effect_produces_finite_emission() {
        std::env::set_var("THYLLORE_FLAME_TRACE_COLS", "5");
        std::env::set_var("THYLLORE_FLAME_TRACE_ROWS", "5");
        let effect = FlameEffect::default();
        let ubo = thyllore_render_core::build_flame_ubo(&effect);
        let view = WallProbeView {
            position: [0.0, 0.5, 3.0],
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_radians: 45f32.to_radians(),
            viewport_size_px: [1280.0, 720.0],
        };
        let trace = trace_flame_field(&ubo, &view);
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
}
