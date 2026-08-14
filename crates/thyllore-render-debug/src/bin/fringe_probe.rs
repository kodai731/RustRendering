//! `fringe_probe` — CLI that reads a flame dump JSON, computes the wave field
//! along view rays across an NDC rect, and writes a JSON report.
//!
//! Usage:
//!   fringe_probe --dump <in.json> --out <out.json> [--res 64] [--rect x0,y0,x1,y1] [--top-modes 12]

use std::fs;
use std::process;

use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};
use thyllore_effect_core::flame_wave::*;
use thyllore_effect_core::{
    build_flame_inverse_model_matrix, build_flame_ubo, flame_bounding_radius,
};
use thyllore_effect_core::{ring_support_span, FlameRadialTaper};

use thyllore_render_debug::dump_effect::{camera_from_dump, flame_from_dump};
use thyllore_render_debug::fringe_field::{flow_warp_with_rate, sample_wave_field, WarpParams};

#[derive(Clone, Debug)]
struct Args {
    dump: String,
    out: String,
    res: usize,
    rect: [f32; 4],
    top_modes: usize,
}

fn parse_args() -> Args {
    let mut dump: Option<String> = None;
    let mut out: Option<String> = None;
    let mut res: usize = 64;
    let mut rect: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
    let mut top_modes: usize = WAVE_MODE_COUNT;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump" => {
                i += 1;
                dump = Some(args[i].clone());
            }
            "--out" => {
                i += 1;
                out = Some(args[i].clone());
            }
            "--res" => {
                i += 1;
                res = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("invalid --res value");
                    process::exit(1);
                });
            }
            "--rect" => {
                i += 1;
                let parts: Vec<f32> = args[i]
                    .split(',')
                    .map(|s| {
                        s.parse().unwrap_or_else(|_| {
                            eprintln!("invalid --rect value");
                            process::exit(1);
                        })
                    })
                    .collect();
                if parts.len() != 4 {
                    eprintln!("--rect must have 4 comma-separated values");
                    process::exit(1);
                }
                rect = [parts[0], parts[1], parts[2], parts[3]];
            }
            "--top-modes" => {
                i += 1;
                top_modes = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("invalid --top-modes value");
                    process::exit(1);
                });
            }
            other => {
                eprintln!("unknown argument: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    let dump = dump.unwrap_or_else(|| {
        eprintln!("--dump is required");
        process::exit(1);
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("--out is required");
        process::exit(1);
    });

    Args {
        dump,
        out,
        res,
        rect,
        top_modes,
    }
}

fn transform_point(matrix: &Matrix4<f32>, point: Vector3<f32>) -> Vector3<f32> {
    (matrix * Vector4::new(point.x, point.y, point.z, 1.0)).truncate()
}

fn transform_vector(matrix: &Matrix4<f32>, vector: Vector3<f32>) -> Vector3<f32> {
    (matrix * Vector4::new(vector.x, vector.y, vector.z, 0.0)).truncate()
}

fn slab_interval(origin_y: f32, direction_y: f32, t_max: f32) -> Option<(f32, f32)> {
    if direction_y.abs() < 1e-6 {
        return (0.0..=1.0).contains(&origin_y).then_some((0.0, t_max));
    }
    let to_floor = (0.0 - origin_y) / direction_y;
    let to_ceiling = (1.0 - origin_y) / direction_y;
    let lo = to_floor.min(to_ceiling).max(0.0);
    let hi = to_floor.max(to_ceiling).min(t_max);
    (hi > lo).then_some((lo, hi))
}

fn main() {
    let args = parse_args();

    // Load dump
    let dump_text = fs::read_to_string(&args.dump).unwrap_or_else(|e| {
        eprintln!("failed to read dump file: {}", e);
        process::exit(1);
    });
    let dump: serde_json::Value = serde_json::from_str(&dump_text).unwrap_or_else(|e| {
        eprintln!("failed to parse dump JSON: {}", e);
        process::exit(1);
    });

    // Reconstruct effect and camera from dump
    let (effect, baked, temporal) = flame_from_dump(&dump["flames"][0]);
    let cam = camera_from_dump(&dump);

    // Check bend_amount is zero (non-zero means not mirrored)
    if effect.bend_amount != 0.0 {
        eprintln!("bend is not mirrored: bend_amount = {}", effect.bend_amount);
        process::exit(1);
    }

    // Build mode table: unified 128-mode table when the unified field is on,
    // else legacy 96 + envelope; sorted by |k| ascending below.
    let mut modes: Vec<WaveMode> = if thyllore_effect_core::read_env_wave_unified() {
        build_unified_erosion_modes(
            WAVE_K_RATIO,
            WAVE_ENV_MU,
            effect.boundary.amp * thyllore_effect_core::read_env_unified_tilt_gain_b(),
            effect.contour_wiggle_amp * thyllore_effect_core::read_env_unified_tilt_gain_w(),
        )
    } else {
        let mut m = generate_wave_modes_with_ratio(WAVE_K_RATIO).to_vec();
        apply_wave_envelope(&mut m, WAVE_ENV_MU);
        m
    };
    let jitter_scale = read_env_wave_jitter();
    for mode in modes.iter_mut() {
        for c in mode.jitter.iter_mut() {
            *c *= jitter_scale;
        }
    }
    let k_mag = |m: &WaveMode| (m.k[0] * m.k[0] + m.k[1] * m.k[1] + m.k[2] * m.k[2]).sqrt();
    modes.sort_by(|a, b| k_mag(a).total_cmp(&k_mag(b)));

    // Warp mode table + medium swirl modes (same combined table the UBO packs).
    // The swirl phase drift is a frame constant, so it folds into the phase here.
    let warp_modes = generate_wave_warp_modes();
    let mut shear_modes = warp_modes.to_vec();
    let drift_time = effect.noise_scroll_speed * effect.time;
    for (i, mut mode) in thyllore_effect_core::flame::build_medium_swirl_modes(&effect, &warp_modes)
        .into_iter()
        .enumerate()
    {
        mode.phase +=
            thyllore_effect_core::flame_wave::medium_swirl_phase_rate(i, mode.k) * drift_time;
        shear_modes.push(mode);
    }

    // Truncate to top_modes (all tracked, so skipped_power = [0.0, 0.0], skipped_env_coeff = 0.0)
    let n_modes = args.top_modes.min(modes.len());
    let modes: Vec<WaveMode> = modes.into_iter().take(n_modes).collect();

    // Camera setup
    let ubo = build_flame_ubo(&effect, &baked, &temporal);
    let inverse_model = build_flame_inverse_model_matrix(&effect);
    let camera_world = Vector3::from(cam.position);
    let camera_local = transform_point(&inverse_model, camera_world);
    let tan_half = (cam.fov_y_degrees * std::f32::consts::PI / 180.0 * 0.5).tan();
    let aspect = (cam.viewport_size_px[0] / cam.viewport_size_px[1].max(1.0)).max(1e-3);
    let forward = Vector3::from(cam.forward);
    let right = Vector3::from(cam.right);
    let up = Vector3::from(cam.up);
    let t_max = camera_local.magnitude() + 4.0;

    // Ring support parameters
    let taper = FlameRadialTaper::from_effect(&effect, &baked);
    let ring_major = if effect.emitter_kind == 1 {
        effect.ring_major_radius / flame_bounding_radius(&effect).max(1e-3)
    } else {
        0.0
    };
    let wiggle_trim = 1.0 + effect.contour_wiggle_amp;

    // Build rays: grid over rect at res x res (pixel center sampling)
    let mut rays = Vec::new();
    for row in 0..args.res {
        for col in 0..args.res {
            let ndc_x =
                args.rect[0] + (col as f32 + 0.5) / args.res as f32 * (args.rect[2] - args.rect[0]);
            let ndc_y =
                args.rect[1] + (row as f32 + 0.5) / args.res as f32 * (args.rect[3] - args.rect[1]);
            let ndc = [ndc_x, ndc_y];

            let direction_world =
                (forward + right * ndc[0] * tan_half * aspect + up * ndc[1] * tan_half).normalize();
            let direction_local = transform_vector(&inverse_model, direction_world).normalize();

            // Compute hit span: slab_interval + ring_support_span
            let span = slab_interval(camera_local.y, direction_local.y, t_max).and_then(
                |(slab_lo, slab_hi)| {
                    ring_support_span(
                        [camera_local.x, camera_local.y, camera_local.z],
                        [direction_local.x, direction_local.y, direction_local.z],
                        slab_lo,
                        slab_hi,
                        ring_major,
                        taper,
                        effect.radial_sharpness,
                        wiggle_trim,
                        effect.support_margin,
                    )
                },
            );

            if let Some((t_enter, t_exit)) = span {
                let t_mid = 0.5 * (t_enter + t_exit);
                let p = [
                    camera_local.x + direction_local.x * t_mid,
                    camera_local.y + direction_local.y * t_mid,
                    camera_local.z + direction_local.z * t_mid,
                ];
                let h = p[1].clamp(0.0, 1.0);

                // Compute advect from wind_direction and rise_speed
                let advect = [
                    effect.wind_direction.x * effect.time,
                    effect.rise_speed * effect.time,
                    effect.wind_direction.y * effect.time,
                ];

                // Flow warp with rate (strain params from the exact UBO packing)
                let warp_params = WarpParams {
                    strain_params: ubo.warp_strain_params,
                    warp_freq: effect.warp_freq,
                    advect,
                    aniso_axis_advect: effect.aniso_axis_advect,
                    height_primitive: ubo.height_primitive_coefficients,
                    mu_zw: [ubo.tip_carve_params[2], ubo.tip_carve_params[3]],
                    displacement_form: ubo.warp_form_params[0] > 0.5,
                };
                let spread_sigma = ubo.spread_params[0]
                    * (-thyllore_render_debug::fringe_field::envelope_remaining_mu(
                        &warp_params,
                        h,
                    ) * ubo.tip_carve_params[1])
                        .exp();
                let spread = (-spread_sigma * h).exp();
                let spread_y = 1.0 / (spread * spread);
                let (q, rate_raw) = flow_warp_with_rate(
                    &shear_modes,
                    &warp_params,
                    [p[0] * spread, p[1] * spread_y, p[2] * spread],
                    [
                        direction_local.x * spread,
                        direction_local.y * spread_y,
                        direction_local.z * spread,
                    ],
                    h,
                );

                // w = anisoCompress(q, noise_aniso_y) * noise_frequency - advect
                // rate = anisoCompress(rate_raw, noise_aniso_y) * noise_frequency
                let w = [
                    q[0] * effect.noise_frequency - advect[0],
                    q[1] * effect.noise_aniso_y * effect.noise_frequency - advect[1],
                    q[2] * effect.noise_frequency - advect[2],
                ];
                let rate = [
                    rate_raw[0] * effect.noise_frequency,
                    rate_raw[1] * effect.noise_aniso_y * effect.noise_frequency,
                    rate_raw[2] * effect.noise_frequency,
                ];

                // node_spacing
                let chord = t_exit - t_enter;
                let segment_dt = chord / FLAME_WAVE_SEGMENTS as f32;

                // Sample wave field
                let sample = sample_wave_field(
                    &modes,
                    w,
                    rate,
                    segment_dt,
                    effect.time * effect.noise_scroll_speed,
                    [0.0, 0.0],
                    0.0,
                );

                rays.push(serde_json::json!({
                    "ndc": [ndc[0], ndc[1]],
                    "hit": true,
                    "t_mid": t_mid,
                    "w": [w[0], w[1], w[2]],
                    "noise": sample.noise,
                    "sigma": sample.sigma,
                    "z": sample.z,
                    "mode_contrib": sample.mode_contrib,
                }));
            } else {
                rays.push(serde_json::json!({
                    "ndc": [ndc[0], ndc[1]],
                    "hit": false,
                    "t_mid": 0.0,
                    "w": [0.0, 0.0, 0.0],
                    "noise": 0.0,
                    "sigma": 0.0,
                    "z": 0.0,
                    "mode_contrib": [],
                }));
            }
        }
    }

    // Build output JSON
    let mode_k: Vec<[f32; 3]> = modes.iter().map(|m| m.k).collect();
    let mode_amp: Vec<f32> = modes.iter().map(|m| m.amplitude).collect();
    let mode_env_coeff: Vec<f32> = modes.iter().map(|m| m.env_coeff).collect();

    let output = serde_json::json!({
        "meta": {
            "res": args.res,
            "rect": [args.rect[0], args.rect[1], args.rect[2], args.rect[3]],
            "camera": {
                "position": cam.position,
                "forward": cam.forward,
                "fov_y_degrees": cam.fov_y_degrees,
            },
            "mode_k": mode_k,
            "mode_amp": mode_amp,
            "mode_env_coeff": mode_env_coeff,
        },
        "rays": rays,
    });

    let out_text = serde_json::to_string_pretty(&output).unwrap();
    fs::write(&args.out, &out_text).unwrap_or_else(|e| {
        eprintln!("failed to write output file: {}", e);
        process::exit(1);
    });

    println!("wrote {} rays to {}", rays.len(), args.out);
}
