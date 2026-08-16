use crate::flame::*;
use crate::flame_trail::{FlameTrailSample, FlameTrailState};
use cgmath::{InnerSpace, Matrix3, Matrix4, Vector3};
use thyllore_math_core::{evaluate_chebyshev, fit_erf_response};

/// Parse a mode mask string into a set of matched indices.
/// Supports: "0-32" (range), "5" (single), "!17" (exclusion: all except 17),
/// comma-separated combinations.
fn parse_mode_mask(mask: &str) -> std::collections::HashSet<usize> {
    let mut result: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for part in mask.split(',') {
        let part = part.trim();
        if part.starts_with('!') {
            // Exclusion mode: "!17" means all modes except 17
            let inner = &part[1..];
            let excluded: std::collections::HashSet<usize> = parse_mode_mask(inner);
            // Collect all indices up to WAVE_MODE_COUNT and subtract excluded
            for i in 0..crate::flame_wave::WAVE_MODE_COUNT {
                if !excluded.contains(&i) {
                    result.insert(i);
                }
            }
        } else if let Some(range) = part.split_once('-') {
            // Range: "0-32"
            if let (Ok(start), Ok(end)) = (range.0.parse::<usize>(), range.1.parse::<usize>()) {
                for i in start..=end {
                    result.insert(i);
                }
            }
        } else if let Ok(single) = part.parse::<usize>() {
            // Single: "5"
            result.insert(single);
        }
    }

    result
}

/// CPU mirror of the shader's `flameAnisoCompress` over the erosion chain
/// (axis optionally bent toward the advection direction by aniso_axis_advect).
fn noise_aniso_compress(effect: &FlameEffect, v: [f32; 3]) -> [f32; 3] {
    let advect = Vector3::new(
        effect.wind.direction.x,
        effect.warp.rise_speed,
        effect.wind.direction.y,
    );
    let mut axis = Vector3::new(0.0, 1.0, 0.0);
    if effect.contour.aniso_axis_advect > 0.0 && advect.magnitude2() > 1e-8 {
        let blend = effect.contour.aniso_axis_advect.clamp(0.0, 1.0);
        axis = (axis * (1.0 - blend) + advect.normalize() * blend).normalize();
    }
    let vector = Vector3::new(v[0], v[1], v[2]);
    let compressed = vector - axis * vector.dot(axis) * (1.0 - effect.noise.aniso_y);
    [compressed.x, compressed.y, compressed.z]
}

/// Medium spread gain alpha (motion_design L3); the reach shares the tip
/// carve inv_reach in the shader.
fn build_medium_spread_params(effect: &FlameEffect) -> FlameSpreadParams {
    FlameSpreadParams {
        gain: effect.spread_gain.max(0.0),
        edge_outer_sharpen: effect.edge.outer_sharpen,
        twist_gain: effect.twist.gain,
        erosion_noise_gain: effect.noise.erosion_gain,
    }
}

fn build_wave_cf_params() -> FlameWaveCfParams {
    let cf_active = read_env_wave_cf();
    FlameWaveCfParams {
        enabled: if cf_active { 1.0 } else { 0.0 },
        shear_layer_count: if cf_active {
            read_env_wave_cf_layers() as f32
        } else {
            0.0
        },
        skipped_power_plain: 0.0,
        skipped_power_env: 0.0,
    }
}

struct WaveUboFields {
    shaping: FlameWaveShaping,
    packed: [[f32; 4]; 2 * crate::flame_wave::WAVE_MODE_SLOTS],
    skipped_power: [f32; 2],
    jitter: [[f32; 4]; crate::flame_wave::WAVE_MODE_COUNT],
    /// Std of the low-octave erosion carrier zLow (the envelope modes), before
    /// the tanh shaping; the mixing window is expressed in these units.
    low_carrier_std: f32,
}

fn build_wave_ubo_fields(effect: &FlameEffect) -> WaveUboFields {
    use crate::flame_wave::{
        build_unified_erosion_modes, generate_wave_cf_shear_layers, generate_wave_detail_modes,
        generate_wave_modes_with_ratio, generate_wave_warp_modes, wave_cf_chebyshev_tables,
        wave_cf_depth_scale, wave_shaping_params, WAVE_CF_CHEB_COEFFS, WAVE_CF_SHEAR_SLOT,
        WAVE_DETAIL_BASE, WAVE_LOW_MODE_COUNT, WAVE_MODE_COUNT, WAVE_MODE_SLOTS, WAVE_WARP_BASE,
    };

    let mut packed = [[0.0f32; 4]; 2 * WAVE_MODE_SLOTS];
    let k_ratio = read_env_wave_k_ratio();
    let unified = read_env_wave_unified();
    let mut erosion_modes: Vec<crate::flame_wave::WaveMode> = if unified {
        build_unified_erosion_modes(
            k_ratio,
            read_env_wave_env_mu(),
            effect.boundary.amp * read_env_unified_tilt_gain_b(),
            effect.contour.wiggle_amp * read_env_unified_tilt_gain_w(),
        )
    } else {
        let mut modes = generate_wave_modes_with_ratio(k_ratio).to_vec();
        crate::flame_wave::apply_wave_envelope(&mut modes, read_env_wave_env_mu());
        modes
    };

    // 5.1 probabilistic reduction: sort by |k| ascending so the tracked prefix
    // is the low-wavenumber part of the spectrum; aggregate the skipped modes'
    // variance per octave class (the high class rides the shared low-octave
    // envelope, whose coefficient is uniform across modes).
    let k_mag = |m: &crate::flame_wave::WaveMode| {
        (m.k[0] * m.k[0] + m.k[1] * m.k[1] + m.k[2] * m.k[2]).sqrt()
    };
    erosion_modes.sort_by(|a, b| k_mag(a).total_cmp(&k_mag(b)));

    // Mode ablation: if THYLLORE_FLAME_WAVE_MODE_MASK is set, zero the amplitude
    // of matched modes (sorted by |k| ascending index). No re-normalization.
    if let Some(mask) = read_env_wave_mode_mask() {
        let matched = parse_mode_mask(&mask);
        for (i, mode) in erosion_modes.iter_mut().enumerate() {
            if matched.contains(&i) {
                mode.amplitude = 0.0;
            }
        }
    }

    let tracked = if unified {
        (read_env_wave_track() + WAVE_LOW_MODE_COUNT).min(erosion_modes.len())
    } else {
        read_env_wave_track()
    };
    let env_coeff = erosion_modes
        .iter()
        .map(|m| m.env_coeff)
        .find(|c| *c != 0.0)
        .unwrap_or(0.0);
    let mut skipped_power = [0.0f32; 2];
    for mode in erosion_modes.iter().skip(tracked) {
        let power = 0.5 * mode.amplitude * mode.amplitude;
        skipped_power[usize::from(mode.env_coeff != 0.0)] += power;
    }
    let low_carrier_std = erosion_modes
        .iter()
        .filter(|mode| mode.env_coeff == 0.0)
        .map(|mode| 0.5 * mode.amplitude * mode.amplitude)
        .sum::<f32>()
        .sqrt();

    let warp_modes = generate_wave_warp_modes();
    let cf_active = read_env_wave_cf() && !unified;
    let depth_scale: Vec<f32> = if cf_active {
        wave_cf_depth_scale(&erosion_modes, &warp_modes, effect.noise.frequency, |v| {
            noise_aniso_compress(effect, v)
        })
        .to_vec()
    } else {
        vec![0.0; erosion_modes.len()]
    };
    for (i, mode) in erosion_modes.iter().enumerate() {
        packed[2 * i] = [mode.k[0], mode.k[1], mode.k[2], mode.amplitude];
        packed[2 * i + 1] = [mode.phase, mode.eddy_rate, mode.env_coeff, depth_scale[i]];
    }
    for (i, mode) in warp_modes.iter().enumerate() {
        let slot = WAVE_WARP_BASE + i;
        packed[2 * slot] = [mode.k[0], mode.k[1], mode.k[2], mode.amplitude];
        packed[2 * slot + 1] = [
            mode.phase,
            mode.curl_direction[0],
            mode.curl_direction[1],
            mode.curl_direction[2],
        ];
    }
    for (i, mode) in generate_wave_detail_modes().iter().enumerate() {
        let slot = WAVE_DETAIL_BASE + i;
        packed[2 * slot] = [mode.k[0], mode.k[1], mode.k[2], mode.amplitude];
        packed[2 * slot + 1] = [mode.phase, mode.eddy_rate, 0.0, 0.0];
    }
    if cf_active {
        let (cheb_sin, cheb_cos) = wave_cf_chebyshev_tables();
        for i in 0..WAVE_CF_CHEB_COEFFS {
            let slot = WAVE_DETAIL_BASE + i;
            packed[2 * slot + 1][2] = cheb_sin[i];
            packed[2 * slot + 1][3] = cheb_cos[i];
        }
        let layers = generate_wave_cf_shear_layers(
            &warp_modes,
            read_env_wave_cf_layers(),
            read_env_wave_cf_shear(),
        );
        for (i, mode) in layers.iter().enumerate() {
            let slot = WAVE_CF_SHEAR_SLOT + i;
            packed[2 * slot] = [mode.k[0], mode.k[1], mode.k[2], mode.amplitude];
            packed[2 * slot + 1] = [
                mode.phase,
                mode.curl_direction[0],
                mode.curl_direction[1],
                mode.curl_direction[2],
            ];
        }
    }
    for (i, mode) in build_medium_swirl_modes(effect, &warp_modes)
        .iter()
        .enumerate()
    {
        let slot = crate::flame_wave::WAVE_MEDIUM_SWIRL_BASE + i;
        packed[2 * slot] = [mode.k[0], mode.k[1], mode.k[2], mode.amplitude];
        // The swirl displacement is horizontal (c.y = 0), so the free .z lane
        // carries the phase-drift rate: [phase, c.x, drift_rate, c.z].
        packed[2 * slot + 1] = [
            mode.phase,
            mode.curl_direction[0],
            crate::flame_wave::medium_swirl_phase_rate(i, mode.k) * effect.swirl.speed.max(0.0),
            mode.curl_direction[2],
        ];
    }
    let tanh_scale = if effect.noise.shaping_scale > 0.0 {
        effect.noise.shaping_scale
    } else {
        read_env_wave_tanh()
    };
    let (inverse_scale, mut amplitude) = if tanh_scale <= 0.0 {
        (0.0, 1.0)
    } else {
        wave_shaping_params(tanh_scale)
    };
    amplitude *= (effect.noise.amplitude.abs() / NOISE_AMPLITUDE_REF).powf(SHAPING_GAMMA);
    let jitter_scale = crate::flame_wave::read_env_wave_jitter();
    let mut jitter = [[0.0f32; 4]; WAVE_MODE_COUNT];
    for (slot, mode) in jitter.iter_mut().zip(erosion_modes.iter()) {
        slot[0] = mode.jitter[0] * jitter_scale;
        slot[1] = mode.jitter[1] * jitter_scale;
        slot[2] = mode.jitter[2] * jitter_scale;
    }
    jitter[0][3] = crate::flame_wave::read_env_wave_jitter_freq();
    WaveUboFields {
        shaping: FlameWaveShaping {
            tracked_count: tracked as f32,
            env_coeff,
            inverse_scale,
            amplitude,
        },
        packed,
        skipped_power,
        jitter,
        low_carrier_std,
    }
}

fn build_segment_params(effect: &FlameEffect) -> FlameSegmentParams {
    let count = wave_segment_count(effect);
    FlameSegmentParams {
        count: count as f32,
        inv_count: 1.0 / count as f32,
        _padding: [0.0; 2],
    }
}

pub fn build_unified_field_params(effect: &FlameEffect) -> FlameUnifiedParams {
    let inactive = FlameUnifiedParams {
        enabled: 0.0,
        sigma_floor: 0.0,
        _padding: [0.0; 2],
    };
    if !read_env_wave_unified() {
        return inactive;
    }
    let amplitude_ratio = (effect.noise.amplitude.abs() / NOISE_AMPLITUDE_REF).powf(SHAPING_GAMMA);
    let std = crate::flame_wave::unified_noise_std(
        read_env_wave_k_ratio(),
        read_env_wave_env_mu(),
        effect.boundary.amp * read_env_unified_tilt_gain_b(),
        effect.contour.wiggle_amp * read_env_unified_tilt_gain_w(),
    );
    let sigma_floor =
        read_env_unified_beta() * effect.noise.amplitude.abs() * std * amplitude_ratio;
    FlameUnifiedParams {
        enabled: 1.0,
        sigma_floor,
        _padding: [0.0; 2],
    }
}

fn build_profile_params(effect: &FlameEffect, baked: &FlameBaked) -> FlameProfileParams {
    let inactive = FlameProfileParams {
        radius_active: 0.0,
        radius_max: 0.0,
        color_active: 0.0,
        _padding: 0.0,
    };
    if baked.radius.is_none() || baked.blend <= 0.0 {
        return inactive;
    }
    let series = thyllore_math_core::ChebyshevSeries::new(
        effect
            .coefficients
            .radius_scale
            .iter()
            .flatten()
            .copied()
            .collect(),
        (0.0, 1.0),
    );
    let mut max_val = 0.0f32;
    for i in 0..=32 {
        let h = i as f32 / 32.0;
        let val = evaluate_chebyshev(&series, h);
        if val > max_val {
            max_val = val;
        }
    }
    let color_flag = if baked.color.is_some() && baked.blend > 0.0 {
        1.0
    } else {
        0.0
    };
    FlameProfileParams {
        radius_active: 1.0,
        radius_max: max_val.max(0.05),
        color_active: color_flag,
        _padding: 0.0,
    }
}

/// Form-matched strain norm: max a|k| for the sequential composition
/// (per-shear bound), RMS for the displacement sum (gradient-sum bound).
fn shear_strain_norm(modes: &[crate::flame_wave::WaveWarpMode]) -> f32 {
    if read_env_warp_form_displacement() {
        crate::flame_wave::warp_strain_norm_rms(modes)
    } else {
        crate::flame_wave::warp_strain_norm(modes)
    }
}

/// The shear table the warp map actually evaluates: cf layers when the
/// closed-form variant is active, the 16 warp modes otherwise.
fn active_shear_table(
    warp_modes: &[crate::flame_wave::WaveWarpMode],
) -> Vec<crate::flame_wave::WaveWarpMode> {
    if read_env_wave_cf() {
        crate::flame_wave::generate_wave_cf_shear_layers(
            warp_modes,
            read_env_wave_cf_layers(),
            read_env_wave_cf_shear(),
        )
        .to_vec()
    } else {
        warp_modes.to_vec()
    }
}

/// Medium swirl modes with amplitudes expressing swirl_gain as a share of the
/// active shear table's strain norm (motion_design L2). Public so the debug
/// harnesses replay the exact packed table.
pub fn build_medium_swirl_modes(
    effect: &FlameEffect,
    warp_modes: &[crate::flame_wave::WaveWarpMode],
) -> [crate::flame_wave::WaveWarpMode; crate::flame_wave::WAVE_MEDIUM_SWIRL_MODE_COUNT] {
    let base_norm = shear_strain_norm(&active_shear_table(warp_modes));
    crate::flame_wave::generate_medium_swirl_modes(
        read_env_swirl_gain(effect.swirl.gain),
        base_norm,
    )
}

/// Asymptotic warp-strain profile. The strain norm is taken over the combined
/// table the warp map evaluates — the active shear table plus the medium swirl
/// modes — so the swirl joins the fixed strain budget: raising the swirl gain
/// thins the carve warp instead of exceeding the cap.
pub fn build_warp_strain_params(effect: &FlameEffect) -> FlameWarpStrainParams {
    let warp_modes = crate::flame_wave::generate_wave_warp_modes();
    let mut table = active_shear_table(&warp_modes);
    table.extend_from_slice(&build_medium_swirl_modes(effect, &warp_modes));
    let [strain_base, strain_tip, inv_reach, inv_strain_norm] =
        crate::flame_wave::warp_strain_params(
            effect.warp.amp,
            effect.warp.reach,
            shear_strain_norm(&table),
        );
    FlameWarpStrainParams {
        strain_base,
        strain_tip,
        inv_reach,
        inv_strain_norm,
    }
}

fn build_warp_form_params(effect: &FlameEffect) -> FlameWarpFormParams {
    FlameWarpFormParams {
        displacement_form: if read_env_warp_form_displacement() {
            1.0
        } else {
            0.0
        },
        burnout_gain: effect.carve.burnout_gain,
        _padding: [0.0; 2],
    }
}

pub fn build_flame_ubo(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
) -> FlameUBO {
    let (color_base, color_mid, color_tip) = resolve_flame_colors(&effect.color);
    let edge_window = effective_edge_window(&effect.edge, &effect.noise);
    let wave_fields = build_wave_ubo_fields(effect);
    FlameUBO {
        model: build_flame_model_matrix(effect),
        inverse_model: build_flame_inverse_model_matrix(effect),
        height_primitive_coefficients: effect.coefficients.height_primitive,
        radial_coefficients: effect.coefficients.radial,
        height_coefficients: effect.coefficients.height,
        time: effect.time,
        sigma_t: effective_sigma_t(effect),
        intensity: effect.intensity,
        height_axis_scale: 1.0,
        noise_amplitude: effect.noise.amplitude,
        noise_frequency: effect.noise.frequency,
        noise_scroll_speed: effect.noise.scroll_speed,
        radial_sharpness: effect.radial_sharpness,
        color_base: FlameColorBase {
            rgb: color_base,
            occlusion_lum_ref: effect.color.occlusion_lum_ref,
        },
        color_mid: FlameColorMid {
            rgb: color_mid,
            _padding: 1.0,
        },
        color_tip: FlameColorTip {
            rgb: color_tip,
            _padding: 0.0,
        },
        temporal_data: FlameTemporalParams {
            accum_weight: temporal.weight,
            frame_index: (temporal.frame_index % 16384) as f32,
            noise_aniso_y: effective_noise_aniso_y(&effect.noise, effect.height, effect.radius),
            warp_y_scale: effect.warp.y_scale,
        },
        light_data: build_light_params(effect),
        warp_style: build_warp_style(&effect.warp),
        edge_style: build_edge_style(&effect.edge, &effect.noise),
        wind_bend: build_wind_bend(&effect.wind),
        trail_unit_inverse: Matrix4::<f32>::from_scale(1.0),
        trail_meta: FlameTrailMeta {
            sample_count: 0.0,
            max_age: 0.0,
            _padding: [0.0; 2],
        },
        trail_coefficients: [[0.0; 4]; 4],
        emitter_params: build_emitter_params(&effect.emitter, effect.radius),
        contour_params: build_contour_params(&effect.contour),
        erosion_response: build_erosion_response(edge_window),
        wave_cf_params: FlameWaveCfParams {
            skipped_power_plain: wave_fields.skipped_power[0],
            skipped_power_env: wave_fields.skipped_power[1],
            ..build_wave_cf_params()
        },
        boundary_params: build_boundary_params(&effect.boundary),
        near_fade_params: build_near_fade_params(&effect.carve, edge_window),
        radius_coefficients: effect.coefficients.radius_scale,
        color_ramp: build_color_ramp(&effect.color, baked),
        temp_ramp: build_temperature_ramp(&effect.color),
        profile_params: build_profile_params(effect, baked),
        wave_params: wave_fields.shaping,
        tip_carve_params: build_tip_carve_params(&effect.carve, &effect.coefficients),
        warp_strain_params: build_warp_strain_params(effect),
        warp_form_params: build_warp_form_params(effect),
        unified_params: build_unified_field_params(effect),
        mix_params: build_mix_params(&effect.mix, wave_fields.low_carrier_std),
        segment_params: build_segment_params(effect),
        thermal_params: build_thermal_params(&effect.thermal, &effect.color),
        spread_params: build_medium_spread_params(effect),
        support_motion: FlameSupportMotion {
            support_margin: effect.support_margin,
            meander_amp: effect.meander.amp,
            swirl_speed: effect.swirl.speed,
            twist_speed: effect.twist.speed,
        },
        twist_field: build_twist_field(&effect.twist, &effect.swirl),
        meander_modes: build_meander_modes(&effect.meander, &effect.swirl),
        branch_field: build_branch_field(effect, baked),
        wave_modes: wave_fields.packed,
        wave_jitter: wave_fields.jitter,
    }
}

fn build_light_params(effect: &FlameEffect) -> FlameLightParams {
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    let relative = effect.light_position_world - effect.position;
    let direction = Vector3::new(
        relative.x / radius,
        relative.y / height,
        relative.z / radius,
    );
    let unit_direction = if direction.dot(direction) < 1e-6 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        direction.normalize()
    };
    FlameLightParams {
        direction: [unit_direction.x, unit_direction.y, unit_direction.z],
        self_shadow_strength: effect.self_shadow_strength,
    }
}

fn build_erosion_response(edge_window: (f32, f32)) -> FlameErosionResponse {
    let model = fit_erf_response(edge_window.0, edge_window.1);
    FlameErosionResponse {
        center: model.center,
        kappa: model.kappa,
        weight1: model.gaussian_weights[0],
        weight2: model.gaussian_weights[1],
    }
}

/// Build the expanded model matrix for flame trail rendering.
/// Computes world AABB of all trail samples + effect position, then builds a rotation-free
/// expansion matrix using the same construction rules as build_flame_model_matrix.
pub fn build_flame_trail_expanded_matrix(
    effect: &FlameEffect,
    samples: &[FlameTrailSample],
) -> Matrix4<f32> {
    assert!(
        !samples.is_empty(),
        "build_flame_trail_expanded_matrix requires at least one sample"
    );

    // Compute world AABB of all samples + effect position
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    // Include effect position
    let ep = &effect.position;
    min_x = min_x.min(ep.x);
    max_x = max_x.max(ep.x);
    min_y = min_y.min(ep.y);
    max_y = max_y.max(ep.y);
    min_z = min_z.min(ep.z);
    max_z = max_z.max(ep.z);

    // Include all sample positions
    for s in samples {
        let p = &s.position;
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
        min_z = min_z.min(p[2]);
        max_z = max_z.max(p[2]);
    }
    // XZ center = AABB center
    let cx = (min_x + max_x) * 0.5;
    let cz = (min_z + max_z) * 0.5;

    // Extension radius = effect.radius + hypot(half_extent_x, half_extent_z)
    let half_extent_x = (max_x - min_x) * 0.5;
    let half_extent_z = (max_z - min_z) * 0.5;
    let extension_radius = flame_bounding_radius(effect)
        + (half_extent_x * half_extent_x + half_extent_z * half_extent_z).sqrt();

    // Extension height = effect.height + (max_y - min_y)
    let extension_height = effect.height + (max_y - min_y);

    // Base y = min_y
    let base_y = min_y;

    // Build rotation-free expansion matrix using same construction as build_flame_model_matrix
    Matrix4::from_translation(Vector3::new(cx, base_y, cz))
        * Matrix4::from_nonuniform_scale(extension_radius, extension_height, extension_radius)
}

/// Build trail UBO fields: (trailUnitInverse, trailMeta, trailCoefficients).
/// trailUnitInverse = inverse of the unit model matrix (without expansion).
/// For each sample i: localDelta_i = trailUnitInverse.linear_part * (sample.position - effect.position), w = fade weight.
/// If count is 0, trailUnitInverse = identity matrix.
pub fn build_flame_trail_ubo_fields(
    effect: &FlameEffect,
    trail: &FlameTrailState,
) -> (Matrix4<f32>, FlameTrailMeta, [[f32; 4]; 4]) {
    let count = trail.samples.len();

    if count == 0 {
        return (
            Matrix4::<f32>::from_scale(1.0),
            FlameTrailMeta {
                sample_count: 0.0,
                max_age: 0.0,
                _padding: [0.0; 2],
            },
            [[0.0; 4]; 4],
        );
    }

    // trailUnitInverse = inverse of unit model matrix (analytical: translation*scale -> inv_scale*inv_translation)
    let radius = flame_bounding_radius(effect);
    let trail_unit_inverse = Matrix4::from_translation(-effect.position)
        * Matrix4::from_nonuniform_scale(1.0 / radius, 1.0 / effect.height, 1.0 / radius);

    // Build local-space sample offsets and their normalized ages (u = age_seconds / fade_seconds)
    let linear = Matrix3::<f32>::from_cols(
        Vector3::new(
            trail_unit_inverse[0][0],
            trail_unit_inverse[1][0],
            trail_unit_inverse[2][0],
        ),
        Vector3::new(
            trail_unit_inverse[0][1],
            trail_unit_inverse[1][1],
            trail_unit_inverse[2][1],
        ),
        Vector3::new(
            trail_unit_inverse[0][2],
            trail_unit_inverse[1][2],
            trail_unit_inverse[2][2],
        ),
    );

    let mut max_u: f32 = 0.0;
    let mut ata: [[f32; 4]; 4] = [[0.0; 4]; 4]; // A^T * A (Vandermonde normal matrix)

    // Build A^T * A (independent of data, depends only on u values)
    for i in 0..count {
        let sample = &trail.samples[i];
        let u = if trail.fade_seconds > 0.0 {
            sample.age_seconds / trail.fade_seconds
        } else {
            0.0
        };
        if u > max_u {
            max_u = u;
        }

        // Vandermonde row: [1, u, u^2, u^3]
        let v = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            for c in 0..4 {
                ata[r][c] += v[r] * v[c];
            }
        }
    }

    // Build A^T * b for each axis (x, y, z) and solve least-squares
    // The system is the same A^T*A for all axes, only b changes.

    // Create augmented matrix [A^T*A | I] and row-reduce to get inverse
    let mut aug: [[f32; 8]; 4] = [[0.0; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            aug[r][c] = ata[r][c];
        }
        aug[r][r + 4] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for col in 0..4 {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }
        // Scale pivot row
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for j in col..8 {
            aug[col][j] /= pivot;
        }
        // Eliminate column
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..8 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Now aug[:, 4..8] is the inverse of A^T*A
    // Compute coefficients for each axis: c = (A^T*A)^{-1} * A^T * b_axis
    let mut atb_x: [f32; 4] = [0.0; 4];
    let mut atb_y: [f32; 4] = [0.0; 4];
    let mut atb_z: [f32; 4] = [0.0; 4];
    for i in 0..count {
        let sample = &trail.samples[i];
        let u = if trail.fade_seconds > 0.0 {
            sample.age_seconds / trail.fade_seconds
        } else {
            0.0
        };
        let diff = Vector3::new(
            sample.position[0] - effect.position.x,
            sample.position[1] - effect.position.y,
            sample.position[2] - effect.position.z,
        );
        let local_delta = linear * diff;

        let v = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            atb_x[r] += v[r] * local_delta.x;
            atb_y[r] += v[r] * local_delta.y;
            atb_z[r] += v[r] * local_delta.z;
        }
    }

    // c = aug_inv * atb for each axis
    let mut coefficients: [[f32; 4]; 4] = [[0.0; 4]; 4];
    for r in 0..4 {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        for c in 0..4 {
            sum_x += aug[r][c + 4] * atb_x[c];
            sum_y += aug[r][c + 4] * atb_y[c];
            sum_z += aug[r][c + 4] * atb_z[c];
        }
        coefficients[r] = [sum_x, sum_y, sum_z, 0.0];
    }

    let meta = FlameTrailMeta {
        sample_count: count as f32,
        max_age: max_u,
        _padding: [0.0; 2],
    };

    (trail_unit_inverse, meta, coefficients)
}

/// Build FlameUBO with optional trail support.
/// If trail is Some AND trail_render_active AND trail.enabled AND samples not empty:
///   - Replace model/inverse_model with the expanded matrix and its inverse.
///   - Fill trail fields using build_flame_trail_ubo_fields.
/// Otherwise, return same as build_flame_ubo.
pub fn build_flame_ubo_with_trail(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
    trail: Option<&FlameTrailState>,
    trail_render_active: bool,
) -> FlameUBO {
    let trail = match trail {
        Some(t) if trail_render_active && t.enabled && !t.samples.is_empty() => t,
        _ => return build_flame_ubo(effect, baked, temporal),
    };

    // Expanded matrix is T*S (translation*scale), so inverse is S^-1 * T^-1
    let model = build_flame_trail_expanded_matrix(effect, &trail.samples);
    let inverse_model =
        Matrix4::from_nonuniform_scale(1.0 / model[0][0], 1.0 / model[1][1], 1.0 / model[2][2])
            * Matrix4::from_translation(-Vector3::new(model[3][0], model[3][1], model[3][2]));

    let (trail_unit_inverse, trail_meta, trail_coefficients) =
        build_flame_trail_ubo_fields(effect, trail);
    FlameUBO {
        model,
        inverse_model,
        trail_unit_inverse,
        trail_meta,
        trail_coefficients,
        ..build_flame_ubo(effect, baked, temporal)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameColorBase {
    pub rgb: [f32; 3],
    pub occlusion_lum_ref: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameColorMid {
    pub rgb: [f32; 3],
    pub _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameColorTip {
    pub rgb: [f32; 3],
    pub _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameTemporalParams {
    pub accum_weight: f32,
    pub frame_index: f32,
    pub noise_aniso_y: f32,
    pub warp_y_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameLightParams {
    pub direction: [f32; 3],
    pub self_shadow_strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWarpStyle {
    pub warp_amp: f32,
    pub warp_freq: f32,
    pub rise_speed: f32,
    pub taper_power: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameEdgeStyle {
    pub radius_tip_ratio: f32,
    pub edge_low: f32,
    pub edge_high: f32,
    pub white_boost: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWindBend {
    pub wind_direction: [f32; 2],
    pub bend_amount: f32,
    pub bend_power: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameTrailMeta {
    pub sample_count: f32,
    pub max_age: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameEmitterParams {
    pub kind: f32,
    pub ring_major_ratio: f32,
    pub ring_angular_speed: f32,
    /// Gaussian half-depth of the SDF billboard slab (emitter kind 2).
    pub sdf_slab_depth: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameContourParams {
    pub wiggle_amp: f32,
    pub aniso_axis_advect: f32,
    pub rte_bands: f32,
    pub sigma_dispersion: f32,
}

/// Bridge model of smoothstep(edge_low, edge_high, x): two gaussians around a
/// center (thyllore_math_core::ErfResponseModel).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameErosionResponse {
    pub center: f32,
    pub kappa: f32,
    pub weight1: f32,
    pub weight2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWaveCfParams {
    pub enabled: f32,
    pub shear_layer_count: f32,
    pub skipped_power_plain: f32,
    pub skipped_power_env: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameBoundaryParams {
    pub amp: f32,
    pub freq: f32,
    pub speed: f32,
    pub radius_ratio: f32,
}

/// Near fade plus the amplitude-scaled effective erosion edge window.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameNearFadeParams {
    pub radius: f32,
    pub carve_residual: f32,
    pub edge_low: f32,
    pub edge_high: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameProfileParams {
    pub radius_active: f32,
    pub radius_max: f32,
    pub color_active: f32,
    pub _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWaveShaping {
    pub tracked_count: f32,
    pub env_coeff: f32,
    pub inverse_scale: f32,
    pub amplitude: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameTipCarveParams {
    pub depth: f32,
    pub inv_reach: f32,
    pub primitive_top: f32,
    pub inv_primitive_range: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWarpStrainParams {
    pub strain_base: f32,
    pub strain_tip: f32,
    pub inv_reach: f32,
    pub inv_strain_norm: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameWarpFormParams {
    pub displacement_form: f32,
    pub burnout_gain: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameUnifiedParams {
    pub enabled: f32,
    pub sigma_floor: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameMixParams {
    pub lo: f32,
    pub hi: f32,
    pub inv_carrier_std: f32,
    pub height_gain: f32,
    /// Wavenumber scale of the mixing eddies relative to the low erosion octave.
    pub scale: f32,
    pub radial_gain: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameThermalParams {
    pub density_exp: f32,
    pub temp_exp: f32,
    pub temp_hot_k: f32,
    pub temp_cold_k: f32,
    pub wien_c_k: f32,
    pub _padding: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameSegmentParams {
    pub count: f32,
    pub inv_count: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameSpreadParams {
    pub gain: f32,
    pub edge_outer_sharpen: f32,
    pub twist_gain: f32,
    pub erosion_noise_gain: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameSupportMotion {
    pub support_margin: f32,
    pub meander_amp: f32,
    pub swirl_speed: f32,
    /// 0 = delegate the twist rate to swirl_speed.
    pub twist_speed: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameTwistMode {
    pub kappa: f32,
    pub omega: f32,
    pub phase: f32,
    pub amp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameTwistField {
    pub modes: [FlameTwistMode; 2],
    pub core_radius_sq: f32,
    pub _padding: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameMeanderMode {
    pub direction: [f32; 2],
    pub kappa: f32,
    pub omega: f32,
    pub phase: f32,
    pub _padding: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameBranchElement {
    pub spawn_time: f32,
    pub side: f32,
    pub azimuth: f32,
    pub spawn_height: f32,
    /// Size multiplier of reach and core (scatter lane).
    pub size: f32,
    /// Tilt of the vortex line out of the horizontal [rad] (scatter lane).
    pub tilt: f32,
    /// Window center shift along the line in reach units (scatter lane).
    pub along_offset: f32,
    pub hash01: f32,
    /// Trunk support radius at the spawn height, in flame-local units; the
    /// reach and core radii are ratios of it.
    pub trunk_radius: f32,
    pub _padding: [f32; 3],
}

/// Age-profile fractions of the vortex transport (winding, burnout), shared with
/// the shader so both sides evaluate the same envelope.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameBranchAgeProfile {
    pub wind_fraction: f32,
    pub burnout_start_fraction: f32,
    pub burnout_release_fraction: f32,
    pub burnout_margin: f32,
    pub burnout_trunk_inner: f32,
    pub _padding: [f32; 3],
}

impl Default for FlameBranchAgeProfile {
    fn default() -> Self {
        Self {
            wind_fraction: BRANCH_WIND_FRACTION,
            burnout_start_fraction: BRANCH_BURNOUT_START_FRACTION,
            burnout_release_fraction: BRANCH_BURNOUT_RELEASE_FRACTION,
            burnout_margin: BRANCH_BURNOUT_MARGIN,
            burnout_trunk_inner: BRANCH_BURNOUT_TRUNK_INNER,
            _padding: [0.0; 3],
        }
    }
}

/// Branch element table (newest first) with the per-effect age-profile
/// constants; `count` = 0 leaves every consumer bit-identical.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlameBranchField {
    pub count: f32,
    pub period: f32,
    pub life: f32,
    pub gain: f32,
    pub rise_rate: f32,
    pub drift_rate: f32,
    pub aspect: f32,
    pub core_radius: f32,
    pub reach_start: f32,
    pub reach_end: f32,
    pub envelope_time: f32,
    pub core_offset: f32,
    pub bounding_pad: f32,
    pub bounding_pad_y: f32,
    pub _padding1: [f32; 2],
    pub age_profile: FlameBranchAgeProfile,
    pub elements: [FlameBranchElement; BRANCH_MAX_ELEMENTS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlameUBO {
    pub model: Matrix4<f32>,
    pub inverse_model: Matrix4<f32>,
    pub height_primitive_coefficients: [[f32; 4]; 3],
    pub radial_coefficients: [[f32; 4]; 2],
    pub height_coefficients: [[f32; 4]; 2],
    pub time: f32,
    pub sigma_t: f32,
    pub intensity: f32,
    pub height_axis_scale: f32,
    pub noise_amplitude: f32,
    pub noise_frequency: f32,
    pub noise_scroll_speed: f32,
    pub radial_sharpness: f32,
    pub color_base: FlameColorBase,
    pub color_mid: FlameColorMid,
    pub color_tip: FlameColorTip,
    pub temporal_data: FlameTemporalParams,
    pub light_data: FlameLightParams,
    pub warp_style: FlameWarpStyle,
    pub edge_style: FlameEdgeStyle,
    pub wind_bend: FlameWindBend,
    pub trail_unit_inverse: Matrix4<f32>,
    pub trail_meta: FlameTrailMeta,
    pub trail_coefficients: [[f32; 4]; 4],
    pub emitter_params: FlameEmitterParams,
    pub contour_params: FlameContourParams,
    pub erosion_response: FlameErosionResponse,
    pub wave_cf_params: FlameWaveCfParams,
    pub boundary_params: FlameBoundaryParams,
    pub near_fade_params: FlameNearFadeParams,
    pub radius_coefficients: [[f32; 4]; 2],
    pub color_ramp: [[f32; 4]; 8],
    /// Planckian chromaticity sampled from temperature_tip_k (index 0) to
    /// temperature_base_k (index 7); the shader interpolates by node temperature.
    pub temp_ramp: [[f32; 4]; 8],
    pub profile_params: FlameProfileParams,
    pub wave_params: FlameWaveShaping,
    pub tip_carve_params: FlameTipCarveParams,
    pub warp_strain_params: FlameWarpStrainParams,
    pub warp_form_params: FlameWarpFormParams,
    pub unified_params: FlameUnifiedParams,
    pub mix_params: FlameMixParams,
    pub segment_params: FlameSegmentParams,
    pub thermal_params: FlameThermalParams,
    pub spread_params: FlameSpreadParams,
    pub support_motion: FlameSupportMotion,
    pub twist_field: FlameTwistField,
    pub meander_modes: [FlameMeanderMode; 2],
    pub branch_field: FlameBranchField,
    pub wave_modes: [[f32; 4]; 2 * crate::flame_wave::WAVE_MODE_SLOTS],
    pub wave_jitter: [[f32; 4]; crate::flame_wave::WAVE_MODE_COUNT],
}

impl Default for FlameUBO {
    fn default() -> Self {
        build_flame_ubo(
            &FlameEffect::default(),
            &FlameBaked::default(),
            &FlameTemporalAccum::default(),
        )
    }
}
