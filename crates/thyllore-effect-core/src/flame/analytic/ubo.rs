use crate::flame::*;
use crate::flame_trail::{FlameTrailSample, FlameTrailState};
use cgmath::{InnerSpace, Matrix3, Matrix4, Vector3};
use thyllore_color_core::blackbody_rgb;
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
        effect.wind_direction.x,
        effect.rise_speed,
        effect.wind_direction.y,
    );
    let mut axis = Vector3::new(0.0, 1.0, 0.0);
    if effect.aniso_axis_advect > 0.0 && advect.magnitude2() > 1e-8 {
        let blend = effect.aniso_axis_advect.clamp(0.0, 1.0);
        axis = (axis * (1.0 - blend) + advect.normalize() * blend).normalize();
    }
    let vector = Vector3::new(v[0], v[1], v[2]);
    let compressed = vector - axis * vector.dot(axis) * (1.0 - effect.noise_aniso_y);
    [compressed.x, compressed.y, compressed.z]
}

/// Effective noise aniso y: noise_scale_mode < 0.5 returns noise_aniso_y as-is;
/// >= 0.5 multiplies by height/radius to compensate for world-to-local scaling.
pub fn effective_noise_aniso_y(effect: &FlameEffect) -> f32 {
    if effect.noise_scale_mode < 0.5 {
        effect.noise_aniso_y
    } else {
        effect.noise_aniso_y * (effect.height / effect.radius.max(1e-4))
    }
}

/// Medium spread gain alpha (motion_design L3); the reach shares the tip
/// carve inv_reach in the shader.
fn build_medium_spread_params(effect: &FlameEffect) -> FlameSpreadParams {
    FlameSpreadParams {
        gain: effect.spread_gain.max(0.0),
        edge_outer_sharpen: effect.edge_outer_sharpen,
        twist_gain: effect.twist.gain,
        erosion_noise_gain: effect.erosion_noise_gain,
    }
}

/// The twist rate scale: twist speed owns the rate when positive, otherwise
/// the rate delegates to the swirl speed.
fn twist_rate_scale(effect: &FlameEffect) -> f32 {
    if effect.twist.speed > 0.0 {
        effect.twist.speed
    } else {
        effect.swirl.speed
    }
}

fn build_twist_field(effect: &FlameEffect) -> FlameTwistField {
    let rate_scale = twist_rate_scale(effect);
    FlameTwistField {
        modes: std::array::from_fn(|j| FlameTwistMode {
            kappa: TWIST_MODE_KAPPA[j],
            omega: TWIST_MODE_SPIN[j] * rate_scale * twist_mode_phase_rate(TWIST_MODE_KAPPA[j]),
            phase: TWIST_MODE_PHASE[j],
            amp: TWIST_MODE_AMP[j],
        }),
        core_radius_sq: TWIST_CORE_RADIUS_SQ,
        _padding: [0.0; 3],
    }
}

fn build_meander_modes(effect: &FlameEffect) -> [FlameMeanderMode; 2] {
    std::array::from_fn(|j| FlameMeanderMode {
        direction: MEANDER_MODE_DIRECTION[j],
        kappa: MEANDER_MODE_KAPPA[j],
        omega: effect.swirl.speed * MEANDER_MODE_RATE_SCALE[j],
        phase: MEANDER_MODE_PHASE[j],
        _padding: [0.0; 3],
    })
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

type WaveUboFields = (
    FlameWaveShaping,
    [[f32; 4]; 2 * crate::flame_wave::WAVE_MODE_SLOTS],
    [f32; 2],
    [[f32; 4]; crate::flame_wave::WAVE_MODE_COUNT],
);

/// Contrast-scaled base edge window: center is fixed, half-width divides by
/// noise_contrast (higher contrast = narrower window = harder carving).
/// Exactly 1.0 returns the authored edge_low/edge_high bytes untouched.
pub fn contrast_scaled_edges(effect: &FlameEffect) -> (f32, f32) {
    let contrast = effect.noise_contrast.clamp(0.25, 4.0);
    if contrast == 1.0 {
        return (effect.edge_low, effect.edge_high);
    }
    let c = 0.5 * (effect.edge_low + effect.edge_high);
    let hw = 0.5 * (effect.edge_high - effect.edge_low) / contrast;
    (c - hw, c + hw)
}

/// Compute the effective edge window (low, high) from noise amplitude.
/// Center c = 0.5*(edge_low + edge_high) is fixed; half-width scales with
/// |noise_amplitude| / NOISE_AMPLITUDE_REF raised to EDGE_WIDTH_GAMMA, clamped
/// to [0.25*hw0, 4.0*hw0] where hw0 = contrast-scaled half-width.
pub fn effective_edge_window(effect: &FlameEffect) -> (f32, f32) {
    let (edge_lo, edge_hi) = contrast_scaled_edges(effect);
    let c = 0.5 * (edge_lo + edge_hi);
    let hw0 = 0.5 * (edge_hi - edge_lo);
    let hw = hw0 * (effect.noise_amplitude.abs() / NOISE_AMPLITUDE_REF).powf(EDGE_WIDTH_GAMMA);
    let hw = hw.max(0.25 * hw0).min(4.0 * hw0);
    (c - hw, c + hw)
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
            effect.contour_wiggle_amp * read_env_unified_tilt_gain_w(),
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

    let warp_modes = generate_wave_warp_modes();
    let cf_active = read_env_wave_cf() && !unified;
    let depth_scale: Vec<f32> = if cf_active {
        wave_cf_depth_scale(&erosion_modes, &warp_modes, effect.noise_frequency, |v| {
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
    let tanh_scale = if effect.noise_shaping_scale > 0.0 {
        effect.noise_shaping_scale
    } else {
        read_env_wave_tanh()
    };
    let (inverse_scale, mut amplitude) = if tanh_scale <= 0.0 {
        (0.0, 1.0)
    } else {
        wave_shaping_params(tanh_scale)
    };
    amplitude *= (effect.noise_amplitude.abs() / NOISE_AMPLITUDE_REF).powf(SHAPING_GAMMA);
    let jitter_scale = crate::flame_wave::read_env_wave_jitter();
    let mut jitter = [[0.0f32; 4]; WAVE_MODE_COUNT];
    for (slot, mode) in jitter.iter_mut().zip(erosion_modes.iter()) {
        slot[0] = mode.jitter[0] * jitter_scale;
        slot[1] = mode.jitter[1] * jitter_scale;
        slot[2] = mode.jitter[2] * jitter_scale;
    }
    jitter[0][3] = crate::flame_wave::read_env_wave_jitter_freq();
    (
        FlameWaveShaping {
            tracked_count: tracked as f32,
            env_coeff,
            inverse_scale,
            amplitude,
        },
        packed,
        skipped_power,
        jitter,
    )
}

/// sigma_floor = relative-window floor coefficient (beta * |A| * shaped noise
/// std; the shader multiplies by lambda * D_mid / 0.30 for the modulation std).
pub fn build_unified_field_params(effect: &FlameEffect) -> FlameUnifiedParams {
    let inactive = FlameUnifiedParams {
        enabled: 0.0,
        sigma_floor: 0.0,
        _padding: [0.0; 2],
    };
    if !read_env_wave_unified() {
        return inactive;
    }
    let amplitude_ratio = (effect.noise_amplitude.abs() / NOISE_AMPLITUDE_REF).powf(SHAPING_GAMMA);
    let std = crate::flame_wave::unified_noise_std(
        read_env_wave_k_ratio(),
        read_env_wave_env_mu(),
        effect.boundary.amp * read_env_unified_tilt_gain_b(),
        effect.contour_wiggle_amp * read_env_unified_tilt_gain_w(),
    );
    let sigma_floor =
        read_env_unified_beta() * effect.noise_amplitude.abs() * std * amplitude_ratio;
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

/// Build the color ramp array for the UBO.
/// When baked_color is Some and baked_blend > 0, each entry is a blend between
/// the legacy 3-point ramp value (same as flameRampColor shader) and the baked color.
/// Otherwise returns all zeros (flag 0 means shader doesn't read it).
fn build_color_ramp(effect: &FlameEffect, baked_state: &FlameBaked) -> [[f32; 4]; 8] {
    let baked = match baked_state.color {
        Some(ref b) if baked_state.blend > 0.0 => b,
        _ => return [[0.0; 4]; 8],
    };
    let blend = baked_state.blend;

    // Compute color_base, color_mid, color_tip same as build_flame_ubo
    let (color_base, color_mid, color_tip) = if effect.use_blackbody {
        let base = blackbody_rgb(effect.temperature_base_k);
        let tip = blackbody_rgb(effect.temperature_tip_k);
        let mid_temp = (effect.temperature_base_k + effect.temperature_tip_k) / 2.0;
        let mid = blackbody_rgb(mid_temp);
        (base, mid, tip)
    } else {
        let base = effect.color_base;
        let tip = effect.color_tip;
        let mid = [
            (base[0] + tip[0]) / 2.0,
            (base[1] + tip[1]) / 2.0,
            (base[2] + tip[2]) / 2.0,
        ];
        (base, mid, tip)
    };

    let mut ramp = [[0.0f32; 4]; 8];
    for i in 0..8 {
        let h = (i as f32 + 0.5) / 8.0;
        // Legacy 3-point ramp (same as flameRampColor shader)
        let legacy = if h < 0.5 {
            let t = h * 2.0;
            [
                color_base[0] + (color_mid[0] - color_base[0]) * t,
                color_base[1] + (color_mid[1] - color_base[1]) * t,
                color_base[2] + (color_mid[2] - color_base[2]) * t,
            ]
        } else {
            let t = (h - 0.5) * 2.0;
            [
                color_mid[0] + (color_tip[0] - color_mid[0]) * t,
                color_mid[1] + (color_tip[1] - color_mid[1]) * t,
                color_mid[2] + (color_tip[2] - color_mid[2]) * t,
            ]
        };
        // Blend: lerp(legacy, baked, blend)
        let r = legacy[0] + (baked[i][0] - legacy[0]) * blend;
        let g = legacy[1] + (baked[i][1] - legacy[1]) * blend;
        let b = legacy[2] + (baked[i][2] - legacy[2]) * blend;
        ramp[i] = [r, g, b, 0.0];
    }
    ramp
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
            effect.warp_amp,
            effect.warp_reach,
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
        burnout_gain: effect.burnout_gain,
        _padding: [0.0; 2],
    }
}

fn build_tip_carve_params(effect: &FlameEffect) -> FlameTipCarveParams {
    let primitive = thyllore_math_core::ChebyshevSeries::new(
        effect
            .coefficients
            .height_primitive
            .iter()
            .flatten()
            .copied()
            .collect(),
        (0.0, 1.0),
    );
    let (at_base, at_top) = thyllore_math_core::chebyshev_endpoint_values(&primitive);
    let total = at_top - at_base;
    let inv_total = if total.abs() > 1e-6 { 1.0 / total } else { 0.0 };
    FlameTipCarveParams {
        depth: effect.tip_carve.depth,
        inv_reach: 1.0 / effect.tip_carve.reach.max(1e-3),
        primitive_top: at_top,
        inv_primitive_range: inv_total,
    }
}

pub fn build_flame_ubo(
    effect: &FlameEffect,
    baked: &FlameBaked,
    temporal: &FlameTemporalAccum,
) -> FlameUBO {
    let (color_base, color_mid, color_tip) = if effect.use_blackbody {
        let base = blackbody_rgb(effect.temperature_base_k);
        let tip = blackbody_rgb(effect.temperature_tip_k);
        let mid_temp = (effect.temperature_base_k + effect.temperature_tip_k) / 2.0;
        let mid = blackbody_rgb(mid_temp);
        (base, mid, tip)
    } else {
        let base = effect.color_base;
        let tip = effect.color_tip;
        let mid = [
            (base[0] + tip[0]) / 2.0,
            (base[1] + tip[1]) / 2.0,
            (base[2] + tip[2]) / 2.0,
        ];
        (base, mid, tip)
    };
    let radius = flame_bounding_radius(effect).max(MIN_FLAME_EXTENT);
    let height = effect.height.max(MIN_FLAME_EXTENT);
    let rel = effect.light_position_world - effect.position;
    let dir = Vector3::new(rel.x / radius, rel.y / height, rel.z / radius);
    let norm_dir = if dir.dot(dir) < 1e-6 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        dir.normalize()
    };
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
        noise_amplitude: effect.noise_amplitude,
        noise_frequency: effect.noise_frequency,
        noise_scroll_speed: effect.noise_scroll_speed,
        radial_sharpness: effect.radial_sharpness,
        color_base: FlameColorBase {
            rgb: color_base,
            occlusion_lum_ref: effect.occlusion_lum_ref,
        },
        color_mid: FlameColorMid {
            rgb: color_mid,
            _padding: 1.0,
        },
        color_tip: FlameColorTip {
            rgb: color_tip,
            edge_temperature_blend: effect.edge_temperature_blend,
        },
        temporal_data: FlameTemporalParams {
            accum_weight: temporal.weight,
            frame_index: (temporal.frame_index % 16384) as f32,
            noise_aniso_y: effective_noise_aniso_y(effect),
            warp_y_scale: effect.warp_y_scale,
        },
        light_data: FlameLightParams {
            direction: [norm_dir.x, norm_dir.y, norm_dir.z],
            self_shadow_strength: effect.self_shadow_strength,
        },
        warp_style: FlameWarpStyle {
            warp_amp: effect.warp_amp,
            warp_freq: effect.warp_freq,
            rise_speed: effect.rise_speed,
            taper_power: effect.taper_power,
        },
        edge_style: {
            let (edge_low, edge_high) = contrast_scaled_edges(effect);
            FlameEdgeStyle {
                radius_tip_ratio: effect.radius_tip_ratio,
                edge_low,
                edge_high,
                white_boost: effect.white_boost,
            }
        },
        wind_bend: FlameWindBend {
            wind_direction: [effect.wind_direction.x, effect.wind_direction.y],
            bend_amount: effect.bend_amount,
            bend_power: effect.bend_power,
        },
        trail_unit_inverse: Matrix4::<f32>::from_scale(1.0),
        trail_meta: FlameTrailMeta {
            sample_count: 0.0,
            max_age: 0.0,
            _padding: [0.0; 2],
        },
        trail_coefficients: [[0.0; 4]; 4],
        emitter_params: FlameEmitterParams {
            kind: effect.emitter_kind as f32,
            ring_major_ratio: if effect.emitter_kind == 1 {
                effect.ring_major_radius / flame_bounding_radius(effect)
            } else {
                0.0
            },
            ring_angular_speed: effect.ring_angular_speed,
            sdf_slab_depth: if effect.emitter_kind == 2 { 0.15 } else { 0.0 },
        },
        contour_params: FlameContourParams {
            wiggle_amp: effect.contour_wiggle_amp,
            aniso_axis_advect: effect.aniso_axis_advect,
            rte_bands: effect.rte_bands,
            sigma_dispersion: effect.sigma_dispersion,
        },
        erosion_response: {
            let (edge_low, edge_high) = effective_edge_window(effect);
            let model = fit_erf_response(edge_low, edge_high);
            FlameErosionResponse {
                center: model.center,
                kappa: model.kappa,
                weight1: model.gaussian_weights[0],
                weight2: model.gaussian_weights[1],
            }
        },
        wave_cf_params: FlameWaveCfParams {
            skipped_power_plain: wave_fields.2[0],
            skipped_power_env: wave_fields.2[1],
            ..build_wave_cf_params()
        },
        boundary_params: FlameBoundaryParams {
            amp: effect.boundary.amp,
            freq: effect.boundary.freq,
            speed: effect.boundary.speed,
            radius_ratio: effect.boundary.radius_ratio,
        },
        near_fade_params: {
            let (edge_low, edge_high) = effective_edge_window(effect);
            FlameNearFadeParams {
                radius: effect.near_fade_radius,
                carve_residual: effect.carve_residual,
                edge_low,
                edge_high,
            }
        },
        radius_coefficients: effect.coefficients.radius_scale,
        color_ramp: build_color_ramp(effect, baked),
        profile_params: build_profile_params(effect, baked),
        wave_params: wave_fields.0,
        tip_carve_params: build_tip_carve_params(effect),
        warp_strain_params: build_warp_strain_params(effect),
        warp_form_params: build_warp_form_params(effect),
        unified_params: build_unified_field_params(effect),
        spread_params: build_medium_spread_params(effect),
        support_motion: FlameSupportMotion {
            support_margin: effect.support_margin,
            meander_amp: effect.meander_amp,
            swirl_speed: effect.swirl.speed,
            twist_speed: effect.twist.speed,
        },
        twist_field: build_twist_field(effect),
        meander_modes: build_meander_modes(effect),
        branch_field: build_branch_field(effect, baked),
        wave_modes: wave_fields.1,
        wave_jitter: wave_fields.3,
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
    pub edge_temperature_blend: f32,
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
    pub kind: f32,
    pub hash01: f32,
    /// Trunk support radius at the spawn height, in flame-local units; the
    /// ring and core radii are ratios of it.
    pub trunk_radius: f32,
    pub _padding: f32,
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
    pub profile_params: FlameProfileParams,
    pub wave_params: FlameWaveShaping,
    pub tip_carve_params: FlameTipCarveParams,
    pub warp_strain_params: FlameWarpStrainParams,
    pub warp_form_params: FlameWarpFormParams,
    pub unified_params: FlameUnifiedParams,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_noise_aniso_y_mode_zero() {
        let mut effect = FlameEffect::default();
        effect.noise_scale_mode = 0.0;
        effect.noise_aniso_y = 0.5;
        assert!((effective_noise_aniso_y(&effect) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_effective_noise_aniso_y_mode_one() {
        let mut effect = FlameEffect::default();
        effect.noise_scale_mode = 1.0;
        effect.noise_aniso_y = 0.5;
        effect.height = 8.0;
        effect.radius = 1.0;
        assert!((effective_noise_aniso_y(&effect) - 4.0).abs() < 1e-6);
    }
}
