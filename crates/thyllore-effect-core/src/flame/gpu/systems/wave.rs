use crate::flame::*;
use cgmath::{InnerSpace, Vector3};

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

pub(super) fn build_wave_cf_params() -> FlameWaveCfParams {
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

pub(super) struct WaveUboFields {
    pub(super) shaping: FlameWaveShaping,
    pub(super) packed: [[f32; 4]; 2 * crate::flame_wave::WAVE_MODE_SLOTS],
    pub(super) skipped_power: [f32; 2],
    pub(super) jitter: [[f32; 4]; crate::flame_wave::WAVE_MODE_COUNT],
    /// Std of the low-octave erosion carrier zLow (the envelope modes), before
    /// the tanh shaping; the mixing window is expressed in these units.
    pub(super) low_carrier_std: f32,
}

pub(super) fn build_wave_ubo_fields(effect: &FlameEffect) -> WaveUboFields {
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
            crate::noise_lobe_shape(&effect.noise),
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

pub(super) fn build_segment_params(effect: &FlameEffect) -> FlameSegmentParams {
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
        crate::noise_lobe_shape(&effect.noise),
    );
    let sigma_floor =
        read_env_unified_beta() * effect.noise.amplitude.abs() * std * amplitude_ratio;
    FlameUnifiedParams {
        enabled: 1.0,
        sigma_floor,
        _padding: [0.0; 2],
    }
}
