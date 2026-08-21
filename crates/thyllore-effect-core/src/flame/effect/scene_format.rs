use crate::flame::ownership::ParameterOwner;
use crate::flame::*;
use cgmath::{Quaternion, Vector2, Vector3};
use thyllore_scene_core::declare_scene_format;

declare_scene_format! {
    component: FlameEffect,
    record: FlameSceneRecord,
    tag: ParameterOwner,
    items {
        tags: PARAMETER_OWNERSHIP,
        snapshot: flame_parameter_snapshot,
        scalars: FLAME_SCALAR_PARAMS,
        overwrite: overwrite_persisted_fields,
    },
    persisted {
        position: [f32; 3] = Frame {
            get: |e| [e.position.x, e.position.y, e.position.z],
            set: |e, v| e.position = Vector3::new(v[0], v[1], v[2]),
        },
        rotation: [f32; 4] = Frame {
            get: |e| [e.rotation.s, e.rotation.v.x, e.rotation.v.y, e.rotation.v.z],
            set: |e, v| e.rotation = Quaternion::new(v[0], v[1], v[2], v[3]),
        },
        height: f32 = Frame { get: |e| e.height, set: |e, v| e.height = v },
        radius: f32 = Frame { get: |e| e.radius, set: |e, v| e.radius = v },
        sigma_t: f32 = Style { get: |e| e.sigma_t, set: |e, v| e.sigma_t = v },
        intensity: f32 = Style { get: |e| e.intensity, set: |e, v| e.intensity = v },
        color_base: [f32; 3] = Style { get: |e| e.color.base, set: |e, v| e.color.base = v },
        color_tip: [f32; 3] = Style { get: |e| e.color.tip, set: |e, v| e.color.tip = v },
        temperature_base_k: f32 = Style {
            get: |e| e.color.temperature_base_k,
            set: |e, v| e.color.temperature_base_k = v,
        },
        temperature_tip_k: f32 = Style {
            get: |e| e.color.temperature_tip_k,
            set: |e, v| e.color.temperature_tip_k = v,
        },
        use_blackbody: bool = Style {
            get: |e| e.color.use_blackbody,
            set: |e, v| e.color.use_blackbody = v,
        },
        noise_amplitude: f32 = Style {
            get: |e| e.noise.amplitude,
            set: |e, v| e.noise.amplitude = v,
        },
        noise_contrast: f32 = Style {
            get: |e| e.noise.contrast,
            set: |e, v| e.noise.contrast = v,
        },
        noise_frequency: f32 = Style {
            get: |e| e.noise.frequency,
            set: |e, v| e.noise.frequency = v,
        },
        noise_scroll_speed: f32 = Style {
            get: |e| e.noise.scroll_speed,
            set: |e, v| e.noise.scroll_speed = v,
        },
        time_scale: f32 = Frame { get: |e| e.time_scale, set: |e, v| e.time_scale = v },
        time_offset: f32 = Frame { get: |e| e.time_offset, set: |e, v| e.time_offset = v },
        warp_amp: f32 = Style { get: |e| e.warp.amp, set: |e, v| e.warp.amp = v },
        warp_freq: f32 = Style { get: |e| e.warp.freq, set: |e, v| e.warp.freq = v },
        rise_speed: f32 = Style {
            get: |e| e.warp.rise_speed,
            set: |e, v| e.warp.rise_speed = v,
        },
        taper_power: f32 = Shape {
            get: |e| e.warp.taper_power,
            set: |e, v| e.warp.taper_power = v,
        },
        radius_tip_ratio: f32 = Shape {
            get: |e| e.edge.radius_tip_ratio,
            set: |e, v| e.edge.radius_tip_ratio = v,
        },
        edge_low: f32 = Style { get: |e| e.edge.low, set: |e, v| e.edge.low = v },
        edge_high: f32 = Style { get: |e| e.edge.high, set: |e, v| e.edge.high = v },
        white_boost: f32 = Style {
            get: |e| e.edge.white_boost,
            set: |e, v| e.edge.white_boost = v,
        },
        wind_direction: [f32; 2] = Frame {
            get: |e| [e.wind.direction.x, e.wind.direction.y],
            set: |e, v| e.wind.direction = Vector2::new(v[0], v[1]),
            scalars {
                wind_x: {
                    get: |e| e.wind.direction.x,
                    set: |e, v| e.wind.direction.x = v,
                },
                wind_z: {
                    get: |e| e.wind.direction.y,
                    set: |e, v| e.wind.direction.y = v,
                },
            },
        },
        bend_amount: f32 = Frame {
            get: |e| e.wind.bend_amount,
            set: |e, v| e.wind.bend_amount = v,
        },
        bend_power: f32 = Frame {
            get: |e| e.wind.bend_power,
            set: |e, v| e.wind.bend_power = v,
        },
        self_shadow_strength: f32 = Style {
            get: |e| e.self_shadow_strength,
            set: |e, v| e.self_shadow_strength = v,
        },
        envelope_peak: f32 = Shape {
            get: |e| e.envelope.peak,
            set: |e, v| e.envelope.peak = v,
            default: 0.35,
        },
        envelope_base: f32 = Shape {
            get: |e| e.envelope.base,
            set: |e, v| e.envelope.base = v,
            default: 0.45,
        },
        envelope_tail: f32 = Shape {
            get: |e| e.envelope.tail,
            set: |e, v| e.envelope.tail = v,
            default: 1.6,
        },
        radial_sharpness: f32 = Shape {
            get: |e| e.radial_sharpness,
            set: |e, v| e.radial_sharpness = v,
        },
        occlusion_lum_ref: f32 = Style {
            get: |e| e.color.occlusion_lum_ref,
            set: |e, v| e.color.occlusion_lum_ref = v,
        },
        contour_wiggle_amp: f32 = Style {
            get: |e| e.contour.wiggle_amp,
            set: |e, v| e.contour.wiggle_amp = v,
        },
        aniso_axis_advect: f32 = Style {
            get: |e| e.contour.aniso_axis_advect,
            set: |e, v| e.contour.aniso_axis_advect = v,
        },
        rte_bands: f32 = Style {
            get: |e| e.contour.rte_bands,
            set: |e, v| e.contour.rte_bands = v,
        },
        sigma_dispersion: f32 = Style {
            get: |e| e.contour.sigma_dispersion,
            set: |e, v| e.contour.sigma_dispersion = v,
        },
        tip_carve_depth: f32 = Style {
            get: |e| e.carve.tip.depth,
            set: |e, v| e.carve.tip.depth = v,
        },
        tip_carve_reach: f32 = Style {
            get: |e| e.carve.tip.reach,
            set: |e, v| e.carve.tip.reach = v,
        },
        warp_reach: f32 = Style { get: |e| e.warp.reach, set: |e, v| e.warp.reach = v },
        swirl_gain: f32 = Style { get: |e| e.swirl.gain, set: |e, v| e.swirl.gain = v },
        swirl_speed: f32 = Style { get: |e| e.swirl.speed, set: |e, v| e.swirl.speed = v },
        spread_gain: f32 = Style { get: |e| e.spread_gain, set: |e, v| e.spread_gain = v },
        support_margin: f32 = Style {
            get: |e| e.support_margin,
            set: |e, v| e.support_margin = v,
        },
        meander_amp: f32 = Style { get: |e| e.meander.amp, set: |e, v| e.meander.amp = v },
        meander_frequency: f32 = Style {
            get: |e| e.meander.frequency,
            set: |e, v| e.meander.frequency = v,
        },
        mix_lo: f32 = Style { get: |e| e.mix.lo, set: |e, v| e.mix.lo = v },
        mix_hi: f32 = Style { get: |e| e.mix.hi, set: |e, v| e.mix.hi = v },
        mix_height_gain: f32 = Style {
            get: |e| e.mix.height_gain,
            set: |e, v| e.mix.height_gain = v,
        },
        mix_scale: f32 = Style { get: |e| e.mix.scale, set: |e, v| e.mix.scale = v },
        mix_radial_gain: f32 = Style {
            get: |e| e.mix.radial_gain,
            set: |e, v| e.mix.radial_gain = v,
        },
        density_exp: f32 = Style {
            get: |e| e.thermal.density_exp,
            set: |e, v| e.thermal.density_exp = v,
        },
        temp_exp: f32 = Style {
            get: |e| e.thermal.temp_exp,
            set: |e, v| e.thermal.temp_exp = v,
        },
        wien_c_k: f32 = Style {
            get: |e| e.thermal.wien_c_k,
            set: |e, v| e.thermal.wien_c_k = v,
        },
        wave_segments: u32 = Frame {
            get: |e| e.wave_segments,
            set: |e, v| e.wave_segments = v,
        },
        noise_aniso_y: f32 = Style {
            get: |e| e.noise.aniso_y,
            set: |e, v| e.noise.aniso_y = v,
        },
        edge_outer_sharpen: f32 = Style {
            get: |e| e.edge.outer_sharpen,
            set: |e, v| e.edge.outer_sharpen = v,
        },
        noise_scale_mode: f32 = Style {
            get: |e| e.noise.scale_mode,
            set: |e, v| e.noise.scale_mode = v,
        },
        erosion_noise_gain: f32 = Style {
            get: |e| e.noise.erosion_gain,
            set: |e, v| e.noise.erosion_gain = v,
        },
        twist_gain: f32 = Style { get: |e| e.twist.gain, set: |e, v| e.twist.gain = v },
        twist_speed: f32 = Style { get: |e| e.twist.speed, set: |e, v| e.twist.speed = v },
        burnout_gain: f32 = Style {
            get: |e| e.carve.burnout_gain,
            set: |e, v| e.carve.burnout_gain = v,
        },
        noise_shaping_scale: f32 = Style {
            get: |e| e.noise.shaping_scale,
            set: |e, v| e.noise.shaping_scale = v,
        },
        optical_depth: f32 = Style {
            get: |e| e.optical_depth,
            set: |e, v| e.optical_depth = v,
        },
        branch_period: f32 = Style {
            get: |e| e.branch.period,
            set: |e, v| e.branch.period = v,
        },
        branch_life: f32 = Style { get: |e| e.branch.life, set: |e, v| e.branch.life = v },
        branch_gain: f32 = Style { get: |e| e.branch.gain, set: |e, v| e.branch.gain = v },
        branch_core_radius: f32 = Style {
            get: |e| e.branch.core_radius,
            set: |e, v| e.branch.core_radius = v,
        },
        branch_core_offset: f32 = Style {
            get: |e| e.branch.core_offset,
            set: |e, v| e.branch.core_offset = v,
        },
        branch_reach: f32 = Style {
            get: |e| e.branch.reach,
            set: |e, v| e.branch.reach = v,
        },
        branch_spread: f32 = Style {
            get: |e| e.branch.spread,
            set: |e, v| e.branch.spread = v,
        },
        branch_spawn_height: f32 = Style {
            get: |e| e.branch.spawn_height,
            set: |e, v| e.branch.spawn_height = v,
        },
        branch_spawn_range: f32 = Style {
            get: |e| e.branch.spawn_range,
            set: |e, v| e.branch.spawn_range = v,
        },
        branch_seed: u32 = Frame { get: |e| e.branch.seed, set: |e, v| e.branch.seed = v },
    },
    runtime {
        time: f32 { get: |e| e.time, set: |e, v| e.time = v },
        warp_y_scale: f32 { get: |e| e.warp.y_scale, set: |e, v| e.warp.y_scale = v },
        emitter_kind: u32 { get: |e| e.emitter.kind, set: |e, v| e.emitter.kind = v },
        ring_major_radius: f32 {
            get: |e| e.emitter.ring_major_radius,
            set: |e, v| e.emitter.ring_major_radius = v,
        },
        ring_angular_speed: f32 {
            get: |e| e.emitter.ring_angular_speed,
            set: |e, v| e.emitter.ring_angular_speed = v,
        },
        boundary_amp: f32 { get: |e| e.boundary.amp, set: |e, v| e.boundary.amp = v },
        boundary_freq: f32 { get: |e| e.boundary.freq, set: |e, v| e.boundary.freq = v },
        boundary_speed: f32 { get: |e| e.boundary.speed, set: |e, v| e.boundary.speed = v },
        boundary_radius_ratio: f32 {
            get: |e| e.boundary.radius_ratio,
            set: |e, v| e.boundary.radius_ratio = v,
        },
        near_fade_radius: f32 {
            get: |e| e.carve.near_fade_radius,
            set: |e, v| e.carve.near_fade_radius = v,
        },
        carve_residual: f32 {
            get: |e| e.carve.residual,
            set: |e, v| e.carve.residual = v,
        },
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_scene_core::find_scalar_param;

    #[test]
    fn test_missing_envelope_keys_take_legacy_scene_defaults() {
        let effect: FlameEffect = serde_json::from_str("{}").expect("all keys defaulted");
        assert_eq!(effect.envelope.peak, 0.35);
        assert_eq!(effect.envelope.base, 0.45);
        assert_eq!(effect.envelope.tail, 1.6);
        assert_eq!(effect.height, FlameEffect::default().height);
    }

    #[test]
    fn test_overwrite_persisted_fields_keeps_runtime_state() {
        let mut loaded = FlameEffect::default();
        loaded.height = 9.0;
        loaded.time = 5.0;
        loaded.emitter.kind = 1;

        let mut target = FlameEffect::default();
        target.time = 2.5;
        target.emitter.kind = 2;
        target.warp.y_scale = 0.9;

        overwrite_persisted_fields(&mut target, &loaded);
        assert_eq!(target.height, 9.0);
        assert_eq!(target.time, 2.5);
        assert_eq!(target.emitter.kind, 2);
        assert_eq!(target.warp.y_scale, 0.9);
    }

    #[test]
    fn test_ron_struct_syntax_roundtrip() {
        let mut effect = FlameEffect::default();
        effect.height = 3.25;
        effect.mix.scale = 0.5;

        let text = ron::ser::to_string_pretty(&effect, ron::ser::PrettyConfig::new())
            .expect("ron serialize");
        let restored: FlameEffect = ron::from_str(&text).expect("ron deserialize");
        assert_eq!(restored.height, 3.25);
        assert_eq!(restored.mix.scale, 0.5);
    }

    #[test]
    fn test_scalar_param_names_are_unique() {
        let mut names: Vec<&str> = FLAME_SCALAR_PARAMS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    #[test]
    fn test_scalar_param_set_then_get_reaches_a_fixpoint() {
        for (i, param) in FLAME_SCALAR_PARAMS.iter().enumerate() {
            let mut effect = FlameEffect::default();
            (param.set)(&mut effect, 3.0 + i as f32);
            let first = (param.get)(&effect);
            (param.set)(&mut effect, first);
            assert_eq!((param.get)(&effect), first, "{}", param.name);
        }
    }

    #[test]
    fn test_bool_scalar_param_maps_zero_and_nonzero() {
        let param = find_scalar_param(FLAME_SCALAR_PARAMS, "use_blackbody").expect("registered");
        let mut effect = FlameEffect::default();
        (param.set)(&mut effect, 1.0);
        assert!(effect.color.use_blackbody);
        (param.set)(&mut effect, 0.0);
        assert!(!effect.color.use_blackbody);
    }

    #[test]
    fn test_wind_aliases_write_wind_direction_components() {
        let mut effect = FlameEffect::default();
        (find_scalar_param(FLAME_SCALAR_PARAMS, "wind_x")
            .expect("registered")
            .set)(&mut effect, 0.25);
        (find_scalar_param(FLAME_SCALAR_PARAMS, "wind_z")
            .expect("registered")
            .set)(&mut effect, -0.5);
        assert_eq!(effect.wind.direction.x, 0.25);
        assert_eq!(effect.wind.direction.y, -0.5);
    }

    #[test]
    fn test_runtime_params_are_not_serialized() {
        let value = serde_json::to_value(FlameEffect::default()).expect("serialize");
        let object = value.as_object().expect("flat object");
        for name in ["time", "warp_y_scale", "emitter_kind", "boundary_amp"] {
            assert!(!object.contains_key(name), "{name} must stay runtime-only");
        }
        assert_eq!(object.len(), PARAMETER_OWNERSHIP.len());
    }
}
