use crate::animation::editable::{
    clip_recalculate_duration, curve_add_keyframe, curve_sample, ClipInstance,
    EditableAnimationClip, SourceClipId,
};
use crate::asset::AssetStorage;
use crate::ecs::component::{ClipSchedule, FlameEffect, FlameParam};
use crate::ecs::resource::ClipLibrary;
use crate::ecs::world::{Entity, World};

pub const FLAME_CLIP_NAME: &str = "Flame";

pub fn find_flame_clip_id(world: &World, flame_entity: Entity) -> Option<SourceClipId> {
    world
        .get_component::<ClipSchedule>(flame_entity)
        .and_then(|schedule| schedule.first_instance().map(|i| i.source_id))
}

/// Returns the flame entity's clip, creating the clip and a schedule instance
/// (start 0, speed 1, so clip-local time equals timeline time) on first use.
pub fn ensure_flame_clip(
    world: &mut World,
    assets: &mut AssetStorage,
    flame_entity: Entity,
) -> SourceClipId {
    if let Some(id) = find_flame_clip_id(world, flame_entity) {
        return id;
    }

    let source_id = {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let editable = EditableAnimationClip::new(0, FLAME_CLIP_NAME.to_string());
        super::clip_library_systems::clip_library_register_and_activate(
            &mut clip_library,
            assets,
            editable,
        )
    };

    let mut schedule = world
        .get_component::<ClipSchedule>(flame_entity)
        .cloned()
        .unwrap_or_default();
    if schedule.next_instance_id == 0 {
        schedule.next_instance_id = 1;
    }
    let instance_id = schedule.next_instance_id;
    schedule.next_instance_id += 1;
    schedule
        .instances
        .push(ClipInstance::new(instance_id, source_id, 0.0));
    world.insert_component(flame_entity, schedule);

    source_id
}

/// Sample every flame scalar curve of `clip` at `time` and write the values
/// into the matching `FlameEffect` fields. Curves clamp at their first/last
/// keys (same semantics as the former FlameTrack sampling).
pub fn apply_flame_scalar_curves(
    clip: &EditableAnimationClip,
    time: f32,
    effect: &mut FlameEffect,
) {
    for curve in &clip.scalar_curves {
        let Some(param) = FlameParam::from_property_type(curve.property_type) else {
            continue;
        };
        if let Some(value) = curve_sample(curve, time) {
            apply_flame_param_value(effect, param, value);
        }
    }
}

pub fn apply_flame_param_value(effect: &mut FlameEffect, param: FlameParam, value: f32) {
    match param {
        FlameParam::Height => effect.height = value,
        FlameParam::Radius => effect.radius = value,
        FlameParam::Intensity => effect.intensity = value,
        FlameParam::SigmaT => effect.sigma_t = value,
        FlameParam::TemperatureBaseK => effect.temperature_base_k = value,
        FlameParam::TemperatureTipK => effect.temperature_tip_k = value,
        FlameParam::WarpAmp => effect.warp_amp = value,
        FlameParam::WarpFreq => effect.warp_freq = value,
        FlameParam::RiseSpeed => effect.rise_speed = value,
        FlameParam::NoiseAmplitude => effect.noise_amplitude = value,
        FlameParam::WhiteBoost => effect.white_boost = value,
        FlameParam::BendAmount => effect.bend_amount = value,
        FlameParam::WindX => effect.wind_direction.x = value,
        FlameParam::WindZ => effect.wind_direction.y = value,
        FlameParam::EdgeLow => effect.edge_low = value,
        FlameParam::EdgeHigh => effect.edge_high = value,
    }
}

pub fn flame_param_value(effect: &FlameEffect, param: FlameParam) -> f32 {
    match param {
        FlameParam::Height => effect.height,
        FlameParam::Radius => effect.radius,
        FlameParam::Intensity => effect.intensity,
        FlameParam::SigmaT => effect.sigma_t,
        FlameParam::TemperatureBaseK => effect.temperature_base_k,
        FlameParam::TemperatureTipK => effect.temperature_tip_k,
        FlameParam::WarpAmp => effect.warp_amp,
        FlameParam::WarpFreq => effect.warp_freq,
        FlameParam::RiseSpeed => effect.rise_speed,
        FlameParam::NoiseAmplitude => effect.noise_amplitude,
        FlameParam::WhiteBoost => effect.white_boost,
        FlameParam::BendAmount => effect.bend_amount,
        FlameParam::WindX => effect.wind_direction.x,
        FlameParam::WindZ => effect.wind_direction.y,
        FlameParam::EdgeLow => effect.edge_low,
        FlameParam::EdgeHigh => effect.edge_high,
    }
}

/// Insert (or overwrite, when a key already sits within `1e-6` of `time`) a
/// key on the flame param's scalar curve.
pub fn flame_clip_insert_key(
    clip: &mut EditableAnimationClip,
    param: FlameParam,
    time: f32,
    value: f32,
) {
    let curve = clip.get_or_add_scalar_curve(param.property_type());
    if let Some(existing) = curve
        .keyframes
        .iter_mut()
        .find(|k| (k.time - time).abs() < 1e-6)
    {
        existing.value = value;
    } else {
        curve_add_keyframe(curve, time, value);
    }
    clip_recalculate_duration(clip);
}

/// Delete all flame scalar keys within `0.02s` of `time` (timeline row delete).
pub fn flame_clip_delete_keys_at(clip: &mut EditableAnimationClip, time: f32) {
    for curve in &mut clip.scalar_curves {
        curve.keyframes.retain(|key| (key.time - time).abs() > 0.02);
    }
    clip.remove_empty_scalar_curves();
    clip_recalculate_duration(clip);
}

pub fn flame_clip_clear_keys(clip: &mut EditableAnimationClip) {
    clip.scalar_curves.clear();
    clip_recalculate_duration(clip);
}

pub const DEBUG_KEYS_PER_CURVE: usize = 4;

/// Fill every flame param curve with `DEBUG_KEYS_PER_CURVE` evenly spaced keys
/// whose values are drawn (deterministically from `seed`) inside
/// `FlameParam::debug_value_range`, so the animated flame stays well-formed.
pub fn flame_clip_insert_debug_keys(
    clip: &mut EditableAnimationClip,
    seed: u64,
    span_seconds: f32,
) {
    let mut rng_state = seed | 1;
    let mut next_unit = move || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 40) as u32) as f32 / 16_777_216.0
    };

    let key_count = DEBUG_KEYS_PER_CURVE;
    for param in FlameParam::ALL {
        let (lo, hi) = param.debug_value_range();
        for i in 0..key_count {
            let time = span_seconds * i as f32 / (key_count - 1) as f32;
            let value = lo + next_unit() * (hi - lo);
            flame_clip_insert_key(clip, param, time, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::PropertyType;

    #[test]
    fn test_insert_overwrites_key_at_same_time() {
        let mut clip = EditableAnimationClip::new(1, FLAME_CLIP_NAME.to_string());
        flame_clip_insert_key(&mut clip, FlameParam::Height, 1.0, 2.0);
        flame_clip_insert_key(&mut clip, FlameParam::Height, 1.0, 3.0);
        let curve = clip
            .get_scalar_curve(FlameParam::Height.property_type())
            .unwrap();
        assert_eq!(curve.keyframes.len(), 1);
        assert!((curve.keyframes[0].value - 3.0).abs() < 1e-6);
        assert!((clip.duration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_scalar_curves_matches_legacy_linear() {
        let mut clip = EditableAnimationClip::new(1, FLAME_CLIP_NAME.to_string());
        flame_clip_insert_key(&mut clip, FlameParam::Height, 0.0, 1.0);
        flame_clip_insert_key(&mut clip, FlameParam::Height, 2.0, 3.0);

        let mut effect = FlameEffect::default();
        apply_flame_scalar_curves(&clip, 1.0, &mut effect);
        assert!((effect.height - 2.0).abs() < 1e-6);

        // Clamped outside the key range, other fields untouched
        apply_flame_scalar_curves(&clip, 5.0, &mut effect);
        assert!((effect.height - 3.0).abs() < 1e-6);
        assert!((effect.radius - FlameEffect::default().radius).abs() < 1e-6);
    }

    #[test]
    fn test_debug_keys_cover_all_params_within_safe_ranges() {
        let mut clip = EditableAnimationClip::new(1, FLAME_CLIP_NAME.to_string());
        flame_clip_insert_debug_keys(&mut clip, 42, 5.0);

        assert_eq!(clip.scalar_curves.len(), FlameParam::ALL.len());
        for param in FlameParam::ALL {
            let curve = clip.get_scalar_curve(param.property_type()).unwrap();
            assert_eq!(curve.keyframes.len(), DEBUG_KEYS_PER_CURVE);
            let (lo, hi) = param.debug_value_range();
            for key in &curve.keyframes {
                assert!(
                    key.value >= lo && key.value <= hi,
                    "{param:?} {}",
                    key.value
                );
                assert!(key.time >= 0.0 && key.time <= 5.0);
            }
        }
        assert!((clip.duration - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_debug_keys_are_deterministic_per_seed() {
        let mut a = EditableAnimationClip::new(1, FLAME_CLIP_NAME.to_string());
        let mut b = EditableAnimationClip::new(2, FLAME_CLIP_NAME.to_string());
        flame_clip_insert_debug_keys(&mut a, 7, 5.0);
        flame_clip_insert_debug_keys(&mut b, 7, 5.0);

        for param in FlameParam::ALL {
            let ka = &a.get_scalar_curve(param.property_type()).unwrap().keyframes;
            let kb = &b.get_scalar_curve(param.property_type()).unwrap().keyframes;
            let va: Vec<f32> = ka.iter().map(|k| k.value).collect();
            let vb: Vec<f32> = kb.iter().map(|k| k.value).collect();
            assert_eq!(va, vb);
        }
    }

    #[test]
    fn test_param_value_mirrors_apply_for_all_params() {
        let mut effect = FlameEffect::default();
        for (i, param) in FlameParam::ALL.into_iter().enumerate() {
            let value = 10.0 + i as f32;
            apply_flame_param_value(&mut effect, param, value);
            assert!(
                (flame_param_value(&effect, param) - value).abs() < 1e-6,
                "{param:?}"
            );
        }
    }

    #[test]
    fn test_delete_keys_at_removes_empty_curves() {
        let mut clip = EditableAnimationClip::new(1, FLAME_CLIP_NAME.to_string());
        flame_clip_insert_key(&mut clip, FlameParam::Height, 1.0, 2.0);
        flame_clip_insert_key(&mut clip, FlameParam::Radius, 4.0, 0.5);
        flame_clip_delete_keys_at(&mut clip, 1.0);
        assert!(clip
            .get_scalar_curve(FlameParam::Height.property_type())
            .is_none());
        assert!(clip
            .get_scalar_curve(PropertyType::Custom(FlameParam::Radius.code()))
            .is_some());
    }
}
