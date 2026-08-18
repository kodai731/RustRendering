use thyllore_anim_core::editable::PropertyType;

use crate::animation::editable::{
    clip_recalculate_duration, curve_add_keyframe, curve_sample, ClipInstance,
    EditableAnimationClip, SourceClipId,
};
use crate::asset::AssetStorage;
use crate::ecs::component::{
    scalar_channel_domains, scalar_domain_for_entity, ClipSchedule, ScalarChannelDomain,
};
use crate::ecs::resource::{ClipLibrary, HierarchyState};
use crate::ecs::world::{Entity, World};

pub fn find_entity_clip_id(world: &World, entity: Entity) -> Option<SourceClipId> {
    world
        .get_component::<ClipSchedule>(entity)
        .and_then(|schedule| schedule.first_instance().map(|i| i.source_id))
}

/// Returns the entity's clip, creating the clip (named after the domain) and a
/// schedule instance (start 0, speed 1, so clip-local time equals timeline
/// time) on first use.
pub fn ensure_entity_clip(
    world: &mut World,
    assets: &mut AssetStorage,
    entity: Entity,
    domain: &ScalarChannelDomain,
) -> SourceClipId {
    if let Some(id) = find_entity_clip_id(world, entity) {
        return id;
    }

    let source_id = {
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let editable = EditableAnimationClip::new(0, domain.name.to_string());
        super::clip_library_systems::clip_library_register_and_activate(
            &mut clip_library,
            assets,
            editable,
        )
    };

    let mut schedule = world
        .get_component::<ClipSchedule>(entity)
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
    world.insert_component(entity, schedule);

    source_id
}

/// The entity the scalar-curve events act on: the selected entity when it
/// belongs to a scalar channel domain, otherwise the first domain entity.
/// Keeping this in one place is what lets the hierarchy selection stay the
/// single source of truth.
pub fn resolve_selected_scalar_entity(
    world: &World,
) -> Option<(Entity, &'static ScalarChannelDomain)> {
    let selected = world
        .get_resource::<HierarchyState>()
        .and_then(|state| state.selected_entity);

    if let Some(entity) = selected {
        if let Some(domain) = scalar_domain_for_entity(world, entity) {
            return Some((entity, domain));
        }
    }

    scalar_channel_domains().iter().find_map(|domain| {
        (domain.entities)(world)
            .first()
            .copied()
            .map(|entity| (entity, *domain))
    })
}

/// Sample every scalar curve of `clip` at `time`. Curves clamp at their
/// first/last keys. The owning domain's system applies the values back to its
/// component fields.
pub fn sampled_scalar_values(clip: &EditableAnimationClip, time: f32) -> Vec<(PropertyType, f32)> {
    clip.scalar_curves
        .iter()
        .filter_map(|curve| curve_sample(curve, time).map(|value| (curve.property_type, value)))
        .collect()
}

/// Insert (or overwrite, when a key already sits within `1e-6` of `time`) a
/// key on the property's scalar curve.
pub fn scalar_clip_insert_key(
    clip: &mut EditableAnimationClip,
    property_type: PropertyType,
    time: f32,
    value: f32,
) {
    let curve = clip.get_or_add_scalar_curve(property_type);
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

/// Delete all scalar keys within `0.02s` of `time` (timeline row delete).
pub fn scalar_clip_delete_keys_at(clip: &mut EditableAnimationClip, time: f32) {
    for curve in &mut clip.scalar_curves {
        curve.keyframes.retain(|key| (key.time - time).abs() > 0.02);
    }
    clip.remove_empty_scalar_curves();
    clip_recalculate_duration(clip);
}

pub fn scalar_clip_clear_keys(clip: &mut EditableAnimationClip) {
    clip.scalar_curves.clear();
    clip_recalculate_duration(clip);
}

pub const DEBUG_KEYS_PER_CURVE: usize = 4;

/// Fill every channel curve of `domain` with `DEBUG_KEYS_PER_CURVE` evenly
/// spaced keys whose values are drawn (deterministically from `seed`) inside
/// the channel's `debug_value_range`, so the animated component stays
/// well-formed.
pub fn scalar_clip_insert_debug_keys(
    clip: &mut EditableAnimationClip,
    domain: &ScalarChannelDomain,
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
    for channel in domain.channels {
        let (lo, hi) = channel.debug_value_range;
        for i in 0..key_count {
            let time = span_seconds * i as f32 / (key_count - 1) as f32;
            let value = lo + next_unit() * (hi - lo);
            scalar_clip_insert_key(clip, channel.property_type(), time, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::{FlameParam, FLAME_DOMAIN};

    #[test]
    fn test_insert_overwrites_key_at_same_time() {
        let mut clip = EditableAnimationClip::new(1, FLAME_DOMAIN.name.to_string());
        let height = FlameParam::Height.property_type();
        scalar_clip_insert_key(&mut clip, height, 1.0, 2.0);
        scalar_clip_insert_key(&mut clip, height, 1.0, 3.0);
        let curve = clip.get_scalar_curve(height).unwrap();
        assert_eq!(curve.keyframes.len(), 1);
        assert!((curve.keyframes[0].value - 3.0).abs() < 1e-6);
        assert!((clip.duration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampled_values_clamp_at_curve_ends() {
        let mut clip = EditableAnimationClip::new(1, FLAME_DOMAIN.name.to_string());
        let height = FlameParam::Height.property_type();
        scalar_clip_insert_key(&mut clip, height, 0.0, 1.0);
        scalar_clip_insert_key(&mut clip, height, 2.0, 3.0);

        let mid = sampled_scalar_values(&clip, 1.0);
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].0, height);
        assert!((mid[0].1 - 2.0).abs() < 1e-6);

        let clamped = sampled_scalar_values(&clip, 5.0);
        assert!((clamped[0].1 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_debug_keys_cover_all_channels_within_safe_ranges() {
        let mut clip = EditableAnimationClip::new(1, FLAME_DOMAIN.name.to_string());
        scalar_clip_insert_debug_keys(&mut clip, &FLAME_DOMAIN, 42, 5.0);

        assert_eq!(clip.scalar_curves.len(), FLAME_DOMAIN.channels.len());
        for channel in FLAME_DOMAIN.channels {
            let curve = clip.get_scalar_curve(channel.property_type()).unwrap();
            assert_eq!(curve.keyframes.len(), DEBUG_KEYS_PER_CURVE);
            let (lo, hi) = channel.debug_value_range;
            for key in &curve.keyframes {
                assert!(
                    key.value >= lo && key.value <= hi,
                    "{} {}",
                    channel.cli_name,
                    key.value
                );
                assert!(key.time >= 0.0 && key.time <= 5.0);
            }
        }
        assert!((clip.duration - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_debug_keys_are_deterministic_per_seed() {
        let mut a = EditableAnimationClip::new(1, FLAME_DOMAIN.name.to_string());
        let mut b = EditableAnimationClip::new(2, FLAME_DOMAIN.name.to_string());
        scalar_clip_insert_debug_keys(&mut a, &FLAME_DOMAIN, 7, 5.0);
        scalar_clip_insert_debug_keys(&mut b, &FLAME_DOMAIN, 7, 5.0);

        for channel in FLAME_DOMAIN.channels {
            let ka = &a
                .get_scalar_curve(channel.property_type())
                .unwrap()
                .keyframes;
            let kb = &b
                .get_scalar_curve(channel.property_type())
                .unwrap()
                .keyframes;
            let va: Vec<f32> = ka.iter().map(|k| k.value).collect();
            let vb: Vec<f32> = kb.iter().map(|k| k.value).collect();
            assert_eq!(va, vb);
        }
    }

    #[test]
    fn test_delete_keys_at_removes_empty_curves() {
        let mut clip = EditableAnimationClip::new(1, FLAME_DOMAIN.name.to_string());
        scalar_clip_insert_key(&mut clip, FlameParam::Height.property_type(), 1.0, 2.0);
        scalar_clip_insert_key(&mut clip, FlameParam::Radius.property_type(), 4.0, 0.5);
        scalar_clip_delete_keys_at(&mut clip, 1.0);
        assert!(clip
            .get_scalar_curve(FlameParam::Height.property_type())
            .is_none());
        assert!(clip
            .get_scalar_curve(FlameParam::Radius.property_type())
            .is_some());
    }
}
