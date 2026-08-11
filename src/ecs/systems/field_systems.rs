use crate::app::FrameContext;
use crate::ecs::component::{FieldAffected, FlameEffect};
use crate::ecs::world::World;
use thyllore_effect_core::{flame_field_manifest, FieldManifest};

/// Single pass for all field compositions: derive, attach, log changes, warn on pending sources.
pub fn field_manifest_sync(ctx: &mut FrameContext) {
    sync_world(ctx.world);
}

pub(crate) fn sync_world(world: &mut World) {
    for entity in world.query_flames() {
        let Some(effect) = world.get_component::<FlameEffect>(entity) else {
            continue;
        };
        let manifest = flame_field_manifest(effect);

        let previous = world
            .get_component::<FieldAffected>(entity)
            .map(|f| f.manifest.summary());
        let summary = manifest.summary();
        if previous.as_deref() == Some(summary.as_str()) {
            continue;
        }

        report_composition(entity, &manifest, previous.is_some());
        world.insert_component(entity, FieldAffected { manifest });
    }
}

fn report_composition(entity_id: u64, manifest: &FieldManifest, is_change: bool) {
    let verb = if is_change { "changed" } else { "declared" };
    log!(
        "[field] entity {entity_id} composition {verb}: {}",
        manifest.summary()
    );
    let pending = manifest.active_unification_pending();
    if !pending.is_empty() {
        let names: Vec<&str> = pending.iter().map(|s| s.as_str()).collect();
        log_warn!(
            "[field] entity {entity_id} uses unification-pending sources: {} \
             (fringe risk — see 20260809_unified_field_redesign.md)",
            names.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::systems::spawn_flame;
    use thyllore_effect_core::FieldSourceKind;

    #[test]
    fn sync_attaches_the_manifest_to_every_flame() {
        let mut world = World::new();
        let mut effect = FlameEffect::default();
        effect.noise_amplitude = 1.5;
        effect.boundary_amp = 0.2;
        let entity = spawn_flame(&mut world, "Flame", effect);

        sync_world(&mut world);

        let field = world
            .get_component::<FieldAffected>(entity)
            .expect("flame gets the FieldAffected attribute");
        let sources = field.manifest.active_sources();
        assert!(sources.contains(&FieldSourceKind::ErosionWaveTable));
    }

    #[test]
    fn sync_tracks_lever_changes() {
        let mut world = World::new();
        let mut effect = FlameEffect::default();
        effect.noise_amplitude = 1.5;
        effect.boundary_amp = 0.2;
        let entity = spawn_flame(&mut world, "Flame", effect);
        sync_world(&mut world);

        world
            .get_component_mut::<FlameEffect>(entity)
            .unwrap()
            .noise_amplitude = 0.0;
        sync_world(&mut world);

        let field = world.get_component::<FieldAffected>(entity).unwrap();
        assert!(
            !field
                .manifest
                .active_sources()
                .contains(&FieldSourceKind::ErosionWaveTable),
            "manifest follows the lever"
        );
    }
}
