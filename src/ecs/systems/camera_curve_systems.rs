use crate::ecs::component::{
    apply_camera_param_value, CameraComponent, CameraParam, CAMERA_DOMAIN,
};
use crate::ecs::resource::ClipLibrary;
use crate::ecs::world::{Entity, Transform, World};

/// Sync camera curve values from clip scalar curves to Transform/CameraComponent.
/// Collects entities, clips, and times first to avoid borrow conflicts, then applies.
pub fn sync_camera_curves(world: &mut World) {
    let camera_entities: Vec<Entity> = world
        .iter_components::<CameraComponent>()
        .map(|(e, _)| e)
        .collect();

    if camera_entities.is_empty() {
        return;
    }

    // Collect clips and times up front to avoid borrow conflicts
    let clips: Vec<(
        Entity,
        crate::animation::editable::EditableAnimationClip,
        f32,
    )> = {
        let clip_library = match world.get_resource::<ClipLibrary>() {
            Some(cl) => cl,
            None => return,
        };
        camera_entities
            .iter()
            .filter_map(|&e| {
                let clip_id =
                    crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(world, e)?;
                let clip = clip_library.get(clip_id)?.clone();
                let time = (CAMERA_DOMAIN.local_time)(world, e)?;
                Some((e, clip, time))
            })
            .collect()
    };

    // Apply sampled values to each camera entity
    for (entity, clip, time) in clips {
        let values: Vec<(thyllore_anim_core::editable::PropertyType, f32)> =
            crate::ecs::systems::scalar_clip_systems::sampled_scalar_values(&clip, time);

        // Clone components to avoid double-mut-borrow, apply values to clones, then write back
        let mut transform = match world.get_component::<Transform>(entity) {
            Some(t) => t.clone(),
            None => continue,
        };
        let mut camera = match world.get_component::<CameraComponent>(entity) {
            Some(c) => c.clone(),
            None => continue,
        };

        for (property_type, value) in values {
            if let Some(param) = CameraParam::from_property_type(property_type) {
                apply_camera_param_value(&mut transform, &mut camera, param, value);
            }
        }

        // Write back the updated components (sequential borrows, no conflict)
        if let Some(t) = world.get_component_mut::<Transform>(entity) {
            *t = transform;
        }
        if let Some(c) = world.get_component_mut::<CameraComponent>(entity) {
            *c = camera;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::AssetStorage;
    use crate::ecs::resource::TimelineState;

    #[test]
    fn test_sync_camera_curves_translation_x() {
        let mut world = World::new();
        let mut assets = AssetStorage::default();
        world.insert_resource(ClipLibrary::default());
        world.insert_resource(TimelineState {
            current_time: 0.5,
            ..Default::default()
        });
        let entity = world.spawn();
        world.insert_component(entity, Transform::default());
        world.insert_component(
            entity,
            CameraComponent {
                fov_y: cgmath::Deg(45.0),
                near_plane: 0.1,
                far_plane: None,
                physical: Default::default(),
            },
        );
        world.insert_component(entity, crate::ecs::component::ClipSchedule::default());

        let clip_id = crate::ecs::systems::scalar_clip_systems::ensure_entity_clip(
            &mut world,
            &mut assets,
            entity,
            &CAMERA_DOMAIN,
        );

        {
            let mut clip_library = world.resource_mut::<ClipLibrary>();
            let mut clip = clip_library.get_mut(clip_id).unwrap();
            crate::ecs::systems::scalar_clip_systems::scalar_clip_insert_key(
                &mut clip,
                CameraParam::TranslationX.property_type(),
                0.0,
                0.0,
            );
            crate::ecs::systems::scalar_clip_systems::scalar_clip_insert_key(
                &mut clip,
                CameraParam::TranslationX.property_type(),
                1.0,
                10.0,
            );
        }

        sync_camera_curves(&mut world);

        let transform = world.get_component::<Transform>(entity).unwrap();
        let x = transform.translation.x;
        // At t=0.5 with linear interpolation between (0,0) and (1,10), expect 5.0
        assert!(
            (x - 5.0).abs() < 1e-3,
            "expected translation.x ~= 5.0, got {}",
            x
        );
    }
}
