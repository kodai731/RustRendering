use crate::ecs::component::{CameraComponent, CameraParam, CAMERA_ACTIVE_THRESHOLD, CAMERA_DOMAIN};
use crate::ecs::resource::{ActiveCamera, ClipLibrary};
use crate::ecs::systems::scalar_clip_systems::find_entity_clip_id;
use crate::ecs::world::{Entity, Name, World};
use thyllore_anim_core::editable::EditableAnimationClip;

/// A key on a camera's `Active` channel that turns the camera on: a cut point.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraSwitchMarker {
    pub time: f32,
    pub entity: Entity,
    pub camera_name: String,
}

/// Value of the last `Active` key at or before `time` (step semantics), so a cut
/// holds until the next key regardless of the curve's interpolation mode.
pub fn stepped_active_value(clip: &EditableAnimationClip, time: f32) -> Option<f32> {
    let curve = clip
        .scalar_curves
        .iter()
        .find(|curve| curve.property_type == CameraParam::Active.property_type())?;
    curve
        .keyframes
        .iter()
        .filter(|key| key.time <= time)
        .max_by(|a, b| a.time.total_cmp(&b.time))
        .map(|key| key.value)
}

fn camera_clips(world: &World) -> Vec<(Entity, EditableAnimationClip, f32)> {
    let Some(clip_library) = world.get_resource::<ClipLibrary>() else {
        return Vec::new();
    };
    let mut cameras: Vec<Entity> = world
        .iter_components::<CameraComponent>()
        .map(|(entity, _)| entity)
        .collect();
    cameras.sort();
    cameras
        .into_iter()
        .filter_map(|entity| {
            let clip_id = find_entity_clip_id(world, entity)?;
            let clip = clip_library.get(clip_id)?.clone();
            let time = (CAMERA_DOMAIN.local_time)(world, entity)?;
            Some((entity, clip, time))
        })
        .collect()
}

/// Switch `ActiveCamera` to the camera whose `Active` channel is on at the current
/// time. Ties resolve to the lowest entity id; with no camera on, the current
/// selection is kept.
pub fn sync_active_camera_switch(world: &mut World) {
    let switched_to = camera_clips(world)
        .into_iter()
        .find(|(_, clip, time)| {
            stepped_active_value(clip, *time).is_some_and(|value| value >= CAMERA_ACTIVE_THRESHOLD)
        })
        .map(|(entity, _, _)| entity);

    let Some(entity) = switched_to else {
        return;
    };
    if let Some(mut active) = world.get_resource_mut::<ActiveCamera>() {
        if active.0 != Some(entity) {
            active.0 = Some(entity);
        }
    }
}

/// Cut points for the timeline ruler: every on-key on any camera's `Active` channel.
pub fn camera_switch_markers(world: &World) -> Vec<CameraSwitchMarker> {
    let mut markers: Vec<CameraSwitchMarker> = camera_clips(world)
        .into_iter()
        .flat_map(|(entity, clip, _)| {
            let camera_name = world
                .get_component::<Name>(entity)
                .map(|name| name.0.clone())
                .unwrap_or_else(|| format!("Camera {entity}"));
            clip.scalar_curves
                .iter()
                .filter(|curve| curve.property_type == CameraParam::Active.property_type())
                .flat_map(|curve| curve.keyframes.iter())
                .filter(|key| key.value >= CAMERA_ACTIVE_THRESHOLD)
                .map(|key| CameraSwitchMarker {
                    time: key.time,
                    entity,
                    camera_name: camera_name.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect();
    markers.sort_by(|a, b| a.time.total_cmp(&b.time).then(a.entity.cmp(&b.entity)));
    markers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::AssetStorage;
    use crate::ecs::component::ClipSchedule;
    use crate::ecs::resource::TimelineState;
    use crate::ecs::systems::scalar_clip_systems::{ensure_entity_clip, scalar_clip_insert_key};
    use crate::ecs::world::Transform;

    fn spawn_camera(
        world: &mut World,
        assets: &mut AssetStorage,
        name: &str,
        cut_times: &[f32],
    ) -> Entity {
        let entity = world.spawn();
        world.insert_component(entity, Name(name.to_string()));
        world.insert_component(entity, Transform::default());
        world.insert_component(entity, CameraComponent::default());
        world.insert_component(entity, ClipSchedule::default());
        let clip_id = ensure_entity_clip(world, assets, entity, &CAMERA_DOMAIN);
        let mut clip_library = world.resource_mut::<ClipLibrary>();
        let mut clip = clip_library.get_mut(clip_id).unwrap();
        for time in cut_times {
            scalar_clip_insert_key(&mut clip, CameraParam::Active.property_type(), *time, 1.0);
        }
        entity
    }

    fn world_with_two_cameras() -> (World, Entity, Entity) {
        let mut world = World::new();
        let mut assets = AssetStorage::default();
        world.insert_resource(ClipLibrary::default());
        world.insert_resource(TimelineState::default());
        world.insert_resource(ActiveCamera(None));
        let first = spawn_camera(&mut world, &mut assets, "CamA", &[0.0]);
        let second = spawn_camera(&mut world, &mut assets, "CamB", &[2.0]);
        let first_clip = find_entity_clip_id(&world, first).unwrap();
        {
            let mut clip_library = world.resource_mut::<ClipLibrary>();
            let mut clip = clip_library.get_mut(first_clip).unwrap();
            scalar_clip_insert_key(&mut clip, CameraParam::Active.property_type(), 2.0, 0.0);
        }
        (world, first, second)
    }

    #[test]
    fn switches_to_the_camera_whose_active_key_holds_at_the_current_time() {
        let (mut world, first, second) = world_with_two_cameras();

        world.resource_mut::<TimelineState>().current_time = 1.0;
        sync_active_camera_switch(&mut world);
        assert_eq!(world.resource::<ActiveCamera>().0, Some(first));

        world.resource_mut::<TimelineState>().current_time = 3.5;
        sync_active_camera_switch(&mut world);
        assert_eq!(world.resource::<ActiveCamera>().0, Some(second));
    }

    #[test]
    fn keeps_the_current_camera_before_any_cut() {
        let (mut world, _, second) = world_with_two_cameras();
        world.resource_mut::<ActiveCamera>().0 = Some(second);

        world.resource_mut::<TimelineState>().current_time = -1.0;
        sync_active_camera_switch(&mut world);

        assert_eq!(world.resource::<ActiveCamera>().0, Some(second));
    }

    #[test]
    fn markers_list_every_cut_in_time_order() {
        let (world, first, second) = world_with_two_cameras();

        let markers = camera_switch_markers(&world);

        let summary: Vec<(f32, Entity, &str)> = markers
            .iter()
            .map(|m| (m.time, m.entity, m.camera_name.as_str()))
            .collect();
        assert_eq!(summary, vec![(0.0, first, "CamA"), (2.0, second, "CamB")]);
    }
}
