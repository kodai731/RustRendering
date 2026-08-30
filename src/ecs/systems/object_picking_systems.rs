use super::flame::find_flame_by_pick_ray;
use super::water::find_water_by_pick_ray;
use crate::asset::AssetStorage;
use crate::ecs::resource::CurveEditorState;
use crate::ecs::resource::{
    ClipLibrary, HierarchyDisplayMode, ObjectIdReadback, PickRay, TimelineState,
};
use crate::ecs::systems::clip_track_systems::resolve_mesh_bone_id;
use crate::ecs::systems::hierarchy_systems::{
    hierarchy_deselect_all, hierarchy_select, hierarchy_toggle_selection,
};
use crate::ecs::world::{Entity, MeshRef, World};

pub fn find_entity_by_object_id(
    world: &World,
    assets: &AssetStorage,
    object_id: u32,
) -> Option<Entity> {
    if object_id == 0 {
        return None;
    }

    let mesh_index = (object_id - 1) as usize;
    let mesh_asset = assets.find_mesh_by_graphics_index(mesh_index)?;
    let target_asset_id = mesh_asset.id;

    world
        .iter_components::<MeshRef>()
        .find(|(_, mesh_ref)| mesh_ref.mesh_asset_id == target_asset_id)
        .map(|(entity, _)| entity)
}

pub fn apply_mesh_selection(
    world: &mut World,
    assets: &AssetStorage,
    readback: &mut ObjectIdReadback,
) {
    let Some(object_id) = readback.last_read_object_id.take() else {
        return;
    };

    let is_shift = readback.is_shift;
    let is_ctrl = readback.is_ctrl;

    let surface_entity = find_entity_by_object_id(world, assets, object_id);
    let picked = resolve_closest_pick(
        world,
        surface_entity,
        readback.pick_ray.as_ref(),
        readback.last_read_world_position,
    );
    log!(
        "pick: object_id={} surface={:?} world_position={:?} selected={:?}",
        object_id,
        surface_entity,
        readback.last_read_world_position,
        picked
    );

    let Some(entity) = picked else {
        if object_id == 0 && !is_shift && !is_ctrl {
            let mut state = world.resource_mut::<crate::ecs::resource::HierarchyState>();
            hierarchy_deselect_all(&mut state);
            state.display_mode = HierarchyDisplayMode::Entities;
        }
        return;
    };

    let mut state = world.resource_mut::<crate::ecs::resource::HierarchyState>();
    if is_shift || is_ctrl {
        hierarchy_toggle_selection(&mut state, entity);
    } else {
        hierarchy_select(&mut state, entity);
    }
    state.display_mode = HierarchyDisplayMode::Entities;
    drop(state);

    sync_curve_editor_on_mesh_select(world, assets, entity);
}

/// Whichever of the two candidates the click actually landed on: the surface reported by the
/// object-id buffer, or a flame/water in front of it.
fn resolve_closest_pick(
    world: &World,
    surface_entity: Option<Entity>,
    ray: Option<&PickRay>,
    surface_world_position: Option<[f32; 3]>,
) -> Option<Entity> {
    let Some(ray) = ray else {
        return surface_entity;
    };

    // Collect flame and water candidates, pick the closest effect
    let flame_candidate = find_flame_by_pick_ray(world, ray);
    let water_candidate = find_water_by_pick_ray(world, ray);

    let effect_candidate: Option<(Entity, f32)> = match (flame_candidate, water_candidate) {
        (Some(f), Some(w)) => {
            if f.1 <= w.1 {
                Some(f)
            } else {
                Some(w)
            }
        }
        (Some(f), None) => Some(f),
        (None, Some(w)) => Some(w),
        (None, None) => None,
    };

    let Some((effect_entity, effect_distance)) = effect_candidate else {
        return surface_entity;
    };

    if surface_entity.is_none() {
        return Some(effect_entity);
    }

    let surface_distance = surface_world_position
        .map(|p| cgmath::Vector3::new(p[0], p[1], p[2]) - ray.origin)
        .map(|to_surface| cgmath::dot(to_surface, ray.direction))
        .unwrap_or(f32::INFINITY);

    if effect_distance < surface_distance {
        Some(effect_entity)
    } else {
        surface_entity
    }
}

fn sync_curve_editor_on_mesh_select(world: &World, assets: &AssetStorage, entity: Entity) {
    let is_open = world
        .get_resource::<CurveEditorState>()
        .map(|s| s.is_open)
        .unwrap_or(false);
    if !is_open {
        return;
    }

    let clip_library = world.resource::<ClipLibrary>();
    let source_id = world.resource::<TimelineState>().current_clip_id;
    let bone_id = resolve_mesh_bone_id(world, entity, assets, &clip_library, source_id);
    drop(clip_library);

    if let Some(bone_id) = bone_id {
        let mut editor = world.resource_mut::<CurveEditorState>();
        editor.select_bone(bone_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::component::FlameEffect;
    use crate::ecs::systems::spawn_flame;
    use cgmath::Vector3;

    const RAY_START_Z: f32 = -10.0;

    fn world_with_flame_at(x: f32) -> (World, Entity) {
        let mut world = World::new();
        let effect = FlameEffect {
            position: Vector3::new(x, 0.0, 0.0),
            ..FlameEffect::default()
        };
        let entity = spawn_flame(&mut world, "Flame", effect);
        (world, entity)
    }

    fn ray_towards_origin() -> PickRay {
        PickRay {
            origin: Vector3::new(0.0, 0.5, RAY_START_Z),
            direction: Vector3::new(0.0, 0.0, 1.0),
        }
    }

    #[test]
    fn a_ray_through_the_flame_finds_it() {
        let (world, flame) = world_with_flame_at(0.0);

        let hit = find_flame_by_pick_ray(&world, &ray_towards_origin());

        assert_eq!(hit.map(|(entity, _)| entity), Some(flame));
    }

    #[test]
    fn a_ray_beside_the_flame_finds_nothing() {
        let (world, _) = world_with_flame_at(50.0);

        assert!(find_flame_by_pick_ray(&world, &ray_towards_origin()).is_none());
    }

    #[test]
    fn the_nearest_flame_wins_when_two_overlap() {
        let (mut world, far_flame) = world_with_flame_at(0.0);
        let near_effect = FlameEffect {
            position: Vector3::new(0.0, 0.0, -5.0),
            ..FlameEffect::default()
        };
        let near_flame = spawn_flame(&mut world, "Flame 2", near_effect);

        let hit = find_flame_by_pick_ray(&world, &ray_towards_origin());

        assert_eq!(hit.map(|(entity, _)| entity), Some(near_flame));
        assert_ne!(hit.map(|(entity, _)| entity), Some(far_flame));
    }

    #[test]
    fn a_surface_in_front_of_the_flame_wins() {
        let (mut world, _) = world_with_flame_at(0.0);
        let ray = ray_towards_origin();
        let surface = world.entity().with_name("mesh").build();
        let in_front_of_the_flame = [0.0, 0.5, -5.0];

        let picked = resolve_closest_pick(
            &world,
            Some(surface),
            Some(&ray),
            Some(in_front_of_the_flame),
        );

        assert_eq!(picked, Some(surface));
    }

    #[test]
    fn a_flame_in_front_of_the_surface_wins() {
        let (mut world, flame) = world_with_flame_at(0.0);
        let ray = ray_towards_origin();
        let surface = world.entity().with_name("mesh").build();
        let behind_the_flame = [0.0, 0.5, 5.0];

        let picked =
            resolve_closest_pick(&world, Some(surface), Some(&ray), Some(behind_the_flame));

        assert_eq!(picked, Some(flame));
    }

    #[test]
    fn a_flame_over_the_background_is_picked() {
        let (world, flame) = world_with_flame_at(0.0);

        let picked = resolve_closest_pick(&world, None, Some(&ray_towards_origin()), None);

        assert_eq!(picked, Some(flame));
    }

    #[test]
    fn without_a_ray_the_surface_decides() {
        let (mut world, _) = world_with_flame_at(0.0);
        let surface = world.entity().with_name("mesh").build();

        assert_eq!(
            resolve_closest_pick(&world, Some(surface), None, None),
            Some(surface)
        );
    }

    #[test]
    fn clicking_empty_space_selects_nothing() {
        let (world, _) = world_with_flame_at(50.0);

        assert_eq!(
            resolve_closest_pick(&world, None, Some(&ray_towards_origin()), None),
            None
        );
    }
}
