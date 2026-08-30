use crate::asset::AssetStorage;
use crate::ecs::component::{EntityIcon, WaterTorusEffect, WATER_DOMAIN};
use crate::ecs::resource::HierarchyState;
use crate::ecs::world::{Entity, Transform, World};

pub const DEFAULT_WATER_NAME: &str = "Water";

/// Spawns a water as a regular scene entity so the hierarchy, inspector and transform gizmo
/// can all reach it through the same components they use for every other object.
pub fn spawn_water(world: &mut World, name: &str, effect: WaterTorusEffect) -> Entity {
    let transform = Transform {
        translation: effect.position,
        rotation: effect.rotation,
        ..Default::default()
    };

    world
        .entity()
        .with_name(name)
        .with_transform(transform)
        .with_editor_display(EntityIcon::Water, false)
        .with_water(effect)
        .build()
}

/// Spawn a water entity together with its (empty) animation clip and schedule
/// instance, so every created water is animatable and shows a Timeline clip
/// lane immediately instead of waiting for the first inserted key.
pub fn spawn_water_with_clip(
    world: &mut World,
    assets: &mut AssetStorage,
    name: &str,
    effect: WaterTorusEffect,
) -> Entity {
    let entity = spawn_water(world, name, effect);
    crate::ecs::systems::scalar_clip_systems::ensure_entity_clip(
        world,
        assets,
        entity,
        &WATER_DOMAIN,
    );
    entity
}

/// The water the UI and the water events act on: the selected entity when it is a water,
/// otherwise the first one. Keeping this in one place is what lets the hierarchy selection
/// stay the single source of truth.
pub fn resolve_selected_water(world: &World) -> Option<Entity> {
    let selected = world
        .get_resource::<HierarchyState>()
        .and_then(|state| state.selected_entity);

    if let Some(entity) = selected {
        if world.get_component::<WaterTorusEffect>(entity).is_some() {
            return Some(entity);
        }
    }

    world.query_waters().first().copied()
}

/// Position and rotation live on the Transform; the effect only mirrors them for the UBO.
pub fn write_water_transform(
    world: &mut World,
    entity: Entity,
    translation: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
) {
    if let Some(transform) = world.get_component_mut::<Transform>(entity) {
        transform.translation = translation;
        transform.rotation = rotation;
    }
}
