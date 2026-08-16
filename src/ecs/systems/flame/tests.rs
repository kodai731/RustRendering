use crate::app::FrameContext;
use crate::asset::AssetStorage;
use crate::ecs::component::{
    apply_flame_param_value, EntityIcon, FlameBaked, FlameBoneAttachment, FlameEffect, FlameParam,
    FlameTemporalAccum, FlameTrail, FLAME_DOMAIN,
};
use crate::ecs::resource::{
    BatchRun, FlameRenderSettings, FlameTemporalSnapshot, FlameTemporalState, HierarchyState,
    LightState, ProjectionData, TimelineState,
};
use crate::ecs::world::{Entity, Transform, World};
use thyllore_effect_core::{advance_flame_time, advance_flame_trail};

use super::*;
use crate::ecs::component::EditorDisplay;
use crate::ecs::world::{GlobalTransform, Name};

fn spawn_default_flame(world: &mut World, name: &str) -> Entity {
    spawn_flame(world, name, FlameEffect::default())
}

#[test]
fn spawned_flame_carries_the_components_the_editor_queries() {
    let mut world = World::new();
    let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);

    assert_eq!(
        world.get_component::<Name>(entity).map(|n| n.0.clone()),
        Some(DEFAULT_FLAME_NAME.to_string())
    );
    assert!(world.get_component::<Transform>(entity).is_some());
    assert!(world.get_component::<GlobalTransform>(entity).is_some());
    assert!(world.get_component::<EditorDisplay>(entity).is_some());
    assert!(world.get_component::<FlameEffect>(entity).is_some());
}

#[test]
fn spawned_flame_appears_as_a_hierarchy_root() {
    let mut world = World::new();
    let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);

    assert!(world.get_root_entities().contains(&entity));
}

#[test]
fn spawn_mirrors_the_effect_position_onto_the_transform() {
    let mut world = World::new();
    let effect = FlameEffect {
        position: cgmath::Vector3::new(1.5, 2.5, -3.5),
        ..FlameEffect::default()
    };
    let entity = spawn_flame(&mut world, "Flame 2", effect);

    let transform = world.get_component::<Transform>(entity).unwrap();
    assert_eq!(transform.translation, cgmath::Vector3::new(1.5, 2.5, -3.5));
}

#[test]
fn selection_falls_back_to_the_first_flame_when_nothing_is_selected() {
    let mut world = World::new();
    world.insert_resource(HierarchyState::default());
    let first = spawn_default_flame(&mut world, "Flame 1");
    spawn_default_flame(&mut world, "Flame 2");

    assert_eq!(resolve_selected_flame(&world), Some(first));
}

#[test]
fn selection_follows_the_hierarchy_when_a_flame_is_selected() {
    let mut world = World::new();
    world.insert_resource(HierarchyState::default());
    spawn_default_flame(&mut world, "Flame 1");
    let second = spawn_default_flame(&mut world, "Flame 2");

    world
        .get_resource_mut::<HierarchyState>()
        .unwrap()
        .selected_entity = Some(second);

    assert_eq!(resolve_selected_flame(&world), Some(second));
}

#[test]
fn selecting_a_non_flame_entity_keeps_editing_the_first_flame() {
    let mut world = World::new();
    world.insert_resource(HierarchyState::default());
    let first = spawn_default_flame(&mut world, "Flame 1");
    let other = world.entity().with_name("Not a flame").build();

    world
        .get_resource_mut::<HierarchyState>()
        .unwrap()
        .selected_entity = Some(other);

    assert_eq!(resolve_selected_flame(&world), Some(first));
}

#[test]
fn write_flame_transform_moves_the_transform_not_the_effect() {
    let mut world = World::new();
    let entity = spawn_default_flame(&mut world, DEFAULT_FLAME_NAME);
    let translation = cgmath::Vector3::new(4.0, 0.0, 2.0);
    let rotation = cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0);

    write_flame_transform(&mut world, entity, translation, rotation);

    let transform = world.get_component::<Transform>(entity).unwrap();
    assert_eq!(transform.translation, translation);
}

#[test]
fn resolve_returns_none_without_any_flame() {
    let mut world = World::new();
    world.insert_resource(HierarchyState::default());

    assert_eq!(resolve_selected_flame(&world), None);
}
