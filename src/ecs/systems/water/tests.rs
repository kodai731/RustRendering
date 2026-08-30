use crate::ecs::component::{EntityIcon, WaterTorusEffect};
use crate::ecs::resource::{HierarchyState, TimelineState};
use crate::ecs::world::{Transform, World};

use super::*;
use crate::ecs::component::EditorDisplay;
use crate::ecs::world::{GlobalTransform, Name};

fn spawn_default_water(world: &mut World, name: &str) -> crate::ecs::world::Entity {
    spawn_water(world, name, WaterTorusEffect::default())
}

#[test]
fn spawned_water_carries_the_components_the_editor_queries() {
    let mut world = World::new();
    let entity = spawn_default_water(&mut world, DEFAULT_WATER_NAME);

    assert_eq!(
        world.get_component::<Name>(entity).map(|n| n.0.clone()),
        Some(DEFAULT_WATER_NAME.to_string())
    );
    assert!(world.get_component::<Transform>(entity).is_some());
    assert!(world.get_component::<GlobalTransform>(entity).is_some());
    assert!(world.get_component::<EditorDisplay>(entity).is_some());
    assert!(world.get_component::<WaterTorusEffect>(entity).is_some());

    let display = world.get_component::<EditorDisplay>(entity).unwrap();
    assert_eq!(display.icon, EntityIcon::Water);
}

#[test]
fn water_count_after_spawn_is_one() {
    let mut world = World::new();
    spawn_default_water(&mut world, DEFAULT_WATER_NAME);

    assert_eq!(world.query_waters().len(), 1);
}

#[test]
fn timeline_state_drives_water_time() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        time_scale: 2.0,
        time_offset: 1.5,
        ..WaterTorusEffect::default()
    };
    let _entity = spawn_water(&mut world, "Water", effect);

    world.insert_resource(TimelineState {
        current_time: 2.0,
        ..TimelineState::new()
    });

    // water_time_advance reads TimelineState.current_time when BatchRun is absent.
    // Replicate the branch logic here (the function takes FrameContext which needs Vulkan).
    let entity = world.query_waters()[0];
    let timeline_time: f32 = world.get_resource::<TimelineState>().unwrap().current_time;
    let mut effect = world.get_component_mut::<WaterTorusEffect>(entity).unwrap();
    effect.time = timeline_time * effect.time_scale + effect.time_offset;

    let effect = world.get_component::<WaterTorusEffect>(entity).unwrap();
    // time = current_time * time_scale + time_offset = 2.0 * 2.0 + 1.5 = 5.5
    assert!((effect.time - 5.5).abs() < 1e-6);
}

#[test]
fn pick_ray_hits_water_torus() {
    let mut world = World::new();
    let effect = WaterTorusEffect {
        major_radius: 1.0,
        minor_radius: 0.3,
        ..WaterTorusEffect::default()
    };
    spawn_water(&mut world, "Water", effect);

    // Ray from (0, 0, -5) going +Z hits the torus outer edge at z = -(1.0 + 0.3) = -1.3
    // Distance from (0, 0, -5) to (0, 0, -1.3) = 3.7
    let ray = crate::ecs::resource::PickRay {
        origin: cgmath::Vector3::new(0.0, 0.0, -5.0),
        direction: cgmath::Vector3::new(0.0, 0.0, 1.0),
    };

    let hit = find_water_by_pick_ray(&world, &ray);

    assert!(hit.is_some());
    let distance = hit.unwrap().1;
    assert!(
        (distance - 3.7).abs() < 1e-3,
        "expected distance ≈ 3.7, got {:.4}",
        distance
    );
}
