use super::{resolve_selected_flame, write_flame_transform};
use crate::ecs::component::FlameEffect;
use crate::ecs::world::World;

/// Apply a named preset to the selected flame's parameter component. The
/// preset table lives in render-core; this system is the only mutation path
/// so UI and batch actions share one behavior.
pub fn apply_flame_preset_to_selected(world: &mut World, name: &str) {
    let Some(target) = resolve_selected_flame(world) else {
        return;
    };
    let Some(mut effect) = world
        .get_component::<FlameEffect>(target)
        .map(|e| e.clone())
    else {
        return;
    };
    if thyllore_effect_core::apply_flame_preset(&mut effect, name) {
        write_flame_transform(world, target, effect.position, effect.rotation);
        world.insert_component(target, effect);
    }
}
