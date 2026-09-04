use super::{resolve_selected_water, write_water_transform};
use crate::ecs::component::{AppliedWaterPreset, WaterTorusEffect};
use crate::ecs::world::World;

/// Apply a named preset to the selected water's parameter component. The
/// preset table lives in render-core; this system is the only mutation path
/// so UI and batch actions share one behavior.
pub fn apply_water_preset_to_selected(world: &mut World, name: &str) {
    let Some(target) = resolve_selected_water(world) else {
        return;
    };
    let Some(mut effect) = world
        .get_component::<WaterTorusEffect>(target)
        .map(|e| e.clone())
    else {
        return;
    };
    if thyllore_effect_core::apply_water_preset(&mut effect, name) {
        write_water_transform(world, target, effect.position, effect.rotation);
        world.insert_component(target, effect);
        world.insert_component(
            target,
            AppliedWaterPreset {
                name: name.to_string(),
            },
        );
    }
}
