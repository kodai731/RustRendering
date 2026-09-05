use super::resolve_selected_wind;
use crate::ecs::component::{AppliedWindPreset, WindTornadoEffect};
use crate::ecs::world::World;

pub fn apply_wind_preset_to_selected(world: &mut World, name: &str) {
    let Some(target) = resolve_selected_wind(world) else {
        return;
    };
    let Some(mut effect) = world
        .get_component::<WindTornadoEffect>(target)
        .map(|e| e.clone())
    else {
        return;
    };
    if thyllore_effect_core::apply_wind_preset(&mut effect, name) {
        world.insert_component(target, effect);
        world.insert_component(
            target,
            AppliedWindPreset {
                name: name.to_string(),
            },
        );
    }
}
