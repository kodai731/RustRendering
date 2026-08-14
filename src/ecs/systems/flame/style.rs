use super::resolve_selected_flame;
use crate::ecs::component::{FlameBaked, FlameEffect};
use crate::ecs::world::World;
use thyllore_effect_core::StyleGroups;

/// Load a FlameStyle file and apply it to the selected flame's parameter
/// component. The parsing and the pure apply live in effect-core / the shared
/// batch helper; this system owns the component I/O so UI and batch share one
/// behavior.
pub fn apply_flame_style_to_selected(world: &mut World, path: &str, groups: StyleGroups) {
    let Some(target) = resolve_selected_flame(world) else {
        return;
    };
    let Some(mut effect) = world
        .get_component::<FlameEffect>(target)
        .map(|e| e.clone())
    else {
        return;
    };
    let baked = world
        .get_component::<FlameBaked>(target)
        .cloned()
        .unwrap_or_default();

    crate::ecs::systems::apply_flame_style_from_path(&mut effect, path, groups);
    thyllore_effect_core::refresh_flame_coefficients(&mut effect, &baked);

    world.insert_component(target, effect);
}
