use super::{resolve_selected_flame, write_flame_transform};
use crate::ecs::component::{FlameBaked, FlameEffect};
use crate::ecs::world::World;
use thyllore_render_core::TextureFitGroups;

/// Run a texture fit against the selected flame and write the results into
/// its parameter and baked components. The heavy fit itself stays in
/// render-core / texture-fit-core; this system owns the component I/O.
pub fn apply_flame_texture_fit_to_selected(
    world: &mut World,
    path: &str,
    blend: f32,
    groups: TextureFitGroups,
    profile: bool,
    route: &str,
) {
    let Some(target) = resolve_selected_flame(world) else {
        return;
    };
    let Some(mut effect) = world
        .get_component::<FlameEffect>(target)
        .map(|e| e.clone())
    else {
        return;
    };
    let mut baked = world
        .get_component::<FlameBaked>(target)
        .cloned()
        .unwrap_or_default();

    crate::ecs::systems::apply_texture_fit_from_path(
        &mut effect,
        &mut baked,
        path,
        blend,
        groups,
        profile,
        route,
    );

    write_flame_transform(world, target, effect.position, effect.rotation);
    world.insert_component(target, effect);
    world.insert_component(target, baked);
}
