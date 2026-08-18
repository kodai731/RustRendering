use super::resolve_selected_flame;
use crate::ecs::component::{AppliedFlameStyle, FlameBaked, FlameEffect};
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

    let Some(style) = crate::ecs::systems::apply_flame_style_from_path(&mut effect, path, groups)
    else {
        return;
    };
    thyllore_effect_core::refresh_flame_coefficients(&mut effect, &baked);

    world.insert_component(target, effect);
    world.insert_component(
        target,
        AppliedFlameStyle {
            name: style.name,
            version: style.version,
        },
    );
}

/// Save the selected flame's current look as a named style file under the
/// styles asset directory, returning the written path.
pub fn save_flame_style_of_selected(world: &World, name: &str) -> Option<String> {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        return None;
    }

    let target = resolve_selected_flame(world)?;
    let effect = world.get_component::<FlameEffect>(target)?;
    let path = format!("{}/{}.style.ron", crate::paths::FLAMES_STYLE_DIR, sanitized);
    crate::ecs::systems::dump_flame_style_to_path(effect, &path);
    Some(path)
}
