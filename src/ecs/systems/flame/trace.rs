use crate::ecs::component::{FlameBaked, FlameEffect, FlameTemporalAccum};
use crate::ecs::resource::{Camera, FlameRenderSettings};
use crate::ecs::systems::flame_dump_systems::{
    write_flame_field_traces, write_flame_wall_probe_dump,
};
use crate::ecs::World;
use thyllore_effect_core::{probe_flame_wall, WallProbeView};
use thyllore_log_core::{log, log_warn};

/// Wall-probe + field-trace dump over every flame entity's components. The
/// sampling mirrors live in thyllore-render-debug; this system owns the
/// component reads and is the shared entry for the interactive UIEvent and
/// the batch run path.
pub fn perform_flame_wall_probe_dump(world: &World, viewport_size: [f32; 2]) {
    use crate::ecs::systems::camera_systems::{
        compute_camera_direction, compute_camera_position, compute_camera_right, compute_camera_up,
    };

    let camera = (*world.resource::<Camera>()).clone();
    let settings = world
        .get_resource::<FlameRenderSettings>()
        .map(|s| *s)
        .unwrap_or_default();
    let view = WallProbeView {
        position: compute_camera_position(&camera).into(),
        forward: compute_camera_direction(&camera).into(),
        right: compute_camera_right(&camera).into(),
        up: compute_camera_up(&camera).into(),
        fov_y_radians: camera.fov_y.0.to_radians(),
        viewport_size_px: viewport_size,
    };

    let flames: Vec<_> = world
        .query_flames()
        .into_iter()
        .filter_map(|entity| {
            let effect = world.get_component::<FlameEffect>(entity)?;
            let baked = world
                .get_component::<FlameBaked>(entity)
                .cloned()
                .unwrap_or_default();
            let temporal = world
                .get_component::<FlameTemporalAccum>(entity)
                .cloned()
                .unwrap_or_default();
            let report = probe_flame_wall(&effect, &baked, &view);
            Some((effect.clone(), baked, temporal, report))
        })
        .collect();
    if flames.is_empty() {
        log_warn!("wall probe dump skipped: no flame entity");
        return;
    }

    match write_flame_wall_probe_dump(&camera, &settings, viewport_size, &flames) {
        Ok(path) => log!("wall probe dumped to {}", path.display()),
        Err(error) => log_warn!("wall probe dump failed: {}", error),
    }

    match write_flame_field_traces(&view, &flames) {
        Ok(paths) => {
            for path in paths {
                log!("flame field trace dumped to {}", path.display());
            }
        }
        Err(error) => log_warn!("flame field trace dump failed: {}", error),
    }
}
