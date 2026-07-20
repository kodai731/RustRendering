#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use thyllore_animation::app::init::instance::cleanup_old_screenshots;
use thyllore_animation::app::App;
use thyllore_animation::ecs::resource::{BatchRun, FlameRenderSettings};
use thyllore_animation::ecs::systems::{batch_run_report, resolve_engine_cli_overrides};
use thyllore_animation::platform;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();

    cleanup_old_screenshots()?;

    let args: Vec<String> = std::env::args().collect();
    let overrides = match resolve_engine_cli_overrides(&args) {
        Ok(overrides) => overrides,
        Err(e) => {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": e.to_string()})
            );
            std::process::exit(1);
        }
    };
    let is_batch_mode = overrides.batch_run.is_some();

    #[cfg(feature = "ml")]
    let curve_copilot_mode =
        thyllore_animation::ecs::systems::curve_copilot_mode_resolve_from_env_args()?;

    let window_title = format!("Thyllore Animation v{}", env!("CARGO_PKG_VERSION"));
    let mut system = platform::init(&window_title);

    #[cfg(feature = "ml")]
    let mut app = unsafe { App::create(&system.window, curve_copilot_mode)? };
    #[cfg(not(feature = "ml"))]
    let mut app = unsafe { App::create(&system.window)? };

    if let Some(batch_run) = overrides.batch_run {
        app.data.ecs_world.insert_resource(batch_run);
    }
    if let Some(shading_mode) = overrides.flame_mode {
        app.data
            .ecs_world
            .resource_mut::<FlameRenderSettings>()
            .shading_mode = shading_mode;
    }
    if let Some(step_count) = overrides.flame_steps {
        app.data
            .ecs_world
            .resource_mut::<FlameRenderSettings>()
            .reference_step_count = step_count;
    }

    unsafe {
        use thyllore_animation::vulkanr::context::{CommandState, RenderTargets};
        let command_pool = app.resource::<CommandState>().pool.clone();
        let rrrender = app.resource::<RenderTargets>().render.clone();
        App::init_imgui_rendering(
            &app.instance,
            &app.rrdevice,
            &mut app.data,
            &mut system.imgui,
            &command_pool,
            &rrrender,
        )?;
    }

    system.main_loop(&mut app);

    if is_batch_mode {
        let batch = app.data.ecs_world.resource::<BatchRun>();
        let (ok, report_line) = batch_run_report(&batch);
        drop(batch);
        println!("{report_line}");
        if !ok {
            std::process::exit(1);
        }
    }

    Ok(())
}
