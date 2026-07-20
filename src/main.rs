#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use thyllore_animation::app::init::instance::cleanup_old_screenshots;
use thyllore_animation::app::App;
use thyllore_animation::ecs::resource::BatchRun;
use thyllore_animation::ecs::systems::{batch_run_report, batch_run_resolve_from_args};
use thyllore_animation::platform;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();

    cleanup_old_screenshots()?;

    let args: Vec<String> = std::env::args().collect();
    let batch_run = match batch_run_resolve_from_args(&args) {
        Ok(batch_run) => batch_run,
        Err(e) => {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": e.to_string()})
            );
            std::process::exit(1);
        }
    };
    let is_batch_mode = batch_run.is_some();

    #[cfg(feature = "ml")]
    let curve_copilot_mode =
        thyllore_animation::ecs::systems::curve_copilot_mode_resolve_from_env_args()?;

    let window_title = format!("Thyllore Animation v{}", env!("CARGO_PKG_VERSION"));
    let mut system = platform::init(&window_title);

    #[cfg(feature = "ml")]
    let mut app = unsafe { App::create(&system.window, curve_copilot_mode)? };
    #[cfg(not(feature = "ml"))]
    let mut app = unsafe { App::create(&system.window)? };

    if let Some(batch_run) = batch_run {
        app.data.ecs_world.insert_resource(batch_run);
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
