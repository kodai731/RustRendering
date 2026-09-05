use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    thyllore_animation::app::cleanup_old_screenshots()?;

    let config = match thyllore_animation::app::config::from_args(env::args().collect())? {
        Some(c) => c,
        None => return Ok(()),
    };

    let window_title = format!("Thyllore Animation v{}", env!("CARGO_PKG_VERSION"));
    let mut system = thyllore_animation::platform::init(&window_title, !config.is_batch_mode);

    #[cfg(feature = "ml")]
    let curve_copilot_mode =
        thyllore_animation::ecs::systems::curve_copilot_mode_resolve_from_env_args()?;

    #[cfg(feature = "ml")]
    let mut app =
        unsafe { thyllore_animation::app::App::create(&system.window, curve_copilot_mode)? };
    #[cfg(not(feature = "ml"))]
    let mut app = unsafe { thyllore_animation::app::App::create(&system.window)? };

    thyllore_animation::app::bootstrap::apply_engine_overrides(&mut app, &config.overrides);

    unsafe {
        thyllore_animation::app::bootstrap::init_flame_sdf_texture(&mut app, &config.overrides)?;
        thyllore_animation::app::bootstrap::init_imgui_rendering(&mut app, &mut system)?;
    }

    system.main_loop(&mut app);

    thyllore_animation::app::bootstrap::finish_run(&app, &config.overrides, config.is_batch_mode);

    Ok(())
}
