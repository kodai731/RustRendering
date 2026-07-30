#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use thyllore_animation::app::init::instance::cleanup_old_screenshots;
use thyllore_animation::app::App;
use thyllore_animation::ecs::component::{FlameEffect, FlameTrail};
use thyllore_animation::ecs::resource::{
    BatchFlameOrbit, BatchRun, Camera, ExposureDumpSink, FlameDumpSink, FlameRenderSettings,
    GpuTimingsSink,
};
use thyllore_animation::ecs::systems::{
    apply_flame_overrides, batch_run_report, resolve_engine_cli_overrides,
};
use thyllore_animation::platform;

use vulkanalia::vk;

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
        let mut settings = app.data.ecs_world.resource_mut::<FlameRenderSettings>();
        settings.reference_step_count = step_count;
        settings.noise_step_count = step_count;
    }
    if let Some(pose) = overrides.camera_pose {
        let mut camera = app.data.ecs_world.resource_mut::<Camera>();
        camera.yaw = pose.yaw_degrees.to_radians();
        camera.pitch = pose.pitch_degrees.to_radians();
        camera.distance = pose.distance;
    }
    if let Some(path) = overrides.flame_dump_path {
        app.data
            .ecs_world
            .insert_resource(FlameDumpSink::new(std::path::PathBuf::from(path)));
    }
    if let Some(path) = overrides.gpu_timings_path {
        app.data
            .ecs_world
            .insert_resource(GpuTimingsSink::new(path));
    }
    if let Some(path) = overrides.exposure_dump_path {
        app.data
            .ecs_world
            .insert_resource(ExposureDumpSink::new(path));
    }
    if let Some(n) = overrides.flame_count {
        if n >= 2 {
            for i in 1..n {
                let mut effect = FlameEffect {
                    position: cgmath::Vector3::new(1.5 * i as f32, 0.0, 0.0),
                    radius: 0.7,
                    height: 0.8,
                    temperature_base_k: 1900.0 - 250.0 * i as f32,
                    temperature_tip_k: 1100.0 - 150.0 * i as f32,
                    ..FlameEffect::default()
                };
                if let Some(name) = overrides.flame_preset.as_deref() {
                    thyllore_render_core::apply_flame_preset(&mut effect, name);
                }
                apply_flame_overrides(&mut effect, &overrides.flame_set);
                thyllore_render_core::refresh_flame_coefficients(&mut effect);
                thyllore_animation::ecs::systems::spawn_flame(
                    &mut app.data.ecs_world,
                    &format!("Flame {}", i + 1),
                    effect,
                );
            }
        }
    }
    {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            if let Some(mut effect) = app.data.ecs_world.get_component_mut::<FlameEffect>(e) {
                if let Some(name) = overrides.flame_preset.as_deref() {
                    thyllore_render_core::apply_flame_preset(&mut effect, name);
                }
                apply_flame_overrides(&mut effect, &overrides.flame_set);
                thyllore_render_core::refresh_flame_coefficients(&mut effect);
            }
        }
    }

    // Apply flame_trail override: insert FlameTrail component on all flame entities
    if let Some(fade) = overrides.flame_trail {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            app.data.ecs_world.insert_component(
                e,
                FlameTrail {
                    state: thyllore_render_core::FlameTrailState {
                        enabled: true,
                        fade_seconds: fade,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
        }
    }

    // Apply flame_orbit override: insert BatchFlameOrbit resource into world
    if let Some((radius, period)) = overrides.flame_orbit {
        app.data.ecs_world.insert_resource(BatchFlameOrbit {
            radius,
            period_seconds: period,
            initial: None,
        });
    }

    // Apply flame_motion override: insert MotionPath component on first flame entity
    if let Some((radius, angular_speed)) = overrides.flame_motion {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        if let Some(&first) = entities.first() {
            let center = app
                .data
                .ecs_world
                .get_component::<FlameEffect>(first)
                .map(|e| e.position)
                .unwrap_or(cgmath::Vector3::new(0.0, 0.0, 0.0));
            app.data.ecs_world.insert_component(
                first,
                thyllore_animation::ecs::component::MotionPath {
                    center,
                    radius,
                    angular_speed,
                    phase_offset: 0.0,
                    enabled: true,
                },
            );
        }
    }
    if let Some(pixel) = overrides.pick_pixel {
        app.data.ecs_world.insert_resource(
            thyllore_animation::ecs::resource::BatchPickRequest::new(pixel),
        );
    }
    // Apply batch_play override: start timeline playback for deterministic batch clip runs
    if overrides.batch_play {
        let first = {
            let clip_library = app
                .data
                .ecs_world
                .resource::<thyllore_animation::ecs::resource::ClipLibrary>();
            let x = clip_library.all_clip_ids().next().copied();
            x
        };
        let mut ts = app
            .data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::TimelineState>();
        ts.playing = true;
        ts.looping = true;
        ts.current_time = 0.0;
        if ts.current_clip_id.is_none() {
            ts.current_clip_id = first;
        }
    }

    // Apply flame_bone override: attach flame entities to a skeleton bone (resolved per frame)
    if let Some(bone) = overrides.flame_bone.clone() {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            app.data.ecs_world.insert_component(
                e,
                thyllore_animation::ecs::component::FlameBoneAttachment { bone: bone.clone() },
            );
        }
    }

    {
        let (pixels, w, h) = match overrides.flame_sdf.as_ref() {
            Some(path) => match thyllore_render_core::flame_sdf::load_flame_sdf(path) {
                Ok(sdf) => {
                    let p = thyllore_render_core::flame_sdf::encode_sdf_rgba8(&sdf);
                    (p, sdf.width, sdf.height)
                }
                Err(e) => {
                    eprintln!("flame_sdf load failed: {e}");
                    (vec![255u8; 4], 1, 1)
                }
            },
            None => (vec![255u8; 4], 1, 1),
        };

        unsafe {
            use thyllore_animation::vulkanr::context::CommandState;
            let command_pool = app.resource::<CommandState>().pool.clone();
            let (image, memory, mips) =
                thyllore_vulkan_core::resource::create_texture_image_pixel_with_format(
                    &app.instance,
                    &app.rrdevice,
                    &command_pool,
                    &pixels,
                    w,
                    h,
                    vk::Format::R8G8B8A8_UNORM,
                )?;
            let image_view = thyllore_vulkan_core::resource::create_image_view(
                &app.rrdevice,
                image,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
                mips,
            )?;
            let sampler =
                thyllore_vulkan_core::resource::create_texture_sampler(&app.rrdevice, mips)?;

            app.data.raytracing.flame_sdf_image = image;
            app.data.raytracing.flame_sdf_image_memory = memory;
            app.data.raytracing.flame_sdf_image_view = image_view;
            app.data.raytracing.flame_sdf_sampler = sampler;

            if let (Some(ref flame_buffer), Some(ref flame_descriptor)) = (
                &app.data.viewport.flame_buffer,
                &app.data.raytracing.flame_descriptor,
            ) {
                if let (Some(ref gbuffer), Some(position_sampler)) = (
                    &app.data.raytracing.gbuffer,
                    app.data.raytracing.gbuffer_sampler,
                ) {
                    flame_descriptor.update_image_views(
                        &app.rrdevice,
                        gbuffer.position_image_view,
                        position_sampler,
                        flame_buffer.accum_image_view,
                        flame_buffer.interval_image_view,
                        flame_buffer.history_image_views,
                        flame_buffer.sampler,
                        image_view,
                        sampler,
                        app.resource::<thyllore_animation::vulkanr::context::RenderTargets>()
                            .render
                            .gbuffer_depth_image_view,
                    );
                }
            }
        }
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
