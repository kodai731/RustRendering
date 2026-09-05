#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use thyllore_animation::app::init::instance::cleanup_old_screenshots;
use thyllore_animation::app::model_loader::find_best_clip;
use thyllore_animation::app::App;
use thyllore_animation::ecs::component::{FlameEffect, FlameTrail, HeatPlume};
use thyllore_animation::ecs::events::{UIEvent, UIEventQueue};
use thyllore_animation::ecs::resource::{
    BatchFlameOrbit, BatchRun, Camera, ExposureDumpSink, FlameDumpSink, FlameRenderSettings,
    GpuTimingsSink,
};
use thyllore_animation::ecs::systems::{
    apply_flame_overrides, apply_flame_style_from_path, apply_texture_fit_from_path,
    batch_anim_dump_write, batch_apply_anim_edits, batch_apply_debug_actions, batch_run_report,
    debug_actions_json, dump_flame_style_to_path, resolve_engine_cli_overrides,
    run_sequence_analyze_from_args, BatchDebugAction, BATCH_LIST_DEBUG_ACTIONS_FLAG,
};
use thyllore_animation::platform;

use thyllore_vulkan_core::FlameImageBindings;
use vulkanalia::vk;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::init();

    cleanup_old_screenshots()?;

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == BATCH_LIST_DEBUG_ACTIONS_FLAG) {
        println!("{}", debug_actions_json());
        return Ok(());
    }
    if let Some(result) = run_sequence_analyze_from_args(args.clone()) {
        result?;
        return Ok(());
    }
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
    let mut system = platform::init(&window_title, !is_batch_mode);

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
    if let Some(debug_view) = overrides.flame_debug_view {
        app.data
            .ecs_world
            .resource_mut::<FlameRenderSettings>()
            .debug_view = debug_view;
    }
    if let Some(debug_view) = overrides.water_debug_view {
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .debug_view = debug_view;
    }
    if let Some(caustic_debug) = overrides.water_caustic_debug {
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .caustic_debug = caustic_debug;
    }
    if let Some(secondary) = overrides.water_secondary {
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .secondary_rays = secondary;
    }
    if let Some(weight) = overrides.water_history_weight {
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .batch_history_weight = Some(weight);
    }
    if let Some(seconds) = overrides.water_fixed_time {
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .batch_fixed_time = Some(seconds);
    }
    if overrides.water_probe_path.is_some() {
        let debug_view = overrides.water_debug_view.unwrap_or(3);
        app.data
            .ecs_world
            .resource_mut::<thyllore_animation::ecs::resource::WaterRenderSettings>()
            .debug_view = debug_view;
    }
    if let Some(pose) = overrides.camera_pose {
        let mut camera = app.data.ecs_world.resource_mut::<Camera>();
        camera.yaw = pose.yaw_degrees.to_radians();
        camera.pitch = pose.pitch_degrees.to_radians();
        camera.distance = pose.distance;
        if let Some(pivot) = pose.pivot {
            camera.pivot = cgmath::Vector3::new(pivot[0], pivot[1], pivot[2]);
        }
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
                    color: thyllore_effect_core::FlameColor {
                        temperature_base_k: 1900.0 - 250.0 * i as f32,
                        temperature_tip_k: 1100.0 - 150.0 * i as f32,
                        ..thyllore_effect_core::FlameColor::default()
                    },
                    ..FlameEffect::default()
                };
                let mut baked = thyllore_effect_core::FlameBaked::default();
                if let Some(name) = overrides.flame_preset.as_deref() {
                    thyllore_effect_core::apply_flame_preset(&mut effect, name);
                }
                if let Some((ref path, blend, profile)) = overrides.flame_texture_fit {
                    apply_texture_fit_from_path(
                        &mut effect,
                        &mut baked,
                        path,
                        blend,
                        thyllore_effect_core::TextureFitGroups::default(),
                        profile,
                        "cli",
                    );
                }
                if let Some((ref path, groups)) = overrides.flame_style {
                    apply_flame_style_from_path(&mut effect, path, groups);
                }
                apply_flame_overrides(&mut effect, &overrides.flame_set);
                thyllore_effect_core::refresh_flame_coefficients(&mut effect, &baked);
                let entity = thyllore_animation::ecs::systems::spawn_flame_with_clip(
                    &mut app.data.ecs_world,
                    &mut app.data.ecs_assets,
                    &format!("Flame {}", i + 1),
                    effect,
                );
                app.data.ecs_world.insert_component(entity, baked);
            }
        }
    }
    {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            let Some(mut effect) = app
                .data
                .ecs_world
                .get_component::<FlameEffect>(e)
                .map(|c| c.clone())
            else {
                continue;
            };
            let mut baked = app
                .data
                .ecs_world
                .get_component::<thyllore_effect_core::FlameBaked>(e)
                .cloned()
                .unwrap_or_default();
            if let Some(name) = overrides.flame_preset.as_deref() {
                thyllore_effect_core::apply_flame_preset(&mut effect, name);
            }
            if let Some((ref path, blend, profile)) = overrides.flame_texture_fit {
                apply_texture_fit_from_path(
                    &mut effect,
                    &mut baked,
                    path,
                    blend,
                    thyllore_effect_core::TextureFitGroups::default(),
                    profile,
                    "cli",
                );
            }
            if let Some((ref path, groups)) = overrides.flame_style {
                apply_flame_style_from_path(&mut effect, path, groups);
            }
            apply_flame_overrides(&mut effect, &overrides.flame_set);
            thyllore_effect_core::refresh_flame_coefficients(&mut effect, &baked);
            if let Some(ref path) = overrides.flame_style_dump {
                dump_flame_style_to_path(&effect, path);
            }
            app.data.ecs_world.insert_component(e, effect);
            app.data.ecs_world.insert_component(e, baked);
        }
    }

    // Apply flame_trail override: insert FlameTrail component on all flame entities
    if let Some(fade) = overrides.flame_trail {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            app.data.ecs_world.insert_component(
                e,
                FlameTrail {
                    state: thyllore_effect_core::FlameTrailState {
                        enabled: true,
                        fade_seconds: fade,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
        }
    }

    // Apply heat_plume override: insert HeatPlume component on all flame entities
    if let Some((gain, amp)) = overrides.heat_plume {
        let entities: Vec<_> = app.data.ecs_world.query_flames();
        for e in entities {
            app.data.ecs_world.insert_component(
                e,
                HeatPlume {
                    distortion_gain: gain,
                    turbulence_amp: amp,
                    ..Default::default()
                },
            );
        }
    }

    // Auto-load model when scene_path override is provided: if the scene restored a model path
    // (via apply_loaded_scene -> ModelState), send UIEvent::LoadModel to load it.
    if let Some(ref _scene_path) = overrides.scene_path {
        let model_state = app
            .data
            .ecs_world
            .resource::<thyllore_animation::ecs::resource::ModelState>();
        if !model_state.model_path.is_empty() && model_state.model_path != "Generated Mesh" {
            let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
            ui_events.send(UIEvent::LoadModel {
                path: model_state.model_path.clone(),
            });
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
    if !overrides.anim_edits.is_empty() {
        batch_apply_anim_edits(
            &mut app.data.ecs_world,
            &mut app.data.ecs_assets,
            &overrides.anim_edits,
        );
    }
    if !overrides.debug_actions.is_empty() {
        let batch_run_owns_dumps = app.data.ecs_world.contains_resource::<BatchRun>();
        let filtered: Vec<_> = overrides
            .debug_actions
            .iter()
            .filter(|a| {
                !batch_run_owns_dumps
                    || !matches!(
                        a,
                        BatchDebugAction::WallProbeDump | BatchDebugAction::WaterDebugDump
                    )
            })
            .cloned()
            .collect();
        batch_apply_debug_actions(&app.data.ecs_world, &filtered);
    }

    // Apply batch_play override: start timeline playback for deterministic batch clip runs.
    // Prefer a clip with bone tracks so the (empty) default flame clip never
    // shadows the model animation the batch run wants to play.
    if overrides.batch_play {
        let first = find_best_clip(&app.data.ecs_world);
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

        // Store play request on BatchRun so model_loader.rs resets can restore it
        if let Some(mut batch_run) = app.data.ecs_world.get_resource_mut::<BatchRun>() {
            batch_run.play_requested = true;
            batch_run.play_clip_id = first;
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
            Some(path) => match thyllore_effect_core::flame_sdf::load_flame_sdf(path) {
                Ok(sdf) => {
                    let p = thyllore_effect_core::flame_sdf::encode_sdf_rgba8(&sdf);
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
                &app.data.effect_targets.flame,
                &app.data.raytracing.flame_descriptor,
            ) {
                flame_descriptor.update_image_views(
                    &app.rrdevice,
                    FlameImageBindings {
                        history_image_views: flame_buffer.history_image_views,
                        flame_sampler: flame_buffer.sampler,
                        sdf_image_view: image_view,
                        sdf_sampler: sampler,
                        scene_depth_view: app
                            .resource::<thyllore_animation::vulkanr::context::RenderTargets>()
                            .render
                            .gbuffer_depth_image_view,
                    },
                )?;
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

    if let Some(ref dump_path) = overrides.anim_dump_path {
        if let Err(e) = batch_anim_dump_write(&app.data.ecs_world, dump_path) {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": format!("anim dump failed: {e}")})
            );
            std::process::exit(1);
        }
    }

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
