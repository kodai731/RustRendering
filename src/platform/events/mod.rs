mod debug_dump;
mod deferred;
mod export;
mod ui_windows;

use std::time::Instant;

use imgui::MouseButton;
use winit::event::{ElementState, Event, WindowEvent};

use super::key_bindings::{default_bindings, dispatch_keyboard_shortcut, ModifierKeys};
use super::platform::System;
#[cfg(debug_assertions)]
use super::ui::{build_click_debug_overlay, DebugWindowState};
use super::ui::{SceneOverlayState, StatusBarState};
use crate::app::App;
use crate::vulkanr::vulkan::*;

use crate::ecs::events::UIEvent;
use crate::ecs::resource::{CameraFlyInput, ImGuiInputCapture, KeyboardModifiers, MouseInput};
use crate::ecs::systems::phases::run_event_dispatch_phase;
use crate::ecs::UIEventQueue;

fn update_mouse_input(world: &crate::ecs::World, ui: &imgui::Ui) {
    let io = ui.io();
    let mut mouse = world.resource_mut::<MouseInput>();
    mouse.position = io.mouse_pos;
    mouse.left_pressed = ui.is_mouse_down(MouseButton::Left);
    mouse.right_pressed = ui.is_mouse_down(MouseButton::Right);
    mouse.middle_pressed = ui.is_mouse_down(MouseButton::Middle);
    drop(mouse);

    let mut modifiers = world.resource_mut::<KeyboardModifiers>();
    modifiers.ctrl = io.key_ctrl;
    modifiers.shift = io.key_shift;
    modifiers.alt = io.key_alt;
    drop(modifiers);

    update_camera_fly_input(world, ui);
}

fn update_camera_fly_input(world: &crate::ecs::World, ui: &imgui::Ui) {
    let io = ui.io();
    let mut fly = world.resource_mut::<CameraFlyInput>();
    fly.delta_seconds = io.delta_time;

    if io.want_text_input {
        fly.forward = 0.0;
        fly.right = 0.0;
        fly.up = 0.0;
        fly.boost = false;
        return;
    }

    let axis = |negative: bool, positive: bool| (positive as i32 - negative as i32) as f32;
    fly.forward = axis(ui.is_key_down(imgui::Key::S), ui.is_key_down(imgui::Key::W));
    fly.right = axis(ui.is_key_down(imgui::Key::A), ui.is_key_down(imgui::Key::D));
    fly.up = axis(ui.is_key_down(imgui::Key::Q), ui.is_key_down(imgui::Key::E));
    fly.boost = io.key_shift;
}

impl System {
    pub fn main_loop(self, app: &mut App) {
        let System {
            event_loop,
            window,
            mut imgui,
            mut platform,
        } = self;
        let mut last_frame = Instant::now();
        let bindings = default_bindings();
        let mut status_bar_state = StatusBarState::default();
        #[cfg(feature = "auto-rig")]
        let mut text_to_mesh_dialog_state = crate::platform::ui::TextToMeshDialogState::default();
        #[cfg(feature = "auto-rig")]
        let mut text_to_animation_dialog_state =
            crate::platform::ui::TextToAnimationDialogState::default();

        event_loop
            .run(move |event, window_target| match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();
                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                }

                Event::AboutToWait => {
                    platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("Failed to prepare frame");
                    window.request_redraw();
                }

                Event::WindowEvent {
                    event: ref window_event,
                    ..
                } => {
                    platform.handle_event(imgui.io_mut(), &window, &event);
                    dispatch_window_event(
                        window_event,
                        window_target,
                        app,
                        &mut imgui,
                        &mut platform,
                        &window,
                        &bindings,
                        &mut status_bar_state,
                        #[cfg(feature = "auto-rig")]
                        &mut text_to_mesh_dialog_state,
                        #[cfg(feature = "auto-rig")]
                        &mut text_to_animation_dialog_state,
                    );
                }

                Event::LoopExiting => {
                    unsafe { app.destroy() };
                }

                _ => {}
            })
            .expect("EventLoop error");
    }
}

fn dispatch_window_event(
    event: &WindowEvent,
    window_target: &winit::event_loop::EventLoopWindowTarget<()>,
    app: &mut App,
    imgui: &mut imgui::Context,
    platform: &mut imgui_winit_support::WinitPlatform,
    window: &winit::window::Window,
    bindings: &[super::key_bindings::KeyBinding],
    status_bar_state: &mut StatusBarState,
    #[cfg(feature = "auto-rig")]
    text_to_mesh_dialog: &mut crate::platform::ui::TextToMeshDialogState,
    #[cfg(feature = "auto-rig")]
    text_to_animation_dialog: &mut crate::platform::ui::TextToAnimationDialogState,
) {
    match event {
        WindowEvent::CloseRequested => window_target.exit(),

        WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
            app.resized = true;
        }

        WindowEvent::CursorMoved { position, .. } => {
            let mut mouse = app.data.ecs_world.resource_mut::<MouseInput>();
            mouse.position = [position.x as f32, position.y as f32];
        }

        WindowEvent::MouseWheel { delta, .. } => {
            let mut mouse = app.data.ecs_world.resource_mut::<MouseInput>();
            mouse.wheel = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
            };
        }

        WindowEvent::DroppedFile(path_buf) => {
            if let Some(path) = path_buf.to_str() {
                if path.to_ascii_lowercase().ends_with(".png") {
                    // A dropped PNG fills the texture-fit path field (selection
                    // only — applying stays on the explicit Apply button).
                    app.data
                        .ecs_world
                        .resource_mut::<crate::ecs::ModelState>()
                        .texture_fit_path = path.to_string();
                } else {
                    let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
                    ui_events.send(UIEvent::LoadModel {
                        path: path.to_string(),
                    });
                }
            }
        }

        WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
            dispatch_keyboard_input(app, event, imgui, bindings);
        }

        WindowEvent::RedrawRequested => {
            handle_redraw_requested(
                imgui,
                platform,
                window,
                app,
                status_bar_state,
                #[cfg(feature = "auto-rig")]
                text_to_mesh_dialog,
                #[cfg(feature = "auto-rig")]
                text_to_animation_dialog,
            );

            if crate::ecs::systems::batch_run_is_completed(&app.data.ecs_world) {
                window_target.exit();
            }
        }

        _ => {}
    }
}

fn dispatch_keyboard_input(
    app: &mut App,
    event: &winit::event::KeyEvent,
    imgui: &imgui::Context,
    bindings: &[super::key_bindings::KeyBinding],
) {
    let modifiers_res = app.data.ecs_world.resource::<KeyboardModifiers>();
    let modifiers = ModifierKeys {
        ctrl: modifiers_res.ctrl,
        shift: modifiers_res.shift,
    };
    drop(modifiers_res);

    let camera_fly_active = app.data.ecs_world.resource::<MouseInput>().right_pressed;

    if let Some(ui_event) = dispatch_keyboard_shortcut(
        &event.logical_key,
        modifiers,
        imgui.io().want_capture_keyboard || camera_fly_active,
        bindings,
    ) {
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        ui_events.send(ui_event);
    }
}

fn handle_redraw_requested(
    imgui: &mut imgui::Context,
    platform: &mut imgui_winit_support::WinitPlatform,
    window: &winit::window::Window,
    app: &mut App,
    status_bar_state: &mut StatusBarState,
    #[cfg(feature = "auto-rig")]
    text_to_mesh_dialog: &mut crate::platform::ui::TextToMeshDialogState,
    #[cfg(feature = "auto-rig")]
    text_to_animation_dialog: &mut crate::platform::ui::TextToAnimationDialogState,
) {
    let dt_ms = if let Some(last) = app.last_frame_instant {
        let elapsed = last.elapsed().as_secs_f32() * 1000.0;
        app.last_frame_instant = Some(Instant::now());
        elapsed
    } else {
        app.last_frame_instant = Some(Instant::now());
        0.0
    };

    let ui = imgui.frame();

    let io = ui.io();
    {
        let mut capture = app.data.ecs_world.resource_mut::<ImGuiInputCapture>();
        capture.wants_mouse = io.want_capture_mouse;
    }

    update_mouse_input(&app.data.ecs_world, ui);

    #[cfg(debug_assertions)]
    let mut debug_state = DebugWindowState {
        debug_view_mode: app
            .resource::<crate::ecs::resource::DebugViewState>()
            .debug_view_mode,
    };

    let model_state = app.resource::<crate::ecs::ModelState>();
    let mut overlay_state = SceneOverlayState {
        model_path: model_state.model_path.clone(),
        load_status: model_state.load_status.clone(),
        flame_preset_index: model_state.flame_preset_index,
        water_preset_index: 0,
        texture_fit_path: model_state.texture_fit_path.clone(),
        texture_fit_blend: model_state.texture_fit_blend,
        texture_fit_groups: model_state.texture_fit_groups,
        texture_fit_profile: model_state.texture_fit_profile,
        texture_fit_scan: model_state.texture_fit_scan.clone(),
        texture_fit_scan_done: model_state.texture_fit_scan_done,
        texture_fit_browser_open: model_state.texture_fit_browser_open,
        texture_fit_browser_dir: model_state.texture_fit_browser_dir.clone(),
        texture_fit_browser_selected: model_state.texture_fit_browser_selected.clone(),
        texture_fit_browser_show_all: model_state.texture_fit_browser_show_all,
        texture_fit_browser_show_hidden: model_state.texture_fit_browser_show_hidden,
        texture_fit_path_validated: model_state.texture_fit_path_validated.clone(),
        texture_fit_path_info: model_state.texture_fit_path_info.clone(),
        flame_style_index: model_state.flame_style_index,
        flame_style_scan: model_state.flame_style_scan.clone(),
        flame_style_scan_done: model_state.flame_style_scan_done,
        flame_style_groups: model_state.flame_style_groups,
        flame_style_save_name: model_state.flame_style_save_name.clone(),
        #[cfg(feature = "auto-rig")]
        open_text_to_mesh_dialog: false,
        #[cfg(feature = "auto-rig")]
        open_text_to_animation_dialog: false,
    };
    drop(model_state);

    ui_windows::build_ui_windows(
        ui,
        app,
        #[cfg(debug_assertions)]
        &mut debug_state,
        &mut overlay_state,
        status_bar_state,
        #[cfg(feature = "auto-rig")]
        text_to_mesh_dialog,
        #[cfg(feature = "auto-rig")]
        text_to_animation_dialog,
    );

    app.resource_mut::<crate::ecs::ModelState>()
        .flame_preset_index = overlay_state.flame_preset_index;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_path = overlay_state.texture_fit_path;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_blend = overlay_state.texture_fit_blend;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_groups = overlay_state.texture_fit_groups;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_profile = overlay_state.texture_fit_profile;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_browser_open = overlay_state.texture_fit_browser_open;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_browser_dir = overlay_state.texture_fit_browser_dir;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_browser_selected = overlay_state.texture_fit_browser_selected;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_browser_show_all = overlay_state.texture_fit_browser_show_all;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_browser_show_hidden = overlay_state.texture_fit_browser_show_hidden;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_path_validated = overlay_state.texture_fit_path_validated;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_path_info = overlay_state.texture_fit_path_info;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_scan = overlay_state.texture_fit_scan;
    app.resource_mut::<crate::ecs::ModelState>()
        .texture_fit_scan_done = overlay_state.texture_fit_scan_done;
    app.resource_mut::<crate::ecs::ModelState>()
        .flame_style_index = overlay_state.flame_style_index;
    app.resource_mut::<crate::ecs::ModelState>()
        .flame_style_scan = overlay_state.flame_style_scan;
    app.resource_mut::<crate::ecs::ModelState>()
        .flame_style_scan_done = overlay_state.flame_style_scan_done;
    app.resource_mut::<crate::ecs::ModelState>()
        .flame_style_groups = overlay_state.flame_style_groups;
    app.resource_mut::<crate::ecs::ModelState>()
        .flame_style_save_name = overlay_state.flame_style_save_name;

    #[cfg(debug_assertions)]
    {
        app.resource_mut::<crate::ecs::resource::DebugViewState>()
            .debug_view_mode = debug_state.debug_view_mode;
    }

    #[cfg(debug_assertions)]
    build_click_debug_overlay(ui, &app.data.ecs_world);

    platform.prepare_render(ui, window);

    let imgui_build_start = Instant::now();
    let draw_data = imgui.render();
    let imgui_build_ms = imgui_build_start.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        process_ui_events_and_render_frame(app, window, draw_data, dt_ms, imgui_build_ms);
    }

    app.data.ecs_world.resource_mut::<MouseInput>().end_frame();
}

unsafe fn process_ui_events_and_render_frame(
    app: &mut App,
    window: &winit::window::Window,
    draw_data: &imgui::DrawData,
    dt_ms: f32,
    imgui_build_ms: f32,
) {
    let model_bounds = app.data.graphics_resources.calculate_model_bounds();
    let (platform_events, deferred_actions) = run_event_dispatch_phase(
        &mut app.data.ecs_world,
        &mut app.data.ecs_assets,
        model_bounds,
    );

    let mut platform_deferred = deferred::process_platform_file_events(&platform_events, app);

    let mut all_deferred = deferred_actions;
    all_deferred.append(&mut platform_deferred);

    for action in all_deferred {
        deferred::execute_deferred_action(app, action);
    }

    app.spawn_pending_debug_primitives();

    render_frame(app, window, draw_data, dt_ms, imgui_build_ms);
}

unsafe fn render_frame(
    app: &mut App,
    window: &winit::window::Window,
    draw_data: &imgui::DrawData,
    dt_ms: f32,
    imgui_build_ms: f32,
) {
    let frame_result = (|| -> anyhow::Result<()> {
        let gpu_wait_start = Instant::now();
        let image_index = app.begin_frame()?;
        let gpu_wait_ms = gpu_wait_start.elapsed().as_secs_f32() * 1000.0;

        let update_start = Instant::now();
        app.update(image_index)?;
        let update_ms = update_start.elapsed().as_secs_f32() * 1000.0;

        let render_cpu_start = Instant::now();
        app.render(image_index, draw_data)?;
        let render_cpu_ms = render_cpu_start.elapsed().as_secs_f32() * 1000.0;

        // Batch run screenshot: after render (which calls queue_present_khr),
        // check if we need to save a screenshot for the batch run.
        if app
            .data
            .ecs_world
            .contains_resource::<crate::ecs::resource::BatchRun>()
        {
            let state = app
                .data
                .ecs_world
                .get_resource::<crate::ecs::resource::BatchRun>()
                .map(|b| b.state.clone());
            let (dump_wall_probe, dump_water_debug) = app
                .data
                .ecs_world
                .get_resource::<crate::ecs::resource::BatchRun>()
                .map(|b| (b.dump_wall_probe, b.dump_water_debug))
                .unwrap_or((false, false));
            if matches!(
                state,
                Some(crate::ecs::resource::BatchRunState::ScreenshotRequested)
            ) {
                app.rrdevice.device.device_wait_idle()?;
                if dump_water_debug {
                    app.dump_water_debug_at(image_index);
                }
                let save_result = app.save_screenshot(image_index);
                crate::ecs::systems::batch_run_record_screenshot(
                    &app.data.ecs_world,
                    save_result.map_err(|e| format!("{e:?}")),
                );
                // Wall probe dump: if this batch run was started with
                // --batch-debug-action dump_wall_probe, dump synchronously
                // at the same frame/time as the screenshot.
                if dump_wall_probe {
                    crate::ecs::systems::perform_flame_wall_probe_dump(
                        &app.data.ecs_world,
                        [1680.0, 840.0],
                    );
                }
                debug_dump::save_flame_history_npy_if_requested(app);
                debug_dump::save_water_probe_if_requested(app);
            }
        }
        app.data
            .ecs_world
            .insert_resource(crate::ecs::resource::CpuFrameTimings {
                frame: app.frame as u64,
                dt_ms,
                stages: vec![
                    ("imgui_build".to_string(), imgui_build_ms),
                    ("gpu_wait".to_string(), gpu_wait_ms),
                    ("update".to_string(), update_ms),
                    ("render_cpu".to_string(), render_cpu_ms),
                ],
                imgui_vtx: draw_data.total_vtx_count as u32,
                imgui_idx: draw_data.total_idx_count as u32,
            });

        Ok(())
    })();

    if let Err(e) = frame_result {
        let msg = e.to_string();
        if msg.contains("SWAPCHAIN_OUT_OF_DATE") {
            if let Err(e) = app.recreate_swapchain(window) {
                log_error!("Failed to recreate swapchain: {:?}", e);
                return;
            }
            // Write swapchain_recreate marker event to exposure dump sink if it exists
            if let Some(mut sink) = app
                .data
                .ecs_world
                .get_resource_mut::<crate::ecs::resource::ExposureDumpSink>()
            {
                let frame = app
                    .data
                    .ecs_world
                    .get_resource::<crate::ecs::resource::BatchRun>()
                    .map(|b| b.frames_rendered)
                    .unwrap_or(sink.last_frame);
                use std::fs::OpenOptions;
                use std::io::Write;
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&sink.path)
                {
                    let _ = writeln!(
                        file,
                        "{{\"event\":\"swapchain_recreate\",\"frame\":{}}}",
                        frame
                    );
                }
            }
        } else {
            log_error!("Frame error: {:?}", e);
        }
    }
}
