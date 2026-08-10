use std::time::Instant;

use imgui::MouseButton;
use winit::event::{ElementState, Event, WindowEvent};

use super::key_bindings::{default_bindings, dispatch_keyboard_shortcut, ModifierKeys};
use super::platform::System;
use super::ui::{
    build_bottom_panel, build_clip_browser_window, build_curve_editor_window,
    build_hierarchy_window, build_inspector_window, build_scene_overlay, build_timeline_window,
    build_viewport_window, draw_status_bar, handle_splitters, LayoutSnapshot, SceneOverlayState,
    StatusBarState, ViewportInfo,
};
#[cfg(debug_assertions)]
use super::ui::{build_click_debug_overlay, DebugWindowState};
use crate::app::App;
use crate::vulkanr::vulkan::*;

use crate::ecs::events::UIEvent;
use crate::ecs::resource::{
    CameraFlyInput, ClipBrowserState, ClipLibrary, CurveEditorBuffer, CurveEditorState,
    HierarchyState, ImGuiInputCapture, KeyboardModifiers, MessageLog, MouseInput, PanelLayout,
    PoseLibrary, TimelineInteractionState, TimelineState, ViewportInput,
};
use crate::ecs::systems::clip_track_systems::query_clip_tracks;
use crate::ecs::systems::phases::run_event_dispatch_phase;
use crate::ecs::{DeferredAction, UIEventQueue};

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
        #[cfg(feature = "auto-rig")]
        open_text_to_mesh_dialog: false,
        #[cfg(feature = "auto-rig")]
        open_text_to_animation_dialog: false,
    };
    drop(model_state);

    build_ui_windows(
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

fn build_ui_windows(
    ui: &imgui::Ui,
    app: &mut App,
    #[cfg(debug_assertions)] debug_state: &mut DebugWindowState,
    overlay_state: &mut SceneOverlayState,
    status_bar_state: &mut StatusBarState,
    #[cfg(feature = "auto-rig")]
    text_to_mesh_dialog: &mut crate::platform::ui::TextToMeshDialogState,
    #[cfg(feature = "auto-rig")]
    text_to_animation_dialog: &mut crate::platform::ui::TextToAnimationDialogState,
) {
    let display_size = ui.io().display_size;

    let layout_snapshot = {
        let mut panel_layout = app.data.ecs_world.resource_mut::<PanelLayout>();
        panel_layout.constrain_to_display(display_size[0], display_size[1]);
        LayoutSnapshot::from_layout(&panel_layout, display_size)
    };

    build_side_panel_windows(
        ui,
        app,
        #[cfg(debug_assertions)]
        debug_state,
        &layout_snapshot,
    );
    let viewport_info = build_viewport_and_update_state(ui, app, &layout_snapshot);

    {
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        build_scene_overlay(
            ui,
            &mut *ui_events,
            overlay_state,
            &app.data.ecs_world,
            &viewport_info,
        );
    }

    build_timeline_and_fixed_overlays(ui, app, status_bar_state, &viewport_info, &layout_snapshot);
    build_curve_editor(ui, app);

    #[cfg(feature = "auto-rig")]
    {
        if overlay_state.open_text_to_mesh_dialog {
            text_to_mesh_dialog.open = true;
            overlay_state.open_text_to_mesh_dialog = false;
        }
        if overlay_state.open_text_to_animation_dialog {
            text_to_animation_dialog.open = true;
            overlay_state.open_text_to_animation_dialog = false;
        }

        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        crate::platform::ui::build_text_to_mesh_dialog(
            ui,
            &mut *ui_events,
            text_to_mesh_dialog,
            &app.data.ecs_world,
        );
        crate::platform::ui::build_text_to_animation_dialog(
            ui,
            &mut *ui_events,
            text_to_animation_dialog,
            &app.data.ecs_world,
        );
    }

    consume_needs_focus(app);
}

fn consume_needs_focus(app: &mut App) {
    let mut curve_editor = app.data.ecs_world.resource_mut::<CurveEditorState>();
    curve_editor.needs_focus = false;
}

fn build_side_panel_windows(
    ui: &imgui::Ui,
    app: &mut App,
    #[cfg(debug_assertions)] debug_state: &mut DebugWindowState,
    layout_snapshot: &LayoutSnapshot,
) {
    {
        let mut msg_log = app.data.ecs_world.resource_mut::<MessageLog>();
        msg_log.sync_from_buffer();
    }

    {
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        let mut msg_log = app.data.ecs_world.resource_mut::<MessageLog>();
        build_bottom_panel(
            ui,
            &mut *ui_events,
            #[cfg(debug_assertions)]
            debug_state,
            &app.data.ecs_world,
            &mut *msg_log,
            layout_snapshot,
        );
    }

    {
        let hierarchy_state = app.data.ecs_world.resource::<HierarchyState>();
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        build_hierarchy_window(
            ui,
            &mut *ui_events,
            &app.data.ecs_world,
            &*hierarchy_state,
            &app.data.ecs_assets,
            layout_snapshot,
        );
    }

    {
        let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
        let mut browser_state = app.data.ecs_world.resource_mut::<ClipBrowserState>();
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        build_clip_browser_window(
            ui,
            &mut *ui_events,
            &*clip_library,
            &mut *browser_state,
            &app.data.ecs_world,
            layout_snapshot,
        );
    }

    {
        let hierarchy_state = app.data.ecs_world.resource::<HierarchyState>();
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        build_inspector_window(
            ui,
            &mut *ui_events,
            &app.data.ecs_world,
            &*hierarchy_state,
            &app.data.ecs_assets,
            &app.data.graphics_resources,
            layout_snapshot,
        );
    }
}

fn build_viewport_and_update_state(
    ui: &imgui::Ui,
    app: &mut App,
    layout_snapshot: &LayoutSnapshot,
) -> ViewportInfo {
    let texture_id = imgui::TextureId::new(app.data.viewport.texture_id());
    let current_size = [
        app.data.viewport.width as f32,
        app.data.viewport.height as f32,
    ];
    let info = build_viewport_window(ui, texture_id, current_size, layout_snapshot);

    app.data.viewport.focused = info.focused;
    app.data.viewport.hovered = info.hovered;

    {
        let mut viewport = app.data.ecs_world.resource_mut::<ViewportInput>();
        viewport.focused = info.focused;
        viewport.hovered = info.hovered;
        viewport.position = info.position;
        viewport.size = info.size;

        let new_width = info.size[0] as u32;
        let new_height = info.size[1] as u32;
        if new_width > 0
            && new_height > 0
            && (new_width != app.data.viewport.width || new_height != app.data.viewport.height)
        {
            viewport.resize_pending = Some((new_width, new_height));
        }
    }

    info
}

fn build_timeline_and_fixed_overlays(
    ui: &imgui::Ui,
    app: &mut App,
    status_bar_state: &mut StatusBarState,
    viewport_info: &ViewportInfo,
    layout_snapshot: &LayoutSnapshot,
) {
    let clip_track_snapshot = {
        let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
        query_clip_tracks(&app.data.ecs_world, &*clip_library, &app.data.ecs_assets)
    };

    {
        let mut timeline_state = app.data.ecs_world.resource_mut::<TimelineState>();
        let mut timeline_interaction = app
            .data
            .ecs_world
            .resource_mut::<TimelineInteractionState>();
        let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
        let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
        let mut curve_editor = app.data.ecs_world.resource_mut::<CurveEditorState>();
        build_timeline_window(
            ui,
            &mut *ui_events,
            &mut *timeline_state,
            &mut *timeline_interaction,
            &*clip_library,
            &mut *curve_editor,
            &clip_track_snapshot,
            layout_snapshot,
        );
    }
    // Batch captures are diffed pixel-by-pixel; the status bar shows wall-clock
    // values (FPS, memory) that would break determinism, so skip it there.
    let is_batch_capture = app
        .data
        .ecs_world
        .contains_resource::<crate::ecs::resource::BatchRun>();
    if !is_batch_capture {
        let delta_time = (app.start.elapsed().as_secs_f32() - app.last_update_time).max(0.001);
        let timeline_state = app.data.ecs_world.resource::<TimelineState>();
        let clip_duration = {
            let lib = app.data.ecs_world.resource::<ClipLibrary>();
            crate::ecs::systems::timeline_effective_duration(&timeline_state, &lib)
        };
        draw_status_bar(
            ui,
            status_bar_state,
            delta_time,
            viewport_info,
            &*timeline_state,
            clip_duration,
        );
    }

    let mut panel_layout = app.data.ecs_world.resource_mut::<PanelLayout>();
    handle_splitters(ui, &mut panel_layout, layout_snapshot);
}

fn build_curve_editor(ui: &imgui::Ui, app: &mut App) {
    let scalar_domain = {
        let world = &app.data.ecs_world;
        let current = world.resource::<TimelineState>().current_clip_id;
        current.and_then(|_| {
            crate::ecs::component::scalar_channel_domains()
                .iter()
                .copied()
                .find(|domain| {
                    (domain.entities)(world).iter().any(|&entity| {
                        crate::ecs::systems::scalar_clip_systems::find_entity_clip_id(world, entity)
                            == current
                    })
                })
        })
    };
    let timeline_state = app.data.ecs_world.resource::<TimelineState>();
    let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
    let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
    let mut curve_editor = app.data.ecs_world.resource_mut::<CurveEditorState>();
    let curve_buffer = app.data.ecs_world.resource::<CurveEditorBuffer>();
    let mut pose_library = app.data.ecs_world.resource_mut::<PoseLibrary>();

    #[cfg(feature = "ml")]
    let suggestion_overlays: Vec<super::ui::SuggestionOverlay> = {
        if let Some(state) = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::CurveSuggestionState>()
        {
            state
                .suggestions
                .iter()
                .map(|s| super::ui::SuggestionOverlay {
                    property_type: s.property_type,
                    time: s.predicted_time,
                    value: s.predicted_value,
                    tangent_in: s.tangent_in,
                    tangent_out: s.tangent_out,
                    confidence: s.confidence,
                })
                .collect()
        } else {
            Vec::new()
        }
    };
    #[cfg(not(feature = "ml"))]
    let suggestion_overlays: Vec<super::ui::SuggestionOverlay> = Vec::new();

    build_curve_editor_window(
        ui,
        &mut *ui_events,
        &*timeline_state,
        &*clip_library,
        &mut *curve_editor,
        &*curve_buffer,
        &suggestion_overlays,
        &mut *pose_library,
        scalar_domain,
    );
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

    let mut platform_deferred = process_platform_file_events(&platform_events, app);

    let mut all_deferred = deferred_actions;
    all_deferred.append(&mut platform_deferred);

    for action in all_deferred {
        execute_deferred_action(app, action);
    }

    render_frame(app, window, draw_data, dt_ms, imgui_build_ms);
}

unsafe fn execute_deferred_action(app: &mut App, action: DeferredAction) {
    match action {
        DeferredAction::LoadModel { path } => {
            if let Err(e) = app.load_model(&path) {
                log_error!("Failed to load model: {:?}", e);
            }
        }

        DeferredAction::LoadModelAdditive { path } => {
            if let Err(e) = app.load_model_additive(&path) {
                log_error!("Failed to add model: {:?}", e);
            }
        }

        DeferredAction::DeleteEntities { entities } => {
            if let Err(e) = app.delete_entities(&entities) {
                log_error!("Failed to delete entities: {:?}", e);
            }
        }

        DeferredAction::TakeScreenshot => {
            log!("Taking screenshot...");
            let image_index = app.frame % crate::app::init::MAX_FRAMES_IN_FLIGHT;
            let save_result = app.save_screenshot(image_index);
            match &save_result {
                Ok(path) => msg_info!("Screenshot saved: {}", path),
                Err(e) => log_error!("Screenshot failed: {:?}", e),
            }
            crate::ecs::systems::batch_run_record_screenshot(
                &app.data.ecs_world,
                save_result.map_err(|e| format!("{e:?}")),
            );
        }

        #[cfg(debug_assertions)]
        DeferredAction::DebugShadowInfo => {
            crate::debugview::log_shadow_debug_info(
                &app.data.ecs_world,
                &app.data.raytracing,
                &app.data.graphics_resources,
            );
        }

        #[cfg(debug_assertions)]
        DeferredAction::DebugBillboardDepth => {
            crate::debugview::collect_and_log_billboard_debug(
                &app.data.ecs_world,
                &app.data.raytracing,
            );
        }

        DeferredAction::DumpDebugInfo => {
            app.dump_debug_info();
        }

        DeferredAction::DumpAnimationDebug => {
            let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
            if let Err(e) = crate::ecs::systems::animation_debug_dump::dump_animation_debug(
                &app.data.ecs_world,
                &app.data.ecs_assets,
                &*clip_library,
            ) {
                log_warn!("Animation debug dump failed: {:?}", e);
            }
        }

        DeferredAction::LoadClipFromFile { path } => {
            let bone_name_to_id = app
                .data
                .ecs_assets
                .skeletons
                .values()
                .next()
                .map(|sa| sa.skeleton.bone_name_to_id.clone());

            let mut clip_library = app.data.ecs_world.resource_mut::<ClipLibrary>();
            match crate::ecs::systems::clip_library_systems::clip_library_load_from_file(
                &mut clip_library,
                &mut app.data.ecs_assets,
                &path,
                bone_name_to_id.as_ref(),
            ) {
                Ok(_) => {}
                Err(e) => msg_error!("Failed to load clip: {:?}", e),
            }
        }

        DeferredAction::SaveClipToFile { source_id, path } => {
            use crate::ecs::systems::clip_library_systems::{
                clip_library_save_to_file, clip_library_update_save_metadata,
            };

            let new_name = extract_clip_name_from_path(&path);
            let mut clip_library = app.data.ecs_world.resource_mut::<ClipLibrary>();
            clip_library_update_save_metadata(
                &mut clip_library,
                source_id,
                new_name.clone(),
                &path,
            );

            match clip_library_save_to_file(&clip_library, source_id, &path) {
                Ok(()) => msg_info!("Saved clip '{}' to {:?}", new_name, path),
                Err(e) => msg_error!("Failed to save clip: {:?}", e),
            }
        }

        DeferredAction::SaveSpringBoneBake { baked_id, path } => {
            use crate::ecs::systems::clip_library_systems::clip_library_save_to_file;

            let clip_library = app.data.ecs_world.resource::<ClipLibrary>();
            match clip_library_save_to_file(&clip_library, baked_id, &path) {
                Ok(()) => msg_info!("Saved spring bone bake to {:?}", path),
                Err(e) => msg_error!("Failed to save spring bone bake: {:?}", e),
            }
        }

        #[cfg(feature = "auto-rig")]
        DeferredAction::LoadModelFromMemory { glb_data, source } => {
            match app.load_model_from_glb(&glb_data) {
                Ok(()) => {
                    log!(
                        "DeferredAction::LoadModelFromMemory: load OK, sending ModelLoadedFromMemory({:?})",
                        source
                    );
                    let mut ui_events = app.data.ecs_world.resource_mut::<UIEventQueue>();
                    ui_events.send(UIEvent::ModelLoadedFromMemory { source });
                }
                Err(e) => {
                    log_error!("Failed to load generated mesh: {}", e);
                    let mut state = app
                        .data
                        .ecs_world
                        .resource_mut::<crate::ecs::resource::TextToMeshState>();
                    state.status = crate::ecs::resource::TextToMeshStatus::Error;
                    state.error_message = Some(format!("Failed to load GLB: {}", e));
                }
            }
        }
    }
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
            let dump_wall_probe = app
                .data
                .ecs_world
                .get_resource::<crate::ecs::resource::BatchRun>()
                .map(|b| b.dump_wall_probe)
                .unwrap_or(false);
            if matches!(
                state,
                Some(crate::ecs::resource::BatchRunState::ScreenshotRequested)
            ) {
                app.rrdevice.device.device_wait_idle()?;
                let save_result = app.save_screenshot(image_index);
                crate::ecs::systems::batch_run_record_screenshot(
                    &app.data.ecs_world,
                    save_result.map_err(|e| format!("{e:?}")),
                );
                // Wall probe dump: if this batch run was started with
                // --batch-debug-action dump_wall_probe, dump synchronously
                // at the same frame/time as the screenshot.
                if dump_wall_probe {
                    crate::ecs::systems::flame_dump_systems::perform_flame_wall_probe_dump(
                        &app.data.ecs_world,
                        [1680.0, 840.0],
                    );
                }
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

fn process_platform_file_events(events: &[UIEvent], app: &mut App) -> Vec<DeferredAction> {
    let mut deferred = Vec::new();

    for event in events {
        match event {
            UIEvent::ClipBrowserLoadFromFile => {
                if let Some(action) = open_clip_load_dialog() {
                    deferred.push(action);
                }
            }
            UIEvent::ClipBrowserSaveToFile(source_id) => {
                if let Some(action) = open_clip_save_dialog(app, *source_id) {
                    deferred.push(action);
                }
            }
            UIEvent::ClipBrowserExportFbx(source_id) => handle_clip_export_fbx(app, *source_id),
            UIEvent::ClipBrowserExportGltf(source_id) => handle_clip_export_gltf(app, *source_id),
            UIEvent::ClipBrowserExportGltfAnimationOnly(source_id) => {
                handle_clip_export_gltf_animation_only(app, *source_id)
            }
            UIEvent::ExportModelGltf => handle_export_model_gltf(app),
            UIEvent::SpringBoneSaveBake => {
                if let Some(action) = open_spring_bone_save_dialog(app) {
                    deferred.push(action);
                }
            }
            _ => {}
        }
    }

    deferred
}

fn open_clip_load_dialog() -> Option<DeferredAction> {
    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .pick_file()?;

    Some(DeferredAction::LoadClipFromFile { path })
}

fn open_clip_save_dialog(app: &App, source_id: u64) -> Option<DeferredAction> {
    let current_name = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "clip".to_string())
    };

    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .set_file_name(format!("{}.anim.ron", current_name))
        .save_file()?;

    Some(DeferredAction::SaveClipToFile { source_id, path })
}

fn handle_clip_export_fbx(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app
        .data
        .ecs_assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.clone());

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };

    let default_filename = format!("{}.fbx", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("FBX Binary", &["fbx"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    let has_fbx_cache = app
        .data
        .ecs_world
        .contains_resource::<crate::ecs::resource::FbxModelCache>();
    let (fbx_model_ref, needs_coord_conversion) = if has_fbx_cache {
        let cache = app
            .data
            .ecs_world
            .resource::<crate::ecs::resource::FbxModelCache>();
        (cache.fbx_model().cloned(), cache.needs_coord_conversion())
    } else {
        (None, false)
    };

    let (axes, fps) = if let Some(ref fbx_model) = fbx_model_ref {
        (fbx_model.axes.clone(), fbx_model.fps)
    } else {
        (crate::loader::fbx::fbx::FbxAxesInfo::default(), 24.0)
    };

    let result = if let Some(ref fbx_model) = fbx_model_ref {
        crate::exporter::fbx_exporter::export_full_fbx(fbx_model, Some(&clip), &skeleton, &path)
    } else {
        crate::exporter::fbx_animation::export_animation_fbx(
            &clip,
            &skeleton,
            &path,
            needs_coord_conversion,
            axes,
            fps,
        )
    };

    match result {
        Ok(()) => msg_info!("FBX exported: {:?}", path),
        Err(e) => msg_error!("FBX export failed: {:?}", e),
    }
}

fn handle_clip_export_gltf(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app
        .data
        .ecs_assets
        .skeletons
        .values()
        .next()
        .map(|sa| sa.skeleton.clone());

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };

    let source_bytes = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::GltfModelCache>()
        .and_then(|cache| resolve_glb_bytes(&*cache));

    let Some(source_bytes) = source_bytes else {
        msg_error!("glTF export failed: no source glTF/GLB model loaded");
        return;
    };

    let default_filename = format!("{}.glb", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match crate::exporter::gltf::export_gltf_animation_from_bytes(
        &source_bytes,
        &clip,
        &skeleton,
        &path,
    ) {
        Ok(()) => msg_info!("glTF exported: {:?}", path),
        Err(e) => msg_error!("glTF export failed: {:?}", e),
    }
}

fn handle_clip_export_gltf_animation_only(app: &mut App, source_id: u64) {
    let clip = {
        let lib = app.data.ecs_world.resource::<ClipLibrary>();
        lib.get(source_id).cloned()
    };
    let skeleton = app.data.ecs_assets.skeletons.values().next();

    let (Some(clip), Some(skeleton)) = (clip, skeleton) else {
        return;
    };
    let default_filename = format!("{}_anim_only.glb", clip.name);
    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match crate::exporter::gltf::export_gltf_animation_only(&clip, &skeleton.skeleton, &path) {
        Ok(()) => msg_info!("Animation-only glTF exported: {:?}", path),
        Err(e) => msg_error!("Animation-only glTF export failed: {:?}", e),
    }
}

fn handle_export_model_gltf(app: &mut App) {
    let cache = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::GltfModelCache>();

    let glb_bytes = match cache {
        Some(c) => resolve_glb_bytes(&*c),
        None => None,
    };

    let Some(glb_bytes) = glb_bytes else {
        msg_error!("glTF export failed: no model data available");
        return;
    };

    let model_name = app
        .data
        .ecs_world
        .get_resource::<crate::ecs::resource::ModelState>()
        .map(|s| s.model_path.clone())
        .unwrap_or_else(|| "model".to_string());

    let default_filename = format!(
        "{}.glb",
        std::path::Path::new(&model_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
    );

    let path = rfd::FileDialog::new()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name(&default_filename)
        .save_file();

    let Some(path) = path else {
        return;
    };

    match std::fs::write(&path, &glb_bytes) {
        Ok(()) => msg_info!("Model exported: {:?}", path),
        Err(e) => msg_error!("Model export failed: {:?}", e),
    }
}

fn resolve_glb_bytes(cache: &crate::ecs::resource::GltfModelCache) -> Option<Vec<u8>> {
    if let Some(ref data) = cache.glb_data {
        return Some(data.clone());
    }

    if let Some(ref path) = cache.source_path {
        return std::fs::read(path).ok();
    }

    None
}

fn extract_clip_name_from_path(path: &std::path::Path) -> String {
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("clip");

    filename
        .strip_suffix(".anim.ron")
        .or_else(|| filename.strip_suffix(".ron"))
        .unwrap_or(filename)
        .to_string()
}

fn open_spring_bone_save_dialog(app: &App) -> Option<DeferredAction> {
    use crate::ecs::resource::SpringBoneState;

    let spring_state = app.data.ecs_world.resource::<SpringBoneState>();
    let baked_id = spring_state.baked_clip_source_id?;
    drop(spring_state);

    let path = rfd::FileDialog::new()
        .add_filter("Animation RON", &["anim.ron", "ron"])
        .set_file_name("spring_baked.anim.ron")
        .save_file()?;

    Some(DeferredAction::SaveSpringBoneBake { baked_id, path })
}
