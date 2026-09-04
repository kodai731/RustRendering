use crate::app::App;
#[cfg(debug_assertions)]
use crate::platform::ui::DebugWindowState;
use crate::platform::ui::{
    build_bottom_panel, build_clip_browser_window, build_curve_editor_window,
    build_hierarchy_window, build_inspector_window, build_scene_overlay, build_timeline_window,
    build_viewport_window, draw_status_bar, handle_splitters, LayoutSnapshot, SceneOverlayState,
    StatusBarState, SuggestionOverlay, ViewportInfo,
};

use crate::ecs::resource::{
    ClipBrowserState, ClipLibrary, CurveEditorBuffer, CurveEditorState, HierarchyState, MessageLog,
    PanelLayout, PoseLibrary, TimelineInteractionState, TimelineState, ViewportInput,
};
use crate::ecs::systems::clip_track_systems::query_clip_tracks;
use crate::ecs::UIEventQueue;

#[cfg(feature = "auto-rig")]
use crate::platform::ui::{build_text_to_animation_dialog, build_text_to_mesh_dialog};

pub(super) fn build_ui_windows(
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
        build_text_to_mesh_dialog(
            ui,
            &mut *ui_events,
            text_to_mesh_dialog,
            &app.data.ecs_world,
        );
        build_text_to_animation_dialog(
            ui,
            &mut *ui_events,
            text_to_animation_dialog,
            &app.data.ecs_world,
        );
    }

    consume_needs_focus(app);
}

pub(super) fn consume_needs_focus(app: &mut App) {
    let mut curve_editor = app.data.ecs_world.resource_mut::<CurveEditorState>();
    curve_editor.needs_focus = false;
}

pub(super) fn build_side_panel_windows(
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

pub(super) fn build_viewport_and_update_state(
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

pub(super) fn build_timeline_and_fixed_overlays(
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

pub(super) fn build_curve_editor(ui: &imgui::Ui, app: &mut App) {
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
    let suggestion_overlays: Vec<SuggestionOverlay> = {
        if let Some(state) = app
            .data
            .ecs_world
            .get_resource::<crate::ecs::resource::CurveSuggestionState>()
        {
            state
                .suggestions
                .iter()
                .map(|s| SuggestionOverlay {
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
    let suggestion_overlays: Vec<SuggestionOverlay> = Vec::new();

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
