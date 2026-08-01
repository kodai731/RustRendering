use std::collections::HashSet;

use imgui::Condition;

use crate::animation::editable::{
    curve_sample, sample_bezier, segment_uses_bezier, BezierHandle, EditableAnimationClip,
    EditableKeyframe, InterpolationType, KeyframeId, PropertyCurve, PropertyType, TangentType,
    TangentWeightMode,
};
use crate::animation::BoneId;
use crate::ecs::component::FlameParam;
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{
    ClipLibrary, CurveEditorBuffer, CurveEditorState, CurveEditorTarget, CurveInteractionMode,
    CurveSelectedKeyframe, CurveTrackRef, DraggingTangent, PoseLibrary, TangentHandleType,
    TimelineState,
};

pub struct SuggestionOverlay {
    pub property_type: PropertyType,
    pub time: f32,
    pub value: f32,
    pub tangent_in: (f32, f32),
    pub tangent_out: (f32, f32),
    pub confidence: f32,
}

const MIN_WINDOW_WIDTH: f32 = 400.0;
const MIN_WINDOW_HEIGHT: f32 = 300.0;
const TRACK_LIST_WIDTH: f32 = 180.0;
const TIME_RULER_HEIGHT: f32 = 30.0;
const CURVE_PADDING: f32 = 10.0;
const KEYFRAME_HIT_RADIUS: f32 = 8.0;
const Y_AXIS_WIDTH: f32 = 50.0;
const PAN_SPEED: f32 = 30.0;

const ALL_PROPERTY_TYPES: &[(PropertyType, [f32; 4], &str)] = &[
    (PropertyType::TranslationX, [1.0, 0.3, 0.3, 1.0], "Pos.X"),
    (PropertyType::TranslationY, [0.3, 1.0, 0.3, 1.0], "Pos.Y"),
    (PropertyType::TranslationZ, [0.3, 0.3, 1.0, 1.0], "Pos.Z"),
    (PropertyType::RotationX, [1.0, 0.6, 0.6, 1.0], "Rot.X"),
    (PropertyType::RotationY, [0.6, 1.0, 0.6, 1.0], "Rot.Y"),
    (PropertyType::RotationZ, [0.6, 0.6, 1.0, 1.0], "Rot.Z"),
    (PropertyType::ScaleX, [1.0, 0.8, 0.4, 1.0], "Scl.X"),
    (PropertyType::ScaleY, [0.8, 1.0, 0.4, 1.0], "Scl.Y"),
    (PropertyType::ScaleZ, [0.4, 0.8, 1.0, 1.0], "Scl.Z"),
];

struct ViewTransform {
    curve_origin: [f32; 2],
    curve_width: f32,
    curve_height: f32,
    duration: f32,
    val_range: f32,
    zoom_x: f32,
    zoom_y: f32,
    view_time_offset: f32,
    view_value_offset: f32,
}

impl ViewTransform {
    fn time_to_x(&self, time: f32) -> f32 {
        self.curve_origin[0]
            + (time - self.view_time_offset) / self.duration.max(0.001)
                * self.zoom_x
                * self.curve_width
    }

    fn value_to_y(&self, value: f32) -> f32 {
        self.curve_origin[1] + self.curve_height
            - (value - self.view_value_offset) / self.val_range.max(0.001)
                * self.zoom_y
                * self.curve_height
    }

    fn x_to_time(&self, x: f32) -> f32 {
        (x - self.curve_origin[0]) / (self.zoom_x * self.curve_width).max(0.001)
            * self.duration.max(0.001)
            + self.view_time_offset
    }

    fn y_to_value(&self, y: f32) -> f32 {
        (self.curve_origin[1] + self.curve_height - y)
            / (self.zoom_y * self.curve_height).max(0.001)
            * self.val_range.max(0.001)
            + self.view_value_offset
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionModifier {
    None,
    Toggle,
    Range,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleasedButton {
    Left,
    Middle,
}

pub fn build_curve_editor_window(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    timeline_state: &TimelineState,
    clip_library: &ClipLibrary,
    editor_state: &mut CurveEditorState,
    curve_buffer: &CurveEditorBuffer,
    suggestion_overlays: &[SuggestionOverlay],
    pose_library: &mut PoseLibrary,
    is_flame_clip: bool,
) {
    if !editor_state.is_open {
        return;
    }

    let display_size = ui.io().display_size;
    let initial_pos = [
        (display_size[0] - editor_state.window_size[0]) * 0.5,
        (display_size[1] - editor_state.window_size[1]) * 0.5,
    ];

    let mut is_open = editor_state.is_open;
    let should_focus = editor_state.needs_focus;

    let mut window = ui
        .window("Curve Editor")
        .position(initial_pos, Condition::FirstUseEver)
        .size(editor_state.window_size, Condition::FirstUseEver)
        .size_constraints(
            [MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT],
            [display_size[0], display_size[1]],
        )
        .bg_alpha(1.0)
        .opened(&mut is_open);

    if should_focus {
        window = window.focused(true);
    }

    window.build(|| {
        editor_state.window_size = ui.window_size();

        let content_region = ui.content_region_avail();

        ui.child_window("left_panel")
            .size([TRACK_LIST_WIDTH, content_region[1]])
            .border(true)
            .build(|| {
                build_track_list(
                    ui,
                    ui_events,
                    timeline_state,
                    clip_library,
                    editor_state,
                    is_flame_clip,
                );
            });

        ui.same_line();

        let curve_view_width = content_region[0] - TRACK_LIST_WIDTH - 10.0;
        ui.child_window("curve_view")
            .size([curve_view_width, content_region[1]])
            .border(true)
            .build(|| {
                build_curve_view(
                    ui,
                    ui_events,
                    timeline_state,
                    clip_library,
                    editor_state,
                    curve_buffer,
                    suggestion_overlays,
                    pose_library,
                );
            });
    });

    editor_state.is_open = is_open;
}

fn get_current_clip<'a>(
    timeline_state: &TimelineState,
    clip_library: &'a ClipLibrary,
) -> Option<&'a EditableAnimationClip> {
    timeline_state
        .current_clip_id
        .and_then(|id| clip_library.get(id))
}

fn build_track_list(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    timeline_state: &TimelineState,
    clip_library: &ClipLibrary,
    editor_state: &mut CurveEditorState,
    is_flame_clip: bool,
) {
    let Some(clip) = get_current_clip(timeline_state, clip_library) else {
        ui.text("No clip selected");
        return;
    };

    if is_flame_clip || !clip.scalar_curves.is_empty() {
        ui.text("Effects:");
        ui.separator();
        let is_selected = editor_state.selected_target == Some(CurveEditorTarget::Scalars);
        if ui.selectable_config("Flame").selected(is_selected).build() {
            editor_state.select_scalars();
            for curve in &clip.scalar_curves {
                editor_state.visible_curves.insert(curve.property_type);
            }
            editor_state.view_initialized = false;
        }
        if is_selected {
            build_scalar_curve_selector_inline(ui, ui_events, clip, editor_state);
        }
        ui.spacing();
    }

    ui.text("Bones:");
    ui.separator();

    let mut sorted_bone_ids: Vec<BoneId> = clip.tracks.keys().copied().collect();
    sorted_bone_ids.sort();

    for bone_id in sorted_bone_ids {
        if let Some(track) = clip.tracks.get(&bone_id) {
            let is_selected = editor_state.selected_bone_id() == Some(bone_id);
            let is_spring_bone = timeline_state.baked_bone_ids.contains(&bone_id);
            let label = if is_spring_bone {
                let name = if track.bone_name.len() > 13 {
                    &track.bone_name[..10]
                } else {
                    &track.bone_name
                };
                format!("[SB] {}", name)
            } else if track.bone_name.len() > 18 {
                format!("{}...", &track.bone_name[..15])
            } else {
                track.bone_name.clone()
            };

            if ui.selectable_config(&label).selected(is_selected).build() {
                editor_state.select_bone(bone_id);
                editor_state.view_initialized = false;
            }

            if is_selected {
                build_curve_selector_inline(ui, track, editor_state);
            }
        }
    }
}

/// Every flame param gets a row, keyed or not: `+` inserts a key at the
/// playhead with the flame's current value (creating the curve on first use),
/// so an empty clip opened in the editor still offers a keying path.
fn build_scalar_curve_selector_inline(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    clip: &EditableAnimationClip,
    editor_state: &mut CurveEditorState,
) {
    ui.indent();

    for param in FlameParam::ALL {
        let property_type = param.property_type();
        let key_count = clip
            .get_scalar_curve(property_type)
            .map(|curve| curve.keyframes.len())
            .unwrap_or(0);

        let (color, name) = scalar_curve_style(property_type);
        let mut visible = editor_state.visible_curves.contains(&property_type);
        ui.text_colored(color, "\u{25CF}");
        ui.same_line();
        let label = if key_count > 0 {
            format!("{name} ({key_count})")
        } else {
            name.to_string()
        };
        if ui.checkbox(&label, &mut visible) {
            if visible {
                editor_state.visible_curves.insert(property_type);
            } else {
                editor_state.visible_curves.remove(&property_type);
            }
        }
        ui.same_line();
        if ui.small_button(&format!("+##key_{}", param.cli_name())) {
            ui_events.send(UIEvent::InsertFlameKeyAtPlayhead { param });
            editor_state.visible_curves.insert(property_type);
        }
        if ui.is_item_hovered() {
            ui.tooltip_text("Insert key at playhead (current value)");
        }
    }

    if ui.small_button("All##scalar") {
        for param in FlameParam::ALL {
            editor_state.visible_curves.insert(param.property_type());
        }
    }
    ui.same_line();
    if ui.small_button("None##scalar") {
        for param in FlameParam::ALL {
            editor_state.visible_curves.remove(&param.property_type());
        }
    }

    ui.unindent();
    ui.spacing();
}

fn build_curve_selector_inline(
    ui: &imgui::Ui,
    track: &crate::animation::editable::BoneTrack,
    editor_state: &mut CurveEditorState,
) {
    ui.indent();

    for (prop_type, color, name) in ALL_PROPERTY_TYPES {
        let curve = track.get_curve(*prop_type);
        if curve.is_empty() {
            continue;
        }

        let mut visible = editor_state.visible_curves.contains(prop_type);
        ui.text_colored(*color, "\u{25CF}");
        ui.same_line();
        if ui.checkbox(name, &mut visible) {
            if visible {
                editor_state.visible_curves.insert(*prop_type);
            } else {
                editor_state.visible_curves.remove(prop_type);
            }
        }
    }

    if ui.small_button("All") {
        for (prop_type, _, _) in ALL_PROPERTY_TYPES {
            editor_state.visible_curves.insert(*prop_type);
        }
    }
    ui.same_line();
    if ui.small_button("None") {
        editor_state.visible_curves.clear();
    }

    ui.unindent();
    ui.spacing();
}

fn build_curve_view(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    timeline_state: &TimelineState,
    clip_library: &ClipLibrary,
    editor_state: &mut CurveEditorState,
    curve_buffer: &CurveEditorBuffer,
    suggestion_overlays: &[SuggestionOverlay],
    pose_library: &mut PoseLibrary,
) {
    build_curve_toolbar(ui, ui_events, curve_buffer, pose_library, clip_library);
    ui.separator();

    let Some(clip) = get_current_clip(timeline_state, clip_library) else {
        ui.text("No clip selected");
        return;
    };

    let curves_to_draw = match editor_state.selected_target {
        Some(CurveEditorTarget::Bone(bone_id)) => {
            let Some(track) = clip.tracks.get(&bone_id) else {
                ui.text("Track not found");
                return;
            };
            collect_visible_curves(track, editor_state)
        }
        Some(CurveEditorTarget::Scalars) => collect_visible_scalar_curves(clip, editor_state),
        None => {
            ui.text("Select a track from the list");
            return;
        }
    };
    let Some(track_ref) = editor_state.selected_track_ref() else {
        return;
    };

    let content_region = ui.content_region_avail();
    let curve_area_width = content_region[0] - Y_AXIS_WIDTH - CURVE_PADDING * 2.0;
    let curve_area_height = content_region[1] - TIME_RULER_HEIGHT - CURVE_PADDING * 2.0;

    if curve_area_width <= 0.0 || curve_area_height <= 0.0 {
        return;
    }
    initialize_view_range(editor_state, &curves_to_draw, clip.duration);
    let cursor_pos = ui.cursor_screen_pos();

    let curve_origin = [
        cursor_pos[0] + Y_AXIS_WIDTH + CURVE_PADDING,
        cursor_pos[1] + TIME_RULER_HEIGHT + CURVE_PADDING,
    ];

    let vt = ViewTransform {
        curve_origin,
        curve_width: curve_area_width,
        curve_height: curve_area_height,
        duration: editor_state.view_duration,
        val_range: editor_state.view_val_range,
        zoom_x: editor_state.zoom_x,
        zoom_y: editor_state.zoom_y,
        view_time_offset: editor_state.view_time_offset,
        view_value_offset: editor_state.view_value_offset,
    };

    draw_curve_area(
        ui,
        &vt,
        cursor_pos,
        curve_area_width,
        curve_area_height,
        timeline_state,
        editor_state,
        &curves_to_draw,
        curve_buffer,
        suggestion_overlays,
        track_ref,
        pose_library,
    );

    let total_width = Y_AXIS_WIDTH + CURVE_PADDING + curve_area_width + CURVE_PADDING;
    let total_height = TIME_RULER_HEIGHT + CURVE_PADDING + curve_area_height + CURVE_PADDING;

    ui.set_cursor_screen_pos([cursor_pos[0], cursor_pos[1]]);
    ui.invisible_button("curve_interaction_area", [total_width, total_height]);

    handle_curve_view_interaction(
        ui,
        ui_events,
        editor_state,
        &vt,
        &curves_to_draw,
        cursor_pos,
        curve_area_width,
        clip.duration,
        track_ref,
    );

    ui.set_cursor_screen_pos([cursor_pos[0], cursor_pos[1] + total_height]);

    #[cfg(feature = "ml")]
    if let Some(bone_id) = track_ref.bone_id() {
        handle_suggestion_keyboard(ui, ui_events, bone_id, editor_state, suggestion_overlays);
    }
}

fn collect_visible_scalar_curves<'a>(
    clip: &'a EditableAnimationClip,
    editor_state: &CurveEditorState,
) -> Vec<(&'a PropertyCurve, [f32; 4], &'static str)> {
    let mut curves = Vec::new();
    for curve in &clip.scalar_curves {
        if !editor_state.visible_curves.contains(&curve.property_type) {
            continue;
        }
        if curve.is_empty() {
            continue;
        }
        let (color, name) = scalar_curve_style(curve.property_type);
        curves.push((curve, color, name));
    }
    curves
}

fn scalar_curve_style(property_type: PropertyType) -> ([f32; 4], &'static str) {
    match FlameParam::from_property_type(property_type) {
        Some(param) => (flame_param_color(param), param.display_name()),
        None => ([0.6, 0.6, 0.6, 1.0], "Custom"),
    }
}

fn flame_param_color(param: FlameParam) -> [f32; 4] {
    // Evenly spaced hues over the 16 params, alternating brightness for
    // neighbor separability.
    let index = param.code() as f32;
    let hue = index / FlameParam::ALL.len() as f32;
    let value = if param.code() % 2 == 0 { 1.0 } else { 0.75 };
    hsv_to_rgba(hue, 0.75, value)
}

fn hsv_to_rgba(h: f32, s: f32, v: f32) -> [f32; 4] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

fn collect_visible_curves<'a>(
    track: &'a crate::animation::editable::BoneTrack,
    editor_state: &CurveEditorState,
) -> Vec<(&'a PropertyCurve, [f32; 4], &'static str)> {
    let mut curves = Vec::new();
    for (prop_type, color, name) in ALL_PROPERTY_TYPES {
        if editor_state.visible_curves.contains(prop_type) {
            let curve = track.get_curve(*prop_type);
            if !curve.is_empty() {
                curves.push((curve, *color, *name));
            }
        }
    }
    curves
}

fn initialize_view_range(
    editor_state: &mut CurveEditorState,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    clip_duration: f32,
) {
    let (global_min, global_max) = calculate_global_value_range(curves_to_draw);
    let display_duration = clip_duration;

    if !editor_state.view_initialized {
        editor_state.view_value_offset = global_min;
        editor_state.view_val_range = global_max - global_min;
        editor_state.view_duration = display_duration;
        editor_state.view_time_offset = 0.0;
        editor_state.zoom_x = 1.0;
        editor_state.zoom_y = 1.0;
        editor_state.view_initialized = true;
    } else {
        editor_state.view_duration = editor_state.view_duration.max(display_duration);
    }
}

fn draw_curve_area(
    ui: &imgui::Ui,
    vt: &ViewTransform,
    cursor_pos: [f32; 2],
    curve_area_width: f32,
    curve_area_height: f32,
    timeline_state: &TimelineState,
    editor_state: &CurveEditorState,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    curve_buffer: &CurveEditorBuffer,
    suggestion_overlays: &[SuggestionOverlay],
    track_ref: CurveTrackRef,
    pose_library: &PoseLibrary,
) {
    let draw_list = ui.get_window_draw_list();

    let y_axis_origin = [
        cursor_pos[0],
        cursor_pos[1] + TIME_RULER_HEIGHT + CURVE_PADDING,
    ];
    draw_y_axis_labels(
        &draw_list,
        y_axis_origin,
        Y_AXIS_WIDTH,
        curve_area_height,
        vt,
    );

    let ruler_pos = [cursor_pos[0] + Y_AXIS_WIDTH + CURVE_PADDING, cursor_pos[1]];
    draw_time_ruler(&draw_list, ruler_pos, curve_area_width, vt);

    let co = vt.curve_origin;
    draw_list
        .add_rect(
            co,
            [co[0] + curve_area_width, co[1] + curve_area_height],
            [0.12, 0.12, 0.15, 1.0],
        )
        .filled(true)
        .build();

    draw_list.with_clip_rect_intersect(
        co,
        [co[0] + curve_area_width, co[1] + curve_area_height],
        || {
            draw_clipped_curve_content(
                ui,
                &draw_list,
                vt,
                curve_area_width,
                curve_area_height,
                timeline_state,
                editor_state,
                curves_to_draw,
                curve_buffer,
                suggestion_overlays,
                track_ref,
                pose_library,
            );
        },
    );
}

fn draw_clipped_curve_content(
    ui: &imgui::Ui,
    draw_list: &imgui::DrawListMut,
    vt: &ViewTransform,
    curve_area_width: f32,
    curve_area_height: f32,
    timeline_state: &TimelineState,
    editor_state: &CurveEditorState,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    curve_buffer: &CurveEditorBuffer,
    suggestion_overlays: &[SuggestionOverlay],
    track_ref: CurveTrackRef,
    pose_library: &PoseLibrary,
) {
    draw_grid(draw_list, curve_area_width, curve_area_height, vt);

    let sample_count = calculate_sample_count(curve_area_width);
    for (curve, color, _name) in curves_to_draw {
        draw_curve_with_keyframes(draw_list, curve, *color, sample_count, vt);
    }

    if !editor_state.selected_keyframes.is_empty() {
        draw_selected_keyframes_highlight(
            draw_list,
            curves_to_draw,
            &editor_state.selected_keyframes,
            vt,
        );
        draw_tangent_handles(
            draw_list,
            curves_to_draw,
            &editor_state.selected_keyframes,
            vt,
        );
    }

    draw_pose_markers(draw_list, vt, curve_area_height, pose_library);

    let playhead_x = vt.time_to_x(timeline_state.current_time);
    draw_list
        .add_line(
            [playhead_x, vt.curve_origin[1]],
            [playhead_x, vt.curve_origin[1] + curve_area_height],
            [1.0, 0.2, 0.2, 1.0],
        )
        .thickness(2.0)
        .build();

    if matches!(
        editor_state.interaction,
        CurveInteractionMode::DraggingKeyframe
    ) {
        draw_keyframe_drag_preview(
            draw_list,
            ui.io().mouse_pos,
            editor_state.drag_start_mouse_pos,
            vt,
            curves_to_draw,
            &editor_state.selected_keyframes,
        );
    }

    if let CurveInteractionMode::DraggingTangent(ref dragging) = editor_state.interaction {
        draw_tangent_drag_curve_preview(draw_list, dragging, ui.io().mouse_pos, curves_to_draw, vt);
    }

    if let Some(bone_id) = track_ref.bone_id() {
        draw_buffer_curve_overlay(
            draw_list,
            curve_buffer,
            bone_id,
            &editor_state.visible_curves,
            vt,
        );
    }

    draw_suggestion_curve_overlay(
        draw_list,
        suggestion_overlays,
        &editor_state.visible_curves,
        vt,
    );
}

fn handle_curve_view_interaction(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    editor_state: &mut CurveEditorState,
    vt: &ViewTransform,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    cursor_pos: [f32; 2],
    curve_area_width: f32,
    clip_duration: f32,
    track_ref: CurveTrackRef,
) {
    let ruler_pos = [cursor_pos[0] + Y_AXIS_WIDTH + CURVE_PADDING, cursor_pos[1]];

    handle_mouse_interaction(
        ui,
        ui_events,
        editor_state,
        vt,
        curves_to_draw,
        ruler_pos,
        curve_area_width,
        clip_duration,
    );

    if ui.is_item_hovered() && ui.is_mouse_clicked(imgui::MouseButton::Right) {
        let mouse_pos = ui.io().mouse_pos;
        if let Some(hit) = find_keyframe_at_position(mouse_pos, curves_to_draw, vt) {
            editor_state.context_menu_keyframe = Some(CurveSelectedKeyframe {
                property_type: hit.0,
                keyframe_id: hit.1,
                original_time: hit.2,
                original_value: hit.3,
            });
            ui.open_popup("keyframe_context_menu");
        } else {
            editor_state.context_menu_click_time = vt.x_to_time(mouse_pos[0]);
            editor_state.context_menu_click_value = vt.y_to_value(mouse_pos[1]);
            ui.open_popup("curve_editor_context_menu");
        }
    }

    build_keyframe_context_menu(ui, ui_events, editor_state, track_ref);
    build_curve_editor_context_menu(ui, ui_events, editor_state, track_ref);
}

fn build_keyframe_context_menu(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    editor_state: &mut CurveEditorState,
    track_ref: CurveTrackRef,
) {
    ui.popup("keyframe_context_menu", || {
        let ctx_kf = match editor_state.context_menu_keyframe.clone() {
            Some(kf) => kf,
            None => return,
        };

        if ui.selectable_config("Delete Key").build() {
            if editor_state.selected_keyframes.len() > 1 {
                for sel in &editor_state.selected_keyframes {
                    ui_events.send(UIEvent::TimelineDeleteKeyframe {
                        track: track_ref,
                        property_type: sel.property_type.clone(),
                        keyframe_id: sel.keyframe_id,
                    });
                }
                editor_state.selected_keyframes.clear();
                editor_state.selection_anchor = None;
            } else {
                ui_events.send(UIEvent::TimelineDeleteKeyframe {
                    track: track_ref,
                    property_type: ctx_kf.property_type.clone(),
                    keyframe_id: ctx_kf.keyframe_id,
                });
                editor_state.selected_keyframes.clear();
                editor_state.selection_anchor = None;
            }
        }

        let section_color = [0.6, 0.8, 1.0, 1.0];

        ui.separator();
        ui.text_colored(section_color, "Interpolation");
        ui.separator();

        if ui.selectable_config("  Linear").build() {
            ui_events.send(UIEvent::TimelineSetKeyframeInterpolation {
                track: track_ref,
                property_type: ctx_kf.property_type,
                keyframe_id: ctx_kf.keyframe_id,
                interpolation: InterpolationType::Linear,
            });
        }

        if ui.selectable_config("  Bezier").build() {
            ui_events.send(UIEvent::TimelineSetKeyframeInterpolation {
                track: track_ref,
                property_type: ctx_kf.property_type,
                keyframe_id: ctx_kf.keyframe_id,
                interpolation: InterpolationType::Bezier,
            });
        }

        if ui.selectable_config("  Stepped").build() {
            ui_events.send(UIEvent::TimelineSetKeyframeInterpolation {
                track: track_ref,
                property_type: ctx_kf.property_type,
                keyframe_id: ctx_kf.keyframe_id,
                interpolation: InterpolationType::Stepped,
            });
        }

        ui.spacing();
        ui.text_colored(section_color, "Tangent");
        ui.separator();

        let tangent_options = [
            ("  Spline", TangentType::Spline),
            ("  Linear", TangentType::Linear),
            ("  Flat", TangentType::Flat),
            ("  Clamped", TangentType::Clamped),
            ("  Plateau", TangentType::Plateau),
            ("  Manual", TangentType::Manual),
        ];

        for (label, tangent_type) in &tangent_options {
            if ui.selectable_config(label).build() {
                ui_events.send(UIEvent::TimelineSetTangentType {
                    track: track_ref,
                    property_type: ctx_kf.property_type,
                    keyframe_id: ctx_kf.keyframe_id,
                    tangent_type: *tangent_type,
                });
            }
        }

        ui.spacing();
        ui.text_colored(section_color, "Weight");
        ui.separator();

        if ui.selectable_config("  Non-Weighted").build() {
            ui_events.send(UIEvent::TimelineSetTangentWeightMode {
                track: track_ref,
                property_type: ctx_kf.property_type,
                keyframe_id: ctx_kf.keyframe_id,
                weight_mode: TangentWeightMode::NonWeighted,
            });
        }

        if ui.selectable_config("  Weighted").build() {
            ui_events.send(UIEvent::TimelineSetTangentWeightMode {
                track: track_ref,
                property_type: ctx_kf.property_type,
                keyframe_id: ctx_kf.keyframe_id,
                weight_mode: TangentWeightMode::Weighted,
            });
        }
    });
}

fn build_curve_editor_context_menu(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    editor_state: &CurveEditorState,
    track_ref: CurveTrackRef,
) {
    ui.popup("curve_editor_context_menu", || {
        if ui.selectable_config("Add Key").build() {
            if let Some(property_type) = add_key_target_property(editor_state, track_ref) {
                ui_events.send(UIEvent::TimelineAddKeyframe {
                    track: track_ref,
                    property_type,
                    time: editor_state.context_menu_click_time.max(0.0),
                    value: editor_state.context_menu_click_value,
                });
            }
        }
    });
}

/// The scalar target only accepts flame params: the visible set can still hold
/// bone property types from a previous bone target, and letting one through
/// would create a curve no flame param answers to (grey "Custom", never
/// sampled). Lowest code wins so the choice is deterministic.
fn add_key_target_property(
    editor_state: &CurveEditorState,
    track_ref: CurveTrackRef,
) -> Option<PropertyType> {
    match track_ref {
        CurveTrackRef::Scalar => editor_state
            .visible_curves
            .iter()
            .filter_map(|p| FlameParam::from_property_type(*p))
            .min_by_key(|p| p.code())
            .map(|p| p.property_type()),
        CurveTrackRef::Bone(_) => editor_state.visible_curves.iter().copied().next(),
    }
}

fn handle_mouse_interaction(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    editor_state: &mut CurveEditorState,
    vt: &ViewTransform,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    ruler_pos: [f32; 2],
    curve_area_width: f32,
    duration: f32,
) {
    let is_hovered = ui.is_item_hovered();
    let mouse_pos = ui.io().mouse_pos;
    let mouse_clicked = ui.is_mouse_clicked(imgui::MouseButton::Left);
    let mouse_down = ui.io().mouse_down[0];
    let mouse_released = ui.is_mouse_released(imgui::MouseButton::Left);
    let middle_clicked = ui.is_mouse_clicked(imgui::MouseButton::Middle);
    let middle_released = ui.is_mouse_released(imgui::MouseButton::Middle);

    let in_ruler_area = mouse_pos[0] >= ruler_pos[0]
        && mouse_pos[0] <= ruler_pos[0] + curve_area_width
        && mouse_pos[1] >= ruler_pos[1]
        && mouse_pos[1] <= ruler_pos[1] + TIME_RULER_HEIGHT;

    let in_curve_area = mouse_pos[0] >= vt.curve_origin[0]
        && mouse_pos[0] <= vt.curve_origin[0] + vt.curve_width
        && mouse_pos[1] >= vt.curve_origin[1]
        && mouse_pos[1] <= vt.curve_origin[1] + vt.curve_height;

    if mouse_released {
        handle_mouse_release(
            ui_events,
            editor_state,
            vt,
            mouse_pos,
            ReleasedButton::Left,
            curves_to_draw,
        );
    }
    if middle_released {
        handle_mouse_release(
            ui_events,
            editor_state,
            vt,
            mouse_pos,
            ReleasedButton::Middle,
            curves_to_draw,
        );
    }

    if is_hovered && mouse_clicked && in_ruler_area {
        editor_state.interaction = CurveInteractionMode::ScrubbingRuler;
        let time = vt.x_to_time(mouse_pos[0]).clamp(0.0, duration);
        ui_events.send(UIEvent::TimelineSetTime(time));
    }

    if is_hovered
        && mouse_clicked
        && in_curve_area
        && matches!(editor_state.interaction, CurveInteractionMode::Idle)
    {
        let modifier = if ui.io().key_ctrl {
            SelectionModifier::Toggle
        } else if ui.io().key_shift {
            SelectionModifier::Range
        } else {
            SelectionModifier::None
        };
        handle_curve_area_click(editor_state, mouse_pos, curves_to_draw, vt, modifier);
    }

    if is_hovered && middle_clicked && in_curve_area {
        editor_state.interaction = CurveInteractionMode::Panning {
            start_mouse_pos: mouse_pos,
            start_offset: [
                editor_state.view_time_offset,
                editor_state.view_value_offset,
            ],
        };
    }

    if matches!(
        editor_state.interaction,
        CurveInteractionMode::Panning { .. }
    ) && ui.io().mouse_down[2]
    {
        handle_panning(editor_state, mouse_pos, vt);
    }

    if matches!(
        editor_state.interaction,
        CurveInteractionMode::ScrubbingRuler
    ) && mouse_down
    {
        let time = vt.x_to_time(mouse_pos[0]).clamp(0.0, duration);
        ui_events.send(UIEvent::TimelineSetTime(time));
    }

    if is_hovered {
        handle_wheel_input(ui, editor_state, mouse_pos, vt);
    }
}

fn handle_mouse_release(
    ui_events: &mut UIEventQueue,
    editor_state: &mut CurveEditorState,
    vt: &ViewTransform,
    mouse_pos: [f32; 2],
    button: ReleasedButton,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
) {
    match button {
        ReleasedButton::Left => {
            if let CurveInteractionMode::DraggingTangent(ref dragging) =
                editor_state.interaction.clone()
            {
                if let Some(track_ref) = editor_state.selected_track_ref() {
                    let (in_tangent, out_tangent) =
                        compute_dragged_tangent(dragging, mouse_pos, curves_to_draw, vt);
                    ui_events.send(UIEvent::TimelineSetKeyframeTangent {
                        track: track_ref,
                        property_type: dragging.property_type,
                        keyframe_id: dragging.keyframe_id,
                        in_tangent,
                        out_tangent,
                    });
                }
            } else if matches!(
                editor_state.interaction,
                CurveInteractionMode::DraggingKeyframe
            ) {
                if let Some(track_ref) = editor_state.selected_track_ref() {
                    let time_delta = vt.x_to_time(mouse_pos[0])
                        - vt.x_to_time(editor_state.drag_start_mouse_pos[0]);
                    let value_delta = vt.y_to_value(mouse_pos[1])
                        - vt.y_to_value(editor_state.drag_start_mouse_pos[1]);

                    for sel in &editor_state.selected_keyframes {
                        ui_events.send(UIEvent::TimelineMoveKeyframe {
                            track: track_ref,
                            property_type: sel.property_type.clone(),
                            keyframe_id: sel.keyframe_id,
                            new_time: (sel.original_time + time_delta).max(0.0),
                            new_value: sel.original_value + value_delta,
                        });
                    }
                }
            }
            editor_state.interaction = CurveInteractionMode::Idle;
        }

        ReleasedButton::Middle => {
            editor_state.interaction = CurveInteractionMode::Idle;
        }
    }
}

fn compute_dragged_tangent(
    dragging: &DraggingTangent,
    mouse_pos: [f32; 2],
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    vt: &ViewTransform,
) -> (BezierHandle, BezierHandle) {
    for (curve, _, _) in curves_to_draw {
        if curve.property_type != dragging.property_type {
            continue;
        }

        let kf = match curve.get_keyframe(dragging.keyframe_id) {
            Some(kf) => kf,
            None => break,
        };

        let mouse_time = vt.x_to_time(mouse_pos[0]);
        let mouse_value = vt.y_to_value(mouse_pos[1]);
        let time_offset = mouse_time - kf.time;
        let value_offset = mouse_value - kf.value;
        let new_handle = BezierHandle::new(time_offset, value_offset);

        return match dragging.handle_type {
            TangentHandleType::In => (new_handle, kf.out_tangent.clone()),
            TangentHandleType::Out => (kf.in_tangent.clone(), new_handle),
        };
    }

    (BezierHandle::linear(), BezierHandle::linear())
}

fn handle_curve_area_click(
    editor_state: &mut CurveEditorState,
    mouse_pos: [f32; 2],
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    vt: &ViewTransform,
    modifier: SelectionModifier,
) {
    if let Some((handle_type, property_type, keyframe_id, original_handle)) =
        find_tangent_handle_at_position(
            mouse_pos,
            curves_to_draw,
            &editor_state.selected_keyframes,
            vt,
        )
    {
        editor_state.interaction = CurveInteractionMode::DraggingTangent(DraggingTangent {
            property_type,
            keyframe_id,
            handle_type,
            original_handle,
        });
        editor_state.drag_start_mouse_pos = mouse_pos;
        return;
    }

    let hit_keyframe = find_keyframe_at_position(mouse_pos, curves_to_draw, vt);

    if let Some((property_type, keyframe_id, time, value)) = hit_keyframe {
        let new_selected = CurveSelectedKeyframe {
            property_type: property_type.clone(),
            keyframe_id,
            original_time: time,
            original_value: value,
        };

        let should_drag = match modifier {
            SelectionModifier::Toggle => {
                let existing = editor_state
                    .selected_keyframes
                    .iter()
                    .position(|s| s.keyframe_id == keyframe_id && s.property_type == property_type);

                let drag = if let Some(pos) = existing {
                    editor_state.selected_keyframes.remove(pos);
                    false
                } else {
                    editor_state.selected_keyframes.push(new_selected.clone());
                    true
                };
                editor_state.selection_anchor = Some((property_type, keyframe_id));
                drag
            }

            SelectionModifier::Range => apply_shift_range_selection(
                editor_state,
                curves_to_draw,
                &property_type,
                keyframe_id,
                time,
                value,
            ),

            SelectionModifier::None => {
                let already_selected = editor_state
                    .selected_keyframes
                    .iter()
                    .any(|s| s.keyframe_id == keyframe_id && s.property_type == property_type);

                if !already_selected {
                    editor_state.selected_keyframes.clear();
                    editor_state.selected_keyframes.push(new_selected.clone());
                    editor_state.selection_anchor = Some((property_type, keyframe_id));
                }
                true
            }
        };

        if should_drag {
            refresh_selected_keyframe_positions(
                &mut editor_state.selected_keyframes,
                curves_to_draw,
            );
            editor_state.interaction = CurveInteractionMode::DraggingKeyframe;
            editor_state.drag_start_mouse_pos = mouse_pos;
        }
    } else if modifier == SelectionModifier::None {
        editor_state.selected_keyframes.clear();
        editor_state.selection_anchor = None;
    }
}

fn refresh_selected_keyframe_positions(
    selected: &mut [CurveSelectedKeyframe],
    curves: &[(&PropertyCurve, [f32; 4], &str)],
) {
    for sel in selected.iter_mut() {
        for (curve, _, _) in curves {
            if curve.property_type != sel.property_type {
                continue;
            }
            if let Some(kf) = curve.get_keyframe(sel.keyframe_id) {
                sel.original_time = kf.time;
                sel.original_value = kf.value;
            }
            break;
        }
    }
}

fn apply_shift_range_selection(
    editor_state: &mut CurveEditorState,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    property_type: &PropertyType,
    keyframe_id: KeyframeId,
    time: f32,
    value: f32,
) -> bool {
    let anchor = editor_state.selection_anchor.clone();

    if let Some((anchor_prop, anchor_id)) = anchor {
        if anchor_prop == *property_type {
            let anchor_time = find_keyframe_time(curves_to_draw, &anchor_prop, anchor_id);
            if let Some(anchor_time) = anchor_time {
                let range_keys =
                    collect_keyframes_in_range(curves_to_draw, property_type, anchor_time, time);
                for key in range_keys {
                    let already_exists = editor_state.selected_keyframes.iter().any(|s| {
                        s.keyframe_id == key.keyframe_id && s.property_type == key.property_type
                    });
                    if !already_exists {
                        editor_state.selected_keyframes.push(key);
                    }
                }
                return true;
            }
        }
    }

    let already_exists = editor_state
        .selected_keyframes
        .iter()
        .any(|s| s.keyframe_id == keyframe_id && s.property_type == *property_type);
    if !already_exists {
        editor_state.selected_keyframes.push(CurveSelectedKeyframe {
            property_type: property_type.clone(),
            keyframe_id,
            original_time: time,
            original_value: value,
        });
    }
    editor_state.selection_anchor = Some((property_type.clone(), keyframe_id));
    true
}

fn find_keyframe_time(
    curves: &[(&PropertyCurve, [f32; 4], &str)],
    property_type: &PropertyType,
    keyframe_id: KeyframeId,
) -> Option<f32> {
    for (curve, _, _) in curves {
        if curve.property_type == *property_type {
            return curve.get_keyframe(keyframe_id).map(|kf| kf.time);
        }
    }
    None
}

fn collect_keyframes_in_range(
    curves: &[(&PropertyCurve, [f32; 4], &str)],
    property_type: &PropertyType,
    time_a: f32,
    time_b: f32,
) -> Vec<CurveSelectedKeyframe> {
    let min_time = time_a.min(time_b);
    let max_time = time_a.max(time_b);

    for (curve, _, _) in curves {
        if curve.property_type == *property_type {
            return curve
                .keyframes
                .iter()
                .filter(|kf| kf.time >= min_time && kf.time <= max_time)
                .map(|kf| CurveSelectedKeyframe {
                    property_type: property_type.clone(),
                    keyframe_id: kf.id,
                    original_time: kf.time,
                    original_value: kf.value,
                })
                .collect();
        }
    }
    Vec::new()
}

fn handle_panning(editor_state: &mut CurveEditorState, mouse_pos: [f32; 2], vt: &ViewTransform) {
    let CurveInteractionMode::Panning {
        start_mouse_pos,
        start_offset,
    } = editor_state.interaction
    else {
        return;
    };

    let dx = mouse_pos[0] - start_mouse_pos[0];
    let dy = mouse_pos[1] - start_mouse_pos[1];

    let time_per_pixel = vt.duration.max(0.001) / (vt.zoom_x * vt.curve_width).max(0.001);
    let value_per_pixel = vt.val_range.max(0.001) / (vt.zoom_y * vt.curve_height).max(0.001);

    editor_state.view_time_offset = start_offset[0] - dx * time_per_pixel;
    editor_state.view_value_offset = start_offset[1] + dy * value_per_pixel;
}

fn handle_wheel_input(
    ui: &imgui::Ui,
    editor_state: &mut CurveEditorState,
    mouse_pos: [f32; 2],
    vt: &ViewTransform,
) {
    let wheel = ui.io().mouse_wheel;
    if wheel == 0.0 {
        return;
    }

    let ctrl = ui.io().key_ctrl;

    if ctrl {
        zoom_at_mouse(editor_state, mouse_pos, wheel, vt);
    } else {
        let shift = ui.io().key_shift;
        pan_with_wheel(editor_state, wheel, shift, vt);
    }
}

fn zoom_at_mouse(
    editor_state: &mut CurveEditorState,
    mouse_pos: [f32; 2],
    wheel: f32,
    vt: &ViewTransform,
) {
    let mouse_time = vt.x_to_time(mouse_pos[0]);
    let mouse_value = vt.y_to_value(mouse_pos[1]);

    let factor = if wheel > 0.0 { 1.15 } else { 1.0 / 1.15 };
    let new_zoom_x = (editor_state.zoom_x * factor).clamp(0.1, 10.0);
    let new_zoom_y = (editor_state.zoom_y * factor).clamp(0.1, 10.0);

    editor_state.view_time_offset = mouse_time
        - (mouse_pos[0] - vt.curve_origin[0]) / (new_zoom_x * vt.curve_width).max(0.001)
            * vt.duration.max(0.001);

    editor_state.view_value_offset = mouse_value
        - (vt.curve_origin[1] + vt.curve_height - mouse_pos[1])
            / (new_zoom_y * vt.curve_height).max(0.001)
            * vt.val_range.max(0.001);

    editor_state.zoom_x = new_zoom_x;
    editor_state.zoom_y = new_zoom_y;
}

fn pan_with_wheel(
    editor_state: &mut CurveEditorState,
    wheel: f32,
    shift: bool,
    vt: &ViewTransform,
) {
    if shift {
        let time_per_pixel = vt.duration.max(0.001) / (vt.zoom_x * vt.curve_width).max(0.001);
        editor_state.view_time_offset -= wheel * PAN_SPEED * time_per_pixel;
    } else {
        let value_per_pixel = vt.val_range.max(0.001) / (vt.zoom_y * vt.curve_height).max(0.001);
        editor_state.view_value_offset += wheel * PAN_SPEED * value_per_pixel;
    }
}

fn calculate_sample_count(width: f32) -> usize {
    let base_samples = 60;
    let samples_per_100px = 15;
    let additional = ((width / 100.0) as usize) * samples_per_100px;
    (base_samples + additional).min(200)
}

fn draw_time_ruler(draw_list: &imgui::DrawListMut, pos: [f32; 2], width: f32, vt: &ViewTransform) {
    draw_list
        .add_rect(
            pos,
            [pos[0] + width, pos[1] + TIME_RULER_HEIGHT],
            [0.18, 0.18, 0.22, 1.0],
        )
        .filled(true)
        .build();

    let visible_duration = vt.duration / vt.zoom_x.max(0.001);
    let tick_interval = compute_nice_step(visible_duration / 8.0);

    let view_start = vt.view_time_offset;
    let view_end = view_start + visible_duration;

    let first_tick = (view_start / tick_interval).floor() * tick_interval;
    let mut time = first_tick;

    while time <= view_end + tick_interval {
        let x = vt.time_to_x(time);
        if x < pos[0] - 10.0 || x > pos[0] + width + 10.0 {
            time += tick_interval;
            continue;
        }

        let is_major = (time / tick_interval).round() as i32 % 4 == 0;
        let tick_height = if is_major { 10.0 } else { 5.0 };

        draw_list
            .add_line(
                [x, pos[1] + TIME_RULER_HEIGHT - tick_height],
                [x, pos[1] + TIME_RULER_HEIGHT],
                [0.6, 0.6, 0.6, 1.0],
            )
            .build();

        if is_major {
            draw_list.add_text(
                [x + 2.0, pos[1] + 2.0],
                [0.7, 0.7, 0.7, 1.0],
                &format!("{:.1}s", time),
            );
        }

        time += tick_interval;
    }
}

fn draw_y_axis_labels(
    draw_list: &imgui::DrawListMut,
    origin: [f32; 2],
    axis_width: f32,
    height: f32,
    vt: &ViewTransform,
) {
    let visible_range = vt.val_range / vt.zoom_y.max(0.001);
    if visible_range.abs() < 0.0001 {
        return;
    }

    let tick_count = calculate_y_tick_count(height);
    let raw_step = visible_range / tick_count as f32;
    let step = compute_nice_step(raw_step);

    let view_bottom = vt.view_value_offset;
    let view_top = view_bottom + visible_range;

    let first_tick = (view_bottom / step).ceil() * step;
    let label_color = [0.6, 0.6, 0.6, 1.0];
    let tick_color = [0.3, 0.3, 0.33, 0.5];

    let mut value = first_tick;
    while value <= view_top + step {
        let y = vt.value_to_y(value);
        if y < origin[1] - 10.0 || y > origin[1] + height + 10.0 {
            value += step;
            continue;
        }

        let label = format_value_label(value);
        draw_list.add_text([origin[0] + 2.0, y - 7.0], label_color, &label);

        draw_list
            .add_line(
                [origin[0] + axis_width - 4.0, y],
                [origin[0] + axis_width, y],
                tick_color,
            )
            .build();

        value += step;
    }
}

fn draw_pose_markers(
    draw_list: &imgui::DrawListMut,
    vt: &ViewTransform,
    curve_area_height: f32,
    pose_library: &PoseLibrary,
) {
    let unselected_color = [0.8, 0.7, 0.2, 0.35];
    let selected_color = [1.0, 0.85, 0.0, 0.9];

    for entry in &pose_library.poses {
        let x = vt.time_to_x(entry.captured_time);
        let is_selected = pose_library.selected_pose_id == Some(entry.id);
        let top = vt.curve_origin[1];
        let bottom = top + curve_area_height;

        if is_selected {
            draw_list
                .add_line([x, top], [x, bottom], selected_color)
                .thickness(2.0)
                .build();
        } else {
            let dash_len = 6.0;
            let gap_len = 4.0;
            let mut y = top;
            while y < bottom {
                let y_end = (y + dash_len).min(bottom);
                draw_list
                    .add_line([x, y], [x, y_end], unselected_color)
                    .thickness(1.0)
                    .build();
                y += dash_len + gap_len;
            }
        }

        let diamond_size = 5.0;
        let diamond_y = top + 8.0;
        let color = if is_selected {
            selected_color
        } else {
            unselected_color
        };
        let top_pt = [x, diamond_y - diamond_size];
        let right_pt = [x + diamond_size, diamond_y];
        let bottom_pt = [x, diamond_y + diamond_size];
        let left_pt = [x - diamond_size, diamond_y];
        if is_selected {
            draw_list
                .add_triangle(top_pt, right_pt, bottom_pt, color)
                .filled(true)
                .build();
            draw_list
                .add_triangle(top_pt, bottom_pt, left_pt, color)
                .filled(true)
                .build();
        } else {
            draw_list.add_line(top_pt, right_pt, color).build();
            draw_list.add_line(right_pt, bottom_pt, color).build();
            draw_list.add_line(bottom_pt, left_pt, color).build();
            draw_list.add_line(left_pt, top_pt, color).build();
        }
    }
}

fn draw_grid(draw_list: &imgui::DrawListMut, width: f32, height: f32, vt: &ViewTransform) {
    let grid_color = [0.25, 0.25, 0.28, 1.0];

    let visible_duration = vt.duration / vt.zoom_x.max(0.001);
    let time_step = compute_nice_step(visible_duration / 8.0);
    let view_start_t = vt.view_time_offset;
    let view_end_t = view_start_t + visible_duration;
    let first_t = (view_start_t / time_step).floor() * time_step;

    let mut time = first_t;
    while time <= view_end_t + time_step {
        let x = vt.time_to_x(time);
        if x >= vt.curve_origin[0] && x <= vt.curve_origin[0] + width {
            draw_list
                .add_line(
                    [x, vt.curve_origin[1]],
                    [x, vt.curve_origin[1] + height],
                    grid_color,
                )
                .build();
        }
        time += time_step;
    }

    let visible_range = vt.val_range / vt.zoom_y.max(0.001);
    let value_step = compute_nice_step(visible_range / 6.0);
    let view_bottom = vt.view_value_offset;
    let view_top = view_bottom + visible_range;
    let first_v = (view_bottom / value_step).ceil() * value_step;

    let mut value = first_v;
    while value <= view_top + value_step {
        let y = vt.value_to_y(value);
        if y >= vt.curve_origin[1] && y <= vt.curve_origin[1] + height {
            let line_color = if value.abs() < value_step * 0.1 {
                [0.4, 0.4, 0.43, 1.0]
            } else {
                grid_color
            };

            draw_list
                .add_line(
                    [vt.curve_origin[0], y],
                    [vt.curve_origin[0] + width, y],
                    line_color,
                )
                .build();
        }
        value += value_step;
    }
}

fn draw_curve_with_keyframes(
    draw_list: &imgui::DrawListMut,
    curve: &PropertyCurve,
    color: [f32; 4],
    _sample_count: usize,
    vt: &ViewTransform,
) {
    if curve.keyframes.is_empty() {
        return;
    }

    if curve.keyframes.len() == 1 {
        let kf = &curve.keyframes[0];
        let x = vt.time_to_x(kf.time);
        let y = vt.value_to_y(kf.value);
        draw_list
            .add_circle([x, y], 5.0, color)
            .filled(true)
            .build();
        draw_list
            .add_circle([x, y], 5.0, [1.0, 1.0, 1.0, 0.8])
            .build();
        return;
    }

    if let Some(first) = curve.keyframes.first() {
        let first_x = vt.time_to_x(first.time);
        let start_x = vt.time_to_x(0.0);
        if start_x < first_x {
            let y = vt.value_to_y(first.value);
            draw_list
                .add_line([start_x, y], [first_x, y], color)
                .thickness(1.5)
                .build();
        }
    }

    for i in 0..curve.keyframes.len() - 1 {
        let k0 = &curve.keyframes[i];
        let k1 = &curve.keyframes[i + 1];

        let segment_samples = if k0.interpolation == InterpolationType::Stepped {
            let x0 = vt.time_to_x(k0.time);
            let y0 = vt.value_to_y(k0.value);
            let x1 = vt.time_to_x(k1.time);
            let y1 = vt.value_to_y(k1.value);
            draw_list
                .add_line([x0, y0], [x1, y0], color)
                .thickness(1.5)
                .build();
            draw_list
                .add_line([x1, y0], [x1, y1], color)
                .thickness(1.5)
                .build();
            continue;
        } else if segment_uses_bezier(k0, k1) {
            20
        } else {
            2
        };

        let mut prev_point: Option<[f32; 2]> = None;
        for s in 0..segment_samples {
            let frac = s as f32 / (segment_samples - 1) as f32;
            let time = k0.time + (k1.time - k0.time) * frac;
            if let Some(value) = curve_sample(curve, time) {
                let point = [vt.time_to_x(time), vt.value_to_y(value)];
                if let Some(prev) = prev_point {
                    draw_list
                        .add_line(prev, point, color)
                        .thickness(1.5)
                        .build();
                }
                prev_point = Some(point);
            }
        }
    }

    if let Some(last) = curve.keyframes.last() {
        let last_x = vt.time_to_x(last.time);
        let end_x = vt.time_to_x(vt.duration);
        if end_x > last_x {
            let y = vt.value_to_y(last.value);
            draw_list
                .add_line([last_x, y], [end_x, y], color)
                .thickness(1.5)
                .build();
        }
    }

    for kf in &curve.keyframes {
        let x = vt.time_to_x(kf.time);
        let y = vt.value_to_y(kf.value);

        draw_list
            .add_circle([x, y], 5.0, color)
            .filled(true)
            .build();

        draw_list
            .add_circle([x, y], 5.0, [1.0, 1.0, 1.0, 0.8])
            .build();
    }
}

fn draw_selected_keyframes_highlight(
    draw_list: &imgui::DrawListMut,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    selected_keyframes: &[CurveSelectedKeyframe],
    vt: &ViewTransform,
) {
    for selected in selected_keyframes {
        for (curve, _, _) in curves_to_draw {
            if curve.property_type == selected.property_type {
                if let Some(kf) = curve.get_keyframe(selected.keyframe_id) {
                    let x = vt.time_to_x(kf.time);
                    let y = vt.value_to_y(kf.value);

                    draw_list
                        .add_circle([x, y], 8.0, [1.0, 1.0, 0.0, 1.0])
                        .thickness(2.0)
                        .build();
                }
                break;
            }
        }
    }
}

fn draw_tangent_handles(
    draw_list: &imgui::DrawListMut,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    selected_keyframes: &[CurveSelectedKeyframe],
    vt: &ViewTransform,
) {
    for selected in selected_keyframes {
        for (curve, color, _) in curves_to_draw {
            if curve.property_type != selected.property_type {
                continue;
            }

            let kf = match curve.get_keyframe(selected.keyframe_id) {
                Some(kf) => kf,
                None => break,
            };

            if kf.interpolation != InterpolationType::Bezier {
                break;
            }

            let kf_x = vt.time_to_x(kf.time);
            let kf_y = vt.value_to_y(kf.value);
            let handle_color = [color[0], color[1], color[2], 0.9];
            let handle_size = 4.0;
            let is_weighted = kf.weight_mode == TangentWeightMode::Weighted;

            let in_x = vt.time_to_x(kf.time + kf.in_tangent.time_offset);
            let in_y = vt.value_to_y(kf.value + kf.in_tangent.value_offset);
            draw_list
                .add_line([kf_x, kf_y], [in_x, in_y], handle_color)
                .thickness(1.0)
                .build();
            if is_weighted {
                draw_list
                    .add_circle([in_x, in_y], handle_size, handle_color)
                    .filled(true)
                    .build();
            } else {
                draw_list
                    .add_rect(
                        [in_x - handle_size, in_y - handle_size],
                        [in_x + handle_size, in_y + handle_size],
                        handle_color,
                    )
                    .filled(true)
                    .build();
            }

            let out_x = vt.time_to_x(kf.time + kf.out_tangent.time_offset);
            let out_y = vt.value_to_y(kf.value + kf.out_tangent.value_offset);
            draw_list
                .add_line([kf_x, kf_y], [out_x, out_y], handle_color)
                .thickness(1.0)
                .build();
            if is_weighted {
                draw_list
                    .add_circle([out_x, out_y], handle_size, handle_color)
                    .filled(true)
                    .build();
            } else {
                draw_list
                    .add_rect(
                        [out_x - handle_size, out_y - handle_size],
                        [out_x + handle_size, out_y + handle_size],
                        handle_color,
                    )
                    .filled(true)
                    .build();
            }

            break;
        }
    }
}

fn draw_keyframe_drag_preview(
    draw_list: &imgui::DrawListMut,
    mouse_pos: [f32; 2],
    drag_start: [f32; 2],
    vt: &ViewTransform,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    selected_keyframes: &[CurveSelectedKeyframe],
) {
    let time_delta = vt.x_to_time(mouse_pos[0]) - vt.x_to_time(drag_start[0]);
    let value_delta = vt.y_to_value(mouse_pos[1]) - vt.y_to_value(drag_start[1]);

    for sel in selected_keyframes {
        let preview_x = vt
            .time_to_x(sel.original_time + time_delta)
            .clamp(vt.curve_origin[0], vt.curve_origin[0] + vt.curve_width);
        let preview_y = vt
            .value_to_y(sel.original_value + value_delta)
            .clamp(vt.curve_origin[1], vt.curve_origin[1] + vt.curve_height);
        let preview_pos = [preview_x, preview_y];

        draw_drag_neighbor_lines(draw_list, preview_pos, vt, curves_to_draw, sel);

        draw_list
            .add_circle(preview_pos, 7.0, [1.0, 1.0, 0.0, 1.0])
            .filled(true)
            .build();

        draw_list
            .add_circle(preview_pos, 7.0, [1.0, 1.0, 1.0, 1.0])
            .thickness(2.0)
            .build();

        let preview_time = (sel.original_time + time_delta).max(0.0);
        let preview_value = sel.original_value + value_delta;

        draw_list.add_text(
            [preview_x + 10.0, preview_y - 10.0],
            [1.0, 1.0, 1.0, 1.0],
            &format!("t={:.2}s v={:.3}", preview_time, preview_value),
        );
    }
}

fn draw_drag_neighbor_lines(
    draw_list: &imgui::DrawListMut,
    preview_pos: [f32; 2],
    vt: &ViewTransform,
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    selected: &CurveSelectedKeyframe,
) {
    for (curve, color, _) in curves_to_draw {
        if curve.property_type != selected.property_type {
            continue;
        }

        let kf_index = match curve
            .keyframes
            .iter()
            .position(|kf| kf.id == selected.keyframe_id)
        {
            Some(idx) => idx,
            None => break,
        };

        let line_color = [color[0], color[1], color[2], 0.6];

        if kf_index > 0 {
            let prev = &curve.keyframes[kf_index - 1];
            let prev_pos = [vt.time_to_x(prev.time), vt.value_to_y(prev.value)];
            draw_list
                .add_line(prev_pos, preview_pos, line_color)
                .thickness(1.5)
                .build();
        }

        if kf_index + 1 < curve.keyframes.len() {
            let next = &curve.keyframes[kf_index + 1];
            let next_pos = [vt.time_to_x(next.time), vt.value_to_y(next.value)];
            draw_list
                .add_line(preview_pos, next_pos, line_color)
                .thickness(1.5)
                .build();
        }

        break;
    }
}

fn draw_tangent_drag_curve_preview(
    draw_list: &imgui::DrawListMut,
    dragging: &DraggingTangent,
    mouse_pos: [f32; 2],
    curves_to_draw: &[(&PropertyCurve, [f32; 4], &str)],
    vt: &ViewTransform,
) {
    let (curve, color) = match curves_to_draw
        .iter()
        .find(|(c, _, _)| c.property_type == dragging.property_type)
    {
        Some((c, col, _)) => (*c, *col),
        None => return,
    };

    let kf_idx = match curve
        .keyframes
        .iter()
        .position(|k| k.id == dragging.keyframe_id)
    {
        Some(idx) => idx,
        None => return,
    };

    let mouse_time = vt.x_to_time(mouse_pos[0]);
    let mouse_value = vt.y_to_value(mouse_pos[1]);
    let kf = &curve.keyframes[kf_idx];
    let new_handle = BezierHandle::new(mouse_time - kf.time, mouse_value - kf.value);

    let preview_in = match dragging.handle_type {
        TangentHandleType::In => new_handle.clone(),
        TangentHandleType::Out => kf.in_tangent.clone(),
    };
    let preview_out = match dragging.handle_type {
        TangentHandleType::In => kf.out_tangent.clone(),
        TangentHandleType::Out => new_handle.clone(),
    };

    let preview_color = [
        (color[0] + 1.0) * 0.5,
        (color[1] + 1.0) * 0.5,
        (color[2] + 1.0) * 0.5,
        0.9,
    ];

    draw_preview_segment_before(draw_list, curve, kf_idx, &preview_in, preview_color, vt);
    draw_preview_segment_after(draw_list, curve, kf_idx, &preview_out, preview_color, vt);

    let kf_x = vt.time_to_x(kf.time);
    let kf_y = vt.value_to_y(kf.value);
    let handle_x = mouse_pos[0];
    let handle_y = mouse_pos[1];

    draw_list
        .add_line([kf_x, kf_y], [handle_x, handle_y], [1.0, 1.0, 0.0, 0.8])
        .thickness(1.0)
        .build();
    draw_list
        .add_rect(
            [handle_x - 5.0, handle_y - 5.0],
            [handle_x + 5.0, handle_y + 5.0],
            [1.0, 1.0, 0.0, 0.8],
        )
        .filled(true)
        .build();
}

fn draw_preview_segment_before(
    draw_list: &imgui::DrawListMut,
    curve: &PropertyCurve,
    kf_idx: usize,
    preview_in: &BezierHandle,
    color: [f32; 4],
    vt: &ViewTransform,
) {
    if kf_idx == 0 {
        return;
    }

    let k0 = &curve.keyframes[kf_idx - 1];
    let k1 = &curve.keyframes[kf_idx];

    if k0.interpolation == InterpolationType::Stepped {
        return;
    }

    if !segment_uses_bezier(k0, k1) {
        return;
    }

    draw_preview_bezier_segment(draw_list, k0, k1, &k0.out_tangent, preview_in, color, vt);
}

fn draw_preview_segment_after(
    draw_list: &imgui::DrawListMut,
    curve: &PropertyCurve,
    kf_idx: usize,
    preview_out: &BezierHandle,
    color: [f32; 4],
    vt: &ViewTransform,
) {
    if kf_idx + 1 >= curve.keyframes.len() {
        return;
    }

    let k0 = &curve.keyframes[kf_idx];
    let k1 = &curve.keyframes[kf_idx + 1];

    if k0.interpolation == InterpolationType::Stepped {
        return;
    }

    if !segment_uses_bezier(k0, k1) {
        return;
    }

    draw_preview_bezier_segment(draw_list, k0, k1, preview_out, &k1.in_tangent, color, vt);
}

fn draw_preview_bezier_segment(
    draw_list: &imgui::DrawListMut,
    k0: &EditableKeyframe,
    k1: &EditableKeyframe,
    out_handle: &BezierHandle,
    in_handle: &BezierHandle,
    color: [f32; 4],
    vt: &ViewTransform,
) {
    let samples = 20;
    let mut prev_point: Option<[f32; 2]> = None;

    for s in 0..samples {
        let frac = s as f32 / (samples - 1) as f32;
        let time = k0.time + (k1.time - k0.time) * frac;

        let dt = k1.time - k0.time;
        let dv = k1.value - k0.value;

        let effective_out = if k0.interpolation == InterpolationType::Bezier {
            out_handle.clone()
        } else {
            BezierHandle::new(dt / 3.0, dv / 3.0)
        };
        let effective_in = if k1.interpolation == InterpolationType::Bezier {
            in_handle.clone()
        } else {
            BezierHandle::new(-dt / 3.0, -dv / 3.0)
        };

        let value = sample_bezier(
            k0.time,
            k0.value,
            &effective_out,
            k1.time,
            k1.value,
            &effective_in,
            time,
        );

        let point = [vt.time_to_x(time), vt.value_to_y(value)];
        if let Some(prev) = prev_point {
            draw_list
                .add_line(prev, point, color)
                .thickness(2.5)
                .build();
        }
        prev_point = Some(point);
    }
}

fn calculate_y_tick_count(height: f32) -> usize {
    ((height / 40.0) as usize).max(2).min(15)
}

fn compute_nice_step(raw_step: f32) -> f32 {
    if raw_step <= 0.0 {
        return 1.0;
    }
    let magnitude = 10.0f32.powf(raw_step.log10().floor());
    let normalized = raw_step / magnitude;

    let nice = if normalized <= 1.0 {
        1.0
    } else if normalized <= 2.0 {
        2.0
    } else if normalized <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice * magnitude
}

fn format_value_label(value: f32) -> String {
    let abs = value.abs();
    if abs >= 100.0 {
        format!("{:.0}", value)
    } else if abs >= 1.0 {
        format!("{:.1}", value)
    } else {
        format!("{:.2}", value)
    }
}

fn calculate_global_value_range(curves: &[(&PropertyCurve, [f32; 4], &str)]) -> (f32, f32) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (curve, _, _) in curves {
        for kf in &curve.keyframes {
            min_val = min_val.min(kf.value);
            max_val = max_val.max(kf.value);
        }
    }

    if min_val == f32::MAX {
        min_val = -1.0;
        max_val = 1.0;
    } else if (max_val - min_val).abs() < 0.001 {
        min_val -= 0.5;
        max_val += 0.5;
    }

    let padding = (max_val - min_val) * 0.1;
    (min_val - padding, max_val + padding)
}

fn find_keyframe_at_position(
    mouse_pos: [f32; 2],
    curves: &[(&PropertyCurve, [f32; 4], &str)],
    vt: &ViewTransform,
) -> Option<(PropertyType, KeyframeId, f32, f32)> {
    for (curve, _, _) in curves {
        for kf in &curve.keyframes {
            let x = vt.time_to_x(kf.time);
            let y = vt.value_to_y(kf.value);

            let dx = mouse_pos[0] - x;
            let dy = mouse_pos[1] - y;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance <= KEYFRAME_HIT_RADIUS {
                return Some((curve.property_type.clone(), kf.id, kf.time, kf.value));
            }
        }
    }

    None
}

const TANGENT_HANDLE_HIT_RADIUS: f32 = 10.0;

fn find_tangent_handle_at_position(
    mouse_pos: [f32; 2],
    curves: &[(&PropertyCurve, [f32; 4], &str)],
    selected_keyframes: &[CurveSelectedKeyframe],
    vt: &ViewTransform,
) -> Option<(TangentHandleType, PropertyType, KeyframeId, BezierHandle)> {
    for selected in selected_keyframes {
        for (curve, _, _) in curves {
            if curve.property_type != selected.property_type {
                continue;
            }

            let kf = match curve.get_keyframe(selected.keyframe_id) {
                Some(kf) => kf,
                None => break,
            };

            if kf.interpolation != InterpolationType::Bezier {
                break;
            }

            let in_x = vt.time_to_x(kf.time + kf.in_tangent.time_offset);
            let in_y = vt.value_to_y(kf.value + kf.in_tangent.value_offset);
            let dx = mouse_pos[0] - in_x;
            let dy = mouse_pos[1] - in_y;
            if (dx * dx + dy * dy).sqrt() <= TANGENT_HANDLE_HIT_RADIUS {
                return Some((
                    TangentHandleType::In,
                    curve.property_type,
                    kf.id,
                    kf.in_tangent.clone(),
                ));
            }

            let out_x = vt.time_to_x(kf.time + kf.out_tangent.time_offset);
            let out_y = vt.value_to_y(kf.value + kf.out_tangent.value_offset);
            let dx = mouse_pos[0] - out_x;
            let dy = mouse_pos[1] - out_y;
            if (dx * dx + dy * dy).sqrt() <= TANGENT_HANDLE_HIT_RADIUS {
                return Some((
                    TangentHandleType::Out,
                    curve.property_type,
                    kf.id,
                    kf.out_tangent.clone(),
                ));
            }

            break;
        }
    }

    None
}

fn draw_buffer_curve_overlay(
    draw_list: &imgui::DrawListMut,
    buffer: &CurveEditorBuffer,
    bone_id: BoneId,
    visible_curves: &HashSet<PropertyType>,
    vt: &ViewTransform,
) {
    if buffer.is_empty() {
        return;
    }

    for (prop_type, color, _name) in ALL_PROPERTY_TYPES {
        if !visible_curves.contains(prop_type) {
            continue;
        }

        let snapshot = match buffer.get_snapshot(bone_id, *prop_type) {
            Some(s) => s,
            None => continue,
        };

        if snapshot.len() < 2 {
            continue;
        }

        let ghost_color = [color[0], color[1], color[2], 0.35];

        for i in 0..snapshot.len() - 1 {
            let (t0, v0) = snapshot[i];
            let (t1, v1) = snapshot[i + 1];

            let x0 = vt.time_to_x(t0);
            let y0 = vt.value_to_y(v0);
            let x1 = vt.time_to_x(t1);
            let y1 = vt.value_to_y(v1);

            draw_list
                .add_line([x0, y0], [x1, y1], ghost_color)
                .thickness(1.5)
                .build();
        }
    }
}

#[cfg(feature = "ml")]
fn handle_suggestion_keyboard(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    bone_id: BoneId,
    editor_state: &CurveEditorState,
    suggestion_overlays: &[SuggestionOverlay],
) {
    let io = ui.io();
    let shift = io.key_shift;

    if shift && ui.is_key_pressed(imgui::Key::C) {
        for property_type in &editor_state.visible_curves {
            ui_events.send(UIEvent::CurveSuggestionRequest {
                bone_id,
                property_type: *property_type,
            });
        }
    }

    if ui.is_key_pressed(imgui::Key::Tab) && !suggestion_overlays.is_empty() {
        ui_events.send(UIEvent::CurveSuggestionAccept);
    }

    if ui.is_key_pressed(imgui::Key::Escape) && !suggestion_overlays.is_empty() {
        ui_events.send(UIEvent::CurveSuggestionDismiss);
    }
}

fn draw_suggestion_curve_overlay(
    draw_list: &imgui::DrawListMut,
    overlays: &[SuggestionOverlay],
    visible_curves: &HashSet<PropertyType>,
    vt: &ViewTransform,
) {
    if overlays.is_empty() {
        return;
    }

    for overlay in overlays {
        if !visible_curves.contains(&overlay.property_type) {
            continue;
        }

        if overlay.confidence < 0.05 {
            continue;
        }

        let alpha = if overlay.confidence > 0.8 {
            0.7
        } else if overlay.confidence > 0.3 {
            0.45
        } else {
            0.25
        };
        let ghost_color = if overlay.confidence > 0.8 {
            [0.3, 1.0, 0.3, alpha]
        } else if overlay.confidence > 0.3 {
            [1.0, 1.0, 0.3, alpha]
        } else {
            [1.0, 0.7, 0.3, alpha]
        };

        let kf_x = vt.time_to_x(overlay.time);
        let kf_y = vt.value_to_y(overlay.value);
        let diamond_size = 6.0;

        draw_list
            .add_line(
                [kf_x, kf_y - diamond_size],
                [kf_x + diamond_size, kf_y],
                ghost_color,
            )
            .thickness(2.0)
            .build();
        draw_list
            .add_line(
                [kf_x + diamond_size, kf_y],
                [kf_x, kf_y + diamond_size],
                ghost_color,
            )
            .thickness(2.0)
            .build();
        draw_list
            .add_line(
                [kf_x, kf_y + diamond_size],
                [kf_x - diamond_size, kf_y],
                ghost_color,
            )
            .thickness(2.0)
            .build();
        draw_list
            .add_line(
                [kf_x - diamond_size, kf_y],
                [kf_x, kf_y - diamond_size],
                ghost_color,
            )
            .thickness(2.0)
            .build();

        let handle_color = [ghost_color[0], ghost_color[1], ghost_color[2], alpha * 0.7];

        let in_x = vt.time_to_x(overlay.time + overlay.tangent_in.0);
        let in_y = vt.value_to_y(overlay.value + overlay.tangent_in.1);
        draw_list
            .add_line([kf_x, kf_y], [in_x, in_y], handle_color)
            .thickness(1.0)
            .build();
        draw_list
            .add_circle([in_x, in_y], 3.0, handle_color)
            .filled(true)
            .build();

        let out_x = vt.time_to_x(overlay.time + overlay.tangent_out.0);
        let out_y = vt.value_to_y(overlay.value + overlay.tangent_out.1);
        draw_list
            .add_line([kf_x, kf_y], [out_x, out_y], handle_color)
            .thickness(1.0)
            .build();
        draw_list
            .add_circle([out_x, out_y], 3.0, handle_color)
            .filled(true)
            .build();
    }
}

fn build_curve_toolbar(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    curve_buffer: &CurveEditorBuffer,
    pose_library: &mut PoseLibrary,
    clip_library: &ClipLibrary,
) {
    if ui.small_button("Capture") {
        ui_events.send(UIEvent::TimelineCaptureBuffer);
    }

    ui.same_line();
    if !curve_buffer.is_empty() {
        if ui.small_button("Swap") {
            ui_events.send(UIEvent::TimelineSwapBuffer);
        }
    } else {
        ui.text_disabled("Swap");
    }

    if !curve_buffer.is_empty() {
        ui.same_line();
        ui.text_colored(
            [0.5, 0.8, 0.5, 1.0],
            &format!("Buf: {}", curve_buffer.snapshots.len()),
        );
    }

    ui.same_line_with_spacing(0.0, 20.0);
    ui.text("|");
    ui.same_line();

    if ui.small_button("Save Pose") {
        let name = format!("Pose {}", pose_library.poses.len() + 1);
        ui_events.send(UIEvent::PoseLibrarySaveCurrent { name });
    }

    ui.same_line();
    if !pose_library.poses.is_empty() {
        let preview = pose_library
            .selected_pose_id
            .and_then(|id| clip_library.get(id))
            .map(|c| c.name.as_str())
            .unwrap_or("(none)");

        ui.set_next_item_width(120.0);
        if let Some(_token) = ui.begin_combo("##pose_select", preview) {
            let pose_ids = pose_library.pose_ids();
            for &pose_id in &pose_ids {
                let name = clip_library
                    .get(pose_id)
                    .map(|c| c.name.as_str())
                    .unwrap_or("(unknown)");

                let is_selected = pose_library.selected_pose_id == Some(pose_id);
                let label = format!("{}##pose_{}", name, pose_id);

                if ui.selectable_config(&label).selected(is_selected).build() {
                    pose_library.selected_pose_id = Some(pose_id);
                }
            }
        }

        ui.same_line();
    }

    if let Some(id) = pose_library.selected_pose_id {
        if ui.small_button("Apply##pose") {
            ui_events.send(UIEvent::PoseLibraryApply(id));
        }
        ui.same_line();
        if ui.small_button("Del##pose") {
            ui_events.send(UIEvent::PoseLibraryDelete(id));
        }
    } else {
        ui.text_disabled("Apply");
        ui.same_line();
        ui.text_disabled("Del");
    }
}
