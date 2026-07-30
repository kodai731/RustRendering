use imgui::Condition;

use crate::animation::editable::{BlendMode, SourceClipId};
use crate::animation::BoneId;
use crate::ecs::component::{
    ClipGroupSnapshot, ClipInstanceSnapshot, ClipTrackEntry, ClipTrackSnapshot,
};
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{
    ClipDragState, ClipDragType, ClipLibrary, CurveEditorState, TimelineInteractionState,
    TimelineState,
};
use crate::ecs::systems::timeline_effective_duration;

use super::layout_snapshot::LayoutSnapshot;

pub(crate) const TRACK_LABEL_WIDTH: f32 = 150.0;
const TIME_RULER_HEIGHT: f32 = 30.0;
pub(crate) const PIXELS_PER_SECOND: f32 = 80.0;
const PLAYHEAD_HANDLE_SIZE: f32 = 10.0;
const CLIP_TRACK_HEIGHT: f32 = 28.0;
const CLIP_EDGE_DRAG_WIDTH: f32 = 5.0;
const CLIP_BLOCK_COLORS: [[f32; 4]; 4] = [
    [0.3, 0.5, 0.8, 0.9],
    [0.5, 0.7, 0.3, 0.9],
    [0.8, 0.4, 0.3, 0.9],
    [0.7, 0.5, 0.7, 0.9],
];

pub fn build_timeline_window(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut TimelineState,
    interaction: &mut TimelineInteractionState,
    clip_library: &ClipLibrary,
    curve_editor_state: &mut CurveEditorState,
    clip_track_snapshot: &ClipTrackSnapshot,
    flame_track: Option<&crate::ecs::component::FlameTrack>,
    layout: &LayoutSnapshot,
) {
    ui.window("Timeline")
        .position([0.0, layout.timeline_y], Condition::Always)
        .size(
            [layout.display_size[0], layout.timeline_height],
            Condition::Always,
        )
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .bring_to_front_on_focus(false)
        .build(|| {
            build_transport_controls(ui, ui_events, state, clip_library, curve_editor_state);
            ui.separator();
            handle_middle_drag_pan(ui, state);
            build_timeline_content(
                ui,
                ui_events,
                state,
                interaction,
                clip_library,
                curve_editor_state,
                clip_track_snapshot,
                flame_track,
            );
            let clip_duration = timeline_effective_duration(state, clip_library);
            handle_timeline_shortcuts(ui, ui_events, state);
            handle_mouse_wheel_zoom(ui, ui_events, state, clip_duration);
        });
}

fn build_transport_controls(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut TimelineState,
    clip_library: &ClipLibrary,
    curve_editor_state: &mut CurveEditorState,
) {
    if state.playing {
        if ui.button("||") {
            ui_events.send(UIEvent::TimelinePause);
        }
    } else if ui.button(">") {
        ui_events.send(UIEvent::TimelinePlay);
    }

    ui.same_line();
    if ui.button("[]") {
        ui_events.send(UIEvent::TimelineStop);
    }

    ui.same_line();
    let mut looping = state.looping;
    if ui.checkbox("Loop", &mut looping) {
        ui_events.send(UIEvent::TimelineToggleLoop);
    }

    ui.same_line();
    let current_clip = state.current_clip_id.and_then(|id| clip_library.get(id));
    let duration = timeline_effective_duration(state, clip_library);

    ui.text(format!(
        "Time: {:.2}s / {:.2}s",
        state.current_time, duration
    ));

    let available_width = state.last_visible_width.max(1.0);
    let (min_zoom, max_zoom) = compute_zoom_limits(
        available_width,
        duration.max(1.0),
        state.snap_settings.frame_rate,
    );

    ui.same_line();
    if ui.button("-") {
        ui_events.send(UIEvent::TimelineZoomOut { min_zoom });
    }
    ui.same_line();
    if ui.button("+") {
        ui_events.send(UIEvent::TimelineZoomIn { max_zoom });
    }
    ui.same_line();
    ui.text(format!("Zoom: {:.1}x", state.zoom_level));

    ui.same_line();
    if ui.button("Curve Editor") {
        curve_editor_state.is_open = true;
        curve_editor_state.needs_focus = true;
        let previous_bone_exists = current_clip
            .zip(curve_editor_state.selected_bone_id())
            .is_some_and(|(c, id)| c.tracks.contains_key(&id));

        if !previous_bone_exists {
            if let Some(first_bone_id) = current_clip.and_then(|c| c.tracks.keys().min().copied()) {
                curve_editor_state.select_bone(first_bone_id);
            }
        }
        curve_editor_state.view_initialized = false;
    }

    build_clip_selector(ui, ui_events, state, clip_library);

    build_snap_controls(ui, ui_events, state);
}

fn build_clip_selector(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &TimelineState,
    clip_library: &ClipLibrary,
) {
    let clip_names =
        crate::ecs::systems::clip_library_systems::clip_library_clip_names(clip_library);

    if clip_names.is_empty() {
        ui.text("No clips available");
        return;
    }

    let current_display = state
        .current_clip_id
        .and_then(|id| clip_library.get(id))
        .map(|c| build_clip_display_name(&c.name, c.source_path.as_deref()))
        .unwrap_or_else(|| "Select Clip".to_string());

    ui.same_line();
    ui.set_next_item_width(200.0);

    if let Some(_token) = ui.begin_combo("##clip_select", &current_display) {
        for (id, name) in &clip_names {
            let is_selected = state.current_clip_id == Some(*id);
            let source_path = clip_library.get(*id).and_then(|c| c.source_path.clone());
            let display = build_clip_display_name(name, source_path.as_deref());
            let label = format!("{}##clip_select_{}", display, id);
            if ui.selectable_config(&label).selected(is_selected).build() {
                ui_events.send(UIEvent::TimelineSelectClip(*id));
            }
        }
    }
}

fn build_timeline_content(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut TimelineState,
    interaction: &mut TimelineInteractionState,
    clip_library: &ClipLibrary,
    curve_editor_state: &mut CurveEditorState,
    clip_track_snapshot: &ClipTrackSnapshot,
    flame_track: Option<&crate::ecs::component::FlameTrack>,
) {
    let content_region = ui.content_region_avail();
    let visible_width = (content_region[0] - TRACK_LABEL_WIDTH).max(1.0);
    state.last_visible_width = visible_width;

    let duration = timeline_effective_duration(state, clip_library);
    let pixels_per_second = PIXELS_PER_SECOND * state.zoom_level;
    let timeline_width = (duration * pixels_per_second).max(visible_width);

    let ruler_child_height = TIME_RULER_HEIGHT + ui.text_line_height_with_spacing() + 4.0;
    let synced_scroll_x = state.scroll_offset;

    ui.child_window("timeline_ruler")
        .size([content_region[0], ruler_child_height])
        .scroll_bar(false)
        .horizontal_scrollbar(true)
        .build(|| {
            ui.set_scroll_x(synced_scroll_x);
            build_time_ruler_with_scrub(
                ui,
                ui_events,
                state,
                interaction,
                timeline_width,
                duration,
            );
        });
    ui.separator();

    let remaining = ui.content_region_avail();
    ui.child_window("timeline_tracks")
        .size(remaining)
        .horizontal_scrollbar(true)
        .build(|| {
            if state.pan_pending_delta_x.abs() >= 0.01 {
                let new_scroll = (ui.scroll_x() + state.pan_pending_delta_x).max(0.0);
                ui.set_scroll_x(new_scroll);
                state.pan_pending_delta_x = 0.0;
            }
            if !clip_track_snapshot.entries.is_empty() {
                build_clip_tracks_section(
                    ui,
                    ui_events,
                    state,
                    interaction,
                    clip_library,
                    curve_editor_state,
                    clip_track_snapshot,
                    timeline_width,
                );
            }
            let ruler_start_x = ui.cursor_screen_pos()[0] + TRACK_LABEL_WIDTH;
            build_flame_track_row(
                ui,
                ui_events,
                flame_track,
                pixels_per_second,
                ruler_start_x,
                state.current_time,
            );
        });
}

fn build_flame_track_row(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    flame_track: Option<&crate::ecs::component::FlameTrack>,
    pixels_per_second: f32,
    ruler_start_x: f32,
    current_time: f32,
) {
    let track = match flame_track {
        Some(t) if !t.channels.is_empty() => t,
        _ => return,
    };

    // Collect all unique key times from all channels
    let mut key_times: Vec<f32> = Vec::new();
    for channel in &track.channels {
        for key in &channel.keys {
            if !key_times.contains(&key.time) {
                key_times.push(key.time);
            }
        }
    }

    if key_times.is_empty() {
        return;
    }

    let cursor_pos = ui.cursor_screen_pos();
    let row_y = cursor_pos[1];
    let row_height: f32 = 18.0;

    // Draw label "Flame" on the left (same style as group headers)
    ui.text_colored([0.7, 0.8, 1.0, 1.0], "Flame");
    ui.same_line_with_pos(TRACK_LABEL_WIDTH);

    let draw_list = ui.get_window_draw_list();

    // Draw a background rect for the row area
    draw_list
        .add_rect(
            [ruler_start_x, row_y],
            [ruler_start_x + 1000.0, row_y + row_height],
            [0.15, 0.15, 0.2, 1.0],
        )
        .filled(true)
        .build();

    // Draw diamond markers at each key position using circles
    for (i, &key_time) in key_times.iter().enumerate() {
        let x = ruler_start_x + key_time * pixels_per_second;
        let y = row_y + row_height / 2.0;

        // Color: white if any channel has a key within 0.02s of current_time, else amber
        let is_under_playhead = (key_time - current_time).abs() < 0.02;
        let color: [f32; 4] = if is_under_playhead {
            [1.0, 1.0, 1.0, 1.0] // White
        } else {
            [1.0, 0.75, 0.375, 1.0] // Amber (0xFFC060FF)
        };

        draw_list
            .add_circle([x, y], 4.0, color)
            .filled(true)
            .build();

        // Interactive hit area: invisible button centered on the marker
        let btn_size = [10.0, 10.0];
        let btn_x = x - btn_size[0] / 2.0;
        let btn_y = y - btn_size[1] / 2.0;
        ui.set_cursor_screen_pos([btn_x, btn_y]);
        let id = format!("flamekey##{}", i);
        ui.invisible_button(&id, btn_size);

        if ui.is_item_hovered() && ui.is_mouse_clicked(imgui::MouseButton::Left) {
            ui_events.send(UIEvent::TimelineSetTime(key_time));
        }

        if ui.is_item_hovered() && ui.is_mouse_clicked(imgui::MouseButton::Right) {
            ui_events.send(UIEvent::DeleteFlameKeysAt { time: key_time });
        }
    }
}

fn build_time_ruler_with_scrub(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut TimelineState,
    interaction: &mut TimelineInteractionState,
    timeline_width: f32,
    display_duration: f32,
) {
    let cursor_pos = ui.cursor_screen_pos();
    let ruler_start_x = cursor_pos[0] + TRACK_LABEL_WIDTH;
    let ruler_width = timeline_width;
    let pixels_per_second = PIXELS_PER_SECOND * state.zoom_level;

    ui.text("Time:");
    ui.same_line_with_pos(TRACK_LABEL_WIDTH);

    let draw_list = ui.get_window_draw_list();

    draw_list
        .add_rect(
            [ruler_start_x, cursor_pos[1]],
            [
                ruler_start_x + ruler_width,
                cursor_pos[1] + TIME_RULER_HEIGHT,
            ],
            [0.2, 0.2, 0.25, 1.0],
        )
        .filled(true)
        .build();

    let tick_interval = calculate_tick_interval(pixels_per_second);
    let mut time = 0.0;
    while time <= display_duration {
        let x = ruler_start_x + time * pixels_per_second;
        let is_major = (time / tick_interval).round() as i32 % 5 == 0;

        let tick_height = if is_major { 12.0 } else { 6.0 };
        let tick_color = if is_major {
            [0.7, 0.7, 0.7, 1.0]
        } else {
            [0.4, 0.4, 0.4, 1.0]
        };

        draw_list
            .add_line(
                [x, cursor_pos[1] + TIME_RULER_HEIGHT - tick_height],
                [x, cursor_pos[1] + TIME_RULER_HEIGHT],
                tick_color,
            )
            .build();

        if is_major {
            draw_list.add_text(
                [x - 10.0, cursor_pos[1] + 2.0],
                [0.8, 0.8, 0.8, 1.0],
                &format!("{:.1}s", time),
            );
        }

        time += tick_interval;
    }

    let playhead_x = ruler_start_x + state.current_time * pixels_per_second;
    draw_playhead_handle(&draw_list, playhead_x, cursor_pos[1], TIME_RULER_HEIGHT);

    let ruler_rect_min = [ruler_start_x, cursor_pos[1]];
    let ruler_rect_max = [
        ruler_start_x + ruler_width,
        cursor_pos[1] + TIME_RULER_HEIGHT,
    ];

    handle_scrub_interaction(
        ui,
        ui_events,
        interaction,
        ruler_rect_min,
        ruler_rect_max,
        display_duration,
        pixels_per_second,
        ruler_start_x,
    );

    ui.dummy([ruler_width + TRACK_LABEL_WIDTH, TIME_RULER_HEIGHT]);
}

fn draw_playhead_handle(draw_list: &imgui::DrawListMut, x: f32, y: f32, ruler_height: f32) {
    draw_list
        .add_triangle(
            [x - PLAYHEAD_HANDLE_SIZE, y],
            [x + PLAYHEAD_HANDLE_SIZE, y],
            [x, y + PLAYHEAD_HANDLE_SIZE + 4.0],
            [1.0, 0.3, 0.3, 1.0],
        )
        .filled(true)
        .build();

    draw_list
        .add_line(
            [x, y + PLAYHEAD_HANDLE_SIZE],
            [x, y + ruler_height],
            [1.0, 0.3, 0.3, 1.0],
        )
        .thickness(2.0)
        .build();
}

fn handle_scrub_interaction(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    interaction: &mut TimelineInteractionState,
    rect_min: [f32; 2],
    rect_max: [f32; 2],
    duration: f32,
    pixels_per_second: f32,
    ruler_start_x: f32,
) {
    let mouse_pos = ui.io().mouse_pos;
    let mouse_down = ui.io().mouse_down[0];

    if !mouse_down {
        interaction.scrubbing = false;
        return;
    }

    let is_mouse_in_ruler = mouse_pos[0] >= rect_min[0]
        && mouse_pos[0] <= rect_max[0]
        && mouse_pos[1] >= rect_min[1]
        && mouse_pos[1] <= rect_max[1];

    if !interaction.scrubbing && !is_mouse_in_ruler {
        return;
    }

    interaction.scrubbing = true;
    let relative_x = mouse_pos[0] - ruler_start_x;
    let new_time = (relative_x / pixels_per_second).clamp(0.0, duration);
    ui_events.send(UIEvent::TimelineSetTime(new_time));
}

fn calculate_tick_interval(pixels_per_second: f32) -> f32 {
    const TARGET_MINOR_SPACING_PX: f32 = 20.0;
    let raw = TARGET_MINOR_SPACING_PX / pixels_per_second.max(0.001);
    nice_round_step(raw)
}

fn nice_round_step(raw: f32) -> f32 {
    if raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let exp = raw.log10().floor();
    let base = 10f32.powf(exp);
    let mantissa = raw / base;

    let nice_mantissa = if mantissa <= 1.0 {
        1.0
    } else if mantissa <= 2.0 {
        2.0
    } else if mantissa <= 2.5 {
        2.5
    } else if mantissa <= 5.0 {
        5.0
    } else {
        10.0
    };

    nice_mantissa * base
}

fn build_clip_tracks_section(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &mut TimelineState,
    interaction: &mut TimelineInteractionState,
    clip_library: &ClipLibrary,
    curve_editor_state: &mut CurveEditorState,
    snapshot: &ClipTrackSnapshot,
    timeline_width: f32,
) {
    let pixels_per_second = PIXELS_PER_SECOND * state.zoom_level;
    let mouse_pos = ui.io().mouse_pos;
    let mouse_down = ui.io().mouse_down[0];
    let mouse_clicked = ui.is_mouse_clicked(imgui::MouseButton::Left);
    let mouse_double_clicked = ui.is_mouse_double_clicked(imgui::MouseButton::Left);

    handle_clip_drag_release(ui, ui_events, interaction, pixels_per_second);

    let mut clicked_any_block = false;

    for (entry_idx, entry) in snapshot.entries.iter().enumerate() {
        build_group_headers(ui, ui_events, entry);

        let cursor_pos = ui.cursor_screen_pos();
        ui.text(&truncate_label(&entry.entity_name, 15));
        ui.same_line_with_pos(TRACK_LABEL_WIDTH);

        let track_origin = [cursor_pos[0] + TRACK_LABEL_WIDTH, cursor_pos[1]];
        let draw_list = ui.get_window_draw_list();

        draw_list
            .add_rect(
                track_origin,
                [
                    track_origin[0] + timeline_width,
                    track_origin[1] + CLIP_TRACK_HEIGHT,
                ],
                [0.15, 0.15, 0.2, 1.0],
            )
            .filled(true)
            .build();

        for (inst_idx, inst) in entry.instances.iter().enumerate() {
            let block_x = track_origin[0] + inst.start_time * pixels_per_second;
            let block_w = (inst.end_time - inst.start_time) * pixels_per_second;
            let block_min = [block_x, track_origin[1] + 2.0];
            let block_max = [block_x + block_w, track_origin[1] + CLIP_TRACK_HEIGHT - 2.0];

            let base_color = CLIP_BLOCK_COLORS[entry_idx % CLIP_BLOCK_COLORS.len()];
            let color = compute_block_color(base_color, inst, interaction, entry.entity);
            let border_color = compute_border_color(inst, state, entry.entity);

            draw_clip_block(&draw_list, block_min, block_max, color, border_color, inst);

            let hit_block = is_point_in_rect(mouse_pos, block_min, block_max);

            if mouse_double_clicked && hit_block {
                clicked_any_block = true;
                ui_events.send(UIEvent::SelectEntity(entry.entity));
                ui_events.send(UIEvent::TimelineSelectClip(inst.source_id));
                open_curve_editor_for_clip(
                    curve_editor_state,
                    clip_library,
                    inst.source_id,
                    entry.mesh_bone_id,
                );
            } else if mouse_clicked && hit_block {
                clicked_any_block = true;
                ui_events.send(UIEvent::SelectEntity(entry.entity));
                ui_events.send(UIEvent::ClipInstanceSelect {
                    entity: entry.entity,
                    instance_id: inst.instance_id,
                });
                begin_clip_drag(
                    interaction,
                    entry.entity,
                    inst,
                    mouse_pos,
                    block_min,
                    block_max,
                    pixels_per_second,
                );
            }

            handle_clip_mute_button(ui, ui_events, entry.entity, inst, inst_idx, entry_idx);
        }

        ui.dummy([timeline_width, CLIP_TRACK_HEIGHT]);

        if let Some(target) = ui.drag_drop_target() {
            let accepted = target
                .accept_payload::<SourceClipId, _>("CLIP_SOURCE", imgui::DragDropFlags::empty());
            if let Some(Ok(payload)) = accepted {
                let source_id = payload.data;
                let drop_x = mouse_pos[0] - track_origin[0];
                let start_time = (drop_x / pixels_per_second).max(0.0);
                ui_events.send(UIEvent::ClipInstanceAdd {
                    entity: entry.entity,
                    source_id,
                    start_time,
                });
            }
        }

        build_clip_instance_properties(ui, ui_events, state, entry);
    }

    if mouse_clicked && !clicked_any_block {
        let section_start_y = ui.cursor_screen_pos()[1]
            - (snapshot.entries.len() as f32
                * (CLIP_TRACK_HEIGHT + ui.text_line_height_with_spacing()));

        if mouse_pos[1] >= section_start_y {
            ui_events.send(UIEvent::ClipInstanceDeselect);
        }
    }

    handle_delete_key(ui, ui_events, state);
    update_clip_drag(interaction, mouse_pos, mouse_down, pixels_per_second);
}

fn build_group_headers(ui: &imgui::Ui, ui_events: &mut UIEventQueue, entry: &ClipTrackEntry) {
    for group in &entry.groups {
        build_single_group_header(ui, ui_events, entry.entity, group);
    }
}

fn build_single_group_header(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    entity: crate::ecs::world::Entity,
    group: &ClipGroupSnapshot,
) {
    let mute_label = if group.muted { "[M]" } else { "[ ]" };
    let header_text = format!(
        "  {} {} (w:{:.2}, {})",
        mute_label,
        group.name,
        group.weight,
        group.instance_ids.len()
    );
    ui.text_colored([0.7, 0.8, 1.0, 1.0], &header_text);

    ui.same_line();
    let mute_btn_id = format!("Mute##grp_{}", group.id);
    if ui.small_button(&mute_btn_id) {
        ui_events.send(UIEvent::ClipGroupToggleMute {
            entity,
            group_id: group.id,
        });
    }

    ui.same_line();
    ui.set_next_item_width(60.0);
    let mut weight = group.weight;
    let slider_id = format!("##grp_w_{}", group.id);
    if imgui::Drag::new(&slider_id)
        .range(0.0, 1.0)
        .speed(0.01)
        .display_format("%.2f")
        .build(ui, &mut weight)
    {
        ui_events.send(UIEvent::ClipGroupSetWeight {
            entity,
            group_id: group.id,
            weight,
        });
    }

    ui.same_line();
    let del_btn_id = format!("X##grp_del_{}", group.id);
    if ui.small_button(&del_btn_id) {
        ui_events.send(UIEvent::ClipGroupDelete {
            entity,
            group_id: group.id,
        });
    }
}

fn build_clip_instance_properties(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &TimelineState,
    entry: &ClipTrackEntry,
) {
    let Some((sel_entity, sel_id)) = state.selected_clip_instance else {
        return;
    };

    if sel_entity != entry.entity {
        return;
    }

    let Some(inst) = entry.instances.iter().find(|i| i.instance_id == sel_id) else {
        return;
    };

    ui.text("  Properties:");
    ui.same_line();

    ui.set_next_item_width(60.0);
    let mut weight = inst.weight;
    if imgui::Drag::new("##inst_weight")
        .range(0.0, 1.0)
        .speed(0.01)
        .display_format("W:%.2f")
        .build(ui, &mut weight)
    {
        ui_events.send(UIEvent::ClipInstanceSetWeight {
            entity: entry.entity,
            instance_id: inst.instance_id,
            weight,
        });
    }

    ui.same_line();
    let blend_names = ["Override", "Additive"];
    let current_idx = match inst.blend_mode {
        BlendMode::Override => 0,
        BlendMode::Additive => 1,
    };

    ui.set_next_item_width(80.0);
    if let Some(_token) = ui.begin_combo("##blend_mode", blend_names[current_idx]) {
        for (idx, &name) in blend_names.iter().enumerate() {
            let is_selected = idx == current_idx;
            if ui.selectable_config(name).selected(is_selected).build() {
                let new_mode = match idx {
                    0 => BlendMode::Override,
                    1 => BlendMode::Additive,
                    _ => continue,
                };
                ui_events.send(UIEvent::ClipInstanceSetBlendMode {
                    entity: entry.entity,
                    instance_id: inst.instance_id,
                    blend_mode: new_mode,
                });
            }
        }
    }

    if !entry.groups.is_empty() {
        ui.same_line();
        let current_group_name = inst
            .group_id
            .and_then(|gid| entry.groups.iter().find(|g| g.id == gid))
            .map(|g| g.name.as_str())
            .unwrap_or("No Group");

        ui.set_next_item_width(100.0);
        if let Some(_token) = ui.begin_combo("##inst_group", current_group_name) {
            if ui
                .selectable_config("No Group")
                .selected(inst.group_id.is_none())
                .build()
            {
                if let Some(gid) = inst.group_id {
                    ui_events.send(UIEvent::ClipGroupRemoveInstance {
                        entity: entry.entity,
                        group_id: gid,
                        instance_id: inst.instance_id,
                    });
                }
            }

            for group in &entry.groups {
                let is_selected = inst.group_id.map(|gid| gid == group.id).unwrap_or(false);
                if ui
                    .selectable_config(&group.name)
                    .selected(is_selected)
                    .build()
                {
                    ui_events.send(UIEvent::ClipGroupAddInstance {
                        entity: entry.entity,
                        group_id: group.id,
                        instance_id: inst.instance_id,
                    });
                }
            }
        }
    }
}

fn handle_clip_drag_release(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    interaction: &mut TimelineInteractionState,
    pixels_per_second: f32,
) {
    if !ui.is_mouse_down(imgui::MouseButton::Left) {
        if let Some(drag) = interaction.dragging_clip.take() {
            let mouse_pos = ui.io().mouse_pos;
            let delta_x = mouse_pos[0] - drag.drag_start_x;
            let delta_time = delta_x / pixels_per_second;

            match drag.drag_type {
                ClipDragType::Move => {
                    let new_start = (drag.original_value + delta_time).max(0.0);
                    ui_events.send(UIEvent::ClipInstanceMove {
                        entity: drag.entity,
                        instance_id: drag.instance_id,
                        new_start_time: new_start,
                    });
                }
                ClipDragType::TrimStart => {
                    let new_clip_in = (drag.original_value + delta_time).max(0.0);
                    ui_events.send(UIEvent::ClipInstanceTrimStart {
                        entity: drag.entity,
                        instance_id: drag.instance_id,
                        new_clip_in,
                    });
                }
                ClipDragType::TrimEnd => {
                    let new_clip_out = (drag.original_value + delta_time).max(0.0);
                    ui_events.send(UIEvent::ClipInstanceTrimEnd {
                        entity: drag.entity,
                        instance_id: drag.instance_id,
                        new_clip_out,
                    });
                }
            }
        }
    }
}

fn begin_clip_drag(
    interaction: &mut TimelineInteractionState,
    entity: crate::ecs::world::Entity,
    inst: &ClipInstanceSnapshot,
    mouse_pos: [f32; 2],
    block_min: [f32; 2],
    block_max: [f32; 2],
    _pixels_per_second: f32,
) {
    let near_left_edge = (mouse_pos[0] - block_min[0]).abs() < CLIP_EDGE_DRAG_WIDTH;
    let near_right_edge = (mouse_pos[0] - block_max[0]).abs() < CLIP_EDGE_DRAG_WIDTH;

    let (drag_type, original_value) = if near_left_edge {
        (ClipDragType::TrimStart, inst.clip_in)
    } else if near_right_edge {
        (ClipDragType::TrimEnd, inst.clip_out)
    } else {
        (ClipDragType::Move, inst.start_time)
    };

    interaction.dragging_clip = Some(ClipDragState {
        entity,
        instance_id: inst.instance_id,
        drag_type,
        original_value,
        drag_start_x: mouse_pos[0],
    });
}

fn update_clip_drag(
    _interaction: &mut TimelineInteractionState,
    _mouse_pos: [f32; 2],
    _mouse_down: bool,
    _pixels_per_second: f32,
) {
}

fn handle_clip_mute_button(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    entity: crate::ecs::world::Entity,
    inst: &ClipInstanceSnapshot,
    inst_idx: usize,
    entry_idx: usize,
) {
    let label = if inst.muted { "M##muted" } else { "M##unmuted" };
    let button_id = format!("{}_{}_{}", label, entry_idx, inst_idx);

    ui.same_line();
    if inst.muted {
        let _color_token = ui.push_style_color(imgui::StyleColor::Button, [0.5, 0.2, 0.2, 1.0]);
        if ui.small_button(&button_id) {
            ui_events.send(UIEvent::ClipInstanceToggleMute {
                entity,
                instance_id: inst.instance_id,
            });
        }
    } else if ui.small_button(&button_id) {
        ui_events.send(UIEvent::ClipInstanceToggleMute {
            entity,
            instance_id: inst.instance_id,
        });
    }
}

fn handle_delete_key(ui: &imgui::Ui, ui_events: &mut UIEventQueue, state: &TimelineState) {
    if ui.is_key_pressed(imgui::Key::Delete) {
        if let Some((entity, instance_id)) = state.selected_clip_instance {
            ui_events.send(UIEvent::ClipInstanceDelete {
                entity,
                instance_id,
            });
        }
    }
}

fn draw_clip_block(
    draw_list: &imgui::DrawListMut,
    block_min: [f32; 2],
    block_max: [f32; 2],
    fill_color: [f32; 4],
    border_color: [f32; 4],
    inst: &ClipInstanceSnapshot,
) {
    draw_list
        .add_rect(block_min, block_max, fill_color)
        .filled(true)
        .build();

    draw_list
        .add_rect(block_min, block_max, border_color)
        .build();

    let text_x = block_min[0] + 4.0;
    let text_y = block_min[1] + 2.0;
    let available_width = block_max[0] - block_min[0] - 8.0;

    if available_width > 10.0 {
        let mode_char = match inst.blend_mode {
            BlendMode::Override => "O",
            BlendMode::Additive => "A",
        };
        let label = format!("{} [{} {:.2}]", inst.clip_name, mode_char, inst.weight);
        let display = truncate_label_by_width(&label, available_width);
        draw_list.add_text([text_x, text_y], [1.0, 1.0, 1.0, 1.0], &display);
    }
}

fn compute_block_color(
    base_color: [f32; 4],
    inst: &ClipInstanceSnapshot,
    interaction: &TimelineInteractionState,
    entity: crate::ecs::world::Entity,
) -> [f32; 4] {
    let alpha = if inst.muted { 0.4 } else { base_color[3] };

    let is_dragging = interaction
        .dragging_clip
        .as_ref()
        .map(|d| d.entity == entity && d.instance_id == inst.instance_id)
        .unwrap_or(false);

    let brightness = if is_dragging { 1.3 } else { 1.0 };

    [
        (base_color[0] * brightness).min(1.0),
        (base_color[1] * brightness).min(1.0),
        (base_color[2] * brightness).min(1.0),
        alpha,
    ]
}

fn compute_border_color(
    inst: &ClipInstanceSnapshot,
    state: &TimelineState,
    entity: crate::ecs::world::Entity,
) -> [f32; 4] {
    let is_selected = state
        .selected_clip_instance
        .map(|(e, id)| e == entity && id == inst.instance_id)
        .unwrap_or(false);

    if is_selected {
        [1.0, 1.0, 0.4, 1.0]
    } else {
        [0.6, 0.6, 0.6, 0.5]
    }
}

fn truncate_label(name: &str, max_chars: usize) -> String {
    if name.len() > max_chars {
        format!("{}...", &name[..max_chars.saturating_sub(3)])
    } else {
        name.to_string()
    }
}

fn truncate_label_by_width(name: &str, available_width: f32) -> String {
    let approx_char_width = 7.0;
    let max_chars = (available_width / approx_char_width) as usize;
    truncate_label(name, max_chars)
}

fn is_point_in_rect(point: [f32; 2], rect_min: [f32; 2], rect_max: [f32; 2]) -> bool {
    point[0] >= rect_min[0]
        && point[0] <= rect_max[0]
        && point[1] >= rect_min[1]
        && point[1] <= rect_max[1]
}

fn handle_timeline_shortcuts(ui: &imgui::Ui, ui_events: &mut UIEventQueue, state: &TimelineState) {
    let io = ui.io();
    if !ui.is_window_focused() {
        return;
    }

    if io.key_ctrl && ui.is_key_pressed(imgui::Key::C) {
        ui_events.send(UIEvent::TimelineCopyKeyframes);
    }

    if io.key_ctrl && !io.key_shift && ui.is_key_pressed(imgui::Key::V) {
        ui_events.send(UIEvent::TimelinePasteKeyframes {
            paste_time: state.current_time,
        });
    }

    if io.key_ctrl && io.key_shift && ui.is_key_pressed(imgui::Key::V) {
        ui_events.send(UIEvent::TimelineMirrorPaste {
            paste_time: state.current_time,
        });
    }

    if ui.is_key_pressed(imgui::Key::Delete) {
        if !state.selected_keyframes.is_empty() {
            ui_events.send(UIEvent::TimelineDeleteSelectedKeyframes);
        }
    }
}

fn compute_zoom_limits(available_width: f32, clip_duration: f32, frame_rate: f32) -> (f32, f32) {
    let width = available_width.max(1.0);
    let fps = frame_rate.max(1.0);
    let duration = clip_duration.max(0.1);

    let three_frames_duration = 3.0 / fps;
    let max_zoom = width / (three_frames_duration * PIXELS_PER_SECOND);

    let four_clips_duration = duration * 4.0;
    let min_zoom = width / (four_clips_duration * PIXELS_PER_SECOND);

    (min_zoom.max(0.01), max_zoom)
}

fn handle_mouse_wheel_zoom(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    state: &TimelineState,
    clip_duration: f32,
) {
    let hovered = ui.is_window_hovered_with_flags(imgui::WindowHoveredFlags::CHILD_WINDOWS);
    if !hovered {
        return;
    }

    let available_width = state.last_visible_width.max(1.0);
    let (min_zoom, max_zoom) = compute_zoom_limits(
        available_width,
        clip_duration,
        state.snap_settings.frame_rate,
    );

    let wheel = ui.io().mouse_wheel;
    if wheel > 0.0 {
        ui_events.send(UIEvent::TimelineZoomIn { max_zoom });
    } else if wheel < 0.0 {
        ui_events.send(UIEvent::TimelineZoomOut { min_zoom });
    }
}

fn handle_middle_drag_pan(ui: &imgui::Ui, state: &mut TimelineState) {
    let hovered = ui.is_window_hovered_with_flags(imgui::WindowHoveredFlags::CHILD_WINDOWS);
    if !hovered {
        return;
    }

    if !ui.io().mouse_down[2] {
        return;
    }

    let delta_x = ui.io().mouse_delta[0];
    if delta_x.abs() < 0.01 {
        return;
    }

    state.pan_pending_delta_x -= delta_x;
}

fn build_snap_controls(ui: &imgui::Ui, ui_events: &mut UIEventQueue, state: &TimelineState) {
    let snap = &state.snap_settings;

    let frame_label = if snap.snap_to_frame {
        "[Snap: F]"
    } else {
        "Snap: F"
    };

    if ui.small_button(frame_label) {
        ui_events.send(UIEvent::TimelineSetSnapToFrame(!snap.snap_to_frame));
    }

    ui.same_line();

    let key_label = if snap.snap_to_key {
        "[Snap: K]"
    } else {
        "Snap: K"
    };

    if ui.small_button(key_label) {
        ui_events.send(UIEvent::TimelineSetSnapToKey(!snap.snap_to_key));
    }

    ui.same_line();

    let fps_options = [24.0_f32, 30.0, 60.0];
    let current_fps_label = format!("{}fps", snap.frame_rate as u32);
    ui.set_next_item_width(70.0);

    if let Some(_token) = ui.begin_combo("##fps_select", &current_fps_label) {
        for fps in &fps_options {
            let label = format!("{}fps", *fps as u32);
            let is_selected = (snap.frame_rate - fps).abs() < 0.1;
            if ui.selectable_config(&label).selected(is_selected).build() {
                ui_events.send(UIEvent::TimelineSetFrameRate(*fps));
            }
        }
    }
}

fn build_clip_display_name(name: &str, source_path: Option<&str>) -> String {
    let filename = source_path
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");

    if filename.is_empty() {
        name.to_string()
    } else {
        format!("{} <{}>", name, filename)
    }
}

fn open_curve_editor_for_clip(
    curve_editor_state: &mut CurveEditorState,
    clip_library: &ClipLibrary,
    source_id: SourceClipId,
    mesh_bone_id: Option<BoneId>,
) {
    curve_editor_state.is_open = true;
    curve_editor_state.needs_focus = true;

    if let Some(clip) = clip_library.get(source_id) {
        let previous_bone_exists = curve_editor_state
            .selected_bone_id()
            .is_some_and(|id| clip.tracks.contains_key(&id));

        if !previous_bone_exists {
            let target_bone = mesh_bone_id.filter(|id| clip.tracks.contains_key(id));
            if let Some(bone_id) = target_bone.or_else(|| clip.tracks.keys().min().copied()) {
                curve_editor_state.select_bone(bone_id);
            }
        }

        curve_editor_state.view_initialized = false;
    }
}
