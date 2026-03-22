use imgui::Condition;

use crate::animation::editable::SourceClipId;
use crate::ecs::events::{UIEvent, UIEventQueue};
use crate::ecs::resource::{ClipBrowserState, ClipLibrary, GltfModelCache};
use crate::ecs::world::World;

use super::layout_snapshot::LayoutSnapshot;

pub fn build_clip_browser_window(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    clip_library: &ClipLibrary,
    browser_state: &mut ClipBrowserState,
    world: &World,
    layout: &LayoutSnapshot,
) {
    ui.window("Clip Browser")
        .position([0.0, layout.clip_browser_y], Condition::Always)
        .size(
            [layout.hierarchy_width, layout.clip_browser_height],
            Condition::Always,
        )
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .build(|| {
            build_toolbar(ui, ui_events, browser_state, world);
            ui.separator();
            build_filter_bar(ui, browser_state);
            ui.separator();
            build_clip_list(ui, ui_events, clip_library, browser_state, world);
        });
}

fn build_toolbar(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    browser_state: &ClipBrowserState,
    world: &World,
) {
    if ui.small_button("+ New") {
        ui_events.send(UIEvent::ClipBrowserCreateEmpty);
    }

    ui.same_line();
    if ui.small_button("Load") {
        ui_events.send(UIEvent::ClipBrowserLoadFromFile);
    }

    ui.same_line();
    let has_selection = browser_state.selected_clip_id.is_some();
    if has_selection {
        if ui.small_button("Save") {
            if let Some(id) = browser_state.selected_clip_id {
                ui_events.send(UIEvent::ClipBrowserSaveToFile(id));
            }
        }
    } else {
        ui.text_disabled("Save");
    }

    ui.same_line();
    if has_selection {
        if ui.small_button("FBX") {
            if let Some(id) = browser_state.selected_clip_id {
                ui_events.send(UIEvent::ClipBrowserExportFbx(id));
            }
        }
    } else {
        ui.text_disabled("FBX");
    }

    ui.same_line();
    let has_model = world
        .get_resource::<GltfModelCache>()
        .map_or(false, |c| c.has_model());
    if has_selection {
        if ui.small_button("glTF") {
            if let Some(id) = browser_state.selected_clip_id {
                ui_events.send(UIEvent::ClipBrowserExportGltf(id));
            }
        }
    } else if has_model {
        if ui.small_button("glTF") {
            ui_events.send(UIEvent::ExportModelGltf);
        }
    } else {
        ui.text_disabled("glTF");
    }

    ui.same_line();
    let can_duplicate = has_selection;
    if can_duplicate {
        if ui.small_button("Dup") {
            if let Some(id) = browser_state.selected_clip_id {
                ui_events.send(UIEvent::ClipBrowserDuplicate(id));
            }
        }
    } else {
        ui.text_disabled("Dup");
    }

    ui.same_line();
    if can_duplicate {
        if ui.small_button("Del") {
            if let Some(id) = browser_state.selected_clip_id {
                ui_events.send(UIEvent::ClipBrowserDelete(id));
            }
        }
    } else {
        ui.text_disabled("Del");
    }
}

fn build_filter_bar(ui: &imgui::Ui, browser_state: &mut ClipBrowserState) {
    ui.set_next_item_width(-1.0);
    ui.input_text("##clip_filter", &mut browser_state.filter_text)
        .hint("Filter...")
        .build();
}

fn build_clip_list(
    ui: &imgui::Ui,
    ui_events: &mut UIEventQueue,
    clip_library: &ClipLibrary,
    browser_state: &mut ClipBrowserState,
    world: &World,
) {
    let clip_names =
        crate::ecs::systems::clip_library_systems::clip_library_clip_names(clip_library);

    if clip_names.is_empty() {
        ui.text_disabled("No clips");
        return;
    }

    let reference_counts =
        crate::ecs::systems::clip_library_systems::clip_library_count_references(world);

    let filter_lower = browser_state.filter_text.to_lowercase();

    let remaining = ui.content_region_avail();
    ui.child_window("##clip_list")
        .size([remaining[0], remaining[1] - 4.0])
        .build(|| {
            for (id, name) in &clip_names {
                if !filter_lower.is_empty() && !name.to_lowercase().contains(&filter_lower) {
                    continue;
                }

                let is_selected = browser_state.selected_clip_id == Some(*id);
                let ref_count = reference_counts
                    .iter()
                    .find(|(sid, _)| *sid == *id)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);

                let clip = clip_library.get(*id);
                let duration = clip.map(|c| c.duration).unwrap_or(0.0);
                let source_filename = extract_source_filename(clip);

                let label = if source_filename.is_empty() {
                    format!("{} ({:.1}s) [{}]##clip_{}", name, duration, ref_count, id)
                } else {
                    format!(
                        "{} ({:.1}s) [{}] <{}>##clip_{}",
                        name, duration, ref_count, source_filename, id
                    )
                };

                if ui.selectable_config(&label).selected(is_selected).build() {
                    browser_state.selected_clip_id = Some(*id);
                }

                build_clip_drag_source(ui, *id, name);
            }
        });
}

fn build_clip_drag_source(ui: &imgui::Ui, clip_id: SourceClipId, _name: &str) {
    let id = clip_id;
    let _source = ui
        .drag_drop_source_config("CLIP_SOURCE")
        .begin_payload(move || id);
}

fn extract_source_filename(
    clip: Option<&crate::animation::editable::EditableAnimationClip>,
) -> String {
    clip.and_then(|c| c.source_path.as_ref())
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string()
}
